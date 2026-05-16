import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.metrics import assignment_metrics, label_subset_metrics, macro_recall
from a3_factcheck.rerank.api import write_predictions


LABELS = ["SUPPORT", "REFUTE", "NEUTRAL"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_ranked(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return row["claim"], row["evidence"], LABEL_TO_ID[row["label"]]


def collate_batch(batch, tokenizer, max_length):
    claims, evidences, labels = zip(*batch)
    encoded = tokenizer(
        list(claims),
        list(evidences),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded


def evaluate_pairs(model, tokenizer, rows, device, batch_size, max_length):
    loader = DataLoader(
        PairDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, max_length),
    )
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, labels=labels)
            total_loss += float(outputs.loss.detach().cpu())
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro"),
        "report": classification_report(
            all_labels,
            all_predictions,
            target_names=LABELS,
            labels=list(range(len(LABELS))),
            zero_division=0,
        ),
    }


def score_ranked(model, tokenizer, claims, evidence, ranked, device, batch_size, max_length, top_k):
    model.eval()
    scored = {}
    all_pairs = []
    for claim_id, claim in claims.items():
        for item in ranked.get(claim_id, [])[:top_k]:
            all_pairs.append((claim_id, claim["claim_text"], item["evidence_id"], evidence[item["evidence_id"]]))

    for start in range(0, len(all_pairs), batch_size):
        chunk = all_pairs[start : start + batch_size]
        encoded = tokenizer(
            [item[1] for item in chunk],
            [item[3] for item in chunk],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=-1).detach().cpu().numpy()
        for (claim_id, _, evidence_id, _), prob in zip(chunk, probs):
            p_support, p_refute, p_neutral = prob.tolist()
            score = max(p_support, p_refute)
            scored.setdefault(claim_id, []).append(
                {
                    "evidence_id": evidence_id,
                    "score": float(score),
                    "p_support": float(p_support),
                    "p_refute": float(p_refute),
                    "p_neutral": float(p_neutral),
                }
            )

    for claim_id, items in scored.items():
        items.sort(key=lambda item: (-item["score"], item["evidence_id"]))
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
    return scored


def predictions_from_scored(claims, train_claims, scored, top_k):
    default_label = majority_label(train_claims)
    predictions = {}
    for claim_id, claim in claims.items():
        items = scored.get(claim_id, [])
        max_support = max([item["p_support"] for item in items[:top_k]], default=0.0)
        max_refute = max([item["p_refute"] for item in items[:top_k]], default=0.0)
        if max_support > 0.5 and max_refute > 0.5:
            label = "DISPUTED"
        elif max_support >= max_refute and max_support > 0.4:
            label = "SUPPORTS"
        elif max_refute > max_support and max_refute > 0.4:
            label = "REFUTES"
        else:
            label = default_label
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": label,
            "evidences": [item["evidence_id"] for item in items[:top_k]],
        }
    return predictions


def write_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train a neural SUPPORT/REFUTE/NEUTRAL verifier.")
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--dev-pairs", required=True)
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--dev-ranked", required=True)
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--output-dir", default="outputs/round10/verifier")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--score-top-k", type=int, default=50)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_pairs)
    dev_rows = read_jsonl(args.dev_pairs)
    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    ranked = load_ranked(args.dev_ranked)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, args.max_length),
    )

    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            train_loss += float(outputs.loss.detach().cpu())
        train_loss /= max(1, len(train_loader))
        pair_metrics = evaluate_pairs(
            model, tokenizer, dev_rows, device, args.batch_size, args.max_length
        )
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": pair_metrics["loss"],
                "dev_accuracy": pair_metrics["accuracy"],
                "dev_macro_f1": pair_metrics["macro_f1"],
            }
        )
        print(json.dumps(curve[-1], indent=2))

    scored = score_ranked(
        model=model,
        tokenizer=tokenizer,
        claims=dev_claims,
        evidence=evidence,
        ranked=ranked,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_k=args.score_top_k,
    )
    write_json(scored, output_dir / "dev-verifier-ranked.json")

    summary_rows = []
    for top_k in [3, 20, 50]:
        predictions = predictions_from_scored(dev_claims, train_claims, scored, top_k)
        prediction_path = output_dir / f"dev-verifier-top{top_k}.json"
        write_predictions(predictions, prediction_path)
        metrics = assignment_metrics(dev_claims, predictions)
        metrics["macro_recall"] = macro_recall(dev_claims, predictions)
        metrics["refutes_recall"] = label_subset_metrics(dev_claims, predictions, "REFUTES")[
            "macro_recall"
        ]
        metrics["top_k"] = top_k
        summary_rows.append(metrics)
    with (output_dir / "verifier_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    write_json({"learning_curve": curve, "pair_report": pair_metrics["report"]}, output_dir / "metrics.json")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
