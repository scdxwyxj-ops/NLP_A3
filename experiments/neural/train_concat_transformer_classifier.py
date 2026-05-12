import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from a3_factcheck.evaluation import run_eval
from a3_factcheck.rerank.api import write_predictions


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_to_text(row, evidence_top_k, evidence_token_budget):
    parts = [f"CLAIM: {row['claim_text']}"]
    for item in row.get("classifier_evidence_context", [])[:evidence_top_k]:
        text = " ".join(item.get("text", "").split()[:evidence_token_budget])
        parts.append(f"EVIDENCE: {text}")
    return "\n".join(parts)


class ClaimDataset(Dataset):
    def __init__(self, rows, evidence_top_k, evidence_token_budget):
        self.rows = rows
        self.evidence_top_k = evidence_top_k
        self.evidence_token_budget = evidence_token_budget

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (
            row_to_text(row, self.evidence_top_k, self.evidence_token_budget),
            LABEL_TO_ID[row["claim_label"]],
        )


def collate_batch(batch, tokenizer, max_length):
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded


def evaluate(model, tokenizer, rows, device, batch_size, max_length, evidence_top_k, evidence_token_budget):
    loader = DataLoader(
        ClaimDataset(rows, evidence_top_k, evidence_token_budget),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, max_length),
    )
    labels = []
    predictions = []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            gold = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, labels=gold)
            total_loss += float(outputs.loss.detach().cpu())
            pred = torch.argmax(outputs.logits, dim=-1)
            labels.extend(gold.detach().cpu().tolist())
            predictions.extend(pred.detach().cpu().tolist())
    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=list(range(len(LABELS))), average="macro"),
        "labels": labels,
        "predictions": predictions,
    }


def build_prediction_json(rows, predicted_ids):
    predictions = {}
    for row, label_id in zip(rows, predicted_ids):
        predictions[row["claim_id"]] = {
            "claim_text": row["claim_text"],
            "claim_label": LABELS[label_id],
            "evidences": row.get("final_evidence_candidates", []),
        }
    return predictions


def write_confusion(labels, predictions, output_path):
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(LABELS))))
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row.tolist()])


def main():
    parser = argparse.ArgumentParser(description="Train transformer concat claim classifier.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--dev-groundtruth", default="data/dev-claims.json")
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--output-dir", default="outputs/round10/claim_classifier")
    parser.add_argument("--name", default="distilroberta_top10")
    parser.add_argument("--evidence-top-k", type=int, default=10)
    parser.add_argument("--evidence-token-budget", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    seed_everything(args.seed)
    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    output_dir = Path(args.output_dir) / args.name
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
        ClaimDataset(train_rows, args.evidence_top_k, args.evidence_token_budget),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, args.max_length),
    )

    curve = []
    dev_eval = None
    best_eval = None
    best_epoch = None
    best_state = None
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
        dev_eval = evaluate(
            model,
            tokenizer,
            dev_rows,
            device,
            args.batch_size,
            args.max_length,
            args.evidence_top_k,
            args.evidence_token_budget,
        )
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_eval["loss"],
                "dev_accuracy": dev_eval["accuracy"],
                "dev_macro_f1": dev_eval["macro_f1"],
            }
        )
        if best_eval is None or (
            dev_eval["accuracy"],
            dev_eval["macro_f1"],
            -dev_eval["loss"],
        ) > (
            best_eval["accuracy"],
            best_eval["macro_f1"],
            -best_eval["loss"],
        ):
            best_eval = {
                key: (value.copy() if isinstance(value, list) else value)
                for key, value in dev_eval.items()
            }
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(curve[-1], indent=2))

    if best_state is not None:
        model.load_state_dict(best_state)
        model.save_pretrained(output_dir / "best_model")
        tokenizer.save_pretrained(output_dir / "best_model")

    prediction_json = build_prediction_json(dev_rows, best_eval["predictions"])
    predictions_path = output_dir / "dev_predictions.json"
    write_predictions(prediction_json, predictions_path)
    assignment_eval = run_eval(args.eval_script, predictions_path, args.dev_groundtruth)
    metrics = {
        "name": args.name,
        "model": args.model,
        "evidence_top_k": args.evidence_top_k,
        "evidence_token_budget": args.evidence_token_budget,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "accuracy": best_eval["accuracy"],
        "macro_f1": best_eval["macro_f1"],
        "assignment_eval": assignment_eval,
        "learning_curve": curve,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_confusion(best_eval["labels"], best_eval["predictions"], output_dir / "confusion_matrix.csv")
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            best_eval["labels"],
            best_eval["predictions"],
            target_names=LABELS,
            labels=list(range(len(LABELS))),
            zero_division=0,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
