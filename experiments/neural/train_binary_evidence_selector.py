import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from a3_factcheck.data import load_json, majority_label  # noqa: E402
from a3_factcheck.metrics import (  # noqa: E402
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)
from a3_factcheck.rerank.api import write_predictions  # noqa: E402


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


TRAIN_SOURCES = [
    ("r09_minilm", "outputs/round09/train-rrf-bm25-char-minilm-ranked-top100.json"),
    ("r09_gbdt", "outputs/round09/fusion_gbdt_top100/train-fusion-ranked-top50.json"),
    ("r09_gbdt_ref2", "outputs/round09/fusion_gbdt_top100_refutes_x2/train-fusion-ranked-top50.json"),
    ("r09_gbdt_ref3", "outputs/round09/fusion_gbdt_top100_refutes_x3/train-fusion-ranked-top50.json"),
    ("r15_gbdt_w1", "outputs/round15/fusion_gbdt_round12_minilm_w1/train-fusion-ranked-top50.json"),
    ("r15_gbdt_w2", "outputs/round15/fusion_gbdt_round12_minilm_w2/train-fusion-ranked-top50.json"),
    ("r15_gbdt_w3", "outputs/round15/fusion_gbdt_round12_minilm_w3/train-fusion-ranked-top50.json"),
    ("r15_logreg_w2", "outputs/round15/fusion_logreg_round12_minilm_w2/train-fusion-ranked-top50.json"),
    ("r14_small_rrf", "outputs/round14/s0_small_only_train_rrf/candidates/rrf_bm25_char_small_smallq_k500_top2000.json"),
]


DEV_SOURCES = [
    ("r15_top3", "outputs/round15/rrf_top3_selector/ranked.json"),
    ("r15_minilm", "outputs/round15/dev-round12-top500-minilm-ranked-top100.json"),
    ("r09_alpha04", "outputs/round09/blend_top100_refutes_x2/alpha_0.4_ranked.json"),
    ("r09_alpha07", "outputs/round09/blend_top50/alpha_0.7_ranked.json"),
    ("r09_alpha08", "outputs/round09/blend_top100_refutes_x2/alpha_0.8_ranked.json"),
    ("r09_minilm", "outputs/round09/dev-rrf-bm25-char-minilm-ranked-top100.json"),
    ("r09_gbdt_ref2", "outputs/round09/fusion_gbdt_top100_refutes_x2/dev-fusion-ranked-top50.json"),
    ("r10_hybrid02", "outputs/round10/hybrid_verifier_e5/gamma_0.2_ranked.json"),
    ("r11_dense_bge_minilm", "outputs/round11/dev-rrf-bm25-char-dense-bge-top1000-minilm-ranked-top100.json"),
    ("r16_s22_rrf", "outputs/round16/baseline/s22_rrf_sparsegate_small_top100.json"),
    ("r16_s22_qprefix", "outputs/round16/baseline/s22_small_qprefix_top100.json"),
    ("r16_a_fixed", "outputs/round16/branch_a_requirement/fixed_rrf_claim_key_ranked_top4500.json"),
]


def load_any(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def normalize_ranked(data):
    ranked = {}
    for claim_id, value in data.items():
        rows = []
        if isinstance(value, dict) and "evidences" in value:
            rows = [
                {"evidence_id": evidence_id, "rank": rank, "score": 1.0 / rank}
                for rank, evidence_id in enumerate(value.get("evidences", []), start=1)
            ]
        elif isinstance(value, list):
            for rank, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    evidence_id = item.get("evidence_id")
                    item_rank = int(item.get("rank", rank))
                    score = float(item.get("score", item.get("fusion_score", 0.0)))
                else:
                    evidence_id = str(item)
                    item_rank = rank
                    score = 1.0 / rank
                if evidence_id:
                    rows.append({"evidence_id": evidence_id, "rank": item_rank, "score": score})
        rows.sort(key=lambda row: (row["rank"], -row["score"], row["evidence_id"]))
        seen = set()
        unique = []
        for row in rows:
            if row["evidence_id"] in seen:
                continue
            seen.add(row["evidence_id"])
            unique.append(row)
        ranked[claim_id] = unique
    return ranked


def load_sources(specs):
    sources = {}
    for name, path in specs:
        if Path(path).exists():
            sources[name] = normalize_ranked(load_any(path))
    return sources


def collect_candidates(claims, sources, per_source_k, include_gold):
    candidates = {}
    source_rank_features = {}
    for claim_id, claim in claims.items():
        seen = set()
        rows = []
        rank_features = {}
        for source_name, source in sources.items():
            for rank, item in enumerate(source.get(claim_id, [])[:per_source_k], start=1):
                evidence_id = item["evidence_id"]
                rank_features.setdefault(evidence_id, {})[f"{source_name}_rank"] = rank
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    rows.append(evidence_id)
        if include_gold:
            for evidence_id in claim.get("evidences", []):
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    rows.append(evidence_id)
        candidates[claim_id] = rows
        source_rank_features[claim_id] = rank_features
    return candidates, source_rank_features


class EvidenceDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(batch, tokenizer, max_length):
    encoded = tokenizer(
        [row["claim_text"] for row in batch],
        [row["evidence_text"] for row in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
    return encoded


def make_train_rows(claims, evidence, candidates, max_negatives, seed):
    rng = random.Random(seed)
    rows = []
    positives = 0
    negatives = 0
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        negs = [evidence_id for evidence_id in candidates[claim_id] if evidence_id not in gold]
        if len(negs) > max_negatives:
            # Keep front-ranked negatives and sample a small tail.
            head = negs[: max_negatives // 2]
            tail = negs[max_negatives // 2 :]
            sampled_tail = rng.sample(tail, min(len(tail), max_negatives - len(head)))
            negs = head + sampled_tail
        for evidence_id in claim.get("evidences", []):
            if evidence_id in evidence:
                rows.append(
                    {
                        "claim_id": claim_id,
                        "evidence_id": evidence_id,
                        "claim_text": claim["claim_text"],
                        "evidence_text": evidence[evidence_id],
                        "label": 1,
                    }
                )
                positives += 1
        for evidence_id in negs:
            if evidence_id in evidence:
                rows.append(
                    {
                        "claim_id": claim_id,
                        "evidence_id": evidence_id,
                        "claim_text": claim["claim_text"],
                        "evidence_text": evidence[evidence_id],
                        "label": 0,
                    }
                )
                negatives += 1
    rng.shuffle(rows)
    return rows, positives, negatives


def score_candidates(model, tokenizer, claims, evidence, candidates, device, batch_size, max_length):
    pairs = []
    for claim_id, claim in claims.items():
        for evidence_id in candidates[claim_id]:
            if evidence_id in evidence:
                pairs.append(
                    {
                        "claim_id": claim_id,
                        "evidence_id": evidence_id,
                        "claim_text": claim["claim_text"],
                        "evidence_text": evidence[evidence_id],
                        "label": 0,
                    }
                )
    loader = DataLoader(
        EvidenceDataset(pairs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer, max_length),
    )
    scored = defaultdict(list)
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            probs = torch.softmax(model(**batch).logits, dim=-1)[:, 1].detach().cpu().tolist()
            chunk = pairs[offset : offset + len(probs)]
            offset += len(probs)
            for row, prob in zip(chunk, probs):
                scored[row["claim_id"]].append(
                    {
                        "evidence_id": row["evidence_id"],
                        "score": float(prob),
                    }
                )
    for claim_id, rows in scored.items():
        rows.sort(key=lambda row: (-row["score"], row["evidence_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    return dict(scored)


def rrf_fuse(sources, weights, rrf_k, cap):
    claim_ids = set()
    for source in sources.values():
        claim_ids.update(source)
    fused = {}
    for claim_id in claim_ids:
        scores = defaultdict(float)
        parts = defaultdict(dict)
        for name, source in sources.items():
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue
            for idx, item in enumerate(source.get(claim_id, [])[:cap], start=1):
                evidence_id = item["evidence_id"]
                rank = int(item.get("rank", idx))
                scores[evidence_id] += weight / (rrf_k + rank)
                parts[evidence_id][f"{name}_rank"] = rank
        rows = [
            {"evidence_id": evidence_id, "score": float(score), **parts[evidence_id]}
            for evidence_id, score in scores.items()
        ]
        rows.sort(key=lambda row: (-row["score"], row["evidence_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        fused[claim_id] = rows
    return fused


def predictions_from_ranked(claims, ranked, label_source, default_label, top_k):
    predictions = {}
    for claim_id, claim in claims.items():
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": label_source.get(claim_id, {}).get("claim_label", default_label),
            "evidences": [
                item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]
            ],
        }
    return predictions


def metric_row(name, claims, predictions):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "macro_recall": macro_recall(claims, predictions),
        "retrieval_f_score": assignment["retrieval_f_score"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
        "micro_recall": aggregate["micro_recall"],
        "precision": aggregate["precision"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
    }
    for label in LABELS:
        row[f"{label.lower()}_macro_recall"] = label_subset_metrics(
            claims, predictions, label
        )["macro_recall"]
    return row


def write_rows(rows, path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train binary gold-evidence selector.")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="outputs/round16/top3_binary_selector")
    parser.add_argument("--train-source-k", type=int, default=30)
    parser.add_argument("--dev-source-k", type=int, default=20)
    parser.add_argument("--max-negatives", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1601)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    default_label = majority_label(train_claims)
    label_source = load_any("outputs/round15/recommended/top3_submission.json")

    train_sources = load_sources(TRAIN_SOURCES)
    dev_sources = load_sources(DEV_SOURCES)
    train_candidates, _ = collect_candidates(
        train_claims, train_sources, args.train_source_k, include_gold=True
    )
    dev_candidates, _ = collect_candidates(
        dev_claims, dev_sources, args.dev_source_k, include_gold=False
    )
    train_rows, positive_rows, negative_rows = make_train_rows(
        train_claims, evidence, train_candidates, args.max_negatives, args.seed
    )
    (output_dir / "train_stats.json").write_text(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "positive_rows": positive_rows,
                "negative_rows": negative_rows,
                "train_sources": list(train_sources),
                "dev_sources": list(dev_sources),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(
        EvidenceDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer, args.max_length),
    )

    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.detach().cpu())
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(loader))}
        curve.append(row)
        print(json.dumps(row))

    neural_ranked = score_candidates(
        model,
        tokenizer,
        dev_claims,
        evidence,
        dev_candidates,
        device,
        args.batch_size,
        args.max_length,
    )
    (output_dir / "dev_binary_selector_ranked.json").write_text(
        json.dumps(neural_ranked, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = []
    neural_top3 = predictions_from_ranked(dev_claims, neural_ranked, label_source, default_label, 3)
    write_predictions(neural_top3, output_dir / "dev_binary_selector_top3.json")
    summary.append(metric_row("binary_selector_top3", dev_claims, neural_top3))

    fusion_sources = {
        "binary": neural_ranked,
        "r15": dev_sources["r15_top3"],
        "s22q": dev_sources["r16_s22_qprefix"],
        "a_fixed": dev_sources["r16_a_fixed"],
    }
    best = None
    fusion_rows = []
    for rrf_k in [1.0, 5.0, 10.0, 20.0, 60.0]:
        for wb in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            for wr15 in [0.5, 1.0, 1.5, 2.0]:
                for ws22 in [0.0, 0.5, 1.0, 1.5]:
                    for wa in [0.0, 0.5, 1.0, 1.5]:
                        weights = {"binary": wb, "r15": wr15, "s22q": ws22, "a_fixed": wa}
                        ranked = rrf_fuse(fusion_sources, weights, rrf_k, cap=20)
                        pred = predictions_from_ranked(dev_claims, ranked, label_source, default_label, 3)
                        row = metric_row("binary_rrf_fusion", dev_claims, pred)
                        row.update(
                            {
                                "rrf_k": rrf_k,
                                "weights": "|".join(f"{k}:{v:g}" for k, v in weights.items() if v > 0),
                            }
                        )
                        fusion_rows.append(row)
                        key = (row["macro_recall"], row["retrieval_f_score"])
                        if best is None or key > best[0]:
                            best = (key, row, pred, ranked)
    fusion_rows.sort(key=lambda row: (-row["macro_recall"], -row["retrieval_f_score"]))
    write_rows(fusion_rows[:100], output_dir / "fusion_sweep_top100.csv")
    best_row = {**best[1], "name": "best_binary_rrf_fusion"}
    summary.append(best_row)
    write_predictions(best[2], output_dir / "best_binary_rrf_fusion_top3.json")
    (output_dir / "best_binary_rrf_fusion_ranked.json").write_text(
        json.dumps(best[3], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_rows(summary, output_dir / "summary.csv")
    (output_dir / "summary.json").write_text(
        json.dumps({"curve": curve, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
