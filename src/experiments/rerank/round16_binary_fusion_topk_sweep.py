import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

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
DEFAULT_SOURCES = [
    ("binary", "outputs/round16/top3_binary_selector_e3_k20_n80/dev_binary_selector_ranked.json"),
    ("r15", "outputs/round15/rrf_top3_selector/ranked.json"),
    ("a_fixed", "outputs/round16/branch_a_requirement/fixed_rrf_claim_key_ranked_top4500.json"),
    ("s22q", "outputs/round16/baseline/s22_small_qprefix_top100.json"),
]


def load_json_file(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def normalize_ranked(data):
    ranked = {}
    for claim_id, value in data.items():
        rows = []
        if isinstance(value, dict) and "evidences" in value:
            rows = [
                {"evidence_id": evidence_id, "rank": rank}
                for rank, evidence_id in enumerate(value.get("evidences", []), start=1)
            ]
        elif isinstance(value, list):
            for rank, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    evidence_id = item.get("evidence_id")
                    item_rank = int(item.get("rank", rank))
                else:
                    evidence_id = str(item)
                    item_rank = rank
                if evidence_id:
                    rows.append({"evidence_id": evidence_id, "rank": item_rank})
        rows.sort(key=lambda row: (row["rank"], row["evidence_id"]))
        seen = set()
        unique = []
        for row in rows:
            if row["evidence_id"] in seen:
                continue
            seen.add(row["evidence_id"])
            unique.append(row)
        ranked[claim_id] = unique
    return ranked


def parse_source(spec):
    parts = spec.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Source must be name,path: {spec}")
    return parts[0], parts[1]


def fuse(claims, sources, weights, rrf_k, cap):
    fused = {}
    for claim_id in claims:
        scores = {}
        for name, source in sources.items():
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue
            for idx, item in enumerate(source.get(claim_id, [])[:cap], start=1):
                evidence_id = item["evidence_id"]
                rank = int(item.get("rank", idx))
                scores[evidence_id] = scores.get(evidence_id, 0.0) + weight / (rrf_k + rank)
        rows = [
            {"evidence_id": evidence_id, "score": float(score)}
            for evidence_id, score in scores.items()
        ]
        rows.sort(key=lambda row: (-row["score"], row["evidence_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        fused[claim_id] = rows
    return fused


def predictions_from_ranked(claims, ranked, label_source, default_label, top_k):
    return {
        claim_id: {
            "claim_text": claim["claim_text"],
            "claim_label": label_source.get(claim_id, {}).get("claim_label", default_label),
            "evidences": [
                item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]
            ],
        }
        for claim_id, claim in claims.items()
    }


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
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Sweep RRF weights for binary selector fusion at top-k.")
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--label-source", default="outputs/round15/recommended/top3_submission.json")
    parser.add_argument("--source", action="append", help="Optional name,path ranked source. Defaults to Round16 four-source fusion.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/round16/top10_binary_selector")
    parser.add_argument("--weight-values", default="0,0.5,1,2,3")
    parser.add_argument("--rrf-k-values", default="10,20,60")
    parser.add_argument("--cap-values", default="20,50,100")
    args = parser.parse_args()

    claims = load_json(args.claims)
    train_claims = load_json(args.train_claims)
    default_label = majority_label(train_claims)
    label_source = load_json_file(args.label_source)
    specs = [parse_source(item) for item in args.source] if args.source else DEFAULT_SOURCES
    sources = {
        name: normalize_ranked(load_json_file(path))
        for name, path in specs
        if Path(path).exists()
    }
    names = list(sources)
    weight_values = [float(value) for value in args.weight_values.split(",") if value]
    rrf_k_values = [float(value) for value in args.rrf_k_values.split(",") if value]
    cap_values = [int(value) for value in args.cap_values.split(",") if value]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    for cap in cap_values:
        for rrf_k in rrf_k_values:
            for values in product(weight_values, repeat=len(names)):
                weights = dict(zip(names, values))
                if sum(weights.values()) <= 0:
                    continue
                ranked = fuse(claims, sources, weights, rrf_k, cap)
                predictions = predictions_from_ranked(
                    claims, ranked, label_source, default_label, args.top_k
                )
                row = metric_row("binary_fusion", claims, predictions)
                row.update(
                    {
                        "top_k": args.top_k,
                        "rrf_k": rrf_k,
                        "cap": cap,
                        "weights": "|".join(
                            f"{name}:{weight:g}"
                            for name, weight in weights.items()
                            if weight > 0
                        ),
                    }
                )
                rows.append(row)
                key = (row["macro_recall"], row["retrieval_f_score"])
                if best is None or key > best[0]:
                    best = (key, row, predictions, ranked)
    rows.sort(key=lambda row: (-row["macro_recall"], -row["retrieval_f_score"]))
    write_rows(rows, output_dir / f"top{args.top_k}_fusion_sweep.csv")
    write_predictions(best[2], output_dir / f"best_top{args.top_k}.json")
    (output_dir / f"best_top{args.top_k}_ranked.json").write_text(
        json.dumps(best[3], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"best_top{args.top_k}_summary.json").write_text(
        json.dumps(best[1], indent=2),
        encoding="utf-8",
    )
    print(json.dumps(best[1], indent=2))


if __name__ == "__main__":
    main()
