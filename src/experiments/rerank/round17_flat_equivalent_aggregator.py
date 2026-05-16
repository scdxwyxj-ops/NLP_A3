import argparse
import csv
import json
import sys
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
                    if not evidence_id:
                        continue
                    score = item.get("score", item.get("fusion_score", item.get("sparse_score", item.get("rrf_score", 0.0))))
                    row = dict(item)
                    row["evidence_id"] = evidence_id
                    row["rank"] = int(item.get("rank", rank))
                    row["score"] = float(score)
                    rows.append(row)
                else:
                    rows.append({"evidence_id": str(item), "rank": rank, "score": 1.0 / rank})
        seen = set()
        unique = []
        for row in sorted(rows, key=lambda item: (item["rank"], -item.get("score", 0.0), item["evidence_id"])):
            if row["evidence_id"] in seen:
                continue
            seen.add(row["evidence_id"])
            unique.append(row)
        for rank, row in enumerate(unique, start=1):
            row["rank"] = rank
        ranked[claim_id] = unique
    return ranked


def index_source(ranked, cap):
    return {
        claim_id: {row["evidence_id"]: row for row in rows[:cap]}
        for claim_id, rows in ranked.items()
    }


def add_rank_features(prefix, source_item, out):
    if not source_item:
        out[f"{prefix}_present"] = 0
        out[f"{prefix}_rank"] = 999999
        out[f"{prefix}_rr"] = 0.0
        out[f"{prefix}_score"] = 0.0
        return
    rank = int(source_item.get("rank", 999999))
    out[f"{prefix}_present"] = 1
    out[f"{prefix}_rank"] = rank
    out[f"{prefix}_rr"] = 1.0 / (20.0 + rank)
    out[f"{prefix}_score"] = float(source_item.get("score", 0.0))


def build_flat_ranked(claims, binary, r15, a_fixed, cap):
    binary_idx = index_source(binary, cap)
    r15_idx = index_source(r15, cap)
    a_fixed_idx = index_source(a_fixed, cap)
    ranked = {}
    feature_rows = []
    for claim_id in claims:
        evidence_ids = set()
        for source in [binary, r15, a_fixed]:
            evidence_ids.update(row["evidence_id"] for row in source.get(claim_id, [])[:cap])
        rows = []
        for evidence_id in evidence_ids:
            row = {"claim_id": claim_id, "evidence_id": evidence_id}
            binary_item = binary_idx.get(claim_id, {}).get(evidence_id)
            r15_item = r15_idx.get(claim_id, {}).get(evidence_id)
            a_fixed_item = a_fixed_idx.get(claim_id, {}).get(evidence_id)

            add_rank_features("binary_selector", binary_item, row)
            add_rank_features("round15_selector", r15_item, row)
            add_rank_features("claim_key_selector", a_fixed_item, row)

            # Expose the nested ingredients as flat diagnostic columns. They are no longer
            # used as nested aggregators here; the final score is produced once below.
            if r15_item:
                for key in [
                    "old50_a07_rank",
                    "old50_a07_score",
                    "old100_a08_rank",
                    "old100_a08_score",
                    "gbdt_w2_rank",
                    "gbdt_w2_score",
                ]:
                    if key in r15_item:
                        row[f"round15_{key}"] = r15_item[key]
            if a_fixed_item:
                for key in ["ranker_rank", "top100_rank", "top3_rank", "gate_rank"]:
                    if key in a_fixed_item:
                        row[f"claim_key_{key}"] = a_fixed_item[key]

            row["final_score"] = (
                2.0 * row["binary_selector_rr"]
                + 0.5 * row["round15_selector_rr"]
                + 0.5 * row["claim_key_selector_rr"]
            )
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "score": float(row["final_score"]),
                    "binary_rank": row["binary_selector_rank"],
                    "r15_rank": row["round15_selector_rank"],
                    "a_fixed_rank": row["claim_key_selector_rank"],
                }
            )
            feature_rows.append(row)
        rows.sort(key=lambda item: (-item["score"], item["evidence_id"]))
        for rank, item in enumerate(rows, start=1):
            item["rank"] = rank
        ranked[claim_id] = rows
    return ranked, feature_rows


def predictions_from_ranked(claims, ranked, label_source, default_label, top_k):
    return {
        claim_id: {
            "claim_text": claim["claim_text"],
            "claim_label": label_source.get(claim_id, {}).get("claim_label", default_label),
            "evidences": [row["evidence_id"] for row in ranked.get(claim_id, [])[:top_k]],
        }
        for claim_id, claim in claims.items()
    }


def metric_row(name, claims, predictions, note):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "note": note,
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
        row[f"{label.lower()}_macro_recall"] = label_subset_metrics(claims, predictions, label)["macro_recall"]
    return row


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows, path):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Flat equivalent single-aggregator version of Round16 best top3.")
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--label-source", default="outputs/round15/recommended/top3_submission.json")
    parser.add_argument("--binary-source", default="outputs/round16/top3_binary_selector_e3_k20_n80/dev_binary_selector_ranked.json")
    parser.add_argument("--round15-source", default="outputs/round15/rrf_top3_selector/ranked.json")
    parser.add_argument("--claim-key-source", default="outputs/round16/branch_a_requirement/fixed_rrf_claim_key_ranked_top4500.json")
    parser.add_argument("--output-dir", default="outputs/round17/flat_equivalent_aggregator")
    parser.add_argument("--cap", type=int, default=20)
    args = parser.parse_args()

    claims = load_json(args.claims)
    train_claims = load_json(args.train_claims)
    label_source = load_any(args.label_source)
    default_label = majority_label(train_claims)
    binary = normalize_ranked(load_any(args.binary_source))
    r15 = normalize_ranked(load_any(args.round15_source))
    a_fixed = normalize_ranked(load_any(args.claim_key_source))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked, feature_rows = build_flat_ranked(claims, binary, r15, a_fixed, args.cap)
    predictions = predictions_from_ranked(claims, ranked, label_source, default_label, top_k=3)
    row = metric_row(
        "flat_equivalent_single_aggregator",
        claims,
        predictions,
        "single flat weighted reciprocal-rank aggregator over binary, Round15, and claim-key selector factors",
    )
    write_json(ranked, output_dir / "flat_equivalent_ranked.json")
    write_predictions(predictions, output_dir / "flat_equivalent_top3.json")
    write_csv(feature_rows, output_dir / "flat_feature_rows.csv")
    write_json({"summary": [row]}, output_dir / "summary.json")
    write_csv([row], output_dir / "summary.csv")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
