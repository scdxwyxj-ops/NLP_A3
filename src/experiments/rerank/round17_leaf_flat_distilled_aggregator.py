import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor

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


LEAF_RANK_FEATURES = [
    "binary_rank",
    "round15_old50_a07_rank",
    "round15_old100_a08_rank",
    "round15_gbdt_w2_rank",
    "claim_key_ranker_rank",
    "claim_key_top100_rank",
    "claim_key_top3_rank",
    "claim_key_gate_rank",
]

LEAF_SCORE_FEATURES = [
    "binary_score",
    "round15_old50_a07_score",
    "round15_old100_a08_score",
    "round15_gbdt_w2_score",
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
                    if not evidence_id:
                        continue
                    row = dict(item)
                    row["evidence_id"] = evidence_id
                    row["rank"] = int(item.get("rank", rank))
                    row["score"] = float(item.get("score", item.get("fusion_score", item.get("sparse_score", 0.0))))
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
        ranked[claim_id] = unique
    return ranked


def index_by_evidence(ranked):
    return {
        claim_id: {row["evidence_id"]: row for row in rows}
        for claim_id, rows in ranked.items()
    }


def add_rank_feature(row, name, value):
    if value is None:
        row[name] = 999999.0
        row[name.replace("_rank", "_present")] = 0.0
        row[name.replace("_rank", "_rr20")] = 0.0
        row[name.replace("_rank", "_rr45")] = 0.0
        row[name.replace("_rank", "_rr60")] = 0.0
        return
    rank = float(value)
    row[name] = rank
    row[name.replace("_rank", "_present")] = 1.0
    row[name.replace("_rank", "_rr20")] = 1.0 / (20.0 + rank)
    row[name.replace("_rank", "_rr45")] = 1.0 / (45.0 + rank)
    row[name.replace("_rank", "_rr60")] = 1.0 / (60.0 + rank)


def row_to_vector(row):
    values = []
    for name in LEAF_RANK_FEATURES:
        values.append(float(row[name]))
        values.append(float(row[name.replace("_rank", "_present")]))
        values.append(float(row[name.replace("_rank", "_rr20")]))
        values.append(float(row[name.replace("_rank", "_rr45")]))
        values.append(float(row[name.replace("_rank", "_rr60")]))
    for name in LEAF_SCORE_FEATURES:
        values.append(float(row.get(name, 0.0)))
    values.extend(
        [
            float(row["leaf_present_count"]),
            float(row["leaf_rr20_sum"]),
            float(row["leaf_rr45_sum"]),
            float(row["leaf_rr60_sum"]),
            float(row["leaf_min_rank"]),
            float(row["leaf_mean_rank"]),
        ]
    )
    return values


def leaf_feature_row(claim_id, evidence_id, binary_idx, r15_idx, claim_key_idx):
    row = {"claim_id": claim_id, "evidence_id": evidence_id}
    binary = binary_idx.get(claim_id, {}).get(evidence_id)
    r15 = r15_idx.get(claim_id, {}).get(evidence_id)
    claim_key = claim_key_idx.get(claim_id, {}).get(evidence_id)

    add_rank_feature(row, "binary_rank", binary.get("rank") if binary else None)
    row["binary_score"] = float(binary.get("score", 0.0)) if binary else 0.0

    add_rank_feature(row, "round15_old50_a07_rank", r15.get("old50_a07_rank") if r15 else None)
    row["round15_old50_a07_score"] = float(r15.get("old50_a07_score", 0.0)) if r15 else 0.0
    add_rank_feature(row, "round15_old100_a08_rank", r15.get("old100_a08_rank") if r15 else None)
    row["round15_old100_a08_score"] = float(r15.get("old100_a08_score", 0.0)) if r15 else 0.0
    add_rank_feature(row, "round15_gbdt_w2_rank", r15.get("gbdt_w2_rank") if r15 else None)
    row["round15_gbdt_w2_score"] = float(r15.get("gbdt_w2_score", 0.0)) if r15 else 0.0

    add_rank_feature(row, "claim_key_ranker_rank", claim_key.get("ranker_rank") if claim_key else None)
    add_rank_feature(row, "claim_key_top100_rank", claim_key.get("top100_rank") if claim_key else None)
    add_rank_feature(row, "claim_key_top3_rank", claim_key.get("top3_rank") if claim_key else None)
    add_rank_feature(row, "claim_key_gate_rank", claim_key.get("gate_rank") if claim_key else None)

    ranks = [row[name] for name in LEAF_RANK_FEATURES if row[name] < 999999.0]
    row["leaf_present_count"] = sum(row[name.replace("_rank", "_present")] for name in LEAF_RANK_FEATURES)
    row["leaf_rr20_sum"] = sum(row[name.replace("_rank", "_rr20")] for name in LEAF_RANK_FEATURES)
    row["leaf_rr45_sum"] = sum(row[name.replace("_rank", "_rr45")] for name in LEAF_RANK_FEATURES)
    row["leaf_rr60_sum"] = sum(row[name.replace("_rank", "_rr60")] for name in LEAF_RANK_FEATURES)
    row["leaf_min_rank"] = min(ranks) if ranks else 999999.0
    row["leaf_mean_rank"] = float(np.mean(ranks)) if ranks else 999999.0
    return row


def build_dataset(claims, binary, r15, claim_key, teacher_ranked, cap):
    binary_idx = index_by_evidence(binary)
    r15_idx = index_by_evidence(r15)
    claim_key_idx = index_by_evidence(claim_key)
    teacher_idx = index_by_evidence(teacher_ranked)
    feature_rows = []
    x_rows = []
    y_rows = []
    groups = []
    row_ids = []
    for claim_id in claims:
        evidence_ids = set()
        for rows in [binary.get(claim_id, [])[:cap], r15.get(claim_id, [])[:cap], claim_key.get(claim_id, [])[:cap]]:
            evidence_ids.update(row["evidence_id"] for row in rows)

        # Add leaf-source candidates that are not necessarily in the composite top cap.
        for row in r15.get(claim_id, []):
            for key in ["old50_a07_rank", "old100_a08_rank", "gbdt_w2_rank"]:
                if key in row and int(row[key]) <= cap:
                    evidence_ids.add(row["evidence_id"])
        for row in claim_key.get(claim_id, []):
            for key in ["ranker_rank", "top100_rank", "top3_rank", "gate_rank"]:
                if key in row and int(row[key]) <= cap:
                    evidence_ids.add(row["evidence_id"])

        groups.append(len(evidence_ids))
        for evidence_id in sorted(evidence_ids):
            row = leaf_feature_row(claim_id, evidence_id, binary_idx, r15_idx, claim_key_idx)
            teacher = teacher_idx.get(claim_id, {}).get(evidence_id)
            row["teacher_score"] = float(teacher.get("score", 0.0)) if teacher else 0.0
            feature_rows.append(row)
            x_rows.append(row_to_vector(row))
            y_rows.append(row["teacher_score"])
            row_ids.append((claim_id, evidence_id))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), groups, row_ids, feature_rows


def rank_from_scores(row_ids, scores):
    ranked = {}
    for (claim_id, evidence_id), score in zip(row_ids, scores):
        ranked.setdefault(claim_id, []).append({"evidence_id": evidence_id, "score": float(score)})
    for claim_id, rows in ranked.items():
        rows.sort(key=lambda row: (-row["score"], row["evidence_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    return ranked


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
    parser = argparse.ArgumentParser(description="Leaf-flat distilled aggregator without composite selector-rank inputs.")
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--label-source", default="outputs/round15/recommended/top3_submission.json")
    parser.add_argument("--binary-source", default="outputs/round16/top3_binary_selector_e3_k20_n80/dev_binary_selector_ranked.json")
    parser.add_argument("--round15-source", default="outputs/round15/rrf_top3_selector/ranked.json")
    parser.add_argument("--claim-key-source", default="outputs/round16/branch_a_requirement/fixed_rrf_claim_key_ranked_top4500.json")
    parser.add_argument("--teacher-source", default="outputs/round16/top3_binary_selector_e3_k20_n80/best_binary_rrf_fusion_ranked.json")
    parser.add_argument("--output-dir", default="outputs/round17/leaf_flat_distilled_aggregator")
    parser.add_argument("--cap", type=int, default=20)
    args = parser.parse_args()

    claims = load_json(args.claims)
    train_claims = load_json(args.train_claims)
    label_source = load_any(args.label_source)
    default_label = majority_label(train_claims)
    binary = normalize_ranked(load_any(args.binary_source))
    r15 = normalize_ranked(load_any(args.round15_source))
    claim_key = normalize_ranked(load_any(args.claim_key_source))
    teacher = normalize_ranked(load_any(args.teacher_source))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, groups, row_ids, feature_rows = build_dataset(claims, binary, r15, claim_key, teacher, args.cap)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=2,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=1703,
        n_jobs=-1,
        verbose=-1,
    )
    sample_weight = np.ones(len(y), dtype=np.float32)
    sample_weight[y > 0] = 20.0
    model.fit(x, y, sample_weight=sample_weight)
    scores = model.predict(x)
    ranked = rank_from_scores(row_ids, scores)
    predictions = predictions_from_ranked(claims, ranked, label_source, default_label, top_k=3)
    row = metric_row(
        "leaf_flat_distilled_single_aggregator",
        claims,
        predictions,
        "single learned aggregator over leaf factors only; no round15_selector_rank or claim_key_selector_rank input",
    )
    write_json(ranked, output_dir / "leaf_flat_distilled_ranked.json")
    write_predictions(predictions, output_dir / "leaf_flat_distilled_top3.json")
    write_csv(feature_rows, output_dir / "leaf_flat_feature_rows.csv")
    write_json({"summary": [row]}, output_dir / "summary.json")
    write_csv([row], output_dir / "summary.csv")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
