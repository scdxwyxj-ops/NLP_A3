import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'%-]*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
CAP_RE = re.compile(r"\b[A-Z][A-Za-z0-9'%-]*(?:\s+[A-Z][A-Za-z0-9'%-]*){0,3}\b")


def parse_ints(value):
    return [int(part) for part in value.split(",") if part]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_pool(pool, output_path):
    write_json(
        output_path,
        {
            claim_id: [asdict(candidate) for candidate in candidates]
            for claim_id, candidates in pool.items()
        },
    )


def source_scores(pool, claim_id, ranks):
    candidates = pool.get(claim_id, [])
    values = []
    for rank in ranks:
        if len(candidates) >= rank:
            values.append(float(candidates[rank - 1].score))
        else:
            values.append(0.0)
    return values


def overlap_at(left, right, claim_id, top_k):
    left_ids = {candidate.evidence_id for candidate in left.get(claim_id, [])[:top_k]}
    right_ids = {candidate.evidence_id for candidate in right.get(claim_id, [])[:top_k]}
    return len(left_ids & right_ids) / float(top_k) if top_k else 0.0


def same_top1(left, right, claim_id):
    left_items = left.get(claim_id, [])
    right_items = right.get(claim_id, [])
    if not left_items or not right_items:
        return 0.0
    return float(left_items[0].evidence_id == right_items[0].evidence_id)


def feature_names():
    names = [
        "query_len_tokens",
        "query_len_chars",
        "num_count",
        "year_count",
        "percent_count",
        "capitalized_count",
        "quote_markup",
    ]
    for source in ["bm25", "char", "rrf"]:
        for rank in [1, 5, 10, 20, 50, 100]:
            names.append(f"{source}_top{rank}_score")
        names.extend([f"{source}_decay_1_10", f"{source}_decay_10_50"])
    for top_k in [20, 50, 100, 200, 500]:
        names.append(f"bm25_char_overlap_{top_k}")
        names.append(f"rrf_bm25_overlap_{top_k}")
        names.append(f"rrf_char_overlap_{top_k}")
    names.extend(["same_top1_bm25_char", "same_top1_rrf_bm25", "same_top1_rrf_char"])
    return names


def claim_features(claim_id, claim_text, bm25, char, rrf):
    tokens = TOKEN_RE.findall(claim_text)
    row = [
        len(tokens),
        len(claim_text),
        len(NUMBER_RE.findall(claim_text)),
        len(YEAR_RE.findall(claim_text)),
        len(PERCENT_RE.findall(claim_text)),
        len(CAP_RE.findall(claim_text)),
        float(any(mark in claim_text for mark in ['"', "'", "[", "]", "...", "“", "”"])),
    ]
    ranks = [1, 5, 10, 20, 50, 100]
    for pool in [bm25, char, rrf]:
        scores = source_scores(pool, claim_id, ranks)
        row.extend(scores)
        row.append(scores[0] - scores[2])
        row.append(scores[2] - scores[4])
    for top_k in [20, 50, 100, 200, 500]:
        row.append(overlap_at(bm25, char, claim_id, top_k))
        row.append(overlap_at(rrf, bm25, claim_id, top_k))
        row.append(overlap_at(rrf, char, claim_id, top_k))
    row.extend(
        [
            same_top1(bm25, char, claim_id),
            same_top1(rrf, bm25, claim_id),
            same_top1(rrf, char, claim_id),
        ]
    )
    return np.asarray(row, dtype=np.float32)


def oracle_bucket(claim, rrf_candidates, buckets):
    gold = set(claim.get("evidences", []))
    if not gold:
        return buckets[0]
    max_bucket = max(buckets)
    available = {
        candidate.evidence_id
        for candidate in rrf_candidates[:max_bucket]
        if candidate.evidence_id in gold
    }
    if not available:
        return buckets[0]
    for bucket in buckets:
        kept = {
            candidate.evidence_id
            for candidate in rrf_candidates[:bucket]
            if candidate.evidence_id in gold
        }
        if kept >= available:
            return bucket
    return max_bucket


def truncate_pool(pool, k_by_claim):
    return {
        claim_id: list(pool.get(claim_id, [])[: k_by_claim.get(claim_id, 0)])
        for claim_id in k_by_claim
    }


def candidate_union_size(pool, k_by_claim):
    seen = set()
    for claim_id, candidates in pool.items():
        for candidate in candidates[: k_by_claim.get(claim_id, 0)]:
            seen.add(candidate.evidence_id)
    return len(seen)


def evaluate_variable_pool(claims, pool, method, k_by_claim):
    truncated = truncate_pool(pool, k_by_claim)
    retained_k = max(k_by_claim.values()) if k_by_claim else 0
    row = evaluate_pool(claims, truncated, method, retained_k, 0.0)
    row["candidate_union"] = candidate_union_size(pool, k_by_claim)
    row["avg_depth"] = sum(k_by_claim.values()) / len(k_by_claim) if k_by_claim else 0.0
    return row


def cv_predict_depths(X, labels, claim_ids, buckets, n_splits):
    predictions = {}
    label_array = np.asarray(labels)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=16)
    for train_idx, test_idx in kfold.split(X):
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=16,
        )
        model.fit(X[train_idx], label_array[train_idx])
        pred = model.predict(X[test_idx])
        for idx, bucket in zip(test_idx, pred):
            predictions[claim_ids[idx]] = int(bucket)
    for claim_id in claim_ids:
        predictions.setdefault(claim_id, buckets[-1])
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a claim-to-K sparse depth router from sparse preview features."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument(
        "--bm25-pool",
        default="outputs/round14/s10_interaction_ranker_blend_train2k_target10k/source_pools/target_bm25_top10000.json",
    )
    parser.add_argument(
        "--char-pool",
        default="outputs/round14/s10_interaction_ranker_blend_train2k_target10k/source_pools/target_char_top10000.json",
    )
    parser.add_argument("--output-dir", default="outputs/round14/s17_sparse_depth_router")
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--buckets", default="500,2000,5000,7000")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    claims = load_json(args.claims)
    bm25 = load_candidate_pool(args.bm25_pool)
    char = load_candidate_pool(args.char_pool)
    rrf = merge_rrf(f"sparse_rrf_k{args.rrf_k}", [bm25, char], top_k=10000, rrf_k=args.rrf_k).pool
    buckets = parse_ints(args.buckets)
    claim_ids = list(claims)

    X = np.vstack(
        [
            claim_features(claim_id, claims[claim_id]["claim_text"], bm25, char, rrf)
            for claim_id in claim_ids
        ]
    )
    labels = [
        oracle_bucket(claims[claim_id], rrf.get(claim_id, []), buckets)
        for claim_id in claim_ids
    ]
    predicted = cv_predict_depths(X, labels, claim_ids, buckets, args.n_splits)
    oracle = dict(zip(claim_ids, labels))

    rows = []
    for bucket in buckets:
        fixed = {claim_id: bucket for claim_id in claim_ids}
        rows.append(evaluate_variable_pool(claims, rrf, f"fixed_{bucket}", fixed))
    rows.append(evaluate_variable_pool(claims, rrf, "router_cv", predicted))
    rows.append(evaluate_variable_pool(claims, rrf, "oracle_bucket", oracle))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    prediction_rows = [
        {
            "claim_id": claim_id,
            "label": claims[claim_id].get("claim_label", ""),
            "oracle_bucket": oracle[claim_id],
            "predicted_bucket": predicted[claim_id],
            "claim_text": claims[claim_id]["claim_text"],
        }
        for claim_id in claim_ids
    ]
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    write_pool(truncate_pool(rrf, predicted), output_dir / "router_cv_pool.json")
    write_json(
        output_dir / "summary.json",
        {
            "feature_names": feature_names(),
            "buckets": buckets,
            "label_counts": dict(Counter(labels)),
            "predicted_counts": dict(Counter(predicted.values())),
            "rows": rows,
        },
    )
    print(f"Wrote {output_dir}")
    for row in rows:
        print(
            f"{row['method']} macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"union={row['candidate_union']} avg_depth={row['avg_depth']:.1f}"
        )


if __name__ == "__main__":
    main()
