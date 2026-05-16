import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, write_pool
from experiments.retrieval.train_component_recall_compressor import (
    build_rows,
    compression_loss,
    feature_names,
    parse_source_specs,
    pool_index,
    rank_predictions,
)


def parse_ints(value):
    return [int(part) for part in value.split(",") if part]


def grouped_indexes(claim_ids):
    groups = defaultdict(list)
    for idx, claim_id in enumerate(claim_ids):
        groups[claim_id].append(idx)
    return groups


def select_negative_indexes(indexes, y, max_negatives):
    negatives = [idx for idx in indexes if not y[idx]]
    if max_negatives <= 0 or len(negatives) <= max_negatives:
        return negatives

    top_budget = max_negatives // 2
    spread_budget = max_negatives - top_budget
    selected = negatives[:top_budget]
    remaining = negatives[top_budget:]
    if remaining and spread_budget > 0:
        stride = max(1, len(remaining) // spread_budget)
        selected.extend(remaining[::stride][:spread_budget])
    return selected


def build_pairwise_matrix(X, y, claim_ids, max_negatives, max_pairs):
    groups = grouped_indexes(claim_ids)
    rows = []
    labels = []
    for claim_id in sorted(groups):
        indexes = groups[claim_id]
        positives = [idx for idx in indexes if y[idx]]
        if not positives:
            continue
        negatives = select_negative_indexes(indexes, y, max_negatives)
        if not negatives:
            continue
        for pos_idx in positives:
            for neg_idx in negatives:
                rows.append(X[pos_idx] - X[neg_idx])
                labels.append(1)
                rows.append(X[neg_idx] - X[pos_idx])
                labels.append(0)
                if max_pairs and len(rows) >= max_pairs:
                    return np.vstack(rows), np.asarray(labels, dtype=np.int8)
    if not rows:
        return np.empty((0, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int8)
    return np.vstack(rows), np.asarray(labels, dtype=np.int8)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train a lightweight pairwise linear ranker for recall compression."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--train-candidate-pool", required=True)
    parser.add_argument("--dev-candidate-pool", required=True)
    parser.add_argument("--train-source", action="append", required=True)
    parser.add_argument("--dev-source", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--output-k-values", default="50,100,200,500")
    parser.add_argument("--max-negatives-per-claim", type=int, default=120)
    parser.add_argument("--max-pairs", type=int, default=300_000)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()
    output_k_values = sorted(set(parse_ints(args.output_k_values)))
    output_k = max(output_k_values)

    train_specs = parse_source_specs(args.train_source)
    dev_specs = parse_source_specs(args.dev_source)
    source_names = [name for name, _path in train_specs]
    if source_names != [name for name, _path in dev_specs]:
        raise ValueError("Train and dev source names must match.")

    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_candidate_pool = load_candidate_pool(args.train_candidate_pool)
    dev_candidate_pool = load_candidate_pool(args.dev_candidate_pool)
    train_pools = {name: load_candidate_pool(path) for name, path in train_specs}
    dev_pools = {name: load_candidate_pool(path) for name, path in dev_specs}
    train_indexes = {name: pool_index(pool) for name, pool in train_pools.items()}
    dev_indexes = {name: pool_index(pool) for name, pool in dev_pools.items()}

    X_train, y_train, _weights, train_row_claims, _train_eids = build_rows(
        train_claims,
        evidence,
        train_candidate_pool,
        train_indexes,
        source_names,
        args.max_candidates,
        train_mode=True,
    )
    X_dev, _y_dev, _dev_weights, dev_row_claims, dev_eids = build_rows(
        dev_claims,
        evidence,
        dev_candidate_pool,
        dev_indexes,
        source_names,
        args.max_candidates,
        train_mode=False,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pairs, y_pairs = build_pairwise_matrix(
        X_train_scaled,
        y_train,
        train_row_claims,
        args.max_negatives_per_claim,
        args.max_pairs,
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=1e-4,
        random_state=14,
        fit_intercept=False,
    )
    model.fit(X_pairs, y_pairs)
    scores = model.decision_function(scaler.transform(X_dev))
    pool = rank_predictions(dev_claims, dev_row_claims, dev_eids, scores, output_k)
    pool_path = candidate_dir / f"pairwise_sgd_top{output_k}.json"
    write_pool(pool, pool_path)

    rows = []
    for retained_k in output_k_values:
        rows.append(evaluate_pool(dev_claims, pool, "pairwise_sgd", retained_k, 0.0))
    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    loss = compression_loss(dev_claims, dev_candidate_pool, pool)
    write_json(
        output_dir / "summary.json",
        {
            "feature_names": feature_names(source_names),
            "source_names": source_names,
            "train_rows": int(X_train.shape[0]),
            "train_positive_rows": int(y_train.sum()),
            "pair_rows": int(X_pairs.shape[0]),
            "dev_rows": int(X_dev.shape[0]),
            "max_candidates": args.max_candidates,
            "output_k": output_k,
            "compression_loss": loss,
            "colab_safe_review": {
                "uses_bge": False,
                "uses_full_dense": False,
                "writes_checkpoints": False,
                "requires_drive": False,
            },
            "rows": rows,
        },
    )
    print(f"Wrote {pool_path}")
    for row in rows:
        print(
            f"pairwise_sgd k={row['retained_k']} "
            f"macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f}"
        )


if __name__ == "__main__":
    main()
