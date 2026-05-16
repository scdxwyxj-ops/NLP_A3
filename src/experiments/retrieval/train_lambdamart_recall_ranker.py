import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
try:
    from lightgbm import LGBMRanker
except ImportError:  # pragma: no cover - optional experiment dependency
    LGBMRanker = None

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


def select_grouped_indexes(y, claim_ids, max_negatives):
    groups = defaultdict(list)
    for idx, claim_id in enumerate(claim_ids):
        groups[claim_id].append(idx)

    keep = []
    group_sizes = []
    for claim_id in sorted(groups):
        indexes = groups[claim_id]
        positives = [idx for idx in indexes if y[idx]]
        negatives = [idx for idx in indexes if not y[idx]]
        selected = list(positives)
        if max_negatives <= 0 or len(negatives) <= max_negatives:
            selected.extend(negatives)
        else:
            top_budget = max_negatives // 2
            spread_budget = max_negatives - top_budget
            selected.extend(negatives[:top_budget])
            remaining = negatives[top_budget:]
            if remaining and spread_budget > 0:
                stride = max(1, len(remaining) // spread_budget)
                selected.extend(remaining[::stride][:spread_budget])
        selected = sorted(set(selected))
        if selected:
            keep.extend(selected)
            group_sizes.append(len(selected))
    return np.asarray(keep, dtype=np.int64), group_sizes


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train a LightGBM LambdaMART ranker for recall compression."
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
    parser.add_argument("--max-negatives-per-claim", type=int, default=800)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=20)
    args = parser.parse_args()
    if LGBMRanker is None:
        raise SystemExit("Install optional dependency first: python3 -m pip install lightgbm")

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
    keep, train_group = select_grouped_indexes(
        y_train, train_row_claims, args.max_negatives_per_claim
    )
    X_fit = X_train[keep]
    y_fit = y_train[keep]
    X_dev, _y_dev, _dev_weights, dev_row_claims, dev_eids = build_rows(
        dev_claims,
        evidence,
        dev_candidate_pool,
        dev_indexes,
        source_names,
        args.max_candidates,
        train_mode=False,
    )

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=14,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_fit, y_fit, group=train_group, eval_at=[50, 100, 200, 500])
    scores = model.predict(X_dev)
    pool = rank_predictions(dev_claims, dev_row_claims, dev_eids, scores, output_k)
    pool_path = candidate_dir / f"lambdamart_top{output_k}.json"
    write_pool(pool, pool_path)

    rows = []
    for retained_k in output_k_values:
        rows.append(evaluate_pool(dev_claims, pool, "lambdamart", retained_k, 0.0))
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
            "train_rows_total": int(X_train.shape[0]),
            "train_rows_fit": int(X_fit.shape[0]),
            "train_positive_rows": int(y_train.sum()),
            "dev_rows": int(X_dev.shape[0]),
            "group_count": len(train_group),
            "max_candidates": args.max_candidates,
            "output_k": output_k,
            "compression_loss": loss,
            "colab_safe_review": {
                "uses_bge": False,
                "uses_full_dense": False,
                "writes_checkpoints": False,
                "requires_drive": False,
                "requires_lightgbm_install": True,
            },
            "rows": rows,
        },
    )
    print(f"Wrote {pool_path}")
    for row in rows:
        print(
            f"lambdamart k={row['retained_k']} "
            f"macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f}"
        )


if __name__ == "__main__":
    main()
