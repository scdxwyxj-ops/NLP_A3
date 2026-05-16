import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, write_pool
from experiments.retrieval.train_recall_compressor import (
    pool_index,
    rank_predictions,
    reciprocal_rank,
    sampled_training_data,
    text_features,
)


def parse_source_specs(specs):
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected source spec name=path, got: {spec}")
        name, path = spec.split("=", 1)
        if not name:
            raise ValueError(f"Source name is empty in spec: {spec}")
        parsed.append((name, path))
    return parsed


def feature_names(source_names):
    return (
        [f"{name}_rr" for name in source_names]
        + [f"{name}_score" for name in source_names]
        + [f"in_{name}" for name in source_names]
        + [
            "source_count",
            "best_rr",
            "sum_rr",
            "token_overlap",
            "token_jaccard",
            "claim_token_recall",
            "number_overlap",
            "evidence_token_count",
        ]
    )


def feature_for(claim_id, claim, evidence_id, evidence, indexes, source_names):
    rr_values = []
    score_values = []
    present_values = []
    for name in source_names:
        item = indexes[name].get(claim_id, {}).get(evidence_id)
        rank = None if item is None else item[0]
        rr_values.append(reciprocal_rank(rank))
        score_values.append(0.0 if item is None else float(item[1]))
        present_values.append(int(rank is not None))

    claim_tokens, claim_numbers, _claim_len = text_features(claim["claim_text"])
    evidence_text = evidence.get(evidence_id, "")
    evidence_tokens, evidence_numbers, evidence_len = text_features(evidence_text)
    token_intersection = claim_tokens & evidence_tokens
    token_union = claim_tokens | evidence_tokens
    number_intersection = claim_numbers & evidence_numbers

    return np.asarray(
        [
            *rr_values,
            *score_values,
            *present_values,
            sum(present_values),
            max(rr_values) if rr_values else 0.0,
            sum(rr_values),
            len(token_intersection),
            len(token_intersection) / len(token_union) if token_union else 0.0,
            len(token_intersection) / len(claim_tokens) if claim_tokens else 0.0,
            len(number_intersection),
            evidence_len,
        ],
        dtype=np.float32,
    )


def build_rows(
    claims,
    evidence,
    candidate_pool,
    indexes,
    source_names,
    max_candidates,
    train_mode,
):
    rows = []
    labels = []
    claim_ids = []
    evidence_ids = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        candidates = candidate_pool.get(claim_id, [])[:max_candidates]
        for candidate in candidates:
            rows.append(
                feature_for(
                    claim_id,
                    claim,
                    candidate.evidence_id,
                    evidence,
                    indexes,
                    source_names,
                )
            )
            if train_mode:
                labels.append(int(candidate.evidence_id in gold))
            claim_ids.append(claim_id)
            evidence_ids.append(candidate.evidence_id)

    names = feature_names(source_names)
    X = np.vstack(rows) if rows else np.empty((0, len(names)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8) if train_mode else None
    return X, y, claim_ids, evidence_ids


def model_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    parser = argparse.ArgumentParser(
        description="Train a recall compressor with arbitrary source rank features."
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
    parser.add_argument("--output-k", type=int, default=500)
    parser.add_argument("--max-negatives-per-claim", type=int, default=0)
    args = parser.parse_args()

    train_specs = parse_source_specs(args.train_source)
    dev_specs = parse_source_specs(args.dev_source)
    source_names = [name for name, _path in train_specs]
    if source_names != [name for name, _path in dev_specs]:
        raise ValueError("Train and dev source names must match and be in the same order.")

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

    X_train, y_train, train_row_claims, _train_eids = build_rows(
        train_claims,
        evidence,
        train_candidate_pool,
        train_indexes,
        source_names,
        args.max_candidates,
        train_mode=True,
    )
    X_fit, y_fit = sampled_training_data(
        X_train, y_train, train_row_claims, args.max_negatives_per_claim
    )
    X_dev, _y_dev, dev_row_claims, dev_eids = build_rows(
        dev_claims,
        evidence,
        dev_candidate_pool,
        dev_indexes,
        source_names,
        args.max_candidates,
        train_mode=False,
    )

    models = {
        "logreg_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=13,
            ),
        ),
        "histgbdt_balanced": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=13,
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        scores = model_score(model, X_dev)
        pool = rank_predictions(
            dev_claims,
            dev_row_claims,
            dev_eids,
            scores,
            args.output_k,
        )
        candidate_path = candidate_dir / f"{name}_top{args.output_k}.json"
        write_pool(pool, candidate_path)
        metrics = evaluate_pool(
            dev_claims,
            pool,
            method=name,
            retained_k=args.output_k,
            build_seconds=0.0,
        )
        rows.append(metrics)
        print(
            f"{name} top{args.output_k} macro={metrics['macro_recall']:.4f} "
            f"hit={metrics['hit_any']:.4f} all={metrics['all_gold']:.4f} "
            f"refutes={metrics['refutes_macro_recall']:.4f}"
        )

    summary = {
        "feature_names": feature_names(source_names),
        "source_names": source_names,
        "train_rows_total": int(X_train.shape[0]),
        "train_rows_fit": int(X_fit.shape[0]),
        "train_positive_rows": int(y_train.sum()),
        "dev_rows": int(X_dev.shape[0]),
        "max_candidates": args.max_candidates,
        "output_k": args.output_k,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
