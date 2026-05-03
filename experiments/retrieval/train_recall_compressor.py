import argparse
import csv
import json
from pathlib import Path

import numpy as np
import re
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, write_pool


FEATURE_NAMES = [
    "bm25_rr",
    "char_rr",
    "dense_rr",
    "dense2_rr",
    "rrf_rr",
    "bm25_score",
    "char_score",
    "dense_score",
    "dense2_score",
    "rrf_score",
    "in_bm25",
    "in_char",
    "in_dense",
    "in_dense2",
    "source_count",
    "best_rr",
    "sum_rr",
    "token_overlap",
    "token_jaccard",
    "claim_token_recall",
    "number_overlap",
    "evidence_token_count",
]


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'%-]*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?|\b(?:18|19|20)\d{2}\b")
TEXT_FEATURE_CACHE = {}


def text_features(text):
    cached = TEXT_FEATURE_CACHE.get(text)
    if cached is not None:
        return cached
    toks = {tok.lower() for tok in TOKEN_RE.findall(text) if len(tok) > 2}
    nums = {tok.lower() for tok in NUMBER_RE.findall(text)}
    value = (toks, nums, len(toks))
    TEXT_FEATURE_CACHE[text] = value
    return value


def pool_index(pool):
    return {
        claim_id: {
            candidate.evidence_id: (candidate.rank, candidate.score)
            for candidate in candidates
        }
        for claim_id, candidates in pool.items()
    }


def reciprocal_rank(rank):
    return 0.0 if rank is None else 1.0 / (1.0 + float(rank))


def feature_for(claim_id, claim, evidence_id, evidence, indexes):
    values = {}
    for name, index in indexes.items():
        item = index.get(claim_id, {}).get(evidence_id)
        values[f"{name}_rank"] = None if item is None else item[0]
        values[f"{name}_score"] = 0.0 if item is None else float(item[1])

    bm25_rr = reciprocal_rank(values["bm25_rank"])
    char_rr = reciprocal_rank(values["char_rank"])
    dense_rr = reciprocal_rank(values["dense_rank"])
    dense2_rr = reciprocal_rank(values.get("dense2_rank"))
    rrf_rr = reciprocal_rank(values["rrf_rank"])
    source_count = sum(
        int(values.get(f"{name}_rank") is not None)
        for name in ["bm25", "char", "dense", "dense2"]
    )
    claim_tokens, claim_numbers, _claim_len = text_features(claim["claim_text"])
    evidence_text = evidence.get(evidence_id, "")
    evidence_tokens, evidence_numbers, evidence_len = text_features(evidence_text)
    token_intersection = claim_tokens & evidence_tokens
    token_union = claim_tokens | evidence_tokens
    number_intersection = claim_numbers & evidence_numbers

    return np.array(
        [
            bm25_rr,
            char_rr,
            dense_rr,
            dense2_rr,
            rrf_rr,
            values["bm25_score"],
            values["char_score"],
            values["dense_score"],
            values.get("dense2_score", 0.0),
            values["rrf_score"],
            int(values["bm25_rank"] is not None),
            int(values["char_rank"] is not None),
            int(values["dense_rank"] is not None),
            int(values.get("dense2_rank") is not None),
            source_count,
            max(bm25_rr, char_rr, dense_rr, dense2_rr, rrf_rr),
            bm25_rr + char_rr + dense_rr + dense2_rr + rrf_rr,
            len(token_intersection),
            len(token_intersection) / len(token_union) if token_union else 0.0,
            len(token_intersection) / len(claim_tokens) if claim_tokens else 0.0,
            len(number_intersection),
            evidence_len,
        ],
        dtype=np.float32,
    )


def build_rows(claims, evidence, candidate_pool, indexes, max_candidates, train_mode):
    rows = []
    labels = []
    claim_ids = []
    evidence_ids = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        candidates = candidate_pool.get(claim_id, [])[:max_candidates]
        for candidate in candidates:
            rows.append(
                feature_for(claim_id, claim, candidate.evidence_id, evidence, indexes)
            )
            if train_mode:
                labels.append(int(candidate.evidence_id in gold))
            claim_ids.append(claim_id)
            evidence_ids.append(candidate.evidence_id)
    X = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8) if train_mode else None
    return X, y, claim_ids, evidence_ids


def sampled_training_data(X, y, claim_ids, max_negatives_per_claim):
    if max_negatives_per_claim <= 0:
        return X, y
    keep = []
    neg_counts = {}
    for idx, (label, claim_id) in enumerate(zip(y, claim_ids)):
        if label:
            keep.append(idx)
            continue
        count = neg_counts.get(claim_id, 0)
        if count < max_negatives_per_claim:
            keep.append(idx)
            neg_counts[claim_id] = count + 1
    keep = np.asarray(keep, dtype=np.int64)
    return X[keep], y[keep]


def rank_predictions(claims, claim_ids, evidence_ids, scores, top_k):
    grouped = {claim_id: [] for claim_id in claims}
    for claim_id, evidence_id, score in zip(claim_ids, evidence_ids, scores):
        grouped[claim_id].append((evidence_id, float(score)))
    pool = {}
    for claim_id, items in grouped.items():
        ranked = sorted(items, key=lambda item: (-item[1], item[0]))[:top_k]
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=score,
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return pool


def model_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    parser = argparse.ArgumentParser(
        description="Train a recall-oriented topN-to-topK candidate compressor."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--train-rrf-pool", required=True)
    parser.add_argument("--dev-rrf-pool", required=True)
    parser.add_argument("--train-bm25-pool", required=True)
    parser.add_argument("--dev-bm25-pool", required=True)
    parser.add_argument("--train-char-pool", required=True)
    parser.add_argument("--dev-char-pool", required=True)
    parser.add_argument("--train-dense-pool", required=True)
    parser.add_argument("--dev-dense-pool", required=True)
    parser.add_argument("--train-dense2-pool", default="")
    parser.add_argument("--dev-dense2-pool", default="")
    parser.add_argument("--output-dir", default="outputs/round11/recall_compressor")
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--output-k", type=int, default=500)
    parser.add_argument("--max-negatives-per-claim", type=int, default=1200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pools = {
        "rrf": load_candidate_pool(args.train_rrf_pool),
        "bm25": load_candidate_pool(args.train_bm25_pool),
        "char": load_candidate_pool(args.train_char_pool),
        "dense": load_candidate_pool(args.train_dense_pool),
    }
    dev_pools = {
        "rrf": load_candidate_pool(args.dev_rrf_pool),
        "bm25": load_candidate_pool(args.dev_bm25_pool),
        "char": load_candidate_pool(args.dev_char_pool),
        "dense": load_candidate_pool(args.dev_dense_pool),
    }
    if args.train_dense2_pool and args.dev_dense2_pool:
        train_pools["dense2"] = load_candidate_pool(args.train_dense2_pool)
        dev_pools["dense2"] = load_candidate_pool(args.dev_dense2_pool)

    train_indexes = {name: pool_index(pool) for name, pool in train_pools.items()}
    dev_indexes = {name: pool_index(pool) for name, pool in dev_pools.items()}

    X_train, y_train, train_row_claims, _train_eids = build_rows(
        train_claims,
        evidence,
        train_pools["rrf"],
        train_indexes,
        args.max_candidates,
        train_mode=True,
    )
    X_fit, y_fit = sampled_training_data(
        X_train, y_train, train_row_claims, args.max_negatives_per_claim
    )
    X_dev, _y_dev, dev_row_claims, dev_eids = build_rows(
        dev_claims,
        evidence,
        dev_pools["rrf"],
        dev_indexes,
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
        "feature_names": FEATURE_NAMES,
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
