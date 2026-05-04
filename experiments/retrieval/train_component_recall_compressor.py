import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, write_pool
from experiments.retrieval.train_recall_compressor import (
    model_score,
    pool_index,
    reciprocal_rank,
)


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'%-]*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
CAP_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
NEGATION_TERMS = {"no", "not", "never", "none", "without", "neither", "nor"}
COMPARISON_TERMS = {
    "more",
    "less",
    "higher",
    "lower",
    "before",
    "after",
    "over",
    "under",
    "largest",
    "smallest",
    "first",
    "last",
}
RELATION_TERMS = {
    "is",
    "was",
    "were",
    "are",
    "has",
    "have",
    "had",
    "won",
    "lost",
    "born",
    "died",
    "located",
    "founded",
    "created",
    "directed",
    "played",
    "served",
}
TEXT_CACHE = {}


def text_components(text):
    cached = TEXT_CACHE.get(text)
    if cached is not None:
        return cached
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text)]
    token_set = {tok for tok in tokens if len(tok) > 2}
    value = {
        "tokens": token_set,
        "numbers": {tok.lower() for tok in NUMBER_RE.findall(text)},
        "years": set(YEAR_RE.findall(text)),
        "percents": {tok.lower() for tok in PERCENT_RE.findall(text)},
        "caps": {tok.lower() for tok in CAP_RE.findall(text)},
        "neg": any(tok in NEGATION_TERMS for tok in tokens),
        "comparison": {tok for tok in tokens if tok in COMPARISON_TERMS},
        "relation": {tok for tok in tokens if tok in RELATION_TERMS},
        "length": len(token_set),
    }
    TEXT_CACHE[text] = value
    return value


def parse_source_specs(specs):
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected source spec name=path, got: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name, path))
    return parsed


def feature_names(source_names):
    names = []
    for source in source_names:
        names.extend([f"{source}_rr", f"{source}_score", f"in_{source}"])
    names.extend(
        [
            "source_count",
            "rr_max",
            "rr_mean",
            "rr_std",
            "rr_sum",
            "rank_disagreement",
            "token_overlap",
            "token_jaccard",
            "claim_token_recall",
            "number_overlap",
            "year_overlap",
            "percent_overlap",
            "capitalized_overlap",
            "negation_xor",
            "comparison_overlap",
            "relation_overlap",
            "evidence_token_count",
            "anchor_overlap",
            "entity_year",
            "entity_number",
            "year_number",
            "entity_negation",
            "relation_negation",
            "dense_token_jaccard",
            "dense_minus_sparse_rr",
            "source_count_token_jaccard",
        ]
    )
    return names


def feature_for(claim_id, claim, evidence_id, evidence, indexes, source_names):
    rr_values = []
    score_values = []
    present_values = []
    ranks = []
    for source in source_names:
        item = indexes[source].get(claim_id, {}).get(evidence_id)
        rank = None if item is None else item[0]
        rr_values.append(reciprocal_rank(rank))
        score_values.append(0.0 if item is None else float(item[1]))
        present_values.append(int(rank is not None))
        if rank is not None:
            ranks.append(float(rank))

    claim_parts = text_components(claim["claim_text"])
    evidence_parts = text_components(evidence.get(evidence_id, ""))
    token_intersection = claim_parts["tokens"] & evidence_parts["tokens"]
    token_union = claim_parts["tokens"] | evidence_parts["tokens"]
    number_overlap = len(claim_parts["numbers"] & evidence_parts["numbers"])
    year_overlap = len(claim_parts["years"] & evidence_parts["years"])
    percent_overlap = len(claim_parts["percents"] & evidence_parts["percents"])
    capitalized_overlap = len(claim_parts["caps"] & evidence_parts["caps"])
    comparison_overlap = len(claim_parts["comparison"] & evidence_parts["comparison"])
    relation_overlap = len(claim_parts["relation"] & evidence_parts["relation"])
    token_jaccard = len(token_intersection) / len(token_union) if token_union else 0.0
    claim_token_recall = (
        len(token_intersection) / len(claim_parts["tokens"])
        if claim_parts["tokens"]
        else 0.0
    )
    negation_xor = int(claim_parts["neg"] != evidence_parts["neg"])
    source_count = sum(present_values)
    rr_array = np.asarray(rr_values, dtype=np.float32)
    sparse_rr = max(
        [rr_values[idx] for idx, source in enumerate(source_names) if source in {"bm25", "char"}],
        default=0.0,
    )
    dense_rr = max(
        [
            rr_values[idx]
            for idx, source in enumerate(source_names)
            if source not in {"bm25", "char", "rrf"}
        ],
        default=0.0,
    )
    rank_disagreement = 0.0
    if len(ranks) > 1:
        rank_disagreement = (max(ranks) - min(ranks)) / max(max(ranks), 1.0)
    anchor_overlap = number_overlap + year_overlap + percent_overlap + capitalized_overlap

    tail = [
        source_count,
        float(rr_array.max()) if rr_array.size else 0.0,
        float(rr_array.mean()) if rr_array.size else 0.0,
        float(rr_array.std()) if rr_array.size else 0.0,
        float(rr_array.sum()) if rr_array.size else 0.0,
        rank_disagreement,
        len(token_intersection),
        token_jaccard,
        claim_token_recall,
        number_overlap,
        year_overlap,
        percent_overlap,
        capitalized_overlap,
        negation_xor,
        comparison_overlap,
        relation_overlap,
        evidence_parts["length"],
        anchor_overlap,
        capitalized_overlap * year_overlap,
        capitalized_overlap * number_overlap,
        year_overlap * number_overlap,
        capitalized_overlap * negation_xor,
        relation_overlap * negation_xor,
        dense_rr * token_jaccard,
        dense_rr - sparse_rr,
        source_count * token_jaccard,
    ]
    return np.asarray(
        [value for triple in zip(rr_values, score_values, present_values) for value in triple]
        + tail,
        dtype=np.float32,
    )


def build_rows(claims, evidence, candidate_pool, indexes, source_names, max_candidates, train_mode):
    rows = []
    labels = []
    claim_ids = []
    evidence_ids = []
    sample_weights = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        label = claim.get("claim_label", "")
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
                is_positive = int(candidate.evidence_id in gold)
                labels.append(is_positive)
                if not is_positive:
                    sample_weights.append(1.0)
                elif label == "NOT_ENOUGH_INFO":
                    sample_weights.append(2.0)
                elif label in {"REFUTES", "DISPUTED"}:
                    sample_weights.append(1.2)
                else:
                    sample_weights.append(1.0)
            claim_ids.append(claim_id)
            evidence_ids.append(candidate.evidence_id)
    names = feature_names(source_names)
    X = np.vstack(rows) if rows else np.empty((0, len(names)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8) if train_mode else None
    weights = np.asarray(sample_weights, dtype=np.float32) if train_mode else None
    return X, y, weights, claim_ids, evidence_ids


def sampled_indexes(y, claim_ids, max_negatives_per_claim):
    if max_negatives_per_claim <= 0:
        return np.arange(len(y), dtype=np.int64)
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
    return np.asarray(keep, dtype=np.int64)


def rank_predictions(claims, claim_ids, evidence_ids, scores, top_k):
    grouped = {claim_id: [] for claim_id in claims}
    for claim_id, evidence_id, score in zip(claim_ids, evidence_ids, scores):
        grouped[claim_id].append((evidence_id, float(score)))
    pool = {}
    for claim_id, items in grouped.items():
        ranked = sorted(items, key=lambda item: (-item[1], item[0]))[:top_k]
        pool[claim_id] = [
            Candidate(claim_id=claim_id, evidence_id=evidence_id, rank=rank, score=score)
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return pool


def quota_pool(claims, scored_pool, source_pools, base_k, quotas, output_k):
    result = {}
    for claim_id in claims:
        selected = []
        seen = set()
        for candidate in scored_pool.get(claim_id, [])[:base_k]:
            selected.append((candidate.evidence_id, float(candidate.score)))
            seen.add(candidate.evidence_id)
        for source_name, quota in quotas:
            for candidate in source_pools[source_name].get(claim_id, [])[:quota]:
                if candidate.evidence_id in seen:
                    continue
                selected.append((candidate.evidence_id, float(candidate.score)))
                seen.add(candidate.evidence_id)
        for candidate in scored_pool.get(claim_id, []):
            if len(selected) >= output_k:
                break
            if candidate.evidence_id in seen:
                continue
            selected.append((candidate.evidence_id, float(candidate.score)))
            seen.add(candidate.evidence_id)
        result[claim_id] = [
            Candidate(claim_id=claim_id, evidence_id=eid, rank=rank, score=score)
            for rank, (eid, score) in enumerate(selected[:output_k], start=1)
        ]
    return result


def compression_loss(claims, wide_pool, top_pool):
    wide_gold = kept_gold = total_gold = 0
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        total_gold += len(gold)
        wide_ids = {candidate.evidence_id for candidate in wide_pool.get(claim_id, [])}
        top_ids = {candidate.evidence_id for candidate in top_pool.get(claim_id, [])}
        wide_gold += len(gold & wide_ids)
        kept_gold += len(gold & top_ids)
    return {
        "total_gold": total_gold,
        "gold_in_wide": wide_gold,
        "gold_kept_topk": kept_gold,
        "dropped_by_compressor": wide_gold - kept_gold,
        "missing_from_wide": total_gold - wide_gold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Round13 component features for topN-to-topK recall compression."
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
    parser.add_argument("--no-nei-weights", action="store_true")
    args = parser.parse_args()

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

    X_train, y_train, weights, train_row_claims, _train_eids = build_rows(
        train_claims,
        evidence,
        train_candidate_pool,
        train_indexes,
        source_names,
        args.max_candidates,
        train_mode=True,
    )
    keep = sampled_indexes(y_train, train_row_claims, args.max_negatives_per_claim)
    X_fit = X_train[keep]
    y_fit = y_train[keep]
    weights_fit = None if args.no_nei_weights else weights[keep]
    X_dev, _y_dev, _weights_dev, dev_row_claims, dev_eids = build_rows(
        dev_claims,
        evidence,
        dev_candidate_pool,
        dev_indexes,
        source_names,
        args.max_candidates,
        train_mode=False,
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=13,
    )
    model.fit(X_fit, y_fit, sample_weight=weights_fit)
    scores = model_score(model, X_dev)
    scored_pool = rank_predictions(dev_claims, dev_row_claims, dev_eids, scores, args.max_candidates)

    rows = []
    loss_rows = []
    variants = {
        "component_histgbdt_top500": rank_predictions(
            dev_claims, dev_row_claims, dev_eids, scores, args.output_k
        ),
    }
    if "baseq" in dev_pools:
        quota400 = [("baseq", 50), ("bm25", 25), ("char", 25)]
        quota450 = [("baseq", 30), ("bm25", 10), ("char", 10)]
    else:
        quota400 = [("smallq", 40), ("small", 30), ("bm25", 15), ("char", 15)]
        quota450 = [("smallq", 20), ("small", 15), ("bm25", 8), ("char", 7)]
    variants["component_quota400_top500"] = quota_pool(
        dev_claims,
        scored_pool,
        dev_pools,
        base_k=min(400, args.output_k),
        quotas=[item for item in quota400 if item[0] in dev_pools],
        output_k=args.output_k,
    )
    variants["component_quota450_top500"] = quota_pool(
        dev_claims,
        scored_pool,
        dev_pools,
        base_k=min(450, args.output_k),
        quotas=[item for item in quota450 if item[0] in dev_pools],
        output_k=args.output_k,
    )
    for name, pool in variants.items():
        candidate_path = candidate_dir / f"{name}.json"
        write_pool(pool, candidate_path)
        metrics = evaluate_pool(dev_claims, pool, name, args.output_k, 0.0)
        rows.append(metrics)
        loss = compression_loss(dev_claims, dev_candidate_pool, pool)
        loss["method"] = name
        loss_rows.append(loss)
        print(
            f"{name} top{args.output_k} macro={metrics['macro_recall']:.4f} "
            f"hit={metrics['hit_any']:.4f} all={metrics['all_gold']:.4f} "
            f"nei={metrics['not_enough_info_macro_recall']:.4f}"
        )

    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "compression_loss.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loss_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loss_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "feature_names": feature_names(source_names),
                "source_names": source_names,
                "train_rows_total": int(X_train.shape[0]),
                "train_rows_fit": int(X_fit.shape[0]),
                "train_positive_rows": int(y_train.sum()),
                "dev_rows": int(X_dev.shape[0]),
                "max_candidates": args.max_candidates,
                "output_k": args.output_k,
                "nei_weights": not args.no_nei_weights,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
