import argparse
import csv
import gc
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf
from experiments.retrieval.rerank_sparse_gate_slots import (
    claim_profile,
    core_tokens,
    coverage,
    jaccard,
    mismatch_penalty,
    proximity_score,
    text_slots,
)
from experiments.retrieval.run_round13_colab_generator import (
    limit_items,
    log_stage,
    query_bm25_pool,
    query_char_tfidf_pool,
    subset_evidence_for_claims,
)
from experiments.retrieval.train_recall_compressor import pool_index, reciprocal_rank


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


def parse_ints(value):
    return [int(part) for part in value.split(",") if part]


def ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)}


def overlap_count(left, right):
    return len(left & right)


def safe_div(num, denom):
    return num / denom if denom else 0.0


def token_idf(claim_profiles):
    df = Counter()
    for profile in claim_profiles.values():
        df.update(profile["core"])
    n_docs = max(len(claim_profiles), 1)
    return {
        token: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
        for token, count in df.items()
    }


def source_rank_features(candidate, claim_id, evidence_id, indexes, source_names, max_rank):
    values = []
    rr_values = []
    present = []
    ranks = []
    for source in source_names:
        item = indexes[source].get(claim_id, {}).get(evidence_id)
        rank = None if item is None else item[0]
        score = 0.0 if item is None else float(item[1])
        rr = reciprocal_rank(rank)
        values.extend(
            [
                rr,
                score,
                0.0 if rank is None else 1.0 - min(rank, max_rank) / max(max_rank, 1),
                int(rank is not None),
            ]
        )
        rr_values.append(rr)
        present.append(int(rank is not None))
        if rank is not None:
            ranks.append(float(rank))
    rr_array = np.asarray(rr_values, dtype=np.float32)
    rank_disagreement = 0.0
    if len(ranks) > 1:
        rank_disagreement = (max(ranks) - min(ranks)) / max(max(ranks), 1.0)
    values.extend(
        [
            reciprocal_rank(candidate.rank),
            1.0 - min(candidate.rank, max_rank) / max(max_rank, 1),
            sum(present),
            float(rr_array.max()) if rr_array.size else 0.0,
            float(rr_array.mean()) if rr_array.size else 0.0,
            float(rr_array.std()) if rr_array.size else 0.0,
            float(rr_array.sum()) if rr_array.size else 0.0,
            rank_disagreement,
        ]
    )
    return values


def phrase_features(claim_tokens, evidence_tokens):
    claim_bigrams = ngrams(claim_tokens, 2)
    evidence_bigrams = ngrams(evidence_tokens, 2)
    claim_trigrams = ngrams(claim_tokens, 3)
    evidence_trigrams = ngrams(evidence_tokens, 3)
    bigram_hits = overlap_count(claim_bigrams, evidence_bigrams)
    trigram_hits = overlap_count(claim_trigrams, evidence_trigrams)
    return [
        bigram_hits,
        safe_div(bigram_hits, len(claim_bigrams)),
        safe_div(bigram_hits, len(claim_bigrams | evidence_bigrams)),
        trigram_hits,
        safe_div(trigram_hits, len(claim_trigrams)),
        safe_div(trigram_hits, len(claim_trigrams | evidence_trigrams)),
    ]


def clause_windows(text):
    parts = re.split(
        r"[,;:()]|\b(?:but|because|although|whereas|while|which|that|and that)\b",
        text,
        flags=re.IGNORECASE,
    )
    windows = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" \t\r\n\"'`.,;:")
        if len(part.split()) >= 3:
            windows.append(part)
    if not windows:
        windows = [text]
    return windows


def best_window_features(claim_profile_value, evidence_text):
    best_core = 0.0
    best_expanded = 0.0
    best_proximity = 0.0
    min_mismatch = 10.0
    for window in clause_windows(evidence_text):
        slots = text_slots(window)
        best_core = max(best_core, coverage(claim_profile_value["core"], slots))
        best_expanded = max(best_expanded, coverage(claim_profile_value["expanded"], slots))
        best_proximity = max(best_proximity, proximity_score(claim_profile_value, slots, 16))
        min_mismatch = min(min_mismatch, mismatch_penalty(claim_profile_value, slots))
    return [best_core, best_expanded, best_proximity, min_mismatch if min_mismatch < 10.0 else 0.0]


def interaction_feature_names(source_names):
    names = []
    for source in source_names:
        names.extend(
            [
                f"{source}_rr",
                f"{source}_score",
                f"{source}_rank_norm",
                f"in_{source}",
            ]
        )
    names.extend(
        [
            "candidate_rr",
            "candidate_rank_norm",
            "source_count",
            "rr_max",
            "rr_mean",
            "rr_std",
            "rr_sum",
            "rank_disagreement",
            "core_coverage",
            "expanded_coverage",
            "token_overlap",
            "token_jaccard",
            "idf_core_coverage",
            "bigram_overlap",
            "bigram_recall",
            "bigram_jaccard",
            "trigram_overlap",
            "trigram_recall",
            "trigram_jaccard",
            "entity_jaccard",
            "entity_overlap",
            "entity_missing",
            "year_jaccard",
            "year_overlap",
            "year_mismatch",
            "number_jaccard",
            "number_overlap",
            "number_mismatch",
            "negation_match",
            "negation_xor",
            "comparison_jaccard",
            "comparison_overlap",
            "comparison_mismatch",
            "proximity",
            "mismatch_penalty",
            "anchor_hits",
            "best_window_core",
            "best_window_expanded",
            "best_window_proximity",
            "best_window_mismatch",
            "dense_placeholder",
            "coverage_x_rr",
            "anchor_x_rr",
            "mismatch_x_rr",
        ]
    )
    return names


def feature_for(
    claim_profile_value,
    evidence_text,
    candidate,
    claim_id,
    source_indexes,
    source_names,
    idf,
    max_rank,
):
    evidence_slots = text_slots(evidence_text)
    claim_core = claim_profile_value["core"]
    evidence_tokens = evidence_slots["tokens"]
    token_intersection = claim_core & evidence_tokens
    token_union = claim_core | evidence_tokens
    idf_total = sum(idf.get(token, 1.0) for token in claim_core)
    idf_hit = sum(idf.get(token, 1.0) for token in token_intersection)
    core_cov = coverage(claim_core, evidence_slots)
    expanded_cov = coverage(claim_profile_value["expanded"], evidence_slots)
    entity_overlap = len(claim_profile_value["caps"] & evidence_slots["caps"])
    year_overlap = len(claim_profile_value["years"] & evidence_slots["years"])
    number_overlap = len(claim_profile_value["numbers"] & evidence_slots["numbers"])
    comparison_overlap = len(
        claim_profile_value["comparison"] & evidence_slots["comparison"]
    )
    year_mismatch = int(
        bool(claim_profile_value["years"])
        and bool(evidence_slots["years"])
        and not year_overlap
    )
    number_mismatch = int(
        bool(claim_profile_value["numbers"])
        and bool(evidence_slots["numbers"])
        and not number_overlap
    )
    negation_xor = int(
        bool(claim_profile_value["negation"]) != bool(evidence_slots["negation"])
    )
    comparison_mismatch = int(
        bool(claim_profile_value["comparison"])
        and bool(evidence_slots["comparison"])
        and not comparison_overlap
    )
    proximity = proximity_score(claim_profile_value, evidence_slots, 16)
    mismatch = mismatch_penalty(claim_profile_value, evidence_slots)
    anchor_hits = entity_overlap + year_overlap + number_overlap
    rr = reciprocal_rank(candidate.rank)
    features = source_rank_features(
        candidate,
        claim_id,
        candidate.evidence_id,
        source_indexes,
        source_names,
        max_rank,
    )
    claim_token_list = sorted(claim_core)
    evidence_token_list = sorted(evidence_tokens)
    features.extend(
        [
            core_cov,
            expanded_cov,
            len(token_intersection),
            safe_div(len(token_intersection), len(token_union)),
            safe_div(idf_hit, idf_total),
            *phrase_features(claim_token_list, evidence_token_list),
            jaccard(claim_profile_value["caps"], evidence_slots["caps"]),
            entity_overlap,
            max(0, len(claim_profile_value["caps"]) - entity_overlap),
            jaccard(claim_profile_value["years"], evidence_slots["years"]),
            year_overlap,
            year_mismatch,
            jaccard(claim_profile_value["numbers"], evidence_slots["numbers"]),
            number_overlap,
            number_mismatch,
            int(bool(claim_profile_value["negation"]) == bool(evidence_slots["negation"])),
            negation_xor,
            jaccard(claim_profile_value["comparison"], evidence_slots["comparison"]),
            comparison_overlap,
            comparison_mismatch,
            proximity,
            mismatch,
            min(anchor_hits, 4),
            *best_window_features(claim_profile_value, evidence_text),
            0.0,
            core_cov * rr,
            min(anchor_hits, 4) * rr,
            mismatch * rr,
        ]
    )
    return np.asarray(features, dtype=np.float32)


def sampled_training_candidates(candidates, gold, max_negatives):
    selected = []
    selected_ids = set()
    for candidate in candidates:
        if candidate.evidence_id in gold:
            selected.append(candidate)
            selected_ids.add(candidate.evidence_id)
    if max_negatives <= 0:
        return selected

    top_budget = max_negatives // 2
    spread_budget = max_negatives - top_budget
    negatives = 0
    for candidate in candidates:
        if candidate.evidence_id in gold or candidate.evidence_id in selected_ids:
            continue
        if negatives < top_budget:
            selected.append(candidate)
            selected_ids.add(candidate.evidence_id)
            negatives += 1
        else:
            break

    remaining = [
        candidate
        for candidate in candidates
        if candidate.evidence_id not in gold and candidate.evidence_id not in selected_ids
    ]
    if remaining and spread_budget > 0:
        stride = max(1, len(remaining) // spread_budget)
        for candidate in remaining[::stride][:spread_budget]:
            selected.append(candidate)
            selected_ids.add(candidate.evidence_id)
    return selected


def build_train_matrix(claims, evidence, candidate_pool, source_pools, args):
    source_names = list(source_pools)
    source_indexes = {name: pool_index(pool) for name, pool in source_pools.items()}
    profiles = {
        claim_id: claim_profile(claim["claim_text"], args.max_expansions_per_token)
        for claim_id, claim in claims.items()
    }
    idf = token_idf(profiles)
    rows = []
    labels = []
    weights = []
    for idx, (claim_id, claim) in enumerate(claims.items()):
        gold = set(claim.get("evidences", []))
        label = claim.get("claim_label", "")
        candidates = candidate_pool.get(claim_id, [])[: args.train_max_candidates]
        candidates = sampled_training_candidates(candidates, gold, args.max_negatives_per_claim)
        for candidate in candidates:
            rows.append(
                feature_for(
                    profiles[claim_id],
                    evidence.get(candidate.evidence_id, ""),
                    candidate,
                    claim_id,
                    source_indexes,
                    source_names,
                    idf,
                    args.train_max_candidates,
                )
            )
            is_positive = int(candidate.evidence_id in gold)
            labels.append(is_positive)
            if not is_positive:
                weights.append(1.0)
            elif label == "NOT_ENOUGH_INFO":
                weights.append(args.nei_positive_weight)
            elif label in {"REFUTES", "DISPUTED"}:
                weights.append(args.hard_label_positive_weight)
            else:
                weights.append(1.0)
        if idx and idx % 100 == 0:
            log_stage(f"built train features {idx}/{len(claims)} claims")
    names = interaction_feature_names(source_names)
    X = np.vstack(rows) if rows else np.empty((0, len(names)), dtype=np.float32)
    return X, np.asarray(labels, dtype=np.int8), np.asarray(weights, dtype=np.float32), names


def model_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
    return model.decision_function(X)


def score_claims(claims, evidence, candidate_pool, source_pools, model, args):
    source_names = list(source_pools)
    source_indexes = {name: pool_index(pool) for name, pool in source_pools.items()}
    profiles = {
        claim_id: claim_profile(claim["claim_text"], args.max_expansions_per_token)
        for claim_id, claim in claims.items()
    }
    idf = token_idf(profiles)
    max_output_k = max(args.output_k_values)
    raw_result = {}
    blend_results = {weight: {} for weight in args.blend_weights}
    for idx, claim_id in enumerate(claims):
        candidates = candidate_pool.get(claim_id, [])[: args.target_max_candidates]
        if not candidates:
            raw_result[claim_id] = []
            for pool in blend_results.values():
                pool[claim_id] = []
            continue
        X = np.vstack(
            [
                feature_for(
                    profiles[claim_id],
                    evidence.get(candidate.evidence_id, ""),
                    candidate,
                    claim_id,
                    source_indexes,
                    source_names,
                    idf,
                    args.target_max_candidates,
                )
                for candidate in candidates
            ]
        )
        scores = model_scores(model, X)
        ranked_raw = sorted(
            zip((candidate.evidence_id for candidate in candidates), scores),
            key=lambda item: (-float(item[1]), item[0]),
        )[:max_output_k]
        raw_result[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=float(score),
            )
            for rank, (evidence_id, score) in enumerate(ranked_raw, start=1)
        ]
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        if score_max > score_min:
            model_norm = (scores - score_min) / (score_max - score_min)
        else:
            model_norm = np.zeros_like(scores, dtype=np.float32)
        sparse_prior = np.asarray(
            [
                1.0 - min(candidate.rank, args.target_max_candidates)
                / max(args.target_max_candidates, 1)
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        evidence_ids = [candidate.evidence_id for candidate in candidates]
        for weight, pool in blend_results.items():
            final_scores = ((1.0 - weight) * sparse_prior) + (weight * model_norm)
            ranked_blend = sorted(
                zip(evidence_ids, final_scores),
                key=lambda item: (-float(item[1]), item[0]),
            )[:max_output_k]
            pool[claim_id] = [
                Candidate(
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    rank=rank,
                    score=float(score),
                )
                for rank, (evidence_id, score) in enumerate(ranked_blend, start=1)
            ]
        if idx and idx % 25 == 0:
            log_stage(f"scored target features {idx}/{len(claims)} claims")
    return raw_result, blend_results


def build_sparse_gate(claims, evidence, top_k, args, prefix):
    if args.source_pool_dir:
        source_dir = Path(args.source_pool_dir)
        bm25_path = source_dir / f"{prefix}_bm25_top{top_k}.json"
        char_path = source_dir / f"{prefix}_char_top{top_k}.json"
        rrf_path = source_dir / f"{prefix}_sparse_gate_top{top_k}.json"
        if bm25_path.exists() and char_path.exists() and rrf_path.exists():
            log_stage(f"loading {prefix} sparse gate top{top_k} from {source_dir}")
            return (
                load_candidate_pool(bm25_path),
                load_candidate_pool(char_path),
                load_candidate_pool(rrf_path),
            )

    log_stage(f"building {prefix} BM25 top{top_k}")
    bm25 = query_bm25_pool(claims, evidence, top_k, k1=1.5, b=0.75)
    gc.collect()
    log_stage(f"building {prefix} char top{top_k}")
    char = query_char_tfidf_pool(
        claims,
        evidence,
        top_k,
        max_claim_fanout=args.query_char_max_claim_fanout,
        max_df_ratio=args.query_char_max_df_ratio,
    )
    gc.collect()
    log_stage(f"building {prefix} RRF top{top_k}")
    rrf = merge_rrf(f"{prefix}_sparse_gate_bm25_char", [bm25, char], top_k, args.rrf_k).pool
    if args.write_source_pools:
        source_dir = Path(args.source_pool_dir) if args.source_pool_dir else Path(args.output_dir) / "source_pools"
        source_dir.mkdir(parents=True, exist_ok=True)
        write_pool(bm25, source_dir / f"{prefix}_bm25_top{top_k}.json")
        write_pool(char, source_dir / f"{prefix}_char_top{top_k}.json")
        write_pool(rrf, source_dir / f"{prefix}_sparse_gate_top{top_k}.json")
    return bm25, char, rrf


def main():
    parser = argparse.ArgumentParser(
        description="Train a Colab-safe interaction-feature recall ranker."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--target-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round14_interaction_ranker")
    parser.add_argument("--train-gate-top-k", type=int, default=4000)
    parser.add_argument("--target-gate-top-k", type=int, default=10000)
    parser.add_argument("--train-max-candidates", type=int, default=4000)
    parser.add_argument("--target-max-candidates", type=int, default=10000)
    parser.add_argument("--max-negatives-per-claim", type=int, default=800)
    parser.add_argument("--output-k-values", default="50,100,200,500")
    parser.add_argument("--blend-weights", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--rrf-k", type=int, default=500)
    parser.add_argument("--query-char-max-claim-fanout", type=int, default=64)
    parser.add_argument("--query-char-max-df-ratio", type=float, default=0.25)
    parser.add_argument("--max-expansions-per-token", type=int, default=3)
    parser.add_argument("--nei-positive-weight", type=float, default=2.0)
    parser.add_argument("--hard-label-positive-weight", type=float, default=1.2)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--smoke-evidence-limit", type=int, default=0)
    parser.add_argument("--smoke-train-claims", type=int, default=0)
    parser.add_argument("--smoke-target-claims", type=int, default=0)
    parser.add_argument("--write-source-pools", action="store_true")
    parser.add_argument(
        "--source-pool-dir",
        default="",
        help="Optional directory for loading/writing train/dev BM25, char, and RRF pools.",
    )
    args = parser.parse_args()
    args.output_k_values = sorted(set(parse_ints(args.output_k_values)))
    args.blend_weights = sorted(set(float(value) for value in args.blend_weights.split(",") if value))

    train_claims = limit_items(load_json(args.train_claims), args.smoke_train_claims)
    target_claims = limit_items(load_json(args.target_claims), args.smoke_target_claims)
    evidence = subset_evidence_for_claims(
        load_json(args.evidence),
        train_claims,
        target_claims,
        args.smoke_evidence_limit,
    )
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    args.train_gate_top_k = min(args.train_gate_top_k, len(evidence))
    args.target_gate_top_k = min(args.target_gate_top_k, len(evidence))
    args.train_max_candidates = min(args.train_max_candidates, args.train_gate_top_k)
    args.target_max_candidates = min(args.target_max_candidates, args.target_gate_top_k)

    config = {
        "train_claims": len(train_claims),
        "target_claims": len(target_claims),
        "evidence": len(evidence),
        "args": vars(args),
        "colab_safe_review": {
            "uses_bge": False,
            "uses_full_dense": False,
            "writes_checkpoints": False,
            "requires_drive": False,
            "large_artifacts_under_outputs_only": True,
        },
    }
    print(json.dumps(config, indent=2), flush=True)

    train_bm25, train_char, train_rrf = build_sparse_gate(
        train_claims, evidence, args.train_gate_top_k, args, "train"
    )
    target_bm25, target_char, target_rrf = build_sparse_gate(
        target_claims, evidence, args.target_gate_top_k, args, "target"
    )
    if args.write_source_pools:
        write_pool(train_rrf, candidate_dir / f"train_sparse_gate_top{args.train_gate_top_k}.json")
        write_pool(target_rrf, candidate_dir / f"target_sparse_gate_top{args.target_gate_top_k}.json")

    log_stage("building supervised interaction train matrix")
    X_train, y_train, weights, names = build_train_matrix(
        train_claims,
        evidence,
        train_rrf,
        {"bm25": train_bm25, "char": train_char, "rrf": train_rrf},
        args,
    )
    log_stage(
        f"training rows={X_train.shape[0]} positives={int(y_train.sum())} "
        f"features={X_train.shape[1]}"
    )
    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        class_weight="balanced",
        random_state=14,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    del X_train, y_train, weights
    gc.collect()

    log_stage("scoring target interaction ranker")
    raw_pool, blend_pools = score_claims(
        target_claims,
        evidence,
        target_rrf,
        {"bm25": target_bm25, "char": target_char, "rrf": target_rrf},
        model,
        args,
    )
    output_k = max(args.output_k_values)
    top_path = candidate_dir / f"interaction_histgbdt_top{output_k}.json"
    write_pool(raw_pool, top_path)
    for weight, pool in blend_pools.items():
        write_pool(
            pool,
            candidate_dir / f"interaction_blend_w{str(weight).replace('.', 'p')}_top{output_k}.json",
        )

    rows = []
    pools = {
        "target_sparse_gate_rrf": target_rrf,
        "interaction_histgbdt": raw_pool,
    }
    for weight, pool in blend_pools.items():
        pools[f"interaction_blend_w{weight}"] = pool
    for method, pool in pools.items():
        for retained_k in args.output_k_values:
            rows.append(evaluate_pool(target_claims, pool, method, retained_k, 0.0))

    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "summary.json",
        {
            "config": config,
            "feature_names": names,
            "top_path": str(top_path),
            "rows": rows,
        },
    )
    print(f"Wrote {summary_path}")
    for row in rows:
        print(
            f"{row['method']} k={row['retained_k']} "
            f"macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f}"
        )


if __name__ == "__main__":
    main()
