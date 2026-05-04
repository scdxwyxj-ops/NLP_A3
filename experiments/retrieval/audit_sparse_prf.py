import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf
from experiments.retrieval.run_round13_colab_generator import (
    limit_items,
    query_bm25_pool,
    subset_evidence_for_claims,
)


def parse_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_pool(output_path, pool):
    write_json(
        output_path,
        {claim_id: [asdict(candidate) for candidate in candidates] for claim_id, candidates in pool.items()},
    )


def candidate_union_size(pool, top_k):
    seen = set()
    for candidates in pool.values():
        for candidate in candidates[:top_k]:
            seen.add(candidate.evidence_id)
    return len(seen)


def trim_pool(pool, top_k):
    return {claim_id: list(candidates[:top_k]) for claim_id, candidates in pool.items()}


def filter_pool_to_evidence(pool, evidence_ids):
    evidence_set = set(evidence_ids)
    filtered = {}
    for claim_id, candidates in pool.items():
        kept = []
        for candidate in candidates:
            if candidate.evidence_id in evidence_set:
                kept.append(replace(candidate, rank=len(kept) + 1))
        filtered[claim_id] = kept
    return filtered


def build_expanded_claims(claims, evidence, base_pool, max_feedback, max_terms, analyzer):
    claim_terms = {}
    for claim_id, claim in claims.items():
        claim_terms[claim_id] = {term for term in analyzer(claim["claim_text"])}

    expansion_counts = {}
    expanded_claims = {}
    for claim_id, claim in claims.items():
        seed_candidates = base_pool.get(claim_id, [])[:max_feedback]
        term_scores = Counter()

        for rank, candidate in enumerate(seed_candidates, start=1):
            text = evidence.get(candidate.evidence_id)
            if not text:
                continue
            weight = 1.0 / (rank + 1.0)
            term_counts = Counter(analyzer(text))
            for term, freq in term_counts.items():
                if term in claim_terms[claim_id]:
                    continue
                term_scores[term] += float(freq) * weight

        top_terms = [term for term, _ in term_scores.most_common(max_terms)]
        expansion_counts[claim_id] = {
            "feedback_docs": len(seed_candidates),
            "expansion_terms": len(top_terms),
            "terms": top_terms,
        }
        if top_terms:
            expanded_text = f"{claim['claim_text']} {' '.join(top_terms)}"
        else:
            expanded_text = claim["claim_text"]
        expanded_claims[claim_id] = {**claim, "claim_text": expanded_text}

    return expanded_claims, expansion_counts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit cheap sparse retrieval with BM25+char RRF baseline and RM3-style PRF "
            "expanded BM25 sparse queries."
        )
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument(
        "--bm25-pool",
        default="outputs/round14/s10_interaction_ranker_blend_train2k_target10k/source_pools/target_bm25_top10000.json",
    )
    parser.add_argument(
        "--char-pool",
        default="outputs/round14/s10_interaction_ranker_blend_train2k_target10k/source_pools/target_char_top10000.json",
    )
    parser.add_argument("--output-dir", default="outputs/round14/s19_sparse_prf")
    parser.add_argument("--pool-top-k", type=int, default=10000)
    parser.add_argument("--top-k-values", default="500,1000,2000,5000,10000")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--feedback-top-k", type=int, default=20)
    parser.add_argument("--prf-max-terms", type=int, default=16)
    parser.add_argument("--smoke-claims", type=int, default=0)
    parser.add_argument("--smoke-evidence-limit", type=int, default=0)
    args = parser.parse_args()

    claims = limit_items(load_json(args.claims), args.smoke_claims)
    evidence = subset_evidence_for_claims(
        load_json(args.evidence), {}, claims, args.smoke_evidence_limit
    )
    pool_top_k = min(args.pool_top_k, len(evidence))
    top_k_values = sorted(set(min(top_k, pool_top_k) for top_k in parse_ints(args.top_k_values)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    analyzer = vectorizer.build_analyzer()

    bm25 = filter_pool_to_evidence(load_candidate_pool(args.bm25_pool), evidence.keys())
    char = filter_pool_to_evidence(load_candidate_pool(args.char_pool), evidence.keys())
    bm25 = trim_pool(bm25, pool_top_k)
    char = trim_pool(char, pool_top_k)

    if args.feedback_top_k > pool_top_k:
        args.feedback_top_k = pool_top_k

    baseline_rrf = merge_rrf(
        f"sparse_rrf_baseline_k{args.rrf_k}",
        [bm25, char],
        top_k=pool_top_k,
        rrf_k=args.rrf_k,
    ).pool

    expanded_claims, expansion_rows = build_expanded_claims(
        claims=claims,
        evidence=evidence,
        base_pool=baseline_rrf,
        max_feedback=args.feedback_top_k,
        max_terms=args.prf_max_terms,
        analyzer=analyzer,
    )
    expanded_bm25 = query_bm25_pool(expanded_claims, evidence, pool_top_k, k1=1.5, b=0.75)
    expanded_bm25 = trim_pool(expanded_bm25, pool_top_k)

    prf_rrf = merge_rrf(
        f"sparse_rrf_prf_k{args.rrf_k}",
        [bm25, char, expanded_bm25],
        top_k=pool_top_k,
        rrf_k=args.rrf_k,
    ).pool

    pools = {
        "bm25": bm25,
        "char": char,
        "baseline_rrf": baseline_rrf,
        "prf_bm25": expanded_bm25,
        "baseline_rrf_plus_prf_bm25": prf_rrf,
    }

    write_pool(output_dir / f"baseline_rrf_top{pool_top_k}.json", baseline_rrf)
    write_pool(output_dir / f"expanded_bm25_top{pool_top_k}.json", expanded_bm25)
    write_pool(output_dir / f"baseline_rrf_plus_prf_bm25_top{pool_top_k}.json", prf_rrf)

    rows = []
    for method, pool in pools.items():
        for retained_k in top_k_values:
            row = evaluate_pool(
                claims=claims,
                pool=pool,
                method=method,
                retained_k=retained_k,
                build_seconds=0.0,
            )
            row["candidate_union"] = candidate_union_size(pool, retained_k)
            row["union_fraction"] = row["candidate_union"] / len(evidence) if evidence else 0.0
            rows.append(row)

    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_terms = sum(item["expansion_terms"] for item in expansion_rows.values()) / len(expansion_rows) if expansion_rows else 0.0
    nonempty = sum(1 for item in expansion_rows.values() if item["expansion_terms"] > 0)
    write_json(
        output_dir / "summary.json",
        {
            "claims": len(claims),
            "evidence": len(evidence),
            "pool_top_k": pool_top_k,
            "top_k_values": top_k_values,
            "rrf_k": args.rrf_k,
            "feedback_top_k": args.feedback_top_k,
            "prf_max_terms": args.prf_max_terms,
            "expansion_row_count": len(expansion_rows),
            "claims_with_expansions": nonempty,
            "avg_expansion_terms": float(avg_terms),
            "rows": rows,
        },
    )
    for row in rows:
        print(
            f"{row['method']} k={row['retained_k']} "
            f"macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"union={row['candidate_union']}"
        )


if __name__ == "__main__":
    main()
