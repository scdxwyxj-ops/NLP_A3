import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from a3_factcheck.retrieval.bm25 import build_bm25_matrix
from a3_factcheck.retrieval.tfidf import top_indices_from_sparse_row


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "between",
    "could",
    "does",
    "during",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "over",
    "same",
    "some",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "under",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def parse_ints(value):
    return [int(part) for part in value.split(",") if part]


def extract_salient_terms(text):
    numbers = re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|per cent|percent|ppm|billion|million|tonnes?|tons?)?\b", text, flags=re.I)
    formulas = re.findall(r"\b(?:CO2|CO 2|CH4|N2O|GHG|IPCC|UNEP|NASA|NOAA)\b", text, flags=re.I)
    capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    long_terms = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z-]{4,}", text.lower())
        if token not in STOPWORDS
    ]

    terms = []
    seen = set()
    for term in [*numbers, *formulas, *capitalized, *long_terms]:
        normalized = " ".join(term.lower().split())
        if normalized and normalized not in seen:
            terms.append(term)
            seen.add(normalized)
    return terms


def boosted_query_text(claim_text, boost):
    salient = extract_salient_terms(claim_text)
    if not salient:
        return claim_text
    return " ".join([claim_text, *salient * boost])


def top_candidates_from_similarities(claims, evidence_ids, similarities, top_k):
    pool = {}
    for row_idx, claim_id in enumerate(claims.keys()):
        row = similarities.getrow(row_idx)
        evidence_indexes = top_indices_from_sparse_row(row, top_k, len(evidence_ids))
        row_scores = dict(zip(row.indices.tolist(), row.data.tolist()))
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_ids[evidence_idx],
                rank=rank,
                score=float(row_scores.get(evidence_idx, 0.0)),
            )
            for rank, evidence_idx in enumerate(evidence_indexes, start=1)
        ]
    return pool


def build_boosted_bm25_pool(claims, evidence, top_k, max_features, k1, b, boost):
    evidence_ids = list(evidence.keys())
    evidence_texts = [evidence[evidence_id] for evidence_id in evidence_ids]
    claim_texts = [
        boosted_query_text(claim["claim_text"], boost=boost)
        for claim in claims.values()
    ]

    bm25_matrix, vectorizer = build_bm25_matrix(
        evidence_texts=evidence_texts,
        max_features=max_features,
        k1=k1,
        b=b,
    )
    query_counts = vectorizer.transform(claim_texts)
    similarities = query_counts @ bm25_matrix.T
    return top_candidates_from_similarities(
        claims=claims,
        evidence_ids=evidence_ids,
        similarities=similarities,
        top_k=top_k,
    )


def merge_rrf(pools, top_k, rrf_k):
    merged = {}
    claim_ids = sorted({claim_id for pool in pools for claim_id in pool.keys()})
    for claim_id in claim_ids:
        scores = defaultdict(float)
        for pool in pools:
            for candidate in pool.get(claim_id, []):
                scores[candidate.evidence_id] += 1.0 / (rrf_k + candidate.rank)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        merged[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=float(score),
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return merged


def evaluate_pool(claims, pool, method, retained_k, build_seconds):
    total_gold = 0
    total_tp = 0
    claim_recalls = []
    hit_any = 0
    all_gold = 0
    label_recalls = defaultdict(list)

    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        candidates = pool.get(claim_id, [])[:retained_k]
        predicted = {candidate.evidence_id for candidate in candidates}
        tp = len(gold & predicted)
        recall = tp / len(gold) if gold else 0.0

        total_gold += len(gold)
        total_tp += tp
        claim_recalls.append(recall)
        hit_any += int(tp > 0)
        all_gold += int(bool(gold) and tp == len(gold))
        label_recalls[claim.get("claim_label")].append(recall)

    row = {
        "method": method,
        "retained_k": retained_k,
        "claims": len(claims),
        "macro_recall": float(np.mean(claim_recalls)) if claim_recalls else 0.0,
        "micro_recall": total_tp / total_gold if total_gold else 0.0,
        "hit_any": hit_any / len(claims) if claims else 0.0,
        "all_gold": all_gold / len(claims) if claims else 0.0,
        "build_seconds": build_seconds,
    }
    for label in LABELS:
        recalls = label_recalls.get(label, [])
        row[f"{label.lower()}_macro_recall"] = (
            float(np.mean(recalls)) if recalls else 0.0
        )
    return row


def write_pool(pool, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonable = {
        claim_id: [asdict(candidate) for candidate in candidates]
        for claim_id, candidates in pool.items()
    }
    output_path.write_text(json.dumps(jsonable, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate salient-query boosted BM25 candidate recall."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round07_query_boost")
    parser.add_argument("--base-pool", action="append", default=[])
    parser.add_argument("--base-name", action="append", default=[])
    parser.add_argument("--top-k-values", default="50,100,200,500")
    parser.add_argument("--pool-top-k", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--boost", type=int, default=4)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_ints(args.top_k_values)
    pool_top_k = max([args.pool_top_k, *top_k_values])

    names = args.base_name or [
        Path(path).stem for path in args.base_pool
    ]
    if len(names) != len(args.base_pool):
        raise ValueError("--base-name must be provided once per --base-pool")

    built = []
    for name, path in zip(names, args.base_pool):
        built.append((name, load_candidate_pool(path), 0.0))

    start = time.perf_counter()
    boosted_pool = build_boosted_bm25_pool(
        claims=claims,
        evidence=evidence,
        top_k=pool_top_k,
        max_features=args.max_features,
        k1=args.k1,
        b=args.b,
        boost=args.boost,
    )
    boosted_seconds = time.perf_counter() - start
    boosted_name = f"bm25_salient_boost{args.boost}_top{pool_top_k}"
    built.append((boosted_name, boosted_pool, boosted_seconds))

    if args.base_pool:
        for name, pool, _ in list(built):
            if name == boosted_name:
                continue
            merged_name = f"rrf_{name}_{boosted_name}"
            built.append(
                (
                    merged_name,
                    merge_rrf([pool, boosted_pool], top_k=pool_top_k, rrf_k=args.rrf_k),
                    boosted_seconds,
                )
            )

        if len(args.base_pool) > 1:
            base_pools = [pool for name, pool, _ in built[: len(args.base_pool)]]
            built.append(
                (
                    f"rrf_all_bases_{boosted_name}",
                    merge_rrf([*base_pools, boosted_pool], top_k=pool_top_k, rrf_k=args.rrf_k),
                    boosted_seconds,
                )
            )

    rows = []
    for name, pool, build_seconds in built:
        write_pool(pool, candidate_dir / f"{name}.json")
        for retained_k in top_k_values:
            rows.append(
                evaluate_pool(
                    claims=claims,
                    pool=pool,
                    method=name,
                    retained_k=retained_k,
                    build_seconds=build_seconds,
                )
            )

    summary_path = output_dir / "query_boost_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote candidate pools: {candidate_dir}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
