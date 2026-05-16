import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate
from a3_factcheck.retrieval.bm25 import build_bm25_similarities
from a3_factcheck.retrieval.tfidf import (
    build_tfidf_similarities,
    top_indices_from_sparse_row,
)


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


@dataclass(frozen=True)
class RankedPool:
    name: str
    pool: dict
    build_seconds: float


def parse_ints(value):
    return [int(part) for part in value.split(",") if part]


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


def build_word_tfidf_pool(claims, evidence, top_k, max_features):
    evidence_ids, similarities = build_tfidf_similarities(
        claims=claims,
        evidence=evidence,
        max_features=max_features,
    )
    return top_candidates_from_similarities(
        claims=claims,
        evidence_ids=evidence_ids,
        similarities=similarities,
        top_k=top_k,
    )


def build_char_tfidf_pool(claims, evidence, top_k, max_features):
    evidence_ids = list(evidence.keys())
    evidence_texts = [evidence[evidence_id] for evidence_id in evidence_ids]
    claim_texts = [claim["claim_text"] for claim in claims.values()]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        dtype=np.float32,
    )
    evidence_matrix = vectorizer.fit_transform(evidence_texts)
    claim_matrix = vectorizer.transform(claim_texts)
    similarities = claim_matrix @ evidence_matrix.T
    return top_candidates_from_similarities(
        claims=claims,
        evidence_ids=evidence_ids,
        similarities=similarities,
        top_k=top_k,
    )


def build_bm25_pool(claims, evidence, top_k, max_features, k1, b):
    evidence_ids, similarities = build_bm25_similarities(
        claims=claims,
        evidence=evidence,
        max_features=max_features,
        k1=k1,
        b=b,
    )
    return top_candidates_from_similarities(
        claims=claims,
        evidence_ids=evidence_ids,
        similarities=similarities,
        top_k=top_k,
    )


def merge_rrf(name, pools, top_k, rrf_k=60):
    merged = {}
    for claim_id in pools[0].keys():
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
    return RankedPool(name=name, pool=merged, build_seconds=0.0)


def evaluate_pool(claims, pool, method, retained_k, build_seconds):
    total_gold = 0
    total_tp = 0
    claim_recalls = []
    hit_any = 0
    all_gold = 0
    candidate_counts = []
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
        candidate_counts.append(len(candidates))
        label_recalls[claim.get("claim_label")].append(recall)

    row = {
        "method": method,
        "retained_k": retained_k,
        "claims": len(claims),
        "macro_recall": float(np.mean(claim_recalls)) if claim_recalls else 0.0,
        "micro_recall": total_tp / total_gold if total_gold else 0.0,
        "hit_any": hit_any / len(claims) if claims else 0.0,
        "all_gold": all_gold / len(claims) if claims else 0.0,
        "avg_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
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
        description="Evaluate candidate recall curves and sparse RRF unions."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round07")
    parser.add_argument("--top-k-values", default="50,100,200,500")
    parser.add_argument("--pool-top-k", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--char-max-features", type=int, default=300_000)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--skip-char", action="store_true")
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = parse_ints(args.top_k_values)
    pool_top_k = max([args.pool_top_k, *top_k_values])

    built = []

    start = time.perf_counter()
    bm25_pool = build_bm25_pool(
        claims=claims,
        evidence=evidence,
        top_k=pool_top_k,
        max_features=args.max_features,
        k1=args.k1,
        b=args.b,
    )
    built.append(
        RankedPool(
            name=f"bm25_top{pool_top_k}",
            pool=bm25_pool,
            build_seconds=time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    word_tfidf_pool = build_word_tfidf_pool(
        claims=claims,
        evidence=evidence,
        top_k=pool_top_k,
        max_features=args.max_features,
    )
    built.append(
        RankedPool(
            name=f"tfidf_word_top{pool_top_k}",
            pool=word_tfidf_pool,
            build_seconds=time.perf_counter() - start,
        )
    )

    if not args.skip_char:
        start = time.perf_counter()
        char_tfidf_pool = build_char_tfidf_pool(
            claims=claims,
            evidence=evidence,
            top_k=pool_top_k,
            max_features=args.char_max_features,
        )
        built.append(
            RankedPool(
                name=f"tfidf_char_top{pool_top_k}",
                pool=char_tfidf_pool,
                build_seconds=time.perf_counter() - start,
            )
        )

    pool_by_name = {ranked.name: ranked for ranked in built}
    bm25_name = f"bm25_top{pool_top_k}"
    word_name = f"tfidf_word_top{pool_top_k}"
    char_name = f"tfidf_char_top{pool_top_k}"
    if bm25_name in pool_by_name and word_name in pool_by_name:
        built.append(
            merge_rrf(
                name=f"rrf_bm25_word_tfidf_top{pool_top_k}",
                pools=[pool_by_name[bm25_name].pool, pool_by_name[word_name].pool],
                top_k=pool_top_k,
            )
        )
    if bm25_name in pool_by_name and char_name in pool_by_name:
        built.append(
            merge_rrf(
                name=f"rrf_bm25_char_tfidf_top{pool_top_k}",
                pools=[pool_by_name[bm25_name].pool, pool_by_name[char_name].pool],
                top_k=pool_top_k,
            )
        )
    if all(key in pool_by_name for key in [bm25_name, word_name, char_name]):
        built.append(
            merge_rrf(
                name=f"rrf_bm25_word_char_tfidf_top{pool_top_k}",
                pools=[
                    pool_by_name[bm25_name].pool,
                    pool_by_name[word_name].pool,
                    pool_by_name[char_name].pool,
                ],
                top_k=pool_top_k,
            )
        )

    rows = []
    for ranked in built:
        write_pool(ranked.pool, candidate_dir / f"{ranked.name}.json")
        for retained_k in top_k_values:
            rows.append(
                evaluate_pool(
                    claims=claims,
                    pool=ranked.pool,
                    method=ranked.name,
                    retained_k=retained_k,
                    build_seconds=ranked.build_seconds,
                )
            )

    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote candidate pools: {candidate_dir}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
