import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate
from a3_factcheck.retrieval.dense import (
    build_embedding_cache,
    device_from_arg,
    encode_texts,
    load_encoder,
    search_dense_topk,
    write_json,
)
from experiments.retrieval.evaluate_candidate_recall import (
    LABELS,
    evaluate_pool,
    parse_ints,
)


def build_dense_pool(claims, evidence_ids, search_results):
    pool = {}
    claim_ids = list(claims.keys())
    for claim_id, (indices, scores) in zip(claim_ids, search_results):
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_ids[index],
                rank=rank,
                score=float(score),
            )
            for rank, (index, score) in enumerate(zip(indices, scores), start=1)
        ]
    return pool


def write_pool(pool, output_path):
    jsonable = {
        claim_id: [asdict(candidate) for candidate in candidates]
        for claim_id, candidates in pool.items()
    }
    write_json(output_path, jsonable)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dense bi-encoder candidate recall with cached evidence embeddings."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round11/dense_bge_small")
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional shared evidence embedding cache directory.",
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k-values", default="50,100,200,500,1000,2000")
    parser.add_argument("--pool-top-k", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--evidence-prefix", default="")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--search-chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = parse_ints(args.top_k_values)
    pool_top_k = max([args.pool_top_k, *top_k_values])
    device = device_from_arg(args.device)

    start = time.perf_counter()
    evidence_ids, evidence_embeddings = build_embedding_cache(
        evidence=evidence,
        model_name=args.model,
        cache_dir=cache_dir,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        text_prefix=args.evidence_prefix,
        dtype=np.dtype(args.cache_dtype),
    )
    cache_seconds = time.perf_counter() - start

    tokenizer, model = load_encoder(args.model, device)
    claim_texts = [
        f"{args.query_prefix}{claim['claim_text']}" for claim in claims.values()
    ]
    query_embeddings = encode_texts(
        texts=claim_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.query_batch_size,
        max_length=args.query_max_length,
        pooling=args.pooling,
    )

    start = time.perf_counter()
    search_results = search_dense_topk(
        query_embeddings=query_embeddings,
        evidence_embeddings=evidence_embeddings,
        top_k=pool_top_k,
        chunk_size=args.search_chunk_size,
    )
    search_seconds = time.perf_counter() - start

    method = f"dense_{args.model.split('/')[-1].replace('-', '_')}_top{pool_top_k}"
    pool = build_dense_pool(claims, evidence_ids, search_results)
    write_pool(pool, candidate_dir / f"{method}.json")

    rows = []
    for retained_k in top_k_values:
        row = evaluate_pool(
            claims=claims,
            pool=pool,
            method=method,
            retained_k=retained_k,
            build_seconds=cache_seconds + search_seconds,
        )
        row["cache_seconds"] = cache_seconds
        row["search_seconds"] = search_seconds
        rows.append(row)

    fieldnames = list(rows[0].keys())
    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote candidate pool: {candidate_dir / f'{method}.json'}")
    for row in rows:
        label_bits = " ".join(
            f"{label.lower()}={row[f'{label.lower()}_macro_recall']:.4f}"
            for label in LABELS
        )
        print(
            f"{method} k={row['retained_k']} macro={row['macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f} all_gold={row['all_gold']:.4f} {label_bits}"
        )


if __name__ == "__main__":
    main()
