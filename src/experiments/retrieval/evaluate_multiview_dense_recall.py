import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from a3_factcheck.retrieval.dense import (
    build_embedding_cache,
    device_from_arg,
    encode_texts,
    load_encoder,
    search_dense_topk,
    write_json,
)
from a3_factcheck.retrieval.query_views import claim_query_views
from experiments.retrieval.evaluate_candidate_recall import (
    evaluate_pool,
    parse_ints,
)


def write_pool(pool, output_path):
    jsonable = {
        claim_id: [asdict(candidate) for candidate in candidates]
        for claim_id, candidates in pool.items()
    }
    write_json(output_path, jsonable)


def build_pool_from_view_results(claim_ids, evidence_ids, view_results_by_claim, top_k):
    pool = {}
    for claim_id in claim_ids:
        scores = defaultdict(float)
        for view_results in view_results_by_claim[claim_id]:
            indices, _raw_scores = view_results
            for rank, evidence_index in enumerate(indices, start=1):
                scores[evidence_ids[evidence_index]] += 1.0 / (60 + rank)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=float(score),
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return pool


def merge_rrf_pools(name, claims, pools, top_k, rrf_k=60):
    merged = {}
    for claim_id in claims:
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
    return name, merged


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-view dense retrieval and optional sparse fusion."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round11/multiview_dense")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--cache-dir", default="outputs/round11/dense_bge_small_top2000/cache")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--view-top-k", type=int, default=1000)
    parser.add_argument("--pool-top-k", type=int, default=2000)
    parser.add_argument("--top-k-values", default="50,100,200,500,1000,2000")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--search-chunk-size", type=int, default=200_000)
    parser.add_argument("--sparse-pool", action="append", default=[])
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = parse_ints(args.top_k_values)
    pool_top_k = max([args.pool_top_k, *top_k_values])
    device = device_from_arg(args.device)

    evidence_ids, evidence_embeddings = build_embedding_cache(
        evidence=evidence,
        model_name=args.model,
        cache_dir=Path(args.cache_dir),
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
    )
    tokenizer, model = load_encoder(args.model, device)

    claim_ids = list(claims.keys())
    views_by_claim = {
        claim_id: claim_query_views(claim["claim_text"], max_views=args.max_views)
        for claim_id, claim in claims.items()
    }
    write_json(output_dir / "claim_query_views.json", views_by_claim)

    flat_views = []
    owners = []
    for claim_id in claim_ids:
        for view in views_by_claim[claim_id]:
            owners.append(claim_id)
            flat_views.append(f"{args.query_prefix}{view}")

    start = time.perf_counter()
    query_embeddings = encode_texts(
        texts=flat_views,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.query_batch_size,
        max_length=args.query_max_length,
        pooling=args.pooling,
    )
    search_results = search_dense_topk(
        query_embeddings=query_embeddings,
        evidence_embeddings=evidence_embeddings,
        top_k=args.view_top_k,
        chunk_size=args.search_chunk_size,
        device=device,
        query_batch_size=args.query_batch_size,
    )
    build_seconds = time.perf_counter() - start

    view_results_by_claim = defaultdict(list)
    for claim_id, result in zip(owners, search_results):
        view_results_by_claim[claim_id].append(result)

    dense_pool = build_pool_from_view_results(
        claim_ids=claim_ids,
        evidence_ids=evidence_ids,
        view_results_by_claim=view_results_by_claim,
        top_k=pool_top_k,
    )
    method = f"multiview_dense_{args.model.split('/')[-1].replace('-', '_')}_views{args.max_views}_viewtop{args.view_top_k}"
    write_pool(dense_pool, candidate_dir / f"{method}.json")

    methods = [(method, dense_pool, build_seconds)]
    sparse_pools = [load_candidate_pool(path) for path in args.sparse_pool]
    if sparse_pools:
        fused_name, fused_pool = merge_rrf_pools(
            name=f"rrf_sparse_{method}",
            claims=claim_ids,
            pools=[*sparse_pools, dense_pool],
            top_k=pool_top_k,
        )
        write_pool(fused_pool, candidate_dir / f"{fused_name}.json")
        methods.append((fused_name, fused_pool, build_seconds))

    rows = []
    for name, pool, seconds in methods:
        for retained_k in top_k_values:
            row = evaluate_pool(
                claims=claims,
                pool=pool,
                method=name,
                retained_k=retained_k,
                build_seconds=seconds,
            )
            row["query_views"] = len(flat_views)
            row["avg_views_per_claim"] = len(flat_views) / len(claim_ids)
            rows.append(row)

    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote views: {output_dir / 'claim_query_views.json'}")
    print(f"Wrote summary: {summary_path}")
    for row in rows:
        print(
            f"{row['method']} k={row['retained_k']} macro={row['macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f} all_gold={row['all_gold']:.4f} "
            f"refutes={row['refutes_macro_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
