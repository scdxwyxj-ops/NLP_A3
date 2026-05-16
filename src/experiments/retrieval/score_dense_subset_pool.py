import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from a3_factcheck.retrieval.dense import (
    build_embedding_cache,
    device_from_arg,
    encode_texts,
    load_encoder,
)
from experiments.retrieval.evaluate_candidate_recall import (
    evaluate_pool,
    parse_ints,
)


def write_pool(pool, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        claim_id: [asdict(candidate) for candidate in candidates]
        for claim_id, candidates in pool.items()
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_embedding_cache(cache_dir):
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    evidence_ids = json.loads((cache_dir / "evidence_ids.json").read_text(encoding="utf-8"))
    dtype = np.dtype(meta.get("dtype", "float32"))
    shape = (int(meta["count"]), int(meta["dim"]))
    embeddings = np.memmap(
        cache_dir / "evidence_embeddings.dat",
        dtype=dtype,
        mode="r",
        shape=shape,
    )
    return evidence_ids, embeddings, meta


def candidate_evidence_subset(candidate_pool, max_candidates):
    seen = set()
    ordered = []
    for candidates in candidate_pool.values():
        for candidate in candidates[:max_candidates]:
            if candidate.evidence_id not in seen:
                ordered.append(candidate.evidence_id)
                seen.add(candidate.evidence_id)
    return ordered


def subset_dict(evidence, evidence_ids):
    return {evidence_id: evidence[evidence_id] for evidence_id in evidence_ids}


def build_ranked_subset_pool(
    claims,
    candidate_pool,
    evidence_ids,
    evidence_embeddings,
    query_embeddings,
    max_candidates,
    output_k,
):
    evidence_index = {evidence_id: idx for idx, evidence_id in enumerate(evidence_ids)}
    claim_ids = list(claims.keys())
    result = {}
    for claim_idx, claim_id in enumerate(claim_ids):
        query = query_embeddings[claim_idx].astype(np.float32, copy=False)
        candidate_ids = []
        candidate_indices = []
        for candidate in candidate_pool.get(claim_id, [])[:max_candidates]:
            evidence_idx = evidence_index.get(candidate.evidence_id)
            if evidence_idx is not None:
                candidate_ids.append(candidate.evidence_id)
                candidate_indices.append(evidence_idx)
        if candidate_indices:
            candidate_matrix = np.asarray(evidence_embeddings[candidate_indices]).astype(
                np.float32, copy=False
            )
            scores = candidate_matrix @ query
            scored = [
                (evidence_id, float(score))
                for evidence_id, score in zip(candidate_ids, scores)
            ]
        else:
            scored = []
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:output_k]
        result[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=score,
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score an existing candidate pool with a dense encoder over only the "
            "candidate evidence subset, avoiding full-corpus dense search."
        )
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--evidence-prefix", default="")
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--reuse-cache-dir",
        default="",
        help="Optional existing full or subset evidence embedding cache to reuse.",
    )
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--pool-top-k", type=int, default=2000)
    parser.add_argument("--top-k-values", default="50,100,200,500,1000,2000")
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    candidate_pool = load_candidate_pool(args.candidate_pool)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_ints(args.top_k_values)
    pool_top_k = max([args.pool_top_k, *top_k_values])
    device = device_from_arg(args.device)

    start = time.perf_counter()
    subset_ids = candidate_evidence_subset(candidate_pool, args.max_candidates)
    if args.reuse_cache_dir:
        evidence_ids, evidence_embeddings, meta = load_existing_embedding_cache(
            args.reuse_cache_dir
        )
        if meta.get("model_name") != args.model:
            raise ValueError(
                f"Cache model {meta.get('model_name')} does not match {args.model}."
            )
    else:
        cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
        evidence_ids, evidence_embeddings = build_embedding_cache(
            evidence=subset_dict(evidence, subset_ids),
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
    query_texts = [
        f"{args.query_prefix}{claim['claim_text']}" for claim in claims.values()
    ]
    start = time.perf_counter()
    query_embeddings = encode_texts(
        texts=query_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.query_batch_size,
        max_length=args.query_max_length,
        pooling=args.pooling,
    )
    query_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pool = build_ranked_subset_pool(
        claims=claims,
        candidate_pool=candidate_pool,
        evidence_ids=evidence_ids,
        evidence_embeddings=evidence_embeddings,
        query_embeddings=query_embeddings,
        max_candidates=args.max_candidates,
        output_k=pool_top_k,
    )
    score_seconds = time.perf_counter() - start

    method = args.name or f"subset_dense_{args.model.split('/')[-1].replace('-', '_')}"
    candidate_path = candidate_dir / f"{method}.json"
    write_pool(pool, candidate_path)

    rows = []
    for retained_k in top_k_values:
        row = evaluate_pool(
            claims=claims,
            pool=pool,
            method=method,
            retained_k=retained_k,
            build_seconds=cache_seconds + query_seconds + score_seconds,
        )
        row["subset_evidence"] = len(subset_ids)
        row["cache_seconds"] = cache_seconds
        row["query_seconds"] = query_seconds
        row["score_seconds"] = score_seconds
        rows.append(row)

    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote candidate pool: {candidate_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Subset evidence: {len(subset_ids)}")
    for row in rows:
        print(
            f"{method} k={row['retained_k']} macro={row['macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f} all_gold={row['all_gold']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
