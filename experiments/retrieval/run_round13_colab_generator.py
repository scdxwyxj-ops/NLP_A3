import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate
from a3_factcheck.retrieval.dense import (
    build_embedding_cache,
    device_from_arg,
    encode_texts,
    load_encoder,
    search_dense_topk,
)
from experiments.retrieval.evaluate_candidate_recall import (
    build_bm25_pool,
    build_char_tfidf_pool,
    evaluate_pool,
    merge_rrf,
    write_pool,
)
from experiments.retrieval.train_recall_compressor import (
    build_rows,
    model_score,
    pool_index,
    rank_predictions,
    sampled_training_data,
)


def limit_items(mapping, limit):
    if limit <= 0:
        return dict(mapping)
    return {key: mapping[key] for key in list(mapping.keys())[:limit]}


def subset_evidence_for_claims(evidence, train_claims, target_claims, limit):
    if limit <= 0 or limit >= len(evidence):
        return dict(evidence)
    keep = []
    seen = set()
    for claims in (train_claims, target_claims):
        for claim in claims.values():
            for evidence_id in claim.get("evidences", []):
                if evidence_id in evidence and evidence_id not in seen:
                    keep.append(evidence_id)
                    seen.add(evidence_id)
    for evidence_id in evidence:
        if len(keep) >= limit:
            break
        if evidence_id not in seen:
            keep.append(evidence_id)
            seen.add(evidence_id)
    return {evidence_id: evidence[evidence_id] for evidence_id in keep}


def build_dense_pool(
    claims,
    evidence,
    model_name,
    cache_dir,
    output_dir,
    method_name,
    query_prefix,
    device,
    pool_top_k,
    batch_size,
    query_batch_size,
    max_length,
    query_max_length,
    pooling,
    search_chunk_size,
    cache_dtype,
):
    start = time.perf_counter()
    evidence_ids, evidence_embeddings = build_embedding_cache(
        evidence=evidence,
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        text_prefix="",
        dtype=np.dtype(cache_dtype),
    )
    cache_seconds = time.perf_counter() - start

    tokenizer, model = load_encoder(model_name, device)
    claim_texts = [f"{query_prefix}{claim['claim_text']}" for claim in claims.values()]
    query_embeddings = encode_texts(
        texts=claim_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=query_batch_size,
        max_length=query_max_length,
        pooling=pooling,
    )

    start = time.perf_counter()
    search_results = search_dense_topk(
        query_embeddings=query_embeddings,
        evidence_embeddings=evidence_embeddings,
        top_k=pool_top_k,
        chunk_size=search_chunk_size,
    )
    search_seconds = time.perf_counter() - start

    pool = {}
    for claim_id, (indices, scores) in zip(claims.keys(), search_results):
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_ids[index],
                rank=rank,
                score=float(score),
            )
            for rank, (index, score) in enumerate(zip(indices, scores), start=1)
        ]
    write_pool(pool, output_dir / "candidates" / f"{method_name}.json")
    return pool, {"cache_seconds": cache_seconds, "search_seconds": search_seconds}


def train_core_compressor(
    train_claims,
    target_claims,
    evidence,
    train_rrf,
    target_rrf,
    train_sources,
    target_sources,
    max_candidates,
    output_k,
    max_negatives_per_claim,
):
    train_indexes = {name: pool_index(pool) for name, pool in train_sources.items()}
    target_indexes = {name: pool_index(pool) for name, pool in target_sources.items()}
    X_train, y_train, train_row_claims, _train_eids = build_rows(
        train_claims,
        evidence,
        train_rrf,
        train_indexes,
        max_candidates,
        train_mode=True,
    )
    X_fit, y_fit = sampled_training_data(
        X_train, y_train, train_row_claims, max_negatives_per_claim
    )
    X_target, _y_target, target_row_claims, target_eids = build_rows(
        target_claims,
        evidence,
        target_rrf,
        target_indexes,
        max_candidates,
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
    model.fit(X_fit, y_fit)
    scores = model_score(model, X_target)
    pool = rank_predictions(target_claims, target_row_claims, target_eids, scores, output_k)
    summary = {
        "train_rows_total": int(X_train.shape[0]),
        "train_rows_fit": int(X_fit.shape[0]),
        "train_positive_rows": int(y_train.sum()),
        "target_rows": int(X_target.shape[0]),
    }
    return pool, summary


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Colab-safe Round13 top500 candidate generator."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--target-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round13_colab_generator")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pool-top-k", type=int, default=2000)
    parser.add_argument("--output-k", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--char-max-features", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--search-chunk-size", type=int, default=50_000)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-negatives-per-claim", type=int, default=0)
    parser.add_argument("--smoke-evidence-limit", type=int, default=0)
    parser.add_argument("--smoke-train-claims", type=int, default=0)
    parser.add_argument("--smoke-target-claims", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_claims = limit_items(load_json(args.train_claims), args.smoke_train_claims)
    target_claims = limit_items(load_json(args.target_claims), args.smoke_target_claims)
    evidence = subset_evidence_for_claims(
        load_json(args.evidence),
        train_claims,
        target_claims,
        args.smoke_evidence_limit,
    )
    pool_top_k = min(args.pool_top_k, len(evidence))
    output_k = min(args.output_k, pool_top_k)
    device = device_from_arg(args.device)

    print(
        json.dumps(
            {
                "train_claims": len(train_claims),
                "target_claims": len(target_claims),
                "evidence": len(evidence),
                "pool_top_k": pool_top_k,
                "output_k": output_k,
                "device": str(device),
                "cache_dtype": args.cache_dtype,
            },
            indent=2,
        )
    )

    train_bm25 = build_bm25_pool(
        train_claims, evidence, pool_top_k, args.max_features, k1=1.5, b=0.75
    )
    target_bm25 = build_bm25_pool(
        target_claims, evidence, pool_top_k, args.max_features, k1=1.5, b=0.75
    )
    train_char = build_char_tfidf_pool(
        train_claims, evidence, pool_top_k, args.char_max_features
    )
    target_char = build_char_tfidf_pool(
        target_claims, evidence, pool_top_k, args.char_max_features
    )
    write_pool(train_bm25, candidate_dir / "train_bm25.json")
    write_pool(target_bm25, candidate_dir / "target_bm25.json")
    write_pool(train_char, candidate_dir / "train_char.json")
    write_pool(target_char, candidate_dir / "target_char.json")

    cache_dir = output_dir / "dense_bge_base_cache"
    train_dense, train_dense_timing = build_dense_pool(
        train_claims,
        evidence,
        args.model,
        cache_dir,
        output_dir,
        "train_bge_base_plain",
        "",
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.cache_dtype,
    )
    target_dense, target_dense_timing = build_dense_pool(
        target_claims,
        evidence,
        args.model,
        cache_dir,
        output_dir,
        "target_bge_base_plain",
        "",
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.cache_dtype,
    )
    train_denseq, train_denseq_timing = build_dense_pool(
        train_claims,
        evidence,
        args.model,
        cache_dir,
        output_dir,
        "train_bge_base_qprefix",
        "Represent this sentence for searching relevant passages: ",
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.cache_dtype,
    )
    target_denseq, target_denseq_timing = build_dense_pool(
        target_claims,
        evidence,
        args.model,
        cache_dir,
        output_dir,
        "target_bge_base_qprefix",
        "Represent this sentence for searching relevant passages: ",
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.cache_dtype,
    )

    train_rrf = merge_rrf(
        "train_rrf_bm25_char_base_baseq_k500",
        [train_bm25, train_char, train_dense, train_denseq],
        top_k=pool_top_k,
        rrf_k=500,
    ).pool
    target_rrf = merge_rrf(
        "target_rrf_bm25_char_base_baseq_k500",
        [target_bm25, target_char, target_dense, target_denseq],
        top_k=pool_top_k,
        rrf_k=500,
    ).pool
    write_pool(train_rrf, candidate_dir / "train_rrf.json")
    write_pool(target_rrf, candidate_dir / "target_rrf.json")

    top_pool, compressor_summary = train_core_compressor(
        train_claims=train_claims,
        target_claims=target_claims,
        evidence=evidence,
        train_rrf=train_rrf,
        target_rrf=target_rrf,
        train_sources={
            "rrf": train_rrf,
            "bm25": train_bm25,
            "char": train_char,
            "dense": train_dense,
            "dense2": train_denseq,
        },
        target_sources={
            "rrf": target_rrf,
            "bm25": target_bm25,
            "char": target_char,
            "dense": target_dense,
            "dense2": target_denseq,
        },
        max_candidates=pool_top_k,
        output_k=output_k,
        max_negatives_per_claim=args.max_negatives_per_claim,
    )
    top_path = candidate_dir / f"round13_colab_core_top{output_k}.json"
    write_pool(top_pool, top_path)

    metrics = None
    if all("evidences" in claim for claim in target_claims.values()):
        metrics = evaluate_pool(
            target_claims,
            top_pool,
            method="round13_colab_core",
            retained_k=output_k,
            build_seconds=0.0,
        )
        with (output_dir / "candidate_recall_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print("METRICS", json.dumps(metrics, indent=2))

    summary = {
        "args": vars(args),
        "train_claims": len(train_claims),
        "target_claims": len(target_claims),
        "evidence": len(evidence),
        "pool_top_k": pool_top_k,
        "output_k": output_k,
        "top_path": str(top_path),
        "dense_timings": {
            "train_plain": train_dense_timing,
            "target_plain": target_dense_timing,
            "train_qprefix": train_denseq_timing,
            "target_qprefix": target_denseq_timing,
        },
        "compressor": compressor_summary,
        "metrics": metrics,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote {top_path}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
