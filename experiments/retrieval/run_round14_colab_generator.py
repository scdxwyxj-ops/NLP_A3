import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from a3_factcheck.data import load_json
from a3_factcheck.retrieval.dense import (
    encode_texts,
    load_encoder,
    device_from_arg,
)
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf, write_pool
from experiments.retrieval.run_round13_colab_generator import (
    build_dense_pool,
    build_restricted_embedding_cache,
    build_sparse_pools,
    combined_claims,
    limit_items,
    log_stage,
    restricted_dense_search,
    split_pool,
    subset_evidence_for_claims,
)
from experiments.retrieval.train_component_recall_compressor import (
    build_rows,
    compression_loss,
    feature_names,
    pool_index,
    quota_pool,
    rank_predictions,
    sampled_indexes,
)
from experiments.retrieval.train_recall_compressor import model_score


BASE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def candidate_evidence_ids(pool, evidence, top_k):
    seen = set()
    ordered = []
    for candidates in pool.values():
        for candidate in candidates[:top_k]:
            if candidate.evidence_id in evidence and candidate.evidence_id not in seen:
                ordered.append(candidate.evidence_id)
                seen.add(candidate.evidence_id)
    return ordered


def build_subset_baseq_pool(
    claims,
    evidence,
    candidate_pool,
    model_name,
    cache_dir,
    device,
    pool_top_k,
    batch_size,
    query_batch_size,
    max_length,
    query_max_length,
    pooling,
    cache_dtype,
):
    log_stage(f"building subset baseq pool {model_name}")
    evidence_ids = candidate_evidence_ids(candidate_pool, evidence, pool_top_k)
    log_stage(f"subset baseq evidence union {len(evidence_ids)}/{len(evidence)}")

    tokenizer, model = load_encoder(model_name, device)
    start = time.perf_counter()
    evidence_ids, evidence_embeddings = build_restricted_embedding_cache(
        evidence=evidence,
        evidence_ids=evidence_ids,
        model_name=model_name,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        cache_dtype=cache_dtype,
    )
    cache_seconds = time.perf_counter() - start

    start = time.perf_counter()
    claim_texts = [f"{BASE_QUERY_PREFIX}{claim['claim_text']}" for claim in claims.values()]
    query_embeddings = encode_texts(
        texts=claim_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=query_batch_size,
        max_length=query_max_length,
        pooling=pooling,
    )
    query_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pool = restricted_dense_search(
        claims=claims,
        prefilter_pool=candidate_pool,
        evidence_ids=evidence_ids,
        evidence_embeddings=evidence_embeddings,
        query_embeddings=query_embeddings,
        top_k=pool_top_k,
        prefilter_top_k=pool_top_k,
    )
    search_seconds = time.perf_counter() - start

    del query_embeddings, model, tokenizer
    gc.collect()
    log_stage("finished subset baseq pool")
    return pool, {
        "cache_seconds": cache_seconds,
        "query_seconds": query_seconds,
        "search_seconds": search_seconds,
        "encoded_evidence": len(evidence_ids),
    }


def train_component_compressor(
    train_claims,
    target_claims,
    evidence,
    train_candidate_pool,
    target_candidate_pool,
    train_sources,
    target_sources,
    max_candidates,
    output_k,
    max_negatives_per_claim,
    use_nei_weights,
):
    source_names = list(train_sources)
    train_indexes = {name: pool_index(pool) for name, pool in train_sources.items()}
    target_indexes = {name: pool_index(pool) for name, pool in target_sources.items()}
    X_train, y_train, weights, train_row_claims, _train_eids = build_rows(
        train_claims,
        evidence,
        train_candidate_pool,
        train_indexes,
        source_names,
        max_candidates,
        train_mode=True,
    )
    keep = sampled_indexes(y_train, train_row_claims, max_negatives_per_claim)
    X_fit = X_train[keep]
    y_fit = y_train[keep]
    weights_fit = weights[keep] if use_nei_weights else None

    X_target, _y_target, _weights_target, target_row_claims, target_eids = build_rows(
        target_claims,
        evidence,
        target_candidate_pool,
        target_indexes,
        source_names,
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
    model.fit(X_fit, y_fit, sample_weight=weights_fit)
    scores = model_score(model, X_target)
    scored_pool = rank_predictions(
        target_claims,
        target_row_claims,
        target_eids,
        scores,
        max_candidates,
    )
    variants = {
        "component_histgbdt_top500": rank_predictions(
            target_claims,
            target_row_claims,
            target_eids,
            scores,
            output_k,
        ),
        "component_quota400_top500": quota_pool(
            target_claims,
            scored_pool,
            target_sources,
            base_k=min(400, output_k),
            quotas=[("baseq", 50), ("bm25", 25), ("char", 25)],
            output_k=output_k,
        ),
        "component_quota450_top500": quota_pool(
            target_claims,
            scored_pool,
            target_sources,
            base_k=min(450, output_k),
            quotas=[("baseq", 30), ("bm25", 10), ("char", 10)],
            output_k=output_k,
        ),
    }
    summary = {
        "feature_names": feature_names(source_names),
        "source_names": source_names,
        "train_rows_total": int(X_train.shape[0]),
        "train_rows_fit": int(X_fit.shape[0]),
        "train_positive_rows": int(y_train.sum()),
        "target_rows": int(X_target.shape[0]),
        "max_candidates": max_candidates,
        "output_k": output_k,
        "nei_weights": use_nei_weights,
    }
    return variants, summary


def main():
    parser = argparse.ArgumentParser(
        description="Round14 Colab-safe small-first top500 candidate generator."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--target-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round14_colab_generator")
    parser.add_argument("--small-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--base-subset-model", default="BAAI/bge-base-en-v1.5")
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
    parser.add_argument("--small-cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--base-cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-negatives-per-claim", type=int, default=0)
    parser.add_argument("--no-nei-weights", action="store_true")
    parser.add_argument(
        "--sparse-mode",
        choices=["exact", "query"],
        default="query",
    )
    parser.add_argument("--query-char-max-claim-fanout", type=int, default=64)
    parser.add_argument("--query-char-max-df-ratio", type=float, default=0.25)
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

    config = {
        "train_claims": len(train_claims),
        "target_claims": len(target_claims),
        "evidence": len(evidence),
        "pool_top_k": pool_top_k,
        "output_k": output_k,
        "device": str(device),
        "small_model": args.small_model,
        "base_subset_model": args.base_subset_model,
        "small_cache_dtype": args.small_cache_dtype,
        "base_cache_dtype": args.base_cache_dtype,
        "sparse_mode": args.sparse_mode,
    }
    print(json.dumps(config, indent=2), flush=True)

    train_bm25, target_bm25, train_char, target_char = build_sparse_pools(
        train_claims,
        target_claims,
        evidence,
        pool_top_k,
        args,
    )
    write_pool(train_bm25, candidate_dir / "train_bm25.json")
    write_pool(target_bm25, candidate_dir / "target_bm25.json")
    write_pool(train_char, candidate_dir / "train_char.json")
    write_pool(target_char, candidate_dir / "target_char.json")

    small_cache_dir = output_dir / "dense_bge_small_cache"
    train_small, train_small_timing = build_dense_pool(
        train_claims,
        evidence,
        args.small_model,
        small_cache_dir,
        output_dir,
        "train_bge_small_plain",
        "",
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.small_cache_dtype,
    )
    target_small, target_small_timing = build_dense_pool(
        target_claims,
        evidence,
        args.small_model,
        small_cache_dir,
        output_dir,
        "target_bge_small_plain",
        "",
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.small_cache_dtype,
    )
    train_smallq, train_smallq_timing = build_dense_pool(
        train_claims,
        evidence,
        args.small_model,
        small_cache_dir,
        output_dir,
        "train_bge_small_qprefix",
        BASE_QUERY_PREFIX,
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.small_cache_dtype,
    )
    target_smallq, target_smallq_timing = build_dense_pool(
        target_claims,
        evidence,
        args.small_model,
        small_cache_dir,
        output_dir,
        "target_bge_small_qprefix",
        BASE_QUERY_PREFIX,
        device,
        pool_top_k,
        args.batch_size,
        args.query_batch_size,
        args.max_length,
        args.query_max_length,
        args.pooling,
        args.search_chunk_size,
        args.small_cache_dtype,
    )

    train_rrf = merge_rrf(
        "train_rrf_bm25_char_small_smallq_k500",
        [train_bm25, train_char, train_small, train_smallq],
        top_k=pool_top_k,
        rrf_k=500,
    ).pool
    target_rrf = merge_rrf(
        "target_rrf_bm25_char_small_smallq_k500",
        [target_bm25, target_char, target_small, target_smallq],
        top_k=pool_top_k,
        rrf_k=500,
    ).pool
    write_pool(train_rrf, candidate_dir / "train_rrf_small_only.json")
    write_pool(target_rrf, candidate_dir / "target_rrf_small_only.json")

    all_claims = combined_claims(train_claims, target_claims)
    all_rrf = {**train_rrf, **target_rrf}
    baseq_all, baseq_timing = build_subset_baseq_pool(
        claims=all_claims,
        evidence=evidence,
        candidate_pool=all_rrf,
        model_name=args.base_subset_model,
        cache_dir=output_dir / "dense_bge_base_subset_cache",
        device=device,
        pool_top_k=pool_top_k,
        batch_size=args.batch_size,
        query_batch_size=args.query_batch_size,
        max_length=args.max_length,
        query_max_length=args.query_max_length,
        pooling=args.pooling,
        cache_dtype=args.base_cache_dtype,
    )
    train_baseq = split_pool(baseq_all, train_claims)
    target_baseq = split_pool(baseq_all, target_claims)
    write_pool(train_baseq, candidate_dir / "train_bge_baseq_subset.json")
    write_pool(target_baseq, candidate_dir / "target_bge_baseq_subset.json")

    train_sources = {
        "bm25": train_bm25,
        "char": train_char,
        "small": train_small,
        "smallq": train_smallq,
        "baseq": train_baseq,
    }
    target_sources = {
        "bm25": target_bm25,
        "char": target_char,
        "small": target_small,
        "smallq": target_smallq,
        "baseq": target_baseq,
    }
    variants, compressor_summary = train_component_compressor(
        train_claims=train_claims,
        target_claims=target_claims,
        evidence=evidence,
        train_candidate_pool=train_rrf,
        target_candidate_pool=target_rrf,
        train_sources=train_sources,
        target_sources=target_sources,
        max_candidates=pool_top_k,
        output_k=output_k,
        max_negatives_per_claim=args.max_negatives_per_claim,
        use_nei_weights=not args.no_nei_weights,
    )

    rows = []
    loss_rows = []
    metrics = None
    for name, pool in variants.items():
        path = candidate_dir / f"{name}.json"
        write_pool(pool, path)
        if all("evidences" in claim for claim in target_claims.values()):
            row = evaluate_pool(target_claims, pool, name, output_k, 0.0)
            rows.append(row)
            loss = compression_loss(target_claims, target_rrf, pool)
            loss["method"] = name
            loss_rows.append(loss)
            if metrics is None or row["macro_recall"] > metrics["macro_recall"]:
                metrics = row
            print("METRICS", json.dumps(row, indent=2), flush=True)

    if rows:
        with (output_dir / "candidate_recall_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / "compression_loss.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(loss_rows[0].keys()))
            writer.writeheader()
            writer.writerows(loss_rows)

    summary = {
        "args": vars(args),
        "config": config,
        "timings": {
            "train_small": train_small_timing,
            "target_small": target_small_timing,
            "train_smallq": train_smallq_timing,
            "target_smallq": target_smallq_timing,
            "baseq_subset": baseq_timing,
        },
        "compressor": compressor_summary,
        "best_metrics": metrics,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
