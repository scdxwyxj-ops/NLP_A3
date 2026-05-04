import argparse
import csv
import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool
from a3_factcheck.retrieval.dense import (
    encode_texts,
    load_encoder,
    device_from_arg,
)
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf, write_pool
from experiments.retrieval.evaluate_candidate_recall import (
    build_bm25_pool,
    build_char_tfidf_pool,
)
from experiments.retrieval.run_round13_colab_generator import (
    build_dense_pool,
    build_restricted_embedding_cache,
    build_sparse_pools,
    combined_claims,
    heaps_to_pool,
    limit_items,
    log_stage,
    prefilter_evidence_ids,
    push_top,
    query_bm25_pool,
    query_char_tfidf_pool,
    restricted_dense_search,
    split_pool,
    subset_evidence_for_claims,
)
from experiments.retrieval.score_dense_subset_pool import load_existing_embedding_cache
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


def trim_pool(pool, top_k):
    return {
        claim_id: list(candidates[:top_k])
        for claim_id, candidates in pool.items()
    }


def validate_reuse_cache(meta, model_name, pooling, max_length, cache_dir):
    if meta.get("model_name") != model_name:
        raise ValueError(
            f"Cache model {meta.get('model_name')} in {cache_dir} "
            f"does not match {model_name}."
        )
    if meta.get("pooling") != pooling:
        raise ValueError(
            f"Cache pooling {meta.get('pooling')} in {cache_dir} "
            f"does not match {pooling}."
        )
    if meta.get("max_length") != max_length:
        log_stage(
            "warning: reused dense cache max_length "
            f"{meta.get('max_length')} differs from requested {max_length}"
        )


def build_sparse_pair(name, claims, evidence, top_k, args):
    if args.source_pool_dir:
        source_dir = Path(args.source_pool_dir)
        bm25_path = source_dir / f"{name}_bm25_top{top_k}.json"
        char_path = source_dir / f"{name}_char_top{top_k}.json"
        if bm25_path.exists() and char_path.exists():
            log_stage(f"loading {name} sparse sources top{top_k} from {source_dir}")
            return load_candidate_pool(bm25_path), load_candidate_pool(char_path)

    if args.sparse_mode == "query":
        log_stage(f"building {name} query BM25 pool top{top_k}")
        bm25 = query_bm25_pool(claims, evidence, top_k, k1=1.5, b=0.75)
        gc.collect()
        log_stage(f"building {name} query char TF-IDF pool top{top_k}")
        char = query_char_tfidf_pool(
            claims,
            evidence,
            top_k,
            max_claim_fanout=args.query_char_max_claim_fanout,
            max_df_ratio=args.query_char_max_df_ratio,
        )
        gc.collect()
        return bm25, char

    log_stage(f"building {name} exact BM25 pool top{top_k}")
    bm25 = build_bm25_pool(
        claims, evidence, top_k, args.max_features, k1=1.5, b=0.75
    )
    gc.collect()
    log_stage(f"building {name} exact char TF-IDF pool top{top_k}")
    char = build_char_tfidf_pool(claims, evidence, top_k, args.char_max_features)
    gc.collect()
    return bm25, char


def query_word_tfidf_pool(all_claims, evidence, top_k):
    """Memory-bounded word TF-IDF over full evidence, restricted to claim terms."""
    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    analyzer = vectorizer.build_analyzer()
    claim_counts = {
        claim_id: Counter(analyzer(claim["claim_text"]))
        for claim_id, claim in all_claims.items()
    }
    term_claims = defaultdict(list)
    for claim_id, counts in claim_counts.items():
        for term in counts:
            term_claims[term].append(claim_id)
    query_vocab = set(term_claims)

    evidence_items = list(evidence.items())
    df = Counter()
    for idx, (_evidence_id, text) in enumerate(evidence_items):
        seen = {term for term in analyzer(text) if term in query_vocab}
        df.update(seen)
        if idx and idx % 100000 == 0:
            log_stage(f"query word TF-IDF df pass {idx}/{len(evidence_items)}")

    n_docs = len(evidence_items)
    idf = {
        term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
        for term, count in df.items()
    }
    claim_ids = list(all_claims.keys())
    claim_index = {claim_id: idx for idx, claim_id in enumerate(claim_ids)}
    claim_weights = {}
    for claim_id, counts in claim_counts.items():
        weights = {
            term: float(tf) * idf[term]
            for term, tf in counts.items()
            if term in idf
        }
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm > 0.0:
            weights = {term: value / norm for term, value in weights.items()}
        claim_weights[claim_id] = weights

    term_claim_weights = defaultdict(list)
    for claim_id, weights in claim_weights.items():
        for term, weight in weights.items():
            term_claim_weights[term].append((claim_index[claim_id], weight))

    heaps = {claim_id: [] for claim_id in all_claims}
    for idx, (evidence_id, text) in enumerate(evidence_items):
        counts = Counter(term for term in analyzer(text) if term in idf)
        if counts:
            weights = {term: float(tf) * idf[term] for term, tf in counts.items()}
            norm = math.sqrt(sum(value * value for value in weights.values()))
            if norm > 0.0:
                scores = defaultdict(float)
                for term, value in weights.items():
                    evidence_weight = value / norm
                    for claim_idx, claim_weight in term_claim_weights[term]:
                        scores[claim_ids[claim_idx]] += evidence_weight * claim_weight
                for claim_id, score in scores.items():
                    push_top(heaps[claim_id], top_k, score, evidence_id)
        if idx and idx % 100000 == 0:
            log_stage(f"query word TF-IDF scoring pass {idx}/{len(evidence_items)}")

    return heaps_to_pool(all_claims, heaps)


def build_word_gate_pair(name, claims, evidence, top_k):
    log_stage(f"building {name} query word TF-IDF gate top{top_k}")
    pool = query_word_tfidf_pool(claims, evidence, top_k)
    gc.collect()
    return pool


def build_sparse_gated_dense_pools(
    claims,
    evidence,
    prefilter_pool,
    model_name,
    cache_dir,
    reuse_cache_dir,
    device,
    pool_top_k,
    prefilter_top_k,
    batch_size,
    query_batch_size,
    max_length,
    query_max_length,
    pooling,
    cache_dtype,
):
    log_stage(
        "building sparse-gated dense pools "
        f"{model_name} (prefilter_top_k={prefilter_top_k})"
    )
    subset_ids = prefilter_evidence_ids(
        claims=claims,
        evidence=evidence,
        prefilter_pool=prefilter_pool,
        prefilter_top_k=prefilter_top_k,
    )
    log_stage(f"sparse-gated dense evidence union {len(subset_ids)}/{len(evidence)}")

    tokenizer, model = load_encoder(model_name, device)
    start = time.perf_counter()
    if reuse_cache_dir:
        evidence_ids, evidence_embeddings, meta = load_existing_embedding_cache(
            reuse_cache_dir
        )
        validate_reuse_cache(meta, model_name, pooling, max_length, reuse_cache_dir)
    else:
        evidence_ids, evidence_embeddings = build_restricted_embedding_cache(
            evidence=evidence,
            evidence_ids=subset_ids,
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

    timings = {}
    pools = {}
    for key, query_prefix in {
        "plain": "",
        "qprefix": BASE_QUERY_PREFIX,
    }.items():
        start = time.perf_counter()
        claim_texts = [
            f"{query_prefix}{claim['claim_text']}" for claim in claims.values()
        ]
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
        pools[key] = restricted_dense_search(
            claims=claims,
            prefilter_pool=prefilter_pool,
            evidence_ids=evidence_ids,
            evidence_embeddings=evidence_embeddings,
            query_embeddings=query_embeddings,
            top_k=pool_top_k,
            prefilter_top_k=prefilter_top_k,
        )
        search_seconds = time.perf_counter() - start
        timings[key] = {
            "cache_seconds": cache_seconds if key == "plain" else 0.0,
            "query_seconds": query_seconds,
            "search_seconds": search_seconds,
            "subset_evidence": len(subset_ids),
            "cache_evidence": len(evidence_ids),
            "prefilter_top_k": prefilter_top_k,
            "reuse_cache_dir": str(reuse_cache_dir) if reuse_cache_dir else "",
        }
        del query_embeddings
        gc.collect()

    del model, tokenizer
    gc.collect()
    log_stage("finished sparse-gated dense pools")
    return pools["plain"], pools["qprefix"], timings["plain"], timings["qprefix"]


def build_subset_baseq_pool(
    claims,
    evidence,
    candidate_pool,
    model_name,
    cache_dir,
    reuse_cache_dir,
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
    subset_evidence = len(evidence_ids)
    if reuse_cache_dir:
        evidence_ids, evidence_embeddings, meta = load_existing_embedding_cache(
            reuse_cache_dir
        )
        validate_reuse_cache(meta, model_name, pooling, max_length, reuse_cache_dir)
    else:
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
        "subset_evidence": subset_evidence,
        "cache_evidence": len(evidence_ids),
        "reuse_cache_dir": str(reuse_cache_dir) if reuse_cache_dir else "",
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
    }
    if "baseq" in target_sources:
        quota400 = [("baseq", 50), ("bm25", 25), ("char", 25)]
        quota450 = [("baseq", 30), ("bm25", 10), ("char", 10)]
    else:
        quota400 = [("smallq", 40), ("small", 30), ("bm25", 15), ("char", 15)]
        quota450 = [("smallq", 20), ("small", 15), ("bm25", 8), ("char", 7)]
    variants["component_quota400_top500"] = quota_pool(
        target_claims,
        scored_pool,
        target_sources,
        base_k=min(400, output_k),
        quotas=[item for item in quota400 if item[0] in target_sources],
        output_k=output_k,
    )
    variants["component_quota450_top500"] = quota_pool(
        target_claims,
        scored_pool,
        target_sources,
        base_k=min(450, output_k),
        quotas=[item for item in quota450 if item[0] in target_sources],
        output_k=output_k,
    )
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
    parser.add_argument(
        "--small-dense-mode",
        choices=["full", "sparse-gate"],
        default="full",
        help=(
            "full searches BGE-small over every evidence item; sparse-gate first "
            "uses a BM25/char RRF gate and scores BGE-small only inside that gate."
        ),
    )
    parser.add_argument("--small-prefilter-top-k", type=int, default=12_000)
    parser.add_argument(
        "--train-small-prefilter-top-k",
        type=int,
        default=0,
        help="Optional smaller sparse gate for training claims. Use 0 to match target.",
    )
    parser.add_argument(
        "--small-reuse-cache-dir",
        default="",
        help="Optional local-only full/subset BGE-small evidence cache for faster experiments.",
    )
    parser.add_argument(
        "--base-reuse-cache-dir",
        default="",
        help="Optional local-only full/subset BGE-base evidence cache for faster experiments.",
    )
    parser.add_argument(
        "--skip-base-subset",
        action="store_true",
        help="Do not run the optional BGE-base subset reranker after the BGE-small stage.",
    )
    parser.add_argument("--write-small-prefilter-pool", action="store_true")
    parser.add_argument(
        "--source-pool-dir",
        default="",
        help="Optional directory containing train/dev BM25 and char source pools.",
    )
    parser.add_argument(
        "--small-gate-word-tfidf",
        action="store_true",
        help="Add a word TF-IDF sparse view to the BGE-small sparse gate.",
    )
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
    target_small_prefilter_top_k = min(args.small_prefilter_top_k, len(evidence))
    train_small_prefilter_top_k = min(
        args.train_small_prefilter_top_k or args.small_prefilter_top_k,
        len(evidence),
    )
    train_sparse_pool_top_k = pool_top_k
    target_sparse_pool_top_k = pool_top_k
    if args.small_dense_mode == "sparse-gate":
        train_sparse_pool_top_k = max(pool_top_k, train_small_prefilter_top_k)
        target_sparse_pool_top_k = max(pool_top_k, target_small_prefilter_top_k)
    sparse_pool_top_k = max(train_sparse_pool_top_k, target_sparse_pool_top_k)

    config = {
        "train_claims": len(train_claims),
        "target_claims": len(target_claims),
        "evidence": len(evidence),
        "pool_top_k": pool_top_k,
        "sparse_pool_top_k": sparse_pool_top_k,
        "train_sparse_pool_top_k": train_sparse_pool_top_k,
        "target_sparse_pool_top_k": target_sparse_pool_top_k,
        "output_k": output_k,
        "device": str(device),
        "small_model": args.small_model,
        "base_subset_model": args.base_subset_model,
        "small_cache_dtype": args.small_cache_dtype,
        "base_cache_dtype": args.base_cache_dtype,
        "small_dense_mode": args.small_dense_mode,
        "small_prefilter_top_k": target_small_prefilter_top_k,
        "train_small_prefilter_top_k": train_small_prefilter_top_k,
        "small_reuse_cache_dir": args.small_reuse_cache_dir,
        "base_reuse_cache_dir": args.base_reuse_cache_dir,
        "skip_base_subset": args.skip_base_subset,
        "source_pool_dir": args.source_pool_dir,
        "small_gate_word_tfidf": args.small_gate_word_tfidf,
        "sparse_mode": args.sparse_mode,
        "colab_safe_review": {
            "stage1_uses_bge": False,
            "stage2_uses_bge_small_only_inside_sparse_gate": (
                args.small_dense_mode == "sparse-gate"
            ),
            "uses_bge_base": not args.skip_base_subset,
            "requires_precomputed_cache_or_checkpoint": bool(
                args.small_reuse_cache_dir or args.base_reuse_cache_dir
            ),
            "writes_runtime_cache": True,
            "runtime_cache_is_submission_dependency": False,
        },
    }
    print(json.dumps(config, indent=2), flush=True)

    if train_sparse_pool_top_k == target_sparse_pool_top_k:
        train_bm25_all, target_bm25_all, train_char_all, target_char_all = build_sparse_pools(
            train_claims,
            target_claims,
            evidence,
            sparse_pool_top_k,
            args,
        )
    else:
        train_bm25_all, train_char_all = build_sparse_pair(
            "train", train_claims, evidence, train_sparse_pool_top_k, args
        )
        target_bm25_all, target_char_all = build_sparse_pair(
            "target", target_claims, evidence, target_sparse_pool_top_k, args
        )
    train_word_gate = None
    target_word_gate = None
    if args.small_dense_mode == "sparse-gate" and args.small_gate_word_tfidf:
        train_word_gate = build_word_gate_pair(
            "train", train_claims, evidence, train_sparse_pool_top_k
        )
        target_word_gate = build_word_gate_pair(
            "target", target_claims, evidence, target_sparse_pool_top_k
        )
    train_bm25 = trim_pool(train_bm25_all, pool_top_k)
    target_bm25 = trim_pool(target_bm25_all, pool_top_k)
    train_char = trim_pool(train_char_all, pool_top_k)
    target_char = trim_pool(target_char_all, pool_top_k)
    write_pool(train_bm25, candidate_dir / "train_bm25.json")
    write_pool(target_bm25, candidate_dir / "target_bm25.json")
    write_pool(train_char, candidate_dir / "train_char.json")
    write_pool(target_char, candidate_dir / "target_char.json")

    small_cache_dir = output_dir / "dense_bge_small_cache"
    if args.small_dense_mode == "sparse-gate":
        train_gate_sources = [train_bm25_all, train_char_all]
        target_gate_sources = [target_bm25_all, target_char_all]
        if train_word_gate is not None and target_word_gate is not None:
            train_gate_sources.append(train_word_gate)
            target_gate_sources.append(target_word_gate)
        train_small_gate = merge_rrf(
            "train_sparse_gate_bm25_char_k500",
            train_gate_sources,
            top_k=train_small_prefilter_top_k,
            rrf_k=500,
        ).pool
        target_small_gate = merge_rrf(
            "target_sparse_gate_bm25_char_k500",
            target_gate_sources,
            top_k=target_small_prefilter_top_k,
            rrf_k=500,
        ).pool
        if args.write_small_prefilter_pool:
            write_pool(train_small_gate, candidate_dir / "train_small_sparse_gate.json")
            write_pool(target_small_gate, candidate_dir / "target_small_sparse_gate.json")
        all_small_gate = {**train_small_gate, **target_small_gate}
        all_claims_for_small = combined_claims(train_claims, target_claims)
        small_all, smallq_all, small_timing, smallq_timing = build_sparse_gated_dense_pools(
            claims=all_claims_for_small,
            evidence=evidence,
            prefilter_pool=all_small_gate,
            model_name=args.small_model,
            cache_dir=small_cache_dir,
            reuse_cache_dir=args.small_reuse_cache_dir,
            device=device,
            pool_top_k=pool_top_k,
            prefilter_top_k=max(
                train_small_prefilter_top_k,
                target_small_prefilter_top_k,
            ),
            batch_size=args.batch_size,
            query_batch_size=args.query_batch_size,
            max_length=args.max_length,
            query_max_length=args.query_max_length,
            pooling=args.pooling,
            cache_dtype=args.small_cache_dtype,
        )
        train_small = split_pool(small_all, train_claims)
        target_small = split_pool(small_all, target_claims)
        train_smallq = split_pool(smallq_all, train_claims)
        target_smallq = split_pool(smallq_all, target_claims)
        train_small_timing = dict(small_timing)
        target_small_timing = dict(small_timing)
        train_smallq_timing = dict(smallq_timing)
        target_smallq_timing = dict(smallq_timing)
        write_pool(train_small, candidate_dir / "train_bge_small_plain.json")
        write_pool(target_small, candidate_dir / "target_bge_small_plain.json")
        write_pool(train_smallq, candidate_dir / "train_bge_small_qprefix.json")
        write_pool(target_smallq, candidate_dir / "target_bge_small_qprefix.json")
        del train_small_gate, target_small_gate, all_small_gate, small_all, smallq_all
        del train_gate_sources, target_gate_sources
        gc.collect()
    else:
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
    del train_bm25_all, target_bm25_all, train_char_all, target_char_all
    if train_word_gate is not None:
        del train_word_gate, target_word_gate
    gc.collect()

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

    baseq_timing = None
    train_baseq = None
    target_baseq = None
    if not args.skip_base_subset:
        all_claims = combined_claims(train_claims, target_claims)
        all_rrf = {**train_rrf, **target_rrf}
        baseq_all, baseq_timing = build_subset_baseq_pool(
            claims=all_claims,
            evidence=evidence,
            candidate_pool=all_rrf,
            model_name=args.base_subset_model,
            cache_dir=output_dir / "dense_bge_base_subset_cache",
            reuse_cache_dir=args.base_reuse_cache_dir,
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
    }
    target_sources = {
        "bm25": target_bm25,
        "char": target_char,
        "small": target_small,
        "smallq": target_smallq,
    }
    if train_baseq is not None and target_baseq is not None:
        train_sources["baseq"] = train_baseq
        target_sources["baseq"] = target_baseq
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
