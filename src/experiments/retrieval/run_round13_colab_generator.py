import argparse
import csv
import gc
import hashlib
import heapq
import json
import math
import resource
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
    TEXT_FEATURE_CACHE,
    build_rows,
    model_score,
    pool_index,
    rank_predictions,
    sampled_training_data,
)


def log_stage(message):
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"[round13] {message} | maxrss={rss_gb:.2f}GB", flush=True)


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


def push_top(heap, top_k, score, evidence_id):
    if score <= 0.0:
        return
    item = (float(score), evidence_id)
    if len(heap) < top_k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def heaps_to_pool(claims, heaps):
    pool = {}
    for claim_id in claims:
        ranked = sorted(heaps[claim_id], key=lambda item: (-item[0], item[1]))
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=float(score),
            )
            for rank, (score, evidence_id) in enumerate(ranked, start=1)
        ]
    return pool


def combined_claims(*claim_mappings):
    merged = {}
    for claims in claim_mappings:
        merged.update(claims)
    return merged


def query_bm25_pool(all_claims, evidence, top_k, k1=1.5, b=0.75):
    """Memory-bounded BM25 over full evidence, restricted to terms used by claims."""
    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    analyzer = vectorizer.build_analyzer()
    claim_terms = {
        claim_id: set(analyzer(claim["claim_text"])) for claim_id, claim in all_claims.items()
    }
    term_claims = defaultdict(list)
    for claim_id, terms in claim_terms.items():
        for term in terms:
            term_claims[term].append(claim_id)
    query_vocab = set(term_claims)

    evidence_items = list(evidence.items())
    doc_lengths = np.empty(len(evidence_items), dtype=np.float32)
    df = Counter()
    for idx, (_evidence_id, text) in enumerate(evidence_items):
        tokens = analyzer(text)
        doc_lengths[idx] = len(tokens)
        seen = {token for token in tokens if token in query_vocab}
        df.update(seen)
        if idx and idx % 100000 == 0:
            log_stage(f"query BM25 df pass {idx}/{len(evidence_items)}")
    avgdl = float(doc_lengths.mean()) if len(doc_lengths) else 0.0
    n_docs = len(evidence_items)
    idf = {
        term: math.log1p((n_docs - count + 0.5) / (count + 0.5))
        for term, count in df.items()
    }

    heaps = {claim_id: [] for claim_id in all_claims}
    for idx, (evidence_id, text) in enumerate(evidence_items):
        counts = Counter(token for token in analyzer(text) if token in idf)
        if counts:
            length_norm = 1.0 - b
            if avgdl > 0.0:
                length_norm += b * (float(doc_lengths[idx]) / avgdl)
            scores = defaultdict(float)
            for term, tf in counts.items():
                denom = float(tf) + k1 * length_norm
                weight = idf[term] * (float(tf) * (k1 + 1.0)) / denom
                for claim_id in term_claims[term]:
                    scores[claim_id] += weight
            for claim_id, score in scores.items():
                push_top(heaps[claim_id], top_k, score, evidence_id)
        if idx and idx % 100000 == 0:
            log_stage(f"query BM25 scoring pass {idx}/{len(evidence_items)}")

    return heaps_to_pool(all_claims, heaps)


def query_char_tfidf_pool(
    all_claims,
    evidence,
    top_k,
    max_claim_fanout=64,
    max_df_ratio=0.25,
):
    """Memory-bounded char TF-IDF over full evidence, restricted to claim ngrams."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
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
            log_stage(f"query char TF-IDF df pass {idx}/{len(evidence_items)}")

    n_docs = len(evidence_items)
    filtered_terms = {
        term
        for term, count in df.items()
        if len(term_claims[term]) <= max_claim_fanout
        and (float(count) / float(n_docs)) <= max_df_ratio
    }
    idf = {
        term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
        for term, count in df.items()
        if term in filtered_terms
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
    term_claim_arrays = {
        term: (
            np.asarray([item[0] for item in items], dtype=np.int32),
            np.asarray([item[1] for item in items], dtype=np.float32),
        )
        for term, items in term_claim_weights.items()
    }

    log_stage(
        "query char TF-IDF filtered "
        f"{len(filtered_terms)}/{len(query_vocab)} ngrams "
        f"(max_claim_fanout={max_claim_fanout}, max_df_ratio={max_df_ratio})"
    )
    heaps = {claim_id: [] for claim_id in all_claims}
    for idx, (evidence_id, text) in enumerate(evidence_items):
        counts = Counter(term for term in analyzer(text) if term in idf)
        if counts:
            weights = {term: float(tf) * idf[term] for term, tf in counts.items()}
            norm = math.sqrt(sum(value * value for value in weights.values()))
            if norm > 0.0:
                scores = np.zeros(len(claim_ids), dtype=np.float32)
                for term, value in weights.items():
                    evidence_weight = value / norm
                    indices, claim_term_weights = term_claim_arrays[term]
                    scores[indices] += evidence_weight * claim_term_weights
                for claim_idx in np.flatnonzero(scores):
                    claim_id = claim_ids[int(claim_idx)]
                    push_top(heaps[claim_id], top_k, float(scores[claim_idx]), evidence_id)
        if idx and idx % 10000 == 0:
            log_stage(f"query char TF-IDF scoring pass {idx}/{len(evidence_items)}")

    return heaps_to_pool(all_claims, heaps)


def split_pool(pool, claims):
    return {claim_id: pool.get(claim_id, []) for claim_id in claims}


def pool_to_id_pool(pool, top_k):
    return {
        claim_id: [candidate.evidence_id for candidate in candidates[:top_k]]
        for claim_id, candidates in pool.items()
    }


def item_evidence_id(item):
    return item if isinstance(item, str) else item.evidence_id


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
    log_stage(f"building dense pool {method_name}")
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
        device=device,
        query_batch_size=query_batch_size,
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
    log_stage(f"finished dense pool {method_name}")
    return pool, {"cache_seconds": cache_seconds, "search_seconds": search_seconds}


def evidence_id_fingerprint(evidence_ids):
    digest = hashlib.sha256()
    for evidence_id in evidence_ids:
        digest.update(evidence_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def prefilter_evidence_ids(claims, evidence, prefilter_pool, prefilter_top_k):
    wanted = set()
    for claim_id in claims:
        for item in prefilter_pool.get(claim_id, [])[:prefilter_top_k]:
            evidence_id = item_evidence_id(item)
            if evidence_id in evidence:
                wanted.add(evidence_id)
    return [evidence_id for evidence_id in evidence if evidence_id in wanted]


def build_restricted_embedding_cache(
    evidence,
    evidence_ids,
    model_name,
    cache_dir,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
    pooling,
    cache_dtype,
):
    dtype = np.dtype(cache_dtype)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path = cache_dir / "evidence_ids.json"
    embeddings_path = cache_dir / "evidence_embeddings.dat"
    meta_path = cache_dir / "metadata.json"
    fingerprint = evidence_id_fingerprint(evidence_ids)

    if ids_path.exists() and embeddings_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model_name") == model_name
            and meta.get("pooling") == pooling
            and meta.get("max_length") == max_length
            and meta.get("dtype") == str(dtype)
            and meta.get("count") == len(evidence_ids)
            and meta.get("fingerprint") == fingerprint
        ):
            shape = (meta["count"], meta["dim"])
            return json.loads(ids_path.read_text(encoding="utf-8")), np.memmap(
                embeddings_path, dtype=dtype, mode="r", shape=shape
            )

    dim = int(model.config.hidden_size)
    embeddings = np.memmap(
        embeddings_path, dtype=dtype, mode="w+", shape=(len(evidence_ids), dim)
    )
    for start in range(0, len(evidence_ids), batch_size):
        batch_ids = evidence_ids[start : start + batch_size]
        batch_texts = [evidence[evidence_id] for evidence_id in batch_ids]
        batch_embeddings = encode_texts(
            texts=batch_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
        )
        embeddings[start : start + len(batch_ids)] = batch_embeddings.astype(dtype)
        if start == 0 or (start // batch_size) % 100 == 0:
            log_stage(
                f"encoded restricted evidence "
                f"{start + len(batch_ids)}/{len(evidence_ids)}"
            )

    embeddings.flush()
    write_json(ids_path, evidence_ids)
    write_json(
        meta_path,
        {
            "model_name": model_name,
            "pooling": pooling,
            "max_length": max_length,
            "count": len(evidence_ids),
            "dim": dim,
            "dtype": str(dtype),
            "fingerprint": fingerprint,
        },
    )
    return evidence_ids, np.memmap(
        embeddings_path, dtype=dtype, mode="r", shape=(len(evidence_ids), dim)
    )


def restricted_dense_search(
    claims,
    prefilter_pool,
    evidence_ids,
    evidence_embeddings,
    query_embeddings,
    top_k,
    prefilter_top_k,
):
    evidence_index = {evidence_id: idx for idx, evidence_id in enumerate(evidence_ids)}
    pool = {}
    for claim_idx, claim_id in enumerate(claims.keys()):
        candidate_ids = []
        seen = set()
        for item in prefilter_pool.get(claim_id, [])[:prefilter_top_k]:
            evidence_id = item_evidence_id(item)
            if evidence_id in evidence_index and evidence_id not in seen:
                candidate_ids.append(evidence_id)
                seen.add(evidence_id)
        if not candidate_ids:
            pool[claim_id] = []
            continue

        positions = np.asarray(
            [evidence_index[evidence_id] for evidence_id in candidate_ids],
            dtype=np.int64,
        )
        candidate_embeddings = np.asarray(evidence_embeddings[positions], dtype=np.float32)
        scores = candidate_embeddings @ query_embeddings[claim_idx]
        local_k = min(top_k, len(candidate_ids))
        if local_k == len(candidate_ids):
            selected = np.argsort(scores)[::-1]
        else:
            selected = np.argpartition(scores, -local_k)[-local_k:]
            selected = selected[np.argsort(scores[selected])[::-1]]
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=candidate_ids[int(index)],
                rank=rank,
                score=float(scores[int(index)]),
            )
            for rank, index in enumerate(selected, start=1)
        ]
        if claim_idx and claim_idx % 100 == 0:
            log_stage(f"restricted dense searched {claim_idx}/{len(claims)} claims")
    return pool


def build_restricted_dense_pools(
    claims,
    evidence,
    prefilter_pool,
    model_name,
    cache_dir,
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
        "building restricted dense pools "
        f"(prefilter_top_k={prefilter_top_k})"
    )
    evidence_ids = prefilter_evidence_ids(
        claims=claims,
        evidence=evidence,
        prefilter_pool=prefilter_pool,
        prefilter_top_k=prefilter_top_k,
    )
    log_stage(
        f"restricted dense evidence union {len(evidence_ids)}/{len(evidence)}"
    )

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

    timings = {}
    pools = {}
    for key, query_prefix in {
        "plain": "",
        "qprefix": "Represent this sentence for searching relevant passages: ",
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
            "encoded_evidence": len(evidence_ids),
            "prefilter_top_k": prefilter_top_k,
        }
        del query_embeddings
        gc.collect()

    log_stage("finished restricted dense pools")
    return pools["plain"], pools["qprefix"], timings["plain"], timings["qprefix"]


def build_bm25_prefilter_pool(claims, evidence, top_k, k1, b, claim_batch_size):
    if claim_batch_size <= 0 or claim_batch_size >= len(claims):
        pool = query_bm25_pool(claims, evidence, top_k, k1=k1, b=b)
        return pool_to_id_pool(pool, top_k)

    pool = {}
    claim_items = list(claims.items())
    for start in range(0, len(claim_items), claim_batch_size):
        batch_claims = dict(claim_items[start : start + claim_batch_size])
        log_stage(
            "building BM25 dense prefilter batch "
            f"{start // claim_batch_size + 1}/"
            f"{math.ceil(len(claim_items) / claim_batch_size)}"
        )
        batch_pool = query_bm25_pool(batch_claims, evidence, top_k, k1=k1, b=b)
        pool.update(pool_to_id_pool(batch_pool, top_k))
        del batch_pool
        gc.collect()
    return pool


def build_sparse_pools(train_claims, target_claims, evidence, pool_top_k, args):
    if args.sparse_mode == "query":
        claims = combined_claims(train_claims, target_claims)
        log_stage("building query BM25 pools")
        bm25_all = query_bm25_pool(
            claims, evidence, pool_top_k, k1=1.5, b=0.75
        )
        train_bm25 = split_pool(bm25_all, train_claims)
        target_bm25 = split_pool(bm25_all, target_claims)
        del bm25_all
        gc.collect()

        log_stage("building query char TF-IDF pools")
        char_all = query_char_tfidf_pool(
            claims,
            evidence,
            pool_top_k,
            max_claim_fanout=args.query_char_max_claim_fanout,
            max_df_ratio=args.query_char_max_df_ratio,
        )
        train_char = split_pool(char_all, train_claims)
        target_char = split_pool(char_all, target_claims)
        del char_all
        gc.collect()
        return train_bm25, target_bm25, train_char, target_char

    log_stage("building exact BM25 pools")
    train_bm25 = build_bm25_pool(
        train_claims, evidence, pool_top_k, args.max_features, k1=1.5, b=0.75
    )
    gc.collect()
    target_bm25 = build_bm25_pool(
        target_claims, evidence, pool_top_k, args.max_features, k1=1.5, b=0.75
    )
    gc.collect()
    log_stage("building exact char TF-IDF pools")
    train_char = build_char_tfidf_pool(
        train_claims, evidence, pool_top_k, args.char_max_features
    )
    gc.collect()
    target_char = build_char_tfidf_pool(
        target_claims, evidence, pool_top_k, args.char_max_features
    )
    gc.collect()
    return train_bm25, target_bm25, train_char, target_char


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
    TEXT_FEATURE_CACHE.clear()
    gc.collect()
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
    TEXT_FEATURE_CACHE.clear()
    gc.collect()
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
    parser.add_argument(
        "--dense-mode",
        choices=["full", "sparse-prefilter"],
        default="full",
        help=(
            "full encodes/searches every evidence item; sparse-prefilter first "
            "uses a sparse pool to restrict each claim to a reproducible lexical "
            "candidate set before BGE scoring."
        ),
    )
    parser.add_argument("--sparse-prefilter-top-k", type=int, default=12_000)
    parser.add_argument(
        "--train-sparse-prefilter-top-k",
        type=int,
        default=0,
        help=(
            "Optional shallower BM25 prefilter for training claims. "
            "Use 0 to match --sparse-prefilter-top-k."
        ),
    )
    parser.add_argument("--prefilter-claim-batch-size", type=int, default=256)
    parser.add_argument("--write-prefilter-pool", action="store_true")
    parser.add_argument(
        "--sparse-mode",
        choices=["exact", "query"],
        default="exact",
        help=(
            "exact uses full sklearn sparse matrices; query streams evidence and "
            "scores only claim terms/ngrams to stay within Colab RAM."
        ),
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
                "dense_mode": args.dense_mode,
            },
            indent=2,
        )
    )

    train_bm25, target_bm25, train_char, target_char = build_sparse_pools(
        train_claims, target_claims, evidence, pool_top_k, args
    )
    write_pool(train_bm25, candidate_dir / "train_bm25.json")
    write_pool(target_bm25, candidate_dir / "target_bm25.json")
    write_pool(train_char, candidate_dir / "train_char.json")
    write_pool(target_char, candidate_dir / "target_char.json")

    if args.dense_mode == "sparse-prefilter":
        target_prefilter_top_k = min(args.sparse_prefilter_top_k, len(evidence))
        train_prefilter_top_k = min(
            args.train_sparse_prefilter_top_k or args.sparse_prefilter_top_k,
            len(evidence),
        )
        prefilter_top_k = max(train_prefilter_top_k, target_prefilter_top_k)
        all_claims = combined_claims(train_claims, target_claims)
        log_stage(
            "building BM25 dense prefilters "
            f"(train_top_k={train_prefilter_top_k}, "
            f"target_top_k={target_prefilter_top_k})"
        )
        train_prefilter_pool = build_bm25_prefilter_pool(
            train_claims,
            evidence,
            top_k=train_prefilter_top_k,
            k1=1.5,
            b=0.75,
            claim_batch_size=args.prefilter_claim_batch_size,
        )
        target_prefilter_pool = build_bm25_prefilter_pool(
            target_claims,
            evidence,
            top_k=target_prefilter_top_k,
            k1=1.5,
            b=0.75,
            claim_batch_size=args.prefilter_claim_batch_size,
        )
        prefilter_pool = {**train_prefilter_pool, **target_prefilter_pool}
        del train_prefilter_pool, target_prefilter_pool
        gc.collect()
        if args.write_prefilter_pool:
            prefilter_path = (
                candidate_dir
                / (
                    "bm25_prefilter_"
                    f"train{train_prefilter_top_k}_target{target_prefilter_top_k}.json"
                )
            )
            write_json(prefilter_path, prefilter_pool)
        dense_all, denseq_all, dense_timing, denseq_timing = build_restricted_dense_pools(
            claims=all_claims,
            evidence=evidence,
            prefilter_pool=prefilter_pool,
            model_name=args.model,
            cache_dir=output_dir / "dense_bge_base_restricted_cache",
            device=device,
            pool_top_k=pool_top_k,
            prefilter_top_k=prefilter_top_k,
            batch_size=args.batch_size,
            query_batch_size=args.query_batch_size,
            max_length=args.max_length,
            query_max_length=args.query_max_length,
            pooling=args.pooling,
            cache_dtype=args.cache_dtype,
        )
        train_dense = split_pool(dense_all, train_claims)
        target_dense = split_pool(dense_all, target_claims)
        train_denseq = split_pool(denseq_all, train_claims)
        target_denseq = split_pool(denseq_all, target_claims)
        del dense_all, denseq_all
        train_dense_timing = dict(dense_timing)
        target_dense_timing = dict(dense_timing)
        train_denseq_timing = dict(denseq_timing)
        target_denseq_timing = dict(denseq_timing)
        write_pool(train_dense, candidate_dir / "train_bge_base_plain.json")
        write_pool(target_dense, candidate_dir / "target_bge_base_plain.json")
        write_pool(train_denseq, candidate_dir / "train_bge_base_qprefix.json")
        write_pool(target_denseq, candidate_dir / "target_bge_base_qprefix.json")
        del prefilter_pool
        gc.collect()
    else:
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
