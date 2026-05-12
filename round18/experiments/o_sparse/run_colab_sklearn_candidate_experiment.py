#!/usr/bin/env python
"""Colab-feasible sklearn/scipy sparse candidate experiment.

This is intentionally separate from the submission notebook. It evaluates whether
a CountVectorizer + BM25 sparse-matrix implementation can recover enough dev
gold evidence at top500 to replace the slower hand-written postings candidate
gate in the self-contained Colab notebook.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = set(ENGLISH_STOP_WORDS)
NEG_WORDS = {"not", "no", "never", "without", "less", "fewer", "decline", "decrease", "false"}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def word_analyzer(text: str) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    toks = [t for t in toks if len(t) > 1 and t not in STOPWORDS]
    return toks + [toks[i] + "_" + toks[i + 1] for i in range(len(toks) - 1)]


def structured_analyzer(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text)
    years = [t for t in toks if re.fullmatch(r"(?:1[5-9]|20)\d\d", t)]
    nums = [t for t in toks if re.search(r"\d", t)]
    caps = [t.lower() for t in toks if len(t) > 2 and t[0].isupper()]
    neg = [t.lower() for t in toks if t.lower() in NEG_WORDS]
    return (
        ["Y_" + x for x in years]
        + ["N_" + x for x in nums]
        + ["E_" + x for x in caps]
        + ["NEG_" + x for x in neg]
    )


def bm25_transform_counts(x_counts: sparse.csr_matrix, *, k1: float = 1.5, b: float = 0.75) -> sparse.csr_matrix:
    """Return a CSR evidence matrix containing BM25 term weights."""
    x = x_counts.tocsr(copy=True).astype(np.float32)
    n_docs, _ = x.shape
    df = np.diff(x.tocsc().indptr).astype(np.float32)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    doc_len = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    doc_len = np.maximum(doc_len, 1.0)
    avgdl = float(doc_len.mean()) if len(doc_len) else 1.0
    row_ids = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(x.indptr))
    denom = x.data + k1 * (1.0 - b + b * doc_len[row_ids] / avgdl)
    x.data = x.data * (k1 + 1.0) / denom
    x = x.multiply(idf).tocsr()
    return x.astype(np.float32)


def topk_from_scores(scores: np.ndarray, evidence_ids: list[str], top_k: int) -> list[dict[str, Any]]:
    if scores.size == 0:
        return []
    k = min(top_k, scores.size)
    if k <= 0:
        return []
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.lexsort((np.asarray(evidence_ids, dtype=object)[idx], -scores[idx]))]
    return [
        {"evidence_id": evidence_ids[int(i)], "score": float(scores[int(i)]), "rank": rank}
        for rank, i in enumerate(idx, start=1)
        if scores[int(i)] > 0
    ]


def score_matrix_topk(
    claim_ids: list[str],
    claim_matrix: sparse.csr_matrix,
    evidence_matrix_bm25: sparse.csr_matrix,
    evidence_ids: list[str],
    *,
    top_k: int,
    batch_size: int,
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    start = time.perf_counter()
    out: dict[str, list[dict[str, Any]]] = {}
    ev_t = evidence_matrix_bm25.T.tocsr()
    for start_idx in range(0, len(claim_ids), batch_size):
        end_idx = min(start_idx + batch_size, len(claim_ids))
        batch_scores = (claim_matrix[start_idx:end_idx] @ ev_t).toarray().astype(np.float32, copy=False)
        for local_idx, cid in enumerate(claim_ids[start_idx:end_idx]):
            out[cid] = topk_from_scores(batch_scores[local_idx], evidence_ids, top_k)
        print(label, "batch", end_idx, "/", len(claim_ids), "elapsed", round(time.perf_counter() - start, 1), "s")
    print(label, "finished in", round(time.perf_counter() - start, 1), "s")
    return out


def rrf_fuse(
    claim_ids: list[str],
    sources: dict[str, dict[str, list[dict[str, Any]]]],
    weights: dict[str, float],
    *,
    top_k: int,
    rrf_k: float,
) -> dict[str, list[dict[str, Any]]]:
    fused: dict[str, list[dict[str, Any]]] = {}
    for cid in claim_ids:
        scores: dict[str, float] = defaultdict(float)
        source_ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for name, pool in sources.items():
            weight = weights.get(name, 0.0)
            for fallback_rank, row in enumerate(pool.get(cid, [])[:top_k], start=1):
                eid = row["evidence_id"]
                rank = int(row.get("rank", fallback_rank))
                scores[eid] += weight / (rrf_k + rank)
                source_ranks[eid][name + "_rank"] = rank
        rows = [
            {"evidence_id": eid, "score": float(score), **source_ranks.get(eid, {})}
            for eid, score in scores.items()
        ]
        rows.sort(key=lambda r: (-r["score"], r["evidence_id"]))
        for rank, row in enumerate(rows[:top_k], start=1):
            row["rank"] = rank
        fused[cid] = rows[:top_k]
    return fused


def retrieval_metrics(claims: dict[str, dict[str, Any]], pool: dict[str, list[dict[str, Any]]], top_k: int) -> dict[str, float]:
    recalls = []
    hit_any = []
    evidence_f = []
    label_recalls: dict[str, list[float]] = defaultdict(list)
    for cid, claim in claims.items():
        gold = set(claim.get("evidences", []))
        pred = [row["evidence_id"] for row in pool.get(cid, [])[:top_k]]
        overlap = gold & set(pred)
        recall = len(overlap) / len(gold) if gold else 0.0
        precision = len(overlap) / len(pred) if pred else 0.0
        f_score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        recalls.append(recall)
        hit_any.append(bool(overlap))
        evidence_f.append(f_score)
        label_recalls[str(claim.get("claim_label", ""))].append(recall)
    return {
        f"macro_recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hit_any@{top_k}": float(np.mean(hit_any)) if hit_any else 0.0,
        f"evidence_f@{top_k}": float(np.mean(evidence_f)) if evidence_f else 0.0,
        "label_macro_recall": {
            label: float(np.mean(values)) if values else 0.0
            for label, values in sorted(label_recalls.items())
        },
    }


def expand_claims_prf(
    claims: dict[str, dict[str, Any]],
    bm25_pool: dict[str, list[dict[str, Any]]],
    evidence: dict[str, str],
    *,
    feedback_k: int,
    max_terms: int,
) -> list[str]:
    expanded = []
    word_cache: dict[str, Counter[str]] = {}
    for cid, claim in claims.items():
        counts: Counter[str] = Counter()
        for row in bm25_pool.get(cid, [])[:feedback_k]:
            eid = row["evidence_id"]
            if eid not in word_cache:
                word_cache[eid] = Counter(word_analyzer(evidence.get(eid, "")))
            counts.update(word_cache[eid])
        original = set(word_analyzer(claim["claim_text"]))
        terms = [
            term.replace("_", " ")
            for term, _ in counts.most_common(max_terms * 3)
            if term not in original and term.isascii()
        ][:max_terms]
        expanded.append(claim["claim_text"] + " " + " ".join(terms))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-claims", type=Path, default=Path("colab_notebooks/train-claims.json"))
    parser.add_argument("--dev-claims", type=Path, default=Path("colab_notebooks/dev-claims.json"))
    parser.add_argument("--evidence", type=Path, default=Path("colab_notebooks/evidence.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("round18/reports/colab_sklearn_candidate_experiment"))
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--word-max-features", type=int, default=200_000)
    parser.add_argument("--char-max-features", type=int, default=160_000)
    parser.add_argument("--char-min-df", type=int, default=2)
    parser.add_argument("--char-max-df", type=float, default=0.25)
    parser.add_argument("--rrf-k", type=float, default=500.0)
    args = parser.parse_args()

    start = time.perf_counter()
    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    evidence_ids = list(evidence)
    evidence_texts = [evidence[eid] for eid in evidence_ids]
    dev_ids = list(dev_claims)
    dev_texts = [dev_claims[cid]["claim_text"] for cid in dev_ids]
    print({"evidence": len(evidence_ids), "dev_claims": len(dev_ids)})

    word_vectorizer = CountVectorizer(
        analyzer=word_analyzer,
        max_features=args.word_max_features,
        dtype=np.float32,
    )
    t = time.perf_counter()
    x_ev_word = word_vectorizer.fit_transform(evidence_texts)
    x_dev_word = word_vectorizer.transform(dev_texts)
    x_ev_word_bm25 = bm25_transform_counts(x_ev_word)
    print("word matrix", x_ev_word.shape, "nnz", x_ev_word.nnz, "time", round(time.perf_counter() - t, 1))

    bm25_pool = score_matrix_topk(
        dev_ids,
        x_dev_word,
        x_ev_word_bm25,
        evidence_ids,
        top_k=args.top_k,
        batch_size=args.batch_size,
        label="word_bm25",
    )
    print("word metrics", retrieval_metrics(dev_claims, bm25_pool, args.top_k))

    t = time.perf_counter()
    prf_texts = expand_claims_prf(dev_claims, bm25_pool, evidence, feedback_k=10, max_terms=12)
    x_dev_prf = word_vectorizer.transform(prf_texts)
    prf_pool = score_matrix_topk(
        dev_ids,
        x_dev_prf,
        x_ev_word_bm25,
        evidence_ids,
        top_k=args.top_k,
        batch_size=args.batch_size,
        label="prf_word_bm25",
    )
    print("prf time", round(time.perf_counter() - t, 1))
    print("prf metrics", retrieval_metrics(dev_claims, prf_pool, args.top_k))

    char_vectorizer = CountVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=args.char_max_features,
        min_df=args.char_min_df,
        max_df=args.char_max_df,
        dtype=np.float32,
    )
    t = time.perf_counter()
    x_ev_char = char_vectorizer.fit_transform(evidence_texts)
    x_dev_char = char_vectorizer.transform(dev_texts)
    x_ev_char_bm25 = bm25_transform_counts(x_ev_char)
    print("char matrix", x_ev_char.shape, "nnz", x_ev_char.nnz, "time", round(time.perf_counter() - t, 1))
    char_pool = score_matrix_topk(
        dev_ids,
        x_dev_char,
        x_ev_char_bm25,
        evidence_ids,
        top_k=args.top_k,
        batch_size=args.batch_size,
        label="char_bm25",
    )
    print("char metrics", retrieval_metrics(dev_claims, char_pool, args.top_k))

    structured_texts = [" ".join(structured_analyzer(text)) for text in evidence_texts]
    structured_dev_texts = [" ".join(structured_analyzer(text)) for text in dev_texts]
    structured_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\S+\b", dtype=np.float32)
    t = time.perf_counter()
    x_ev_struct = structured_vectorizer.fit_transform(structured_texts)
    x_dev_struct = structured_vectorizer.transform(structured_dev_texts)
    x_ev_struct_bm25 = bm25_transform_counts(x_ev_struct)
    print("structured matrix", x_ev_struct.shape, "nnz", x_ev_struct.nnz, "time", round(time.perf_counter() - t, 1))
    struct_pool = score_matrix_topk(
        dev_ids,
        x_dev_struct,
        x_ev_struct_bm25,
        evidence_ids,
        top_k=args.top_k,
        batch_size=args.batch_size,
        label="structured_bm25",
    )
    print("structured metrics", retrieval_metrics(dev_claims, struct_pool, args.top_k))

    fusion_sources = {"bm25": bm25_pool, "char": char_pool, "structured": struct_pool, "prf": prf_pool}
    weight_configs = {
        "fused": {"bm25": 1.0, "char": 1.5, "structured": 1.0, "prf": 1.0},
        "fused_no_structured": {"bm25": 1.0, "char": 1.5, "prf": 1.0},
        "fused_struct_025": {"bm25": 1.0, "char": 1.5, "structured": 0.25, "prf": 1.0},
        "fused_struct_050": {"bm25": 1.0, "char": 1.5, "structured": 0.5, "prf": 1.0},
        "fused_char_200_struct_025_prf_050": {"bm25": 0.75, "char": 2.0, "structured": 0.25, "prf": 0.5},
        "fused_char_250_struct_025_prf_050": {"bm25": 0.75, "char": 2.5, "structured": 0.25, "prf": 0.5},
        "fused_char_300_struct_025_prf_025": {"bm25": 0.5, "char": 3.0, "structured": 0.25, "prf": 0.25},
    }
    fused_outputs = {
        name: rrf_fuse(dev_ids, fusion_sources, weights, top_k=args.top_k, rrf_k=args.rrf_k)
        for name, weights in weight_configs.items()
    }
    metrics = {
        "bm25": retrieval_metrics(dev_claims, bm25_pool, args.top_k),
        "char": retrieval_metrics(dev_claims, char_pool, args.top_k),
        "structured": retrieval_metrics(dev_claims, struct_pool, args.top_k),
        "prf": retrieval_metrics(dev_claims, prf_pool, args.top_k),
    }
    metrics.update(
        {
            name: retrieval_metrics(dev_claims, pool, args.top_k)
            for name, pool in fused_outputs.items()
        }
    )
    summary = {
        "config": vars(args),
        "metrics": metrics,
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "weight_configs": weight_configs,
        "matrix_shapes": {
            "word": list(x_ev_word.shape),
            "char": list(x_ev_char.shape),
            "structured": list(x_ev_struct.shape),
        },
        "matrix_nnz": {
            "word": int(x_ev_word.nnz),
            "char": int(x_ev_char.nnz),
            "structured": int(x_ev_struct.nnz),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "bm25_top500.json", bm25_pool)
    write_json(args.output_dir / "char_top500.json", char_pool)
    write_json(args.output_dir / "structured_top500.json", struct_pool)
    write_json(args.output_dir / "prf_top500.json", prf_pool)
    write_json(args.output_dir / "fused_top500.json", fused_outputs["fused"])
    write_json(args.output_dir / "fused_no_structured_top500.json", fused_outputs["fused_no_structured"])
    write_json(
        args.output_dir / "fused_char_heavy_top500.json",
        fused_outputs["fused_char_200_struct_025_prf_050"],
    )
    print(json.dumps(summary["metrics"], indent=2))
    print("Wrote", args.output_dir)


if __name__ == "__main__":
    main()
