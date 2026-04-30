import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from a3_factcheck.retrieval.tfidf import top_indices_from_sparse_row


def build_bm25_matrix(
    evidence_texts,
    max_features=200_000,
    k1=1.5,
    b=0.75,
):
    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        dtype=np.float32,
    )
    counts = vectorizer.fit_transform(evidence_texts).tocsr()
    doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
    avg_doc_length = float(doc_lengths.mean()) if len(doc_lengths) else 0.0

    df = np.diff(counts.tocsc().indptr)
    n_docs = counts.shape[0]
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    bm25 = counts.tocoo(copy=True)
    row_lengths = doc_lengths[bm25.row]
    length_norm = 1.0 - b
    if avg_doc_length > 0:
        length_norm = length_norm + b * (row_lengths / avg_doc_length)
    denominator = bm25.data + k1 * length_norm
    bm25.data = idf[bm25.col] * (bm25.data * (k1 + 1.0)) / denominator
    return bm25.tocsr(), vectorizer


def build_bm25_similarities(
    claims,
    evidence,
    max_features=200_000,
    k1=1.5,
    b=0.75,
):
    evidence_ids = list(evidence.keys())
    evidence_texts = [evidence[evidence_id] for evidence_id in evidence_ids]
    claim_texts = [claim["claim_text"] for claim in claims.values()]

    bm25_matrix, vectorizer = build_bm25_matrix(
        evidence_texts=evidence_texts,
        max_features=max_features,
        k1=k1,
        b=b,
    )
    query_counts = vectorizer.transform(claim_texts)
    query_binary = query_counts.copy()
    query_binary.data = np.ones_like(query_binary.data, dtype=np.float32)
    similarities = query_binary @ bm25_matrix.T
    return evidence_ids, similarities


def build_predictions(claims, evidence_ids, similarities, top_k, default_label):
    predictions = {}
    for row_idx, (claim_id, claim) in enumerate(claims.items()):
        row = similarities.getrow(row_idx)
        evidence_indexes = top_indices_from_sparse_row(row, top_k, len(evidence_ids))
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": default_label,
            "evidences": [evidence_ids[i] for i in evidence_indexes],
        }
    return predictions


def write_predictions(predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

