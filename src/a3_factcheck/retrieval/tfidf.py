import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def top_indices_from_sparse_row(row, k, fallback_count):
    if row.nnz == 0:
        return list(range(min(k, fallback_count)))
    if row.nnz <= k:
        order = np.argsort(row.data)[::-1]
    else:
        selected = np.argpartition(row.data, -k)[-k:]
        order = selected[np.argsort(row.data[selected])[::-1]]
    return row.indices[order].tolist()


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


def build_tfidf_similarities(claims, evidence, max_features=200_000):
    evidence_ids = list(evidence.keys())
    evidence_texts = [evidence[evidence_id] for evidence_id in evidence_ids]
    claim_texts = [claim["claim_text"] for claim in claims.values()]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        dtype=np.float32,
    )
    evidence_matrix = vectorizer.fit_transform(evidence_texts)
    claim_matrix = vectorizer.transform(claim_texts)
    similarities = claim_matrix @ evidence_matrix.T
    return evidence_ids, similarities


def write_predictions(predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

