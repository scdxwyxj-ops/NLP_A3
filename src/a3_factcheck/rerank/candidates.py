import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from a3_factcheck.retrieval.bm25 import build_bm25_similarities
from a3_factcheck.retrieval.tfidf import top_indices_from_sparse_row


@dataclass(frozen=True)
class Candidate:
    claim_id: str
    evidence_id: str
    rank: int
    score: float


def build_bm25_candidate_pool(
    claims,
    evidence,
    top_k=50,
    max_features=200_000,
    k1=1.5,
    b=0.75,
):
    """Return BM25 candidates grouped by claim id.

    The returned mapping is intentionally text-free and compact. Downstream
    dataset builders can join against the original claim/evidence dictionaries.
    """
    evidence_ids, similarities = build_bm25_similarities(
        claims=claims,
        evidence=evidence,
        max_features=max_features,
        k1=k1,
        b=b,
    )
    pool = {}
    for row_idx, claim_id in enumerate(claims.keys()):
        row = similarities.getrow(row_idx)
        evidence_indexes = top_indices_from_sparse_row(row, top_k, len(evidence_ids))
        row_scores = dict(zip(row.indices.tolist(), row.data.tolist()))
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_ids[evidence_idx],
                rank=rank,
                score=float(row_scores.get(evidence_idx, 0.0)),
            )
            for rank, evidence_idx in enumerate(evidence_indexes, start=1)
        ]
    return pool


def pool_to_jsonable(pool):
    return {
        claim_id: [asdict(candidate) for candidate in candidates]
        for claim_id, candidates in pool.items()
    }


def write_candidate_pool(pool, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pool_to_jsonable(pool), f, ensure_ascii=False, indent=2)


def load_candidate_pool(path):
    raw_pool = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        claim_id: [Candidate(**candidate) for candidate in candidates]
        for claim_id, candidates in raw_pool.items()
    }


def candidate_recall_at_k(claims, pool, k):
    recalls = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        predicted = {candidate.evidence_id for candidate in pool.get(claim_id, [])[:k]}
        recalls.append(len(gold & predicted) / len(gold))
    return float(np.mean(recalls)) if recalls else 0.0
