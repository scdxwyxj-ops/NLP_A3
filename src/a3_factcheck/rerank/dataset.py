import json
from pathlib import Path


def pair_record(
    claim_id,
    evidence_id,
    claim_text,
    evidence_text,
    label,
    source,
    bm25_rank=None,
    bm25_score=None,
):
    return {
        "claim_id": claim_id,
        "evidence_id": evidence_id,
        "claim_text": claim_text,
        "evidence_text": evidence_text,
        "label": int(label),
        "source": source,
        "bm25_rank": bm25_rank,
        "bm25_score": bm25_score,
    }


def build_hard_negative_pairs(
    claims,
    evidence,
    candidate_pool,
    negatives_per_claim=5,
    include_positive_candidates=False,
):
    """Build supervised claim-evidence pairs for a binary reranker.

    Positives always come from gold evidence ids so training is not limited by
    BM25 recall. Negatives come from high-ranked BM25 candidates that are not
    gold evidence, making them harder than random corpus passages.
    """
    rows = []
    for claim_id, claim in claims.items():
        claim_text = claim["claim_text"]
        gold_ids = list(dict.fromkeys(claim.get("evidences", [])))
        candidates = candidate_pool.get(claim_id, [])
        candidate_by_evidence = {
            candidate.evidence_id: candidate for candidate in candidates
        }

        for evidence_id in gold_ids:
            if evidence_id not in evidence:
                continue
            candidate = candidate_by_evidence.get(evidence_id)
            rows.append(
                pair_record(
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    claim_text=claim_text,
                    evidence_text=evidence[evidence_id],
                    label=1,
                    source="gold"
                    if candidate is None
                    else "gold_candidate",
                    bm25_rank=None if candidate is None else candidate.rank,
                    bm25_score=None if candidate is None else candidate.score,
                )
            )

        added_negatives = 0
        gold_set = set(gold_ids)
        for candidate in candidates:
            if candidate.evidence_id in gold_set and not include_positive_candidates:
                continue
            if candidate.evidence_id not in evidence:
                continue
            rows.append(
                pair_record(
                    claim_id=claim_id,
                    evidence_id=candidate.evidence_id,
                    claim_text=claim_text,
                    evidence_text=evidence[candidate.evidence_id],
                    label=0,
                    source="bm25_hard_negative",
                    bm25_rank=candidate.rank,
                    bm25_score=candidate.score,
                )
            )
            added_negatives += 1
            if added_negatives >= negatives_per_claim:
                break
    return rows


def write_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
