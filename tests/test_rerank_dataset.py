from a3_factcheck.rerank.candidates import Candidate
from a3_factcheck.rerank.dataset import build_hard_negative_pairs


def test_build_hard_negative_pairs_injects_gold_and_caps_negatives():
    claims = {
        "claim-1": {
            "claim_text": "Climate claim",
            "evidences": ["evidence-gold"],
        }
    }
    evidence = {
        "evidence-gold": "Gold evidence",
        "evidence-neg-1": "Hard negative one",
        "evidence-neg-2": "Hard negative two",
    }
    pool = {
        "claim-1": [
            Candidate("claim-1", "evidence-neg-1", rank=1, score=9.0),
            Candidate("claim-1", "evidence-neg-2", rank=2, score=8.0),
        ]
    }

    rows = build_hard_negative_pairs(
        claims=claims,
        evidence=evidence,
        candidate_pool=pool,
        negatives_per_claim=1,
    )

    assert len(rows) == 2
    assert rows[0]["evidence_id"] == "evidence-gold"
    assert rows[0]["label"] == 1
    assert rows[0]["source"] == "gold"
    assert rows[1]["evidence_id"] == "evidence-neg-1"
    assert rows[1]["label"] == 0
    assert rows[1]["bm25_rank"] == 1
