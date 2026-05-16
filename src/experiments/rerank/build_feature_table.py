import argparse
import csv
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool
from a3_factcheck.semantic import extract_semantic_features


def load_ranked(path):
    import json

    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def pool_index(pool):
    return {
        claim_id: {
            candidate.evidence_id: {
                "rank": candidate.rank,
                "score": candidate.score,
            }
            for candidate in candidates
        }
        for claim_id, candidates in pool.items()
    }


def overlap_count(left, right):
    return len({item.lower() for item in left} & {item.lower() for item in right})


def jaccard(left, right):
    left_set = {item.lower() for item in left}
    right_set = {item.lower() for item in right}
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def feature_row(
    claim_id,
    claim,
    evidence_id,
    evidence_text,
    minilm_item,
    bm25_by_claim,
    char_by_claim,
    rrf_by_claim,
):
    claim_features = extract_semantic_features(claim["claim_text"])
    evidence_features = extract_semantic_features(evidence_text)
    bm25_item = bm25_by_claim.get(claim_id, {}).get(evidence_id)
    char_item = char_by_claim.get(claim_id, {}).get(evidence_id)
    rrf_item = rrf_by_claim.get(claim_id, {}).get(evidence_id)
    in_bm25 = int(bm25_item is not None)
    in_char = int(char_item is not None)

    return {
        "claim_id": claim_id,
        "claim_label": claim.get("claim_label"),
        "evidence_id": evidence_id,
        "label_is_gold": int(evidence_id in set(claim.get("evidences", []))),
        "minilm_rank": minilm_item.get("rank", 999999),
        "minilm_score": minilm_item.get("score", 0.0),
        "rrf_rank": rrf_item["rank"] if rrf_item else 999999,
        "rrf_score": rrf_item["score"] if rrf_item else 0.0,
        "bm25_rank": bm25_item["rank"] if bm25_item else 999999,
        "bm25_score": bm25_item["score"] if bm25_item else 0.0,
        "char_tfidf_rank": char_item["rank"] if char_item else 999999,
        "char_tfidf_score": char_item["score"] if char_item else 0.0,
        "in_bm25": in_bm25,
        "in_char_tfidf": in_char,
        "source_count": in_bm25 + in_char,
        "entity_overlap_count": overlap_count(
            claim_features["entities"], evidence_features["entities"]
        ),
        "entity_jaccard": jaccard(
            claim_features["entities"], evidence_features["entities"]
        ),
        "percentage_overlap_count": overlap_count(
            claim_features["percentages"], evidence_features["percentages"]
        ),
        "quantity_overlap_count": overlap_count(
            claim_features["quantities"], evidence_features["quantities"]
        ),
        "year_overlap_count": overlap_count(
            claim_features["years"], evidence_features["years"]
        ),
        "claim_has_negation": int(bool(claim_features["negation_cues"])),
        "evidence_has_negation": int(bool(evidence_features["negation_cues"])),
        "negation_presence_match": int(
            bool(claim_features["negation_cues"])
            == bool(evidence_features["negation_cues"])
        ),
        "negation_xor": int(
            bool(claim_features["negation_cues"])
            != bool(evidence_features["negation_cues"])
        ),
        "comparison_cue_overlap": overlap_count(
            claim_features["comparison_cues"], evidence_features["comparison_cues"]
        ),
        "causality_cue_overlap": overlap_count(
            claim_features["causality_cues"], evidence_features["causality_cues"]
        ),
        "relation_verb_overlap": overlap_count(
            claim_features["relation_verbs"], evidence_features["relation_verbs"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a claim-evidence feature table for fusion reranking."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--bm25-pool", required=True)
    parser.add_argument("--char-pool", required=True)
    parser.add_argument("--rrf-pool", required=True)
    parser.add_argument("--ranked-candidates", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    bm25_by_claim = pool_index(load_candidate_pool(args.bm25_pool))
    char_by_claim = pool_index(load_candidate_pool(args.char_pool))
    rrf_by_claim = pool_index(load_candidate_pool(args.rrf_pool))
    ranked = load_ranked(args.ranked_candidates)

    rows = []
    for claim_id, claim in claims.items():
        for minilm_item in ranked.get(claim_id, [])[: args.top_k]:
            evidence_id = minilm_item["evidence_id"]
            rows.append(
                feature_row(
                    claim_id=claim_id,
                    claim=claim,
                    evidence_id=evidence_id,
                    evidence_text=evidence[evidence_id],
                    minilm_item=minilm_item,
                    bm25_by_claim=bm25_by_claim,
                    char_by_claim=char_by_claim,
                    rrf_by_claim=rrf_by_claim,
                )
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    positives = sum(row["label_is_gold"] for row in rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Positive rows: {positives}")


if __name__ == "__main__":
    main()
