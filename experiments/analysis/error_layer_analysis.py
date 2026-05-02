import argparse
import csv
import json
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool


LAYERS = [
    "gold_not_in_top500",
    "gold_in_top500_not_top50",
    "gold_in_top50_not_top20",
    "gold_in_top20_not_top3",
]


def load_ranked(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def ids_from_pool(pool, claim_id, top_k):
    return [candidate.evidence_id for candidate in pool.get(claim_id, [])[:top_k]]


def ids_from_ranked(ranked, claim_id, top_k):
    return [item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]]


def row_for_claim(claim_id, claim, candidate_pool, ranked):
    gold = set(claim.get("evidences", []))
    top500 = set(ids_from_pool(candidate_pool, claim_id, 500))
    top50 = set(ids_from_ranked(ranked, claim_id, 50))
    top20 = set(ids_from_ranked(ranked, claim_id, 20))
    top3 = set(ids_from_ranked(ranked, claim_id, 3))

    gold_in_top500 = gold & top500
    gold_in_top50 = gold & top50
    gold_in_top20 = gold & top20
    gold_in_top3 = gold & top3

    return {
        "claim_id": claim_id,
        "claim_label": claim.get("claim_label"),
        "gold_count": len(gold),
        "gold_not_in_top500": len(gold - top500),
        "gold_in_top500_not_top50": len(gold_in_top500 - top50),
        "gold_in_top50_not_top20": len(gold_in_top50 - top20),
        "gold_in_top20_not_top3": len(gold_in_top20 - top3),
        "gold_in_top500": len(gold_in_top500),
        "gold_in_top50": len(gold_in_top50),
        "gold_in_top20": len(gold_in_top20),
        "gold_in_top3": len(gold_in_top3),
        "hit_top500": int(bool(gold_in_top500)),
        "hit_top50": int(bool(gold_in_top50)),
        "hit_top20": int(bool(gold_in_top20)),
        "hit_top3": int(bool(gold_in_top3)),
    }


def summarize(rows):
    total_claims = len(rows)
    total_gold = sum(row["gold_count"] for row in rows)
    summary_rows = []
    fixes = {
        "gold_not_in_top500": "dense supplement / candidate retrieval",
        "gold_in_top500_not_top50": "score more candidates / fusion reranker",
        "gold_in_top50_not_top20": "feature fusion reranker",
        "gold_in_top20_not_top3": "final evidence selection",
    }
    for layer in LAYERS:
        count = sum(row[layer] for row in rows)
        affected_claims = sum(1 for row in rows if row[layer] > 0)
        summary_rows.append(
            {
                "layer": layer,
                "count": count,
                "claim_percent": affected_claims / total_claims if total_claims else 0.0,
                "gold_evidence_percent": count / total_gold if total_gold else 0.0,
                "main_fix": fixes[layer],
            }
        )

    for top_k in [500, 50, 20, 3]:
        hit_count = sum(row[f"hit_top{top_k}"] for row in rows)
        gold_count = sum(row[f"gold_in_top{top_k}"] for row in rows)
        summary_rows.append(
            {
                "layer": f"coverage_top{top_k}",
                "count": gold_count,
                "claim_percent": hit_count / total_claims if total_claims else 0.0,
                "gold_evidence_percent": gold_count / total_gold if total_gold else 0.0,
                "main_fix": "coverage diagnostic",
            }
        )
    return summary_rows


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Break evidence loss into candidate/rerank/final-selection layers."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument(
        "--candidate-pool",
        default="outputs/round07/candidates/rrf_bm25_char_tfidf_top500.json",
    )
    parser.add_argument(
        "--ranked-candidates",
        default="outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json",
    )
    parser.add_argument("--output-dir", default="outputs/round08")
    args = parser.parse_args()

    claims = load_json(args.claims)
    pool = load_candidate_pool(args.candidate_pool)
    ranked = load_ranked(args.ranked_candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row_for_claim(claim_id, claim, pool, ranked)
        for claim_id, claim in claims.items()
    ]
    summary_rows = summarize(rows)

    write_csv(rows, output_dir / "error_layer_by_claim.csv")
    write_csv(summary_rows, output_dir / "error_layer_summary.csv")

    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
