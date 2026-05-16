import argparse
import json
import random
from pathlib import Path

from a3_factcheck.data import load_json


LABELS = {"SUPPORTS": "SUPPORT", "REFUTES": "REFUTE"}


def load_ranked(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(claims, evidence, ranked, max_neutral_per_claim, refute_weight, seed):
    rng = random.Random(seed)
    rows = []
    evidence_ids = list(evidence.keys())
    for claim_id, claim in claims.items():
        claim_label = claim.get("claim_label")
        gold = set(claim.get("evidences", []))

        if claim_label in LABELS:
            pair_label = LABELS[claim_label]
            repeat = refute_weight if pair_label == "REFUTE" else 1
            for evidence_id in gold:
                for _ in range(repeat):
                    rows.append(
                        {
                            "claim_id": claim_id,
                            "evidence_id": evidence_id,
                            "claim": claim["claim_text"],
                            "evidence": evidence[evidence_id],
                            "label": pair_label,
                            "source": "gold",
                        }
                    )

        ranked_negatives = [
            item["evidence_id"]
            for item in ranked.get(claim_id, [])
            if item["evidence_id"] not in gold
        ][:max_neutral_per_claim]
        random_negatives = []
        while len(random_negatives) < max(1, max_neutral_per_claim // 3):
            evidence_id = rng.choice(evidence_ids)
            if evidence_id not in gold:
                random_negatives.append(evidence_id)

        for evidence_id in [*ranked_negatives, *random_negatives]:
            rows.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": evidence_id,
                    "claim": claim["claim_text"],
                    "evidence": evidence[evidence_id],
                    "label": "NEUTRAL",
                    "source": "ranked_or_random_negative",
                }
            )
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build SUPPORT/REFUTE/NEUTRAL verifier pairs.")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--ranked-candidates", required=True)
    parser.add_argument("--max-neutral-per-claim", type=int, default=8)
    parser.add_argument("--refute-weight", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    ranked = load_ranked(args.ranked_candidates)
    rows = build_rows(
        claims=claims,
        evidence=evidence,
        ranked=ranked,
        max_neutral_per_claim=args.max_neutral_per_claim,
        refute_weight=args.refute_weight,
        seed=args.seed,
    )
    write_jsonl(rows, args.output)
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(json.dumps({"rows": len(rows), "label_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
