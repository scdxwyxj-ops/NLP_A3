import argparse
import json
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.metrics import load_predictions
from a3_factcheck.semantic import extract_semantic_features


def write_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_examples(rows, output_path, limit=20):
    lines = ["# Semantic Feature Examples", ""]
    for row in rows[:limit]:
        lines.append(f"## {row['claim_id']} -> {row['evidence_id']}")
        lines.append("")
        lines.append(f"Claim: {row['claim_text']}")
        lines.append("")
        lines.append(f"Evidence: {row['evidence_text']}")
        lines.append("")
        lines.append("Claim features:")
        lines.append("```json")
        lines.append(json.dumps(row["claim_features"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
        lines.append("Evidence features:")
        lines.append("```json")
        lines.append(json.dumps(row["evidence_features"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Extract lightweight semantic features for claim-evidence pairs."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument(
        "--predictions", default="outputs/round05/dev-msmarco-minilm-zero-shot-top5.json"
    )
    parser.add_argument("--output", default="outputs/round06/semantic_features.jsonl")
    parser.add_argument(
        "--examples", default="outputs/round06/semantic_feature_examples.md"
    )
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    predictions = load_predictions(args.predictions)

    rows = []
    for claim_id, prediction in predictions.items():
        claim = claims[claim_id]
        for evidence_id in prediction.get("evidences", []):
            if evidence_id not in evidence:
                continue
            rows.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": evidence_id,
                    "claim_label": claim.get("claim_label"),
                    "is_gold": evidence_id in set(claim.get("evidences", [])),
                    "claim_text": claim["claim_text"],
                    "evidence_text": evidence[evidence_id],
                    "claim_features": extract_semantic_features(claim["claim_text"]),
                    "evidence_features": extract_semantic_features(
                        evidence[evidence_id]
                    ),
                }
            )

    write_jsonl(rows, args.output)
    write_examples(rows, args.examples)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
