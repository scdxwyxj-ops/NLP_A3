import argparse
import csv
import json
from pathlib import Path

from a3_factcheck.data import load_json


def load_ranked(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def hit(gold, ranked, top_k):
    return bool(set(gold) & {item["evidence_id"] for item in ranked[:top_k]})


def main():
    parser = argparse.ArgumentParser(description="Compare two ranked outputs by gain/loss.")
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--left-name", default="minilm")
    parser.add_argument("--left-ranked", required=True)
    parser.add_argument("--right-name", default="fusion")
    parser.add_argument("--right-ranked", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/round09/gain_loss")
    args = parser.parse_args()

    claims = load_json(args.claims)
    left = load_ranked(args.left_ranked)
    right = load_ranked(args.right_ranked)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    examples = []
    for claim_id, claim in claims.items():
        gold = claim.get("evidences", [])
        left_hit = hit(gold, left.get(claim_id, []), args.top_k)
        right_hit = hit(gold, right.get(claim_id, []), args.top_k)
        if left_hit and right_hit:
            bucket = "both_hit"
        elif left_hit and not right_hit:
            bucket = f"{args.left_name}_only"
        elif right_hit and not left_hit:
            bucket = f"{args.right_name}_only"
        else:
            bucket = "both_miss"
        rows.append(
            {
                "claim_id": claim_id,
                "claim_label": claim.get("claim_label"),
                "bucket": bucket,
                "left_hit": int(left_hit),
                "right_hit": int(right_hit),
            }
        )
        if bucket != "both_hit":
            examples.append(
                {
                    **rows[-1],
                    "claim_text": claim["claim_text"],
                    "gold": gold,
                    "left_topk": [item["evidence_id"] for item in left.get(claim_id, [])[: args.top_k]],
                    "right_topk": [item["evidence_id"] for item in right.get(claim_id, [])[: args.top_k]],
                }
            )

    summary = {}
    for row in rows:
        key = (row["claim_label"], row["bucket"])
        summary[key] = summary.get(key, 0) + 1
    summary_rows = [
        {"claim_label": label, "bucket": bucket, "count": count}
        for (label, bucket), count in sorted(summary.items())
    ]

    with (output_dir / "gain_loss_by_label.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_label", "bucket", "count"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "gain_loss_by_claim.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "examples.jsonl").open("w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
