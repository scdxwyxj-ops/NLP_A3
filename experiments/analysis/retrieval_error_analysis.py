import argparse
import csv
import json
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool


def load_predictions(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def confusion_for_claim(gold_ids, predicted_ids, candidate_ids=None):
    gold = set(gold_ids)
    predicted = set(predicted_ids)
    if candidate_ids is None:
        candidate_set = predicted | gold
    else:
        candidate_set = set(candidate_ids) | gold

    tp_ids = sorted(predicted & gold)
    fp_ids = sorted(predicted - gold)
    fn_ids = sorted(gold - predicted)
    tn_ids = sorted(candidate_set - predicted - gold)
    return {
        "tp_ids": tp_ids,
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "tn_ids": tn_ids,
        "tp": len(tp_ids),
        "fp": len(fp_ids),
        "fn": len(fn_ids),
        "tn": len(tn_ids),
    }


def truncate(text, max_chars=260):
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def write_summary(rows, output_path):
    totals = {
        "claims": len(rows),
        "tp": sum(row["tp"] for row in rows),
        "fp": sum(row["fp"] for row in rows),
        "fn": sum(row["fn"] for row in rows),
        "tn": sum(row["tn"] for row in rows),
        "hit_any_claims": sum(1 for row in rows if row["tp"] > 0),
        "all_gold_claims": sum(1 for row in rows if row["fn"] == 0),
    }
    totals["precision"] = (
        totals["tp"] / (totals["tp"] + totals["fp"])
        if totals["tp"] + totals["fp"]
        else 0.0
    )
    totals["recall"] = (
        totals["tp"] / (totals["tp"] + totals["fn"])
        if totals["tp"] + totals["fn"]
        else 0.0
    )
    totals["hit_any_rate"] = totals["hit_any_claims"] / totals["claims"]
    totals["all_gold_rate"] = totals["all_gold_claims"] / totals["claims"]
    output_path.write_text(json.dumps(totals, indent=2), encoding="utf-8")
    return totals


def write_claim_rows(rows, output_path):
    fieldnames = [
        "claim_id",
        "claim_label",
        "tp",
        "fp",
        "fn",
        "tn",
        "gold_count",
        "predicted_count",
        "candidate_count",
        "tp_ids",
        "fp_ids",
        "fn_ids",
        "claim_text",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(row[key])
                    if key.endswith("_ids")
                    else row[key]
                    for key in fieldnames
                }
            )


def write_examples(rows, evidence, output_path, limit=20):
    misses = sorted(rows, key=lambda row: (row["tp"] > 0, -row["fn"], -row["fp"]))
    hits = sorted(rows, key=lambda row: (-row["tp"], row["fp"], row["fn"]))

    lines = ["# Retrieval Error Examples", ""]
    for title, selected in [
        ("Hard misses", misses[:limit]),
        ("Good hits", hits[:limit]),
    ]:
        lines.extend([f"## {title}", ""])
        for row in selected:
            lines.append(
                f"### {row['claim_id']} ({row['claim_label']}) "
                f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}"
            )
            lines.append("")
            lines.append(f"Claim: {row['claim_text']}")
            lines.append("")
            lines.append(f"Gold: {', '.join(row['gold_ids'])}")
            lines.append(f"Predicted: {', '.join(row['predicted_ids'])}")
            if row["tp_ids"]:
                lines.append("")
                lines.append("TP evidence:")
                for evidence_id in row["tp_ids"]:
                    lines.append(
                        f"- `{evidence_id}`: {truncate(evidence.get(evidence_id, ''))}"
                    )
            if row["fp_ids"]:
                lines.append("")
                lines.append("FP evidence:")
                for evidence_id in row["fp_ids"][:5]:
                    lines.append(
                        f"- `{evidence_id}`: {truncate(evidence.get(evidence_id, ''))}"
                    )
            if row["fn_ids"]:
                lines.append("")
                lines.append("FN evidence:")
                for evidence_id in row["fn_ids"]:
                    lines.append(
                        f"- `{evidence_id}`: {truncate(evidence.get(evidence_id, ''))}"
                    )
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Create TP/FP/FN/TN retrieval error analysis files."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument(
        "--candidate-pool",
        default="outputs/round05/dev-bm25-top50.json",
        help="Candidate pool used for candidate-level TN counts.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Named prediction JSON. Can be supplied multiple times.",
    )
    parser.add_argument("--output-dir", default="outputs/round05_analysis")
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    pool = load_candidate_pool(args.candidate_pool)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, prediction_path in args.prediction:
        predictions = load_predictions(prediction_path)
        method_dir = output_dir / name
        method_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for claim_id, claim in claims.items():
            predicted_ids = predictions.get(claim_id, {}).get("evidences", [])
            candidate_ids = [
                candidate.evidence_id for candidate in pool.get(claim_id, [])
            ]
            confusion = confusion_for_claim(
                gold_ids=claim.get("evidences", []),
                predicted_ids=predicted_ids,
                candidate_ids=candidate_ids,
            )
            rows.append(
                {
                    "claim_id": claim_id,
                    "claim_label": claim["claim_label"],
                    "claim_text": claim["claim_text"],
                    "gold_ids": claim.get("evidences", []),
                    "predicted_ids": predicted_ids,
                    "gold_count": len(claim.get("evidences", [])),
                    "predicted_count": len(predicted_ids),
                    "candidate_count": len(candidate_ids),
                    **confusion,
                }
            )

        totals = write_summary(rows, method_dir / "summary.json")
        write_claim_rows(rows, method_dir / "claim_confusion.csv")
        write_examples(rows, evidence, method_dir / "examples.md")
        summary_rows.append({"method": name, **totals})

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
