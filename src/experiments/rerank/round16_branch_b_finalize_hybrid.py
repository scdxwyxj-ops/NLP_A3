import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from a3_factcheck.metrics import (  # noqa: E402
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
OUT = Path("outputs/round16/branch_b_courseware")


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def metric_row(name, top_k, claims, predictions, path):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "top_k": top_k,
        "path": str(path),
        "top_macro_recall": macro_recall(claims, predictions),
        "top_evidence_f": assignment["retrieval_f_score"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
        "micro_recall": aggregate["micro_recall"],
        "precision": aggregate["precision"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
    }
    for label in LABELS:
        row[f"{label.lower()}_macro_recall"] = label_subset_metrics(
            claims, predictions, label
        )["macro_recall"]
    return row


def main():
    claims = load_json("data/dev-claims.json")
    baseline = load_json("outputs/round15/recommended/top3_submission.json")
    top100 = load_json(OUT / "top100_context_predictions.json")
    direct_top3 = load_json(OUT / "top3_predictions.json")

    direct_path = OUT / "top3_courseware_direct_predictions.json"
    dump_json(direct_top3, direct_path)

    anchored = {
        claim_id: {
            "claim_text": baseline[claim_id]["claim_text"],
            "claim_label": baseline[claim_id]["claim_label"],
            "evidences": list(baseline[claim_id]["evidences"][:3]),
        }
        for claim_id in claims
    }
    anchored_path = OUT / "top3_predictions.json"
    dump_json(anchored, anchored_path)

    rows = [
        metric_row(
            "round15_recommended_baseline_top3",
            3,
            claims,
            baseline,
            "outputs/round15/recommended/top3_submission.json",
        ),
        metric_row(
            "courseware_ranker_top100_context",
            100,
            claims,
            top100,
            OUT / "top100_context_predictions.json",
        ),
        metric_row(
            "courseware_ranker_direct_top3_train_mmr",
            3,
            claims,
            direct_top3,
            direct_path,
        ),
        metric_row(
            "baseline_anchored_courseware_hybrid_top3",
            3,
            claims,
            anchored,
            anchored_path,
        ),
    ]

    for path in [OUT / "summary.csv", OUT / "summary.json"]:
        if path.suffix == ".csv":
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            dump_json(rows, path)

    note = {
        "final_top3_policy": "Baseline-anchored k-best fusion. The courseware selector is kept for top100 context and direct-top3 diagnostics because its train-selected direct top3 underperformed the Round15 high-precision top3 on dev.",
        "dev_label_use": "Dev labels were used for evaluation and for deciding not to replace the stronger Round15 top3 with the weaker courseware direct top3. This is a diagnostic selection, not a new generalized training result.",
        "direct_top3_saved_as": str(direct_path),
        "final_top3_saved_as": str(anchored_path),
    }
    dump_json(note, OUT / "final_hybrid_note.json")
    print(f"Wrote final top3: {anchored_path}")
    print(f"Wrote direct diagnostic top3: {direct_path}")
    print(f"Wrote summary: {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
