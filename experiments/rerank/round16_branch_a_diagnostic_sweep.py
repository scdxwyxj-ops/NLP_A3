import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.metrics import (
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)
from a3_factcheck.rerank.api import write_predictions


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
OUT = Path("outputs/round16/branch_a_requirement")


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def pred_source(predictions):
    return {
        claim_id: [
            {"evidence_id": evidence_id, "rank": rank}
            for rank, evidence_id in enumerate(pred.get("evidences", []), start=1)
        ]
        for claim_id, pred in predictions.items()
    }


def fuse_claim(claim_id, sources, weights, rrf_k, cap_by_source):
    scores = defaultdict(float)
    for name, source in sources.items():
        weight = weights.get(name, 0.0)
        if weight == 0:
            continue
        for idx, item in enumerate(source.get(claim_id, [])[: cap_by_source.get(name, 100)], start=1):
            rank = int(item.get("rank", idx))
            scores[item["evidence_id"]] += weight / (rrf_k + rank)
    rows = [{"evidence_id": evidence_id, "score": score} for evidence_id, score in scores.items()]
    rows.sort(key=lambda item: (-item["score"], item["evidence_id"]))
    return rows


def predictions_from_sources(claims, sources, weights, rrf_k, cap_by_source, label_source, default_label, top_k):
    pred = {}
    for claim_id, claim in claims.items():
        ranked = fuse_claim(claim_id, sources, weights, rrf_k, cap_by_source)
        pred[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": label_source.get(claim_id, {}).get("claim_label", default_label),
            "evidences": [item["evidence_id"] for item in ranked[:top_k]],
        }
    return pred


def summary_row(name, top_k, path, claims, predictions, note):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "top_k": top_k,
        "path": str(path),
        "note": note,
        "top100_macro_recall": macro_recall(claims, predictions) if top_k == 100 else "",
        "top3_macro_recall": macro_recall(claims, predictions) if top_k == 3 else "",
        "top3_evidence_f": assignment["retrieval_f_score"] if top_k == 3 else "",
        "macro_recall": macro_recall(claims, predictions),
        "micro_recall": aggregate["micro_recall"],
        "precision": aggregate["precision"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
    }
    for label in LABELS:
        row[f"{label.lower()}_macro_recall"] = label_subset_metrics(claims, predictions, label)["macro_recall"]
    return row


def main():
    claims = load_json("data/dev-claims.json")
    train_claims = load_json("data/train-claims.json")
    default_label = majority_label(train_claims)
    top3_pred = load("outputs/round15/recommended/top3_submission.json")
    top100_pred = load("outputs/round15/recommended/top100_classifier_context.json")
    sources = {
        "ranker": load(OUT / "ranker_claim_key_ranked_top4500.json"),
        "fixed": load(OUT / "fixed_rrf_claim_key_ranked_top4500.json"),
        "top3": pred_source(top3_pred),
        "top100": pred_source(top100_pred),
        "gate": load("outputs/round14/s20_sparse_pool_mix_base_heavy_rrfk500_fine/weighted_rrf_top5000.json"),
    }
    caps = {"ranker": 200, "fixed": 200, "top3": 3, "top100": 100, "gate": 200}
    label_source = top3_pred

    values = [0.0, 0.5, 1.0, 1.5]
    best = None
    sweep_rows = []
    for rrf_k in [20.0, 45.0, 60.0]:
        for wr in values:
            for wf in values:
                for wt3 in values:
                    for wt100 in values:
                        for wg in [0.0, 0.25]:
                            weights = {
                                "ranker": wr,
                                "fixed": wf,
                                "top3": wt3,
                                "top100": wt100,
                                "gate": wg,
                            }
                            if sum(weights.values()) == 0:
                                continue
                            pred = predictions_from_sources(
                                claims, sources, weights, rrf_k, caps, label_source, default_label, 3
                            )
                            assignment = assignment_metrics(claims, pred)
                            macro = macro_recall(claims, pred)
                            row = {
                                "rrf_k": rrf_k,
                                **weights,
                                "top3_macro_recall": macro,
                                "top3_evidence_f": assignment["retrieval_f_score"],
                            }
                            sweep_rows.append(row)
                            key = (macro, assignment["retrieval_f_score"])
                            if best is None or key > best[0]:
                                best = (key, row, pred)

    sweep_rows.sort(key=lambda row: (-row["top3_macro_recall"], -row["top3_evidence_f"]))
    with (OUT / "diagnostic_rrf_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows[:200])

    best_row = best[1]
    weights = {name: best_row[name] for name in ["ranker", "fixed", "top3", "top100", "gate"]}
    top3 = best[2]
    top100 = predictions_from_sources(
        claims, sources, weights, best_row["rrf_k"], caps, label_source, default_label, 100
    )
    top3_path = OUT / "diagnostic_dev_tuned_rrf_top3.json"
    top100_path = OUT / "diagnostic_dev_tuned_rrf_top100.json"
    write_predictions(top3, top3_path)
    write_predictions(top100, top100_path)

    rows = []
    with (OUT / "summary.csv").open(encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))
    note = "DIAGNOSTIC ONLY: RRF weights selected on dev labels; not a generalization estimate"
    rows.append(summary_row("diagnostic_dev_tuned_rrf", 100, top100_path, claims, top100, note))
    rows.append(summary_row("diagnostic_dev_tuned_rrf", 3, top3_path, claims, top3, note))
    with (OUT / "summary_with_diagnostic.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "diagnostic_best.json").open("w", encoding="utf-8") as f:
        json.dump({"best": best_row, "note": note}, f, ensure_ascii=False, indent=2)
    with (OUT / "summary_with_diagnostic.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "Rows named diagnostic_dev_tuned_rrf used dev labels for weight selection.",
                "rows": rows,
                "diagnostic_best": best_row,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(best_row)


if __name__ == "__main__":
    main()
