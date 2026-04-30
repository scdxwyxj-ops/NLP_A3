import json
from pathlib import Path

import numpy as np


def load_predictions(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def assignment_metrics(claims, predictions):
    """Compute the same aggregate metrics as the assignment eval script."""
    f_scores = []
    accuracies = []
    for claim_id, claim in sorted(claims.items()):
        prediction = predictions.get(claim_id, {})
        if "claim_label" not in prediction or "evidences" not in prediction:
            continue

        accuracies.append(
            1.0 if prediction["claim_label"] == claim.get("claim_label") else 0.0
        )

        predicted_ids = prediction["evidences"]
        evidence_fscore = 0.0
        if isinstance(predicted_ids, list) and predicted_ids:
            predicted_set = set(predicted_ids)
            correct = len(set(claim.get("evidences", [])) & predicted_set)
            if correct > 0:
                recall = correct / len(claim.get("evidences", []))
                precision = correct / len(predicted_ids)
                evidence_fscore = (2 * precision * recall) / (precision + recall)
        f_scores.append(evidence_fscore)

    mean_f = float(np.mean(f_scores if f_scores else [0.0]))
    mean_accuracy = float(np.mean(accuracies if accuracies else [0.0]))
    harmonic_mean = (
        0.0
        if mean_f == 0.0 and mean_accuracy == 0.0
        else (2 * mean_f * mean_accuracy) / (mean_f + mean_accuracy)
    )
    return {
        "retrieval_f_score": mean_f,
        "claim_accuracy": mean_accuracy,
        "harmonic_mean": harmonic_mean,
    }


def confusion_rows(claims, predictions, candidate_pool=None):
    rows = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        predicted_ids = predictions.get(claim_id, {}).get("evidences", [])
        predicted = set(predicted_ids)
        if candidate_pool is None:
            candidate_set = gold | predicted
        else:
            candidate_set = {
                candidate.evidence_id for candidate in candidate_pool.get(claim_id, [])
            } | gold

        tp = gold & predicted
        fp = predicted - gold
        fn = gold - predicted
        tn = candidate_set - predicted - gold
        rows.append(
            {
                "claim_id": claim_id,
                "claim_label": claim.get("claim_label"),
                "tp": len(tp),
                "fp": len(fp),
                "fn": len(fn),
                "tn": len(tn),
                "gold_count": len(gold),
                "predicted_count": len(predicted_ids),
            }
        )
    return rows


def aggregate_confusion(rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    tn = sum(row["tn"] for row in rows)
    claims = len(rows)
    hit_any = sum(1 for row in rows if row["tp"] > 0)
    all_gold = sum(1 for row in rows if row["fn"] == 0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "hit_any": hit_any / claims if claims else 0.0,
        "all_gold": all_gold / claims if claims else 0.0,
    }


def macro_recall(claims, predictions):
    recalls = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        predicted = set(predictions.get(claim_id, {}).get("evidences", []))
        recalls.append(len(gold & predicted) / len(gold))
    return float(np.mean(recalls)) if recalls else 0.0


def label_subset_metrics(claims, predictions, label):
    subset = {
        claim_id: claim
        for claim_id, claim in claims.items()
        if claim.get("claim_label") == label
    }
    rows = confusion_rows(subset, predictions)
    aggregate = aggregate_confusion(rows)
    aggregate["macro_recall"] = macro_recall(subset, predictions)
    return aggregate
