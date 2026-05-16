import argparse
import csv
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.metrics import (
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    load_predictions,
    macro_recall,
)
from a3_factcheck.rerank.candidates import candidate_recall_at_k, load_candidate_pool


FIELDNAMES = [
    "method_id",
    "candidate_source",
    "reranker",
    "negative_strategy",
    "semantic_features",
    "final_output_top_k",
    "classifier_context_top_k",
    "retrieval_f_score",
    "claim_accuracy_placeholder",
    "harmonic_mean",
    "precision",
    "macro_recall",
    "micro_recall",
    "hit_any",
    "all_gold",
    "refutes_recall",
    "refutes_hit_any",
    "topical_fp_pattern",
    "runtime_notes",
    "decision",
    "interpretation",
    "metric_source",
]


BASELINE_ROWS = [
    {
        "method_id": "R05-BM25-T5",
        "candidate_source": "BM25 top-5",
        "reranker": "none",
        "negative_strategy": "none",
        "semantic_features": "none",
        "final_output_top_k": "5",
        "classifier_context_top_k": "5",
        "prediction_path": "outputs/round04/dev-bm25-top5.json",
        "topical_fp_pattern": "high",
        "runtime_notes": "fast sparse lexical retrieval",
        "decision": "baseline",
        "interpretation": "Weak lexical baseline; useful reference row.",
    },
    {
        "method_id": "R05-MINILM-ZS-T3",
        "candidate_source": "BM25 top-50",
        "reranker": "zero-shot cross-encoder/ms-marco-MiniLM-L6-v2",
        "negative_strategy": "none",
        "semantic_features": "none",
        "final_output_top_k": "3",
        "classifier_context_top_k": "3",
        "prediction_path": "outputs/round05/dev-msmarco-minilm-zero-shot-top3.json",
        "topical_fp_pattern": "medium",
        "runtime_notes": "GPU-friendly cross-encoder reranking",
        "decision": "keep as final evidence baseline",
        "interpretation": "Best Round05 final evidence F-score.",
    },
    {
        "method_id": "R05-MINILM-ZS-T5",
        "candidate_source": "BM25 top-50",
        "reranker": "zero-shot cross-encoder/ms-marco-MiniLM-L6-v2",
        "negative_strategy": "none",
        "semantic_features": "none",
        "final_output_top_k": "5",
        "classifier_context_top_k": "5",
        "prediction_path": "outputs/round05/dev-msmarco-minilm-zero-shot-top5.json",
        "topical_fp_pattern": "medium",
        "runtime_notes": "GPU-friendly cross-encoder reranking",
        "decision": "needs classifier test",
        "interpretation": "Higher recall than top-3 but lower final F-score.",
    },
    {
        "method_id": "R05-MINILM-ZS-T20",
        "candidate_source": "BM25 top-50",
        "reranker": "zero-shot cross-encoder/ms-marco-MiniLM-L6-v2",
        "negative_strategy": "none",
        "semantic_features": "none",
        "final_output_top_k": "20",
        "classifier_context_top_k": "20",
        "prediction_path": "outputs/round05/dev-msmarco-minilm-zero-shot-extra-top20.json",
        "topical_fp_pattern": "high",
        "runtime_notes": "more classifier context, poor final precision",
        "decision": "classifier input candidate",
        "interpretation": "Not good for final output, useful for classifier context recall.",
    },
    {
        "method_id": "R05-MINILM-BCE-T5",
        "candidate_source": "BM25 top-50",
        "reranker": "MiniLM single-logit BCE",
        "negative_strategy": "BM25 non-gold",
        "semantic_features": "none",
        "final_output_top_k": "5",
        "classifier_context_top_k": "5",
        "prediction_path": "outputs/round05/dev-minilm-bce-reranked-top5.json",
        "topical_fp_pattern": "medium",
        "runtime_notes": "fine-tuned for 1 epoch on Round05 hard negatives",
        "decision": "reject as final output",
        "interpretation": "Beats BM25 but underperforms zero-shot MiniLM.",
    },
    {
        "method_id": "R06-TA-MINILM-N10-LR5E6-E05-T3",
        "candidate_source": "BM25 top-50",
        "reranker": "task-aware MiniLM single-logit BCE",
        "negative_strategy": "MiniLM-mined non-gold top-100 negatives",
        "semantic_features": "none",
        "final_output_top_k": "3",
        "classifier_context_top_k": "3",
        "prediction_path": "outputs/round06/dev-minilm-task-aware-neg10-lr5e6-ep05-top3.json",
        "topical_fp_pattern": "medium",
        "runtime_notes": "0.5 epoch, lr 5e-6, negatives_per_claim 10",
        "decision": "keep for comparison, not current best",
        "interpretation": "Task-aware fine-tuning nearly matches zero-shot but does not beat it.",
    },
    {
        "method_id": "R06-TA-MINILM-N10-LR5E6-E05-T5",
        "candidate_source": "BM25 top-50",
        "reranker": "task-aware MiniLM single-logit BCE",
        "negative_strategy": "MiniLM-mined non-gold top-100 negatives",
        "semantic_features": "none",
        "final_output_top_k": "5",
        "classifier_context_top_k": "5",
        "prediction_path": "outputs/round06/dev-minilm-task-aware-neg10-lr5e6-ep05-top5.json",
        "topical_fp_pattern": "medium",
        "runtime_notes": "0.5 epoch, lr 5e-6, negatives_per_claim 10",
        "decision": "keep for comparison, not current best",
        "interpretation": "Slightly below zero-shot top-5, but stronger than Round05 BCE.",
    },
    {
        "method_id": "R06-TA-MINILM-N10-LR5E6-E05-T20",
        "candidate_source": "BM25 top-50",
        "reranker": "task-aware MiniLM single-logit BCE",
        "negative_strategy": "MiniLM-mined non-gold top-100 negatives",
        "semantic_features": "none",
        "final_output_top_k": "20",
        "classifier_context_top_k": "20",
        "prediction_path": "outputs/round06/dev-minilm-task-aware-neg10-lr5e6-ep05-top20.json",
        "topical_fp_pattern": "high",
        "runtime_notes": "0.5 epoch, lr 5e-6, negatives_per_claim 10",
        "decision": "classifier input candidate",
        "interpretation": "Wide context row for classifier-readiness comparison.",
    },
]


def pending_row(method_id, candidate_source, context_top_k, recall, metric_source):
    return {
        "method_id": method_id,
        "candidate_source": candidate_source,
        "reranker": "none",
        "negative_strategy": "none",
        "semantic_features": "none",
        "final_output_top_k": "not_applicable",
        "classifier_context_top_k": str(context_top_k),
        "retrieval_f_score": "not_applicable",
        "claim_accuracy_placeholder": "not_applicable",
        "harmonic_mean": "not_applicable",
        "precision": "not_applicable",
        "macro_recall": f"{recall:.6f}",
        "micro_recall": "pending",
        "hit_any": "pending",
        "all_gold": "pending",
        "refutes_recall": "pending",
        "refutes_hit_any": "pending",
        "topical_fp_pattern": "not_an_output",
        "runtime_notes": "candidate pool upper-bound row",
        "decision": "candidate recall reference",
        "interpretation": "Recall ceiling for wider classifier context.",
        "metric_source": metric_source,
    }


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def build_prediction_row(claims, row_spec):
    prediction_path = Path(row_spec["prediction_path"])
    if not prediction_path.exists():
        row = {key: row_spec.get(key, "pending") for key in FIELDNAMES}
        row["metric_source"] = str(prediction_path)
        row.pop("prediction_path", None)
        return row

    predictions = load_predictions(prediction_path)
    assignment = assignment_metrics(claims, predictions)
    rows = confusion_rows(claims, predictions)
    aggregate = aggregate_confusion(rows)
    refutes = label_subset_metrics(claims, predictions, "REFUTES")

    row = {key: row_spec.get(key, "none") for key in FIELDNAMES}
    row.update(
        {
            "retrieval_f_score": assignment["retrieval_f_score"],
            "claim_accuracy_placeholder": assignment["claim_accuracy"],
            "harmonic_mean": assignment["harmonic_mean"],
            "precision": aggregate["precision"],
            "macro_recall": macro_recall(claims, predictions),
            "micro_recall": aggregate["micro_recall"],
            "hit_any": aggregate["hit_any"],
            "all_gold": aggregate["all_gold"],
            "refutes_recall": refutes["macro_recall"],
            "refutes_hit_any": refutes["hit_any"],
            "metric_source": row_spec["prediction_path"],
        }
    )
    row.pop("prediction_path", None)
    return {key: format_value(row.get(key, "")) for key in FIELDNAMES}


def main():
    parser = argparse.ArgumentParser(
        description="Build the Round06 comparison table from existing results."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument(
        "--bm25-top50-pool", default="outputs/round05/dev-bm25-top50.json"
    )
    parser.add_argument(
        "--bm25-top100-pool", default="outputs/round05_top100/dev-bm25-top100.json"
    )
    parser.add_argument("--output", default="outputs/round06/comparison_table.csv")
    args = parser.parse_args()

    claims = load_json(args.claims)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [build_prediction_row(claims, row) for row in BASELINE_ROWS]

    pool_50 = load_candidate_pool(args.bm25_top50_pool)
    rows.append(
        pending_row(
            method_id="R05-BM25-POOL-T50",
            candidate_source="BM25 top-50",
            context_top_k=50,
            recall=candidate_recall_at_k(claims, pool_50, 50),
            metric_source=args.bm25_top50_pool,
        )
    )
    pool_100 = load_candidate_pool(args.bm25_top100_pool)
    rows.append(
        pending_row(
            method_id="R05-BM25-POOL-T100",
            candidate_source="BM25 top-100",
            context_top_k=100,
            recall=candidate_recall_at_k(claims, pool_100, 100),
            metric_source=args.bm25_top100_pool,
        )
    )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
