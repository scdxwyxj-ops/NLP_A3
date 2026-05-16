import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from a3_factcheck.data import load_json
from a3_factcheck.metrics import (
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)
from a3_factcheck.rerank.api import write_predictions


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


EVIDENCE_SYSTEMS = [
    ("TF-IDF top3", "outputs/round04/dev-tfidf-top3.json", "Sparse lexical"),
    ("BM25 top3", "outputs/round04/dev-bm25-top3.json", "Sparse lexical"),
    (
        "MiniLM zero-shot top3",
        "outputs/round05/dev-msmarco-minilm-zero-shot-top3.json",
        "Cross-encoder rerank",
    ),
    (
        "RRF + MiniLM top3",
        "outputs/round07/dev-rrf-bm25-char-minilm-top3.json",
        "Candidate rescue",
    ),
    (
        "Fusion GBDT top3",
        "outputs/round08/fusion_gbdt/dev-fusion-top3.json",
        "Feature fusion",
    ),
    (
        "Round09 blend top3",
        "outputs/round09/blend_top100_refutes_x2/alpha_0.4-top3.json",
        "Best evidence selector",
    ),
    (
        "Neural verifier top3",
        "outputs/round10/verifier_distilroberta_e5/dev-verifier-top3.json",
        "Verifier rerank",
    ),
    (
        "Hybrid verifier gamma0.2 top3",
        "outputs/round10/hybrid_verifier_e5/gamma_0.2-top3.json",
        "Verifier hybrid",
    ),
]


CLASSIFIER_SYSTEMS = [
    (
        "Round08 TF-IDF logreg",
        "outputs/round08_classifier/fusion_gbdt_top50_plain_logreg/dev_predictions.json",
        "Sparse classifier",
    ),
    (
        "Round09 TF-IDF logreg",
        "outputs/round09_classifier/blend_top100_x2_alpha04_top50_plain_logreg/dev_predictions.json",
        "Sparse classifier",
    ),
    (
        "DistilRoBERTa fusion top10",
        "outputs/round10/claim_classifier/distilroberta_fusion_top10_e5/dev_predictions.json",
        "Neural classifier",
    ),
    (
        "DistilRoBERTa blend top10",
        "outputs/round10/claim_classifier/distilroberta_blend_top10_e5/dev_predictions.json",
        "Neural classifier",
    ),
]


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_predictions(claims, predictions, system, family):
    assignment = assignment_metrics(claims, predictions)
    rows = confusion_rows(claims, predictions)
    aggregate = aggregate_confusion(rows)
    y_true = [claim["claim_label"] for claim in claims.values()]
    y_pred = [
        predictions.get(claim_id, {}).get("claim_label", "")
        for claim_id in claims.keys()
    ]
    result = {
        "system": system,
        "family": family,
        "evidence_f": assignment["retrieval_f_score"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
        "claim_macro_f1": f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "macro_recall": macro_recall(claims, predictions),
        "micro_recall": aggregate["micro_recall"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
        "precision": aggregate["precision"],
    }
    for label in LABELS:
        key = label.lower().replace("not_enough_info", "nei")
        result[f"{key}_recall"] = label_subset_metrics(claims, predictions, label)[
            "macro_recall"
        ]
    return result


def combine_label_and_evidence(claims, label_path, evidence_path, output_path):
    label_predictions = read_json(label_path)
    evidence_predictions = read_json(evidence_path)
    combined = {}
    for claim_id, claim in claims.items():
        label_record = label_predictions.get(claim_id, {})
        evidence_record = evidence_predictions.get(claim_id, {})
        combined[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": label_record.get("claim_label", "SUPPORTS"),
            "evidences": evidence_record.get("evidences", []),
        }
    write_predictions(combined, output_path)
    return combined


def plot_grouped_bars(rows, metrics, output_path, title, ylabel):
    names = [row["system"] for row in rows]
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.05), 5.5))
    colors = ["#3366AA", "#D55E00", "#009E73", "#7A5195"]
    for index, metric in enumerate(metrics):
        values = [row[metric] for row in rows]
        offset = (index - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=metric.replace("_", " "), color=colors[index % len(colors)])
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(0.6, max(max(row[m] for row in rows) for m in metrics) * 1.18))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_learning_curve(curve, output_path, title):
    epochs = [row["epoch"] for row in curve]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in curve], marker="o", label="train loss")
    axes[0].plot(epochs, [row["dev_loss"] for row in curve], marker="o", label="dev loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["dev_accuracy"] for row in curve], marker="o", label="dev accuracy")
    axes[1].plot(epochs, [row["dev_macro_f1"] for row in curve], marker="o", label="dev macro-F1")
    axes[1].set_title("Dev metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 0.6)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_candidate_recall(candidate_csv, output_path):
    rows = []
    with Path(candidate_csv).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] in {"bm25_top500", "tfidf_char_top500", "rrf_bm25_char_tfidf_top500"}:
                rows.append(row)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ["bm25_top500", "tfidf_char_top500", "rrf_bm25_char_tfidf_top500"]:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: int(row["retained_k"]))
        ax.plot(
            [int(row["retained_k"]) for row in method_rows],
            [float(row["macro_recall"]) for row in method_rows],
            marker="o",
            label=method.replace("_top500", "").replace("_", " "),
        )
    ax.set_title("Candidate Recall Improves With Wide Sparse Pools", fontsize=14)
    ax.set_xlabel("retained candidates")
    ax.set_ylabel("macro recall")
    ax.set_ylim(0.25, 0.72)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_layers(error_csv, output_path):
    rows = []
    with Path(error_csv).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["layer"].startswith("gold_"):
                rows.append(row)
    labels = [
        "not in top500",
        "top500 not top50",
        "top50 not top20",
        "top20 not top3",
    ]
    values = [float(row["gold_evidence_percent"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color=["#3366AA", "#D55E00", "#009E73", "#7A5195"])
    ax.set_title("Where Gold Evidence Is Lost", fontsize=14)
    ax.set_ylabel("share of gold evidence")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(claims, predictions, output_path, title):
    y_true = [claim["claim_label"] for claim in claims.values()]
    y_pred = [predictions[claim_id]["claim_label"] for claim_id in claims.keys()]
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=25, ha="right")
    ax.set_yticklabels(LABELS)
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_confusion_csv(claims, predictions, output_path):
    y_true = [claim["claim_label"] for claim in claims.values()]
    y_pred = [predictions[claim_id]["claim_label"] for claim_id in claims.keys()]
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    rows = []
    for label, values in zip(LABELS, matrix):
        rows.append({"gold\\pred": label, **dict(zip(LABELS, values.tolist()))})
    write_csv(rows, output_path)


def write_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    output_dir = Path("outputs/round10/full_experiment")
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    claims = load_json("data/dev-claims.json")
    evidence_path = "outputs/round09/blend_top100_refutes_x2/alpha_0.4-top3.json"
    fusion_label_path = (
        "outputs/round10/claim_classifier/distilroberta_fusion_top10_e5/dev_predictions.json"
    )
    blend_label_path = (
        "outputs/round10/claim_classifier/distilroberta_blend_top10_e5/dev_predictions.json"
    )

    final_decoupled = combine_label_and_evidence(
        claims,
        fusion_label_path,
        evidence_path,
        predictions_dir / "final_round09_evidence_distilroberta_fusion_labels.json",
    )
    final_aligned = combine_label_and_evidence(
        claims,
        blend_label_path,
        evidence_path,
        predictions_dir / "final_round09_evidence_distilroberta_blend_labels.json",
    )

    evidence_rows = []
    for system, path, family in EVIDENCE_SYSTEMS:
        if Path(path).exists():
            evidence_rows.append(evaluate_predictions(claims, read_json(path), system, family))
    evidence_rows.append(
        evaluate_predictions(
            claims,
            final_decoupled,
            "Final decoupled system",
            "Round09 evidence + neural label",
        )
    )
    write_csv(evidence_rows, output_dir / "evidence_system_comparison.csv")

    classifier_rows = []
    for system, path, family in CLASSIFIER_SYSTEMS:
        if Path(path).exists():
            classifier_rows.append(evaluate_predictions(claims, read_json(path), system, family))
    classifier_rows.append(
        evaluate_predictions(
            claims,
            final_decoupled,
            "Final: Round09 evidence + fusion-label classifier",
            "Final combined",
        )
    )
    classifier_rows.append(
        evaluate_predictions(
            claims,
            final_aligned,
            "Final: Round09 evidence + blend-label classifier",
            "Final combined",
        )
    )
    write_csv(classifier_rows, output_dir / "classifier_system_comparison.csv")

    final_rows = [
        row
        for row in classifier_rows
        if row["system"]
        in {
            "Round08 TF-IDF logreg",
            "Round09 TF-IDF logreg",
            "DistilRoBERTa fusion top10",
            "DistilRoBERTa blend top10",
            "Final: Round09 evidence + fusion-label classifier",
            "Final: Round09 evidence + blend-label classifier",
        }
    ]
    write_csv(final_rows, output_dir / "final_system_comparison.csv")

    plot_candidate_recall(
        "outputs/round07/candidate_recall_summary.csv",
        figures_dir / "candidate_recall_curve.png",
    )
    plot_error_layers(
        "outputs/round08/error_layer_summary.csv",
        figures_dir / "error_layer_breakdown.png",
    )
    plot_grouped_bars(
        evidence_rows,
        ["evidence_f", "macro_recall", "refutes_recall"],
        figures_dir / "evidence_progression.png",
        "Evidence Selection Progression",
        "score",
    )
    plot_grouped_bars(
        classifier_rows,
        ["claim_accuracy", "claim_macro_f1", "harmonic_mean"],
        figures_dir / "classifier_and_final_progression.png",
        "Claim Classification and Final Score",
        "score",
    )

    for name, metrics_path in [
        (
            "classifier_fusion_learning_curve",
            "outputs/round10/claim_classifier/distilroberta_fusion_top10_e5/metrics.json",
        ),
        (
            "classifier_blend_learning_curve",
            "outputs/round10/claim_classifier/distilroberta_blend_top10_e5/metrics.json",
        ),
        (
            "verifier_learning_curve",
            "outputs/round10/verifier_distilroberta_e5/metrics.json",
        ),
    ]:
        metrics = read_json(metrics_path)
        plot_learning_curve(
            metrics["learning_curve"],
            figures_dir / f"{name}.png",
            name.replace("_", " ").title(),
        )

    plot_confusion(
        claims,
        final_decoupled,
        figures_dir / "final_decoupled_confusion_matrix.png",
        "Final Decoupled Claim Confusion Matrix",
    )
    plot_confusion(
        claims,
        final_aligned,
        figures_dir / "final_aligned_confusion_matrix.png",
        "Final Aligned Claim Confusion Matrix",
    )
    write_confusion_csv(
        claims,
        final_decoupled,
        output_dir / "final_decoupled_confusion_matrix.csv",
    )
    write_confusion_csv(
        claims,
        final_aligned,
        output_dir / "final_aligned_confusion_matrix.csv",
    )

    best_by_h = max(final_rows, key=lambda row: row["harmonic_mean"])
    best_by_macro_f1 = max(final_rows, key=lambda row: row["claim_macro_f1"])
    summary = {
        "best_by_assignment_harmonic": best_by_h,
        "best_by_claim_macro_f1": best_by_macro_f1,
        "final_decoupled_prediction_path": str(
            predictions_dir / "final_round09_evidence_distilroberta_fusion_labels.json"
        ),
        "final_aligned_prediction_path": str(
            predictions_dir / "final_round09_evidence_distilroberta_blend_labels.json"
        ),
        "figures": sorted(str(path) for path in figures_dir.glob("*.png")),
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
