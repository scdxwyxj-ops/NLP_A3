import argparse
import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from a3_factcheck.evaluation import run_eval
from a3_factcheck.rerank.api import write_predictions


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def semantic_summary(prefix, features):
    if not features:
        return ""
    parts = []
    for key in [
        "entities",
        "percentages",
        "quantities",
        "years",
        "negation_cues",
        "comparison_cues",
        "causality_cues",
        "relation_verbs",
    ]:
        values = features.get(key) or []
        if values:
            parts.append(f"{prefix}_{key}=" + " ".join(values))
    return " ".join(parts)


def row_to_text(row, evidence_top_k, include_semantic):
    parts = [f"CLAIM: {row['claim_text']}"]
    if include_semantic:
        parts.append(semantic_summary("claim", row.get("claim_semantic_features")))

    for item in row.get("classifier_evidence_context", [])[:evidence_top_k]:
        parts.append(f"EVIDENCE_{item.get('rank')}: {item.get('text', '')}")
        if include_semantic:
            parts.append(semantic_summary("evidence", item.get("semantic_features")))
    return "\n".join(part for part in parts if part)


def build_prediction_json(rows, predicted_labels):
    predictions = {}
    for row, label in zip(rows, predicted_labels):
        predictions[row["claim_id"]] = {
            "claim_text": row["claim_text"],
            "claim_label": label,
            "evidences": row.get("final_evidence_candidates", []),
        }
    return predictions


def write_confusion(labels, predictions, output_path):
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row.tolist()])


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple claim-label classifier over retrieved evidence context."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--dev-groundtruth", default="data/dev-claims.json")
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument("--output-dir", default="outputs/round07_classifier")
    parser.add_argument("--name", default="concat_tfidf_logreg")
    parser.add_argument("--evidence-top-k", type=int, default=20)
    parser.add_argument("--include-semantic", action="store_true")
    parser.add_argument("--max-features", type=int, default=200_000)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    train_texts = [
        row_to_text(row, args.evidence_top_k, args.include_semantic)
        for row in train_rows
    ]
    dev_texts = [
        row_to_text(row, args.evidence_top_k, args.include_semantic)
        for row in dev_rows
    ]
    train_labels = [row["claim_label"] for row in train_rows]
    dev_labels = [row["claim_label"] for row in dev_rows]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=args.max_features,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=13,
                ),
            ),
        ]
    )
    model.fit(train_texts, train_labels)
    predicted = model.predict(dev_texts)

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "dev_predictions.json"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.txt"
    confusion_path = output_dir / "confusion_matrix.csv"

    prediction_json = build_prediction_json(dev_rows, predicted)
    write_predictions(prediction_json, predictions_path)
    write_confusion(dev_labels, predicted, confusion_path)

    assignment_eval = run_eval(
        eval_script=Path(args.eval_script),
        predictions_path=predictions_path,
        groundtruth_path=Path(args.dev_groundtruth),
    )
    metrics = {
        "name": args.name,
        "evidence_top_k": args.evidence_top_k,
        "include_semantic": args.include_semantic,
        "accuracy": accuracy_score(dev_labels, predicted),
        "macro_f1": f1_score(dev_labels, predicted, labels=LABELS, average="macro"),
        "assignment_eval": assignment_eval,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(
        classification_report(dev_labels, predicted, labels=LABELS),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote confusion matrix: {confusion_path}")


if __name__ == "__main__":
    main()
