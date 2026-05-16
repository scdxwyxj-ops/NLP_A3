import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.evaluation import run_eval
from a3_factcheck.rerank.api import write_predictions


FEATURES = [
    "minilm_rank",
    "minilm_score",
    "rrf_rank",
    "rrf_score",
    "bm25_rank",
    "bm25_score",
    "char_tfidf_rank",
    "char_tfidf_score",
    "in_bm25",
    "in_char_tfidf",
    "source_count",
    "entity_overlap_count",
    "entity_jaccard",
    "percentage_overlap_count",
    "quantity_overlap_count",
    "year_overlap_count",
    "claim_has_negation",
    "evidence_has_negation",
    "negation_presence_match",
    "negation_xor",
    "comparison_cue_overlap",
    "causality_cue_overlap",
    "relation_verb_overlap",
]


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in FEATURES:
            row[key] = float(row[key])
        row["label_is_gold"] = int(row["label_is_gold"])
    return rows


def matrix(rows):
    return np.asarray([[row[key] for key in FEATURES] for row in rows], dtype=np.float32)


def labels(rows):
    return np.asarray([row["label_is_gold"] for row in rows], dtype=np.int32)


def group_ranked(rows, scores):
    grouped = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row["claim_id"], []).append(
            {
                "evidence_id": row["evidence_id"],
                "score": float(score),
                "fusion_score": float(score),
                "minilm_rank": int(float(row["minilm_rank"])),
                "minilm_score": float(row["minilm_score"]),
                "rrf_rank": int(float(row["rrf_rank"])),
                "rrf_score": float(row["rrf_score"]),
                "bm25_rank": int(float(row["bm25_rank"])),
                "bm25_score": float(row["bm25_score"]),
                "char_tfidf_rank": int(float(row["char_tfidf_rank"])),
                "char_tfidf_score": float(row["char_tfidf_score"]),
            }
        )
    for claim_id, items in grouped.items():
        items.sort(key=lambda item: (-item["score"], item["evidence_id"]))
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
            item["fusion_rank"] = rank
    return grouped


def output_path_for_k(base_output_path, top_k, multi_k):
    if not multi_k:
        return base_output_path
    return base_output_path.with_name(
        f"{base_output_path.stem}-top{top_k}{base_output_path.suffix}"
    )


def write_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prediction_from_ranked(claims, ranked, default_label, top_k):
    predictions = {}
    for claim_id, claim in claims.items():
        evidences = [item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]]
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": default_label,
            "evidences": evidences,
        }
    return predictions


def train_model(name, train_x, train_y):
    if name == "gbdt":
        return GradientBoostingClassifier(random_state=13)
    if name == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=2000)
    raise ValueError(f"Unsupported model: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a lightweight feature fusion reranker."
    )
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--dev-table", required=True)
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument("--model", choices=["logreg", "gbdt"], default="logreg")
    parser.add_argument("--refutes-positive-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/round08/fusion_logreg")
    parser.add_argument("--top-k-values", default="3,20,50")
    args = parser.parse_args()

    train_rows = read_rows(args.train_table)
    dev_rows = read_rows(args.dev_table)
    train_x = matrix(train_rows)
    train_y = labels(train_rows)
    dev_x = matrix(dev_rows)
    dev_y = labels(dev_rows)

    scaler = None
    if args.model == "logreg":
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        dev_x = scaler.transform(dev_x)

    sample_weight = np.ones(len(train_rows), dtype=np.float32)
    if args.refutes_positive_weight != 1.0:
        for index, row in enumerate(train_rows):
            if row["claim_label"] == "REFUTES" and row["label_is_gold"]:
                sample_weight[index] = args.refutes_positive_weight

    model = train_model(args.model, train_x, train_y)
    model.fit(train_x, train_y, sample_weight=sample_weight)
    if hasattr(model, "predict_proba"):
        train_scores = model.predict_proba(train_x)[:, 1]
        dev_scores = model.predict_proba(dev_x)[:, 1]
    else:
        train_scores = model.decision_function(train_x)
        dev_scores = model.decision_function(dev_x)

    metrics = {
        "model": args.model,
        "train_rows": len(train_rows),
        "train_positive_rows": int(train_y.sum()),
        "dev_rows": len(dev_rows),
        "dev_positive_rows": int(dev_y.sum()),
        "refutes_positive_weight": args.refutes_positive_weight,
        "dev_average_precision": float(average_precision_score(dev_y, dev_scores)),
    }
    try:
        metrics["dev_roc_auc"] = float(roc_auc_score(dev_y, dev_scores))
    except ValueError:
        metrics["dev_roc_auc"] = None

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    default_label = majority_label(train_claims)
    train_ranked = group_ranked(train_rows, train_scores)
    dev_ranked = group_ranked(dev_rows, dev_scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(train_ranked, output_dir / "train-fusion-ranked-top50.json")
    write_json(dev_ranked, output_dir / "dev-fusion-ranked-top50.json")

    top_k_values = [int(value) for value in args.top_k_values.split(",") if value]
    base_output = output_dir / "dev-fusion.json"
    eval_outputs = {}
    for top_k in top_k_values:
        predictions = prediction_from_ranked(
            claims=dev_claims,
            ranked=dev_ranked,
            default_label=default_label,
            top_k=top_k,
        )
        output_path = output_path_for_k(base_output, top_k, len(top_k_values) > 1)
        write_predictions(predictions, output_path)
        eval_text = run_eval(args.eval_script, output_path, args.dev_claims)
        eval_outputs[f"top{top_k}"] = eval_text
        print(f"\nTop-k: {top_k}")
        print(eval_text)

    metrics["assignment_eval"] = eval_outputs
    write_json(metrics, output_dir / "metrics.json")
    if hasattr(model, "feature_importances_"):
        importances = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(FEATURES, model.feature_importances_)
        ]
        importances.sort(key=lambda item: item["importance"], reverse=True)
        write_json(importances, output_dir / "feature_importance.json")
    elif hasattr(model, "coef_"):
        coefficients = [
            {"feature": feature, "coefficient": float(coef)}
            for feature, coef in zip(FEATURES, model.coef_[0])
        ]
        coefficients.sort(key=lambda item: abs(item["coefficient"]), reverse=True)
        write_json(coefficients, output_dir / "feature_coefficients.json")
    print(f"Wrote ranked candidates and metrics to {output_dir}")


if __name__ == "__main__":
    main()
