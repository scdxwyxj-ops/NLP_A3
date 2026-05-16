import argparse
import csv
import json
from pathlib import Path

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.evaluation import run_eval
from a3_factcheck.metrics import (
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)
from a3_factcheck.rerank.api import write_predictions


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


def parse_ranked_spec(spec):
    parts = spec.split(",", 2)
    if len(parts) != 3:
        raise ValueError(
            "--ranked entries must use the format name,path,weight "
            f"(got: {spec})"
        )
    name, path, weight = parts
    return name, Path(path), float(weight)


def load_ranked(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def output_path_for_k(base_output_path, top_k, multi_k):
    if not multi_k:
        return base_output_path
    return base_output_path.with_name(
        f"{base_output_path.stem}-top{top_k}{base_output_path.suffix}"
    )


def fuse_claim(items_by_source, weights, rrf_k):
    scores = {}
    parts = {}
    for source_name, items in items_by_source.items():
        weight = weights[source_name]
        if weight == 0:
            continue
        for rank, item in enumerate(items, start=1):
            evidence_id = item["evidence_id"]
            scores[evidence_id] = scores.get(evidence_id, 0.0) + weight / (
                rrf_k + rank
            )
            parts.setdefault(evidence_id, {})[f"{source_name}_rank"] = rank
            parts[evidence_id][f"{source_name}_score"] = float(item.get("score", 0.0))

    rows = []
    for evidence_id, score in scores.items():
        rows.append(
            {
                "evidence_id": evidence_id,
                "score": float(score),
                **parts.get(evidence_id, {}),
            }
        )
    rows.sort(key=lambda item: (-item["score"], item["evidence_id"]))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def build_predictions(claims, ranked, default_label, top_k):
    predictions = {}
    for claim_id, claim in claims.items():
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": default_label,
            "evidences": [
                item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]
            ],
        }
    return predictions


def recall_row(name, top_k, output_path, claims, predictions, eval_text):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "top_k": top_k,
        "path": str(output_path),
        "retrieval_f_score": assignment["retrieval_f_score"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
        "macro_recall": macro_recall(claims, predictions),
        "micro_recall": aggregate["micro_recall"],
        "precision": aggregate["precision"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
        "eval_output": eval_text.strip().replace("\n", " | "),
    }
    for label in LABELS:
        key = label.lower()
        row[f"{key}_macro_recall"] = label_subset_metrics(claims, predictions, label)[
            "macro_recall"
        ]
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Fuse ranked evidence lists with weighted reciprocal rank fusion."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument(
        "--ranked",
        action="append",
        required=True,
        help="Ranked source as name,path,weight. Repeat for multiple sources.",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--top-k-values", default="3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-name", default="rrf-fusion")
    args = parser.parse_args()

    claims = load_json(args.claims)
    train_claims = load_json(args.train_claims)
    default_label = majority_label(train_claims)
    specs = [parse_ranked_spec(spec) for spec in args.ranked]
    ranked_sources = {name: load_ranked(path) for name, path, _ in specs}
    weights = {name: weight for name, _, weight in specs}

    ranked = {}
    for claim_id in claims.keys():
        ranked[claim_id] = fuse_claim(
            items_by_source={
                name: source.get(claim_id, [])
                for name, source in ranked_sources.items()
            },
            weights=weights,
            rrf_k=args.rrf_k,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = output_dir / f"{args.base_name}-ranked.json"
    ranked_path.write_text(
        json.dumps(ranked, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote ranked candidates: {ranked_path}")

    top_k_values = [int(value) for value in args.top_k_values.split(",") if value]
    base_output = output_dir / f"{args.base_name}.json"
    summary_rows = []
    for top_k in top_k_values:
        predictions = build_predictions(claims, ranked, default_label, top_k)
        output_path = output_path_for_k(base_output, top_k, len(top_k_values) > 1)
        write_predictions(predictions, output_path)
        eval_text = run_eval(args.eval_script, output_path, args.claims)
        summary_rows.append(
            recall_row(
                name=args.base_name,
                top_k=top_k,
                output_path=output_path,
                claims=claims,
                predictions=predictions,
                eval_text=eval_text,
            )
        )
        print(f"\nTop-k: {top_k}")
        print(eval_text)

    summary_path = output_dir / f"{args.base_name}-summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
