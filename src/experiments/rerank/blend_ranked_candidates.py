import argparse
import csv
import json
from pathlib import Path

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.evaluation import run_eval
from a3_factcheck.rerank.api import write_predictions


def load_ranked(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def normalize_scores(items, key):
    values = [float(item.get(key, 0.0)) for item in items]
    low = min(values) if values else 0.0
    high = max(values) if values else 0.0
    if high == low:
        return {item["evidence_id"]: 0.0 for item in items}
    return {
        item["evidence_id"]: (float(item.get(key, 0.0)) - low) / (high - low)
        for item in items
    }


def blend_for_claim(minilm_items, fusion_items, alpha):
    by_id = {}
    for item in minilm_items:
        by_id.setdefault(item["evidence_id"], {})["minilm"] = item
    for item in fusion_items:
        by_id.setdefault(item["evidence_id"], {})["fusion"] = item

    minilm_norm = normalize_scores(minilm_items, "score")
    fusion_norm = normalize_scores(fusion_items, "score")
    rows = []
    for evidence_id, parts in by_id.items():
        score = alpha * minilm_norm.get(evidence_id, 0.0) + (1 - alpha) * fusion_norm.get(evidence_id, 0.0)
        rows.append(
            {
                "evidence_id": evidence_id,
                "score": float(score),
                "blend_score": float(score),
                "minilm_score": float(parts.get("minilm", {}).get("score", 0.0)),
                "fusion_score": float(parts.get("fusion", {}).get("score", 0.0)),
                "minilm_rank": int(parts.get("minilm", {}).get("rank", 999999)),
                "fusion_rank": int(parts.get("fusion", {}).get("rank", 999999)),
            }
        )
    rows.sort(key=lambda item: (-item["score"], item["evidence_id"]))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def output_path_for_k(base_output_path, top_k, multi_k):
    if not multi_k:
        return base_output_path
    return base_output_path.with_name(
        f"{base_output_path.stem}-top{top_k}{base_output_path.suffix}"
    )


def parse_eval(text):
    import re

    return {
        "retrieval_f_score": float(re.search(r"F\)\s*=\s*([0-9.]+)", text).group(1)),
        "claim_accuracy": float(re.search(r"Accuracy \(A\)\s*=\s*([0-9.]+)", text).group(1)),
        "harmonic_mean": float(re.search(r"Harmonic Mean.*=\s*([0-9.]+)", text).group(1)),
    }


def main():
    parser = argparse.ArgumentParser(description="Alpha-blend MiniLM and fusion ranked candidates.")
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--minilm-ranked", required=True)
    parser.add_argument("--fusion-ranked", required=True)
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--top-k-values", default="3,20")
    parser.add_argument("--output-dir", default="outputs/round09/blend")
    args = parser.parse_args()

    claims = load_json(args.claims)
    train_claims = load_json(args.train_claims)
    default_label = majority_label(train_claims)
    minilm_ranked = load_ranked(args.minilm_ranked)
    fusion_ranked = load_ranked(args.fusion_ranked)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(value) for value in args.alphas.split(",") if value]
    top_k_values = [int(value) for value in args.top_k_values.split(",") if value]

    summary_rows = []
    for alpha in alphas:
        ranked = {
            claim_id: blend_for_claim(
                minilm_ranked.get(claim_id, []),
                fusion_ranked.get(claim_id, []),
                alpha,
            )
            for claim_id in claims.keys()
        }
        alpha_name = f"alpha_{alpha:.1f}"
        ranked_path = output_dir / f"{alpha_name}_ranked.json"
        ranked_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
        base_output = output_dir / f"{alpha_name}.json"

        for top_k in top_k_values:
            predictions = {}
            for claim_id, claim in claims.items():
                predictions[claim_id] = {
                    "claim_text": claim["claim_text"],
                    "claim_label": default_label,
                    "evidences": [item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]],
                }
            output_path = output_path_for_k(base_output, top_k, len(top_k_values) > 1)
            write_predictions(predictions, output_path)
            eval_text = run_eval(args.eval_script, output_path, args.claims)
            summary_rows.append(
                {
                    "alpha_minilm": alpha,
                    "top_k": top_k,
                    "prediction_path": str(output_path),
                    **parse_eval(eval_text),
                }
            )

    summary_path = output_dir / "blend_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
