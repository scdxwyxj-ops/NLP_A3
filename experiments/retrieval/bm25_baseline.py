import argparse
from pathlib import Path

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.evaluation import run_eval
from a3_factcheck.retrieval.bm25 import (
    build_bm25_similarities,
    build_predictions,
    write_predictions,
)


def parse_top_k_values(top_k_values, top_k):
    if top_k_values:
        return [int(value) for value in top_k_values.split(",") if value]
    return [top_k]


def output_path_for_k(base_output_path, top_k, multi_k):
    if not multi_k:
        return base_output_path
    return base_output_path.with_name(
        f"{base_output_path.stem}-top{top_k}{base_output_path.suffix}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="BM25 lexical retrieval baseline for COMP90042 A3."
    )
    parser.add_argument("--train", default="data/train-claims.json")
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument("--output", default="outputs/round04/dev-bm25.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--top-k-values",
        default="",
        help="Optional comma-separated k values. When set, writes one output per k.",
    )
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    train_claims = load_json(args.train)
    claims = load_json(args.claims)
    evidence = load_json(args.evidence)

    default_label = majority_label(train_claims)
    evidence_ids, similarities = build_bm25_similarities(
        claims=claims,
        evidence=evidence,
        max_features=args.max_features,
        k1=args.k1,
        b=args.b,
    )

    print(f"Default label: {default_label}")
    top_k_values = parse_top_k_values(args.top_k_values, args.top_k)
    base_output_path = Path(args.output)

    for top_k in top_k_values:
        predictions = build_predictions(
            claims=claims,
            evidence_ids=evidence_ids,
            similarities=similarities,
            top_k=top_k,
            default_label=default_label,
        )
        output_path = output_path_for_k(
            base_output_path=base_output_path,
            top_k=top_k,
            multi_k=len(top_k_values) > 1,
        )
        write_predictions(predictions, output_path)

        print(f"\nTop-k: {top_k}")
        print(f"Wrote predictions: {output_path}")
        print(
            run_eval(
                eval_script=Path(args.eval_script),
                predictions_path=output_path,
                groundtruth_path=Path(args.claims),
            )
        )


if __name__ == "__main__":
    main()

