import argparse
from pathlib import Path

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.evaluation import run_eval
from a3_factcheck.rerank.api import EvidenceReranker, write_predictions
from a3_factcheck.rerank.candidates import load_candidate_pool


def output_path_for_k(base_output_path, top_k, multi_k):
    if not multi_k:
        return base_output_path
    return base_output_path.with_name(
        f"{base_output_path.stem}-top{top_k}{base_output_path.suffix}"
    )


def parse_top_k_values(top_k_values, top_k):
    if top_k_values:
        return [int(value) for value in top_k_values.split(",") if value]
    return [top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Rerank BM25 candidates with a trained cross-encoder."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument(
        "--candidate-pool", default="outputs/round05/dev-bm25-top50.json"
    )
    parser.add_argument("--model", default="models/round05/minilm-reranker")
    parser.add_argument("--eval-script", default="eval.py")
    parser.add_argument("--output", default="outputs/round05/dev-minilm-reranked.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-values", default="1,3,5,10")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_claims = load_json(args.train_claims)
    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    pool = load_candidate_pool(args.candidate_pool)
    default_label = majority_label(train_claims)
    reranker = EvidenceReranker.from_pretrained(
        args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    top_k_values = parse_top_k_values(args.top_k_values, args.top_k)
    base_output_path = Path(args.output)
    for top_k in top_k_values:
        predictions = reranker.rerank_claims(
            claims=claims,
            evidence=evidence,
            candidate_pool=pool,
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
