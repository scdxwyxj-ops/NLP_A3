import argparse
import json
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import (
    build_bm25_candidate_pool,
    candidate_recall_at_k,
    write_candidate_pool,
)
from a3_factcheck.rerank.dataset import build_hard_negative_pairs, write_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Build BM25 candidate pools and hard-negative pairs for reranking."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--negatives-per-claim", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--output-dir", default="outputs/round05")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)

    train_pool = build_bm25_candidate_pool(
        claims=train_claims,
        evidence=evidence,
        top_k=args.top_k,
        max_features=args.max_features,
        k1=args.k1,
        b=args.b,
    )
    dev_pool = build_bm25_candidate_pool(
        claims=dev_claims,
        evidence=evidence,
        top_k=args.top_k,
        max_features=args.max_features,
        k1=args.k1,
        b=args.b,
    )

    train_pairs = build_hard_negative_pairs(
        claims=train_claims,
        evidence=evidence,
        candidate_pool=train_pool,
        negatives_per_claim=args.negatives_per_claim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_pool(train_pool, output_dir / f"train-bm25-top{args.top_k}.json")
    write_candidate_pool(dev_pool, output_dir / f"dev-bm25-top{args.top_k}.json")
    write_jsonl(
        train_pairs,
        output_dir
        / f"train-reranker-pairs-top{args.top_k}-neg{args.negatives_per_claim}.jsonl",
    )

    stats = {
        "top_k": args.top_k,
        "negatives_per_claim": args.negatives_per_claim,
        "train_claims": len(train_claims),
        "dev_claims": len(dev_claims),
        "train_pairs": len(train_pairs),
        "train_positive_pairs": sum(row["label"] == 1 for row in train_pairs),
        "train_negative_pairs": sum(row["label"] == 0 for row in train_pairs),
        "train_recall_at_top_k": candidate_recall_at_k(
            train_claims, train_pool, args.top_k
        ),
        "dev_recall_at_top_k": candidate_recall_at_k(dev_claims, dev_pool, args.top_k),
    }
    stats_path = output_dir / "reranker_dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
