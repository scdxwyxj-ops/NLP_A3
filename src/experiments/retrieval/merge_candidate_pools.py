import argparse
import csv
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import (
    evaluate_pool,
    merge_rrf,
    parse_ints,
    write_pool,
)


def main():
    parser = argparse.ArgumentParser(
        description="Merge stored candidate pools with RRF and evaluate recall."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", action="append", required=True)
    parser.add_argument("--top-k-values", default="50,100,200,500,1000,2000")
    parser.add_argument("--pool-top-k", type=int, default=2000)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    claims = load_json(args.claims)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    pools = [load_candidate_pool(path) for path in args.pool]
    merged = merge_rrf(
        name=args.name,
        pools=pools,
        top_k=args.pool_top_k,
        rrf_k=args.rrf_k,
    )

    candidate_path = candidate_dir / f"{args.name}.json"
    write_pool(merged.pool, candidate_path)

    rows = [
        evaluate_pool(
            claims=claims,
            pool=merged.pool,
            method=args.name,
            retained_k=retained_k,
            build_seconds=0.0,
        )
        for retained_k in parse_ints(args.top_k_values)
    ]

    summary_path = output_dir / f"{args.name}_candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote merged pool: {candidate_path}")
    print(f"Wrote summary: {summary_path}")
    for row in rows:
        print(
            f"{args.name} k={row['retained_k']} macro={row['macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f} all_gold={row['all_gold']:.4f} "
            f"refutes={row['refutes_macro_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
