import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool


def parse_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_pool(pool, output_path):
    write_json(
        output_path,
        {
            claim_id: [asdict(candidate) for candidate in candidates]
            for claim_id, candidates in pool.items()
        },
    )


def candidate_union_size(pool, top_k):
    seen = set()
    for candidates in pool.values():
        for candidate in candidates[:top_k]:
            seen.add(candidate.evidence_id)
    return len(seen)


def parse_pool_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Pool specs must use name:weight:path, for example bm25:1.0:path/to/pool.json"
        )
    name, weight, path = parts
    return name, float(weight), path


def weighted_rrf(claims, pool_specs, top_k, rrf_k):
    scores_by_claim = {claim_id: defaultdict(float) for claim_id in claims}
    source_count_by_claim = {claim_id: defaultdict(int) for claim_id in claims}
    for _name, weight, pool in pool_specs:
        if weight <= 0.0:
            continue
        for claim_id in claims:
            for candidate in pool.get(claim_id, []):
                scores_by_claim[claim_id][candidate.evidence_id] += weight / (
                    rrf_k + candidate.rank
                )
                source_count_by_claim[claim_id][candidate.evidence_id] += 1

    mixed = {}
    for claim_id, scores in scores_by_claim.items():
        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                -source_count_by_claim[claim_id][item[0]],
                item[0],
            ),
        )[:top_k]
        mixed[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=float(score),
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return mixed


def main():
    parser = argparse.ArgumentParser(
        description="Fuse saved sparse candidate pools with weighted RRF and evaluate recall."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--output-dir", default="outputs/round14/s20_sparse_pool_mix")
    parser.add_argument("--pool-top-k", type=int, default=5000)
    parser.add_argument("--top-k-values", default="500,1000,2000,3000,5000")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--pool",
        action="append",
        required=True,
        help="Pool spec as name:weight:path. Repeat for multiple pools.",
    )
    args = parser.parse_args()

    claims = load_json(args.claims)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for spec in args.pool:
        name, weight, path = parse_pool_spec(spec)
        loaded.append((name, weight, load_candidate_pool(path)))

    mixed = weighted_rrf(claims, loaded, args.pool_top_k, args.rrf_k)
    output_pool = output_dir / f"weighted_rrf_top{args.pool_top_k}.json"
    write_pool(mixed, output_pool)

    rows = []
    for retained_k in parse_ints(args.top_k_values):
        row = evaluate_pool(claims, mixed, "weighted_rrf", retained_k, 0.0)
        row["candidate_union"] = candidate_union_size(mixed, retained_k)
        rows.append(row)

    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "summary.json",
        {
            "pool_top_k": args.pool_top_k,
            "rrf_k": args.rrf_k,
            "pool_specs": [
                {"name": name, "weight": weight, "size_claims": len(pool)}
                for name, weight, pool in loaded
            ],
            "output_pool": str(output_pool),
            "rows": rows,
        },
    )
    for row in rows:
        print(
            f"{row['method']} k={row['retained_k']} "
            f"macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"union={row['candidate_union']}"
        )


if __name__ == "__main__":
    main()
