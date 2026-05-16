import argparse
import csv
import json
from pathlib import Path

from a3_factcheck.data import load_json
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf
from experiments.retrieval.run_round13_colab_generator import (
    limit_items,
    log_stage,
    query_bm25_pool,
    query_char_tfidf_pool,
    subset_evidence_for_claims,
)
from experiments.retrieval.run_round14_colab_generator import query_word_tfidf_pool


def parse_ints(value):
    return [int(part) for part in value.split(",") if part]


def candidate_union_size(pool, top_k):
    seen = set()
    for candidates in pool.values():
        for candidate in candidates[:top_k]:
            seen.add(candidate.evidence_id)
    return len(seen)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit sparse BM25/char gates before running dense scoring. "
            "This is intended for cheap local top-k selection."
        )
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round14_sparse_gate_audit")
    parser.add_argument("--pool-top-k", type=int, default=12_000)
    parser.add_argument("--top-k-values", default="2000,4000,8000,12000")
    parser.add_argument("--rrf-k", type=int, default=500)
    parser.add_argument("--query-char-max-claim-fanout", type=int, default=64)
    parser.add_argument("--query-char-max-df-ratio", type=float, default=0.25)
    parser.add_argument("--include-word-tfidf", action="store_true")
    parser.add_argument("--smoke-evidence-limit", type=int, default=0)
    parser.add_argument("--smoke-claims", type=int, default=0)
    parser.add_argument("--write-rrf-pool", action="store_true")
    args = parser.parse_args()

    claims = limit_items(load_json(args.claims), args.smoke_claims)
    evidence = subset_evidence_for_claims(
        load_json(args.evidence),
        {},
        claims,
        args.smoke_evidence_limit,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_top_k = min(args.pool_top_k, len(evidence))
    top_k_values = [min(value, pool_top_k) for value in parse_ints(args.top_k_values)]
    top_k_values = sorted(set(top_k_values))

    log_stage(f"building BM25 sparse gate top{pool_top_k}")
    bm25 = query_bm25_pool(claims, evidence, pool_top_k, k1=1.5, b=0.75)
    log_stage(f"building char sparse gate top{pool_top_k}")
    char = query_char_tfidf_pool(
        claims,
        evidence,
        pool_top_k,
        max_claim_fanout=args.query_char_max_claim_fanout,
        max_df_ratio=args.query_char_max_df_ratio,
    )
    word = None
    if args.include_word_tfidf:
        log_stage(f"building word sparse gate top{pool_top_k}")
        word = query_word_tfidf_pool(claims, evidence, pool_top_k)
    log_stage(f"merging sparse gate top{pool_top_k}")
    rrf_sources = [bm25, char]
    if word is not None:
        rrf_sources.append(word)
    rrf = merge_rrf(
        "sparse_gate_bm25_char_rrf",
        rrf_sources,
        top_k=pool_top_k,
        rrf_k=args.rrf_k,
    ).pool

    rows = []
    pools = {
        "bm25": bm25,
        "char": char,
    }
    if word is not None:
        pools["word_tfidf"] = word
    pools["sparse_gate_rrf"] = rrf
    for name, pool in pools.items():
        for retained_k in top_k_values:
            row = evaluate_pool(
                claims=claims,
                pool=pool,
                method=name,
                retained_k=retained_k,
                build_seconds=0.0,
            )
            row["candidate_union"] = candidate_union_size(pool, retained_k)
            row["union_fraction"] = row["candidate_union"] / len(evidence) if evidence else 0.0
            rows.append(row)

    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.write_rrf_pool:
        from experiments.retrieval.evaluate_candidate_recall import write_pool

        write_pool(rrf, output_dir / f"sparse_gate_rrf_top{pool_top_k}.json")

    write_json(
        output_dir / "summary.json",
        {
            "claims": len(claims),
            "evidence": len(evidence),
            "pool_top_k": pool_top_k,
            "top_k_values": top_k_values,
            "rrf_k": args.rrf_k,
            "query_char_max_claim_fanout": args.query_char_max_claim_fanout,
            "query_char_max_df_ratio": args.query_char_max_df_ratio,
            "include_word_tfidf": args.include_word_tfidf,
        },
    )
    print(f"Wrote {summary_path}")
    for row in rows:
        print(
            f"{row['method']} k={row['retained_k']} "
            f"macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"union={row['candidate_union']}/{len(evidence)}"
        )


if __name__ == "__main__":
    main()
