import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'%-]*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
CAP_RE = re.compile(r"\b[A-Z][A-Za-z0-9'%-]*(?:\s+[A-Z][A-Za-z0-9'%-]*){0,3}\b")

UNIT_TERMS = {
    "c",
    "co2",
    "co₂",
    "degree",
    "degrees",
    "fahrenheit",
    "celsius",
    "ppm",
    "ppb",
    "mm",
    "cm",
    "meter",
    "metre",
    "meters",
    "metres",
    "ton",
    "tons",
    "tonne",
    "tonnes",
    "billion",
    "million",
    "percent",
    "percentage",
}
CLIMATE_TERMS = {
    "carbon",
    "climate",
    "co2",
    "co₂",
    "dioxide",
    "emission",
    "emissions",
    "greenhouse",
    "ice",
    "ocean",
    "ppm",
    "temperature",
    "warming",
}
NEGATION_TERMS = {"no", "not", "never", "none", "without", "cannot", "neither", "nor"}
COMPARISON_TERMS = {
    "above",
    "after",
    "before",
    "below",
    "cooler",
    "decrease",
    "decline",
    "greater",
    "higher",
    "increase",
    "less",
    "lower",
    "more",
    "over",
    "smaller",
    "under",
    "warmer",
}


def parse_ints(value):
    return [int(part) for part in value.split(",") if part]


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


def candidate_union_size(pool, k_by_claim):
    seen = set()
    for claim_id, candidates in pool.items():
        top_k = k_by_claim.get(claim_id, 0)
        for candidate in candidates[:top_k]:
            seen.add(candidate.evidence_id)
    return len(seen)


def fixed_k(claims, top_k):
    return {claim_id: top_k for claim_id in claims}


def truncate_pool(pool, k_by_claim):
    return {
        claim_id: list(pool.get(claim_id, [])[: k_by_claim.get(claim_id, 0)])
        for claim_id in k_by_claim
    }


def pool_index(pool):
    return {
        claim_id: {candidate.evidence_id: candidate.rank for candidate in candidates}
        for claim_id, candidates in pool.items()
    }


def top_overlap(left, right, claim_id, top_k):
    left_ids = {candidate.evidence_id for candidate in left.get(claim_id, [])[:top_k]}
    right_ids = {candidate.evidence_id for candidate in right.get(claim_id, [])[:top_k]}
    union = left_ids | right_ids
    return len(left_ids & right_ids) / len(union) if union else 0.0


def claim_features(claim_text, bm25, char, claim_id):
    lowered = claim_text.lower()
    tokens = [token.lower().strip("'") for token in TOKEN_RE.findall(claim_text)]
    token_set = set(tokens)
    caps = CAP_RE.findall(claim_text)
    return {
        "tokens": len(tokens),
        "numbers": len(NUMBER_RE.findall(claim_text)),
        "years": len(YEAR_RE.findall(claim_text)),
        "caps": len(caps),
        "unit_terms": len(token_set & UNIT_TERMS),
        "climate_terms": len(token_set & CLIMATE_TERMS),
        "negation_terms": len(token_set & NEGATION_TERMS),
        "comparison_terms": len(token_set & COMPARISON_TERMS),
        "has_quote_markup": int(
            any(mark in claim_text for mark in ['"', "'", "[", "]", "...", "“", "”"])
        ),
        "clauses": len(re.split(r"[,;:]|\b(?:but|because|although|whereas|while)\b", lowered)),
        "bm25_char_overlap_200": top_overlap(bm25, char, claim_id, 200),
        "bm25_char_overlap_500": top_overlap(bm25, char, claim_id, 500),
    }


def heuristic_depth(features, variant):
    if variant == "v1":
        depth = 3000
        if features["numbers"] or features["years"] or features["unit_terms"]:
            depth += 1200
        if features["climate_terms"] >= 2:
            depth += 1200
        if features["has_quote_markup"]:
            depth += 1000
        if features["negation_terms"] or features["comparison_terms"]:
            depth += 800
        if features["caps"] >= 3 or features["clauses"] >= 3:
            depth += 800
        if features["bm25_char_overlap_500"] < 0.08:
            depth += 1200
        return min(depth, 7000)
    if variant == "v2_conservative":
        depth = 2000
        if features["numbers"] or features["years"] or features["unit_terms"]:
            depth += 1000
        if features["climate_terms"] >= 2:
            depth += 1000
        if features["has_quote_markup"]:
            depth += 1000
        if features["negation_terms"] or features["comparison_terms"]:
            depth += 500
        if features["bm25_char_overlap_500"] < 0.06:
            depth += 1500
        return min(depth, 6000)
    if variant == "v3_aggressive":
        depth = 3500
        if features["numbers"] or features["years"] or features["unit_terms"]:
            depth += 1500
        if features["climate_terms"]:
            depth += 1000
        if features["has_quote_markup"]:
            depth += 1500
        if features["negation_terms"] or features["comparison_terms"]:
            depth += 1000
        if features["caps"] >= 3 or features["clauses"] >= 3:
            depth += 1000
        if features["bm25_char_overlap_500"] < 0.10:
            depth += 1500
        return min(depth, 10000)
    raise ValueError(f"Unknown heuristic variant: {variant}")


def oracle_depth(claims, pool, cap):
    rank_by_claim = pool_index(pool)
    result = {}
    for claim_id, claim in claims.items():
        gold = claim.get("evidences", [])
        ranks = [
            rank_by_claim.get(claim_id, {}).get(evidence_id)
            for evidence_id in gold
        ]
        ranks = [rank for rank in ranks if rank is not None and rank <= cap]
        result[claim_id] = min(max(ranks), cap) if ranks else 0
    return result


def branch_preserve_pool(claims, bm25, char, rrf, branch_k, output_k):
    pool = {}
    for claim_id in claims:
        selected = []
        seen = set()
        for source in (bm25, char):
            for candidate in source.get(claim_id, [])[:branch_k]:
                if candidate.evidence_id in seen:
                    continue
                selected.append((candidate.evidence_id, candidate.score))
                seen.add(candidate.evidence_id)
                if len(selected) >= output_k:
                    break
            if len(selected) >= output_k:
                break
        for candidate in rrf.get(claim_id, []):
            if len(selected) >= output_k:
                break
            if candidate.evidence_id in seen:
                continue
            selected.append((candidate.evidence_id, candidate.score))
            seen.add(candidate.evidence_id)
        pool[claim_id] = [
            Candidate(claim_id=claim_id, evidence_id=evidence_id, rank=rank, score=float(score))
            for rank, (evidence_id, score) in enumerate(selected[:output_k], start=1)
        ]
    return pool


def add_eval_row(rows, claims, pool, method, retained_k, k_by_claim):
    row = evaluate_pool(claims, pool, method, retained_k, 0.0)
    row["candidate_union"] = candidate_union_size(pool, k_by_claim)
    row["avg_depth"] = sum(k_by_claim.values()) / len(k_by_claim) if k_by_claim else 0.0
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(
        description="Audit sparse fusion and adaptive-depth strategies from saved source pools."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument(
        "--bm25-pool",
        default="outputs/round14/s10_interaction_ranker_blend_train2k_target10k/source_pools/target_bm25_top10000.json",
    )
    parser.add_argument(
        "--char-pool",
        default="outputs/round14/s10_interaction_ranker_blend_train2k_target10k/source_pools/target_char_top10000.json",
    )
    parser.add_argument("--output-dir", default="outputs/round14/s16_sparse_adaptive_depth")
    parser.add_argument("--rrf-k-values", default="20,60,100,200,500,1000")
    parser.add_argument("--fixed-k-values", default="500,1000,2000,3000,4000,5000,6000,7000,10000")
    parser.add_argument("--oracle-caps", default="3000,5000,7000,10000")
    args = parser.parse_args()

    claims = load_json(args.claims)
    bm25 = load_candidate_pool(args.bm25_pool)
    char = load_candidate_pool(args.char_pool)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rrf_rows = []
    best_rrf = None
    for rrf_k in parse_ints(args.rrf_k_values):
        rrf = merge_rrf(f"sparse_rrf_k{rrf_k}", [bm25, char], top_k=10000, rrf_k=rrf_k).pool
        for retained_k in parse_ints(args.fixed_k_values):
            k_by = fixed_k(claims, retained_k)
            add_eval_row(
                rrf_rows,
                claims,
                rrf,
                f"sparse_rrf_k{rrf_k}",
                retained_k,
                k_by,
            )
        if rrf_k == 20:
            best_rrf = rrf
    if best_rrf is None:
        best_rrf = merge_rrf("sparse_rrf_k20", [bm25, char], top_k=10000, rrf_k=20).pool

    adaptive_rows = []
    for variant in ["v1", "v2_conservative", "v3_aggressive"]:
        k_by = {
            claim_id: heuristic_depth(claim_features(claim["claim_text"], bm25, char, claim_id), variant)
            for claim_id, claim in claims.items()
        }
        pool = truncate_pool(best_rrf, k_by)
        add_eval_row(adaptive_rows, claims, pool, f"heuristic_{variant}", max(k_by.values()), k_by)

    for cap in parse_ints(args.oracle_caps):
        k_by = oracle_depth(claims, best_rrf, cap)
        pool = truncate_pool(best_rrf, k_by)
        add_eval_row(adaptive_rows, claims, pool, f"oracle_cap{cap}", cap, k_by)

    branch_rows = []
    for branch_k in [50, 100, 150, 200, 250]:
        pool = branch_preserve_pool(claims, bm25, char, best_rrf, branch_k, 500)
        k_by = fixed_k(claims, 500)
        add_eval_row(branch_rows, claims, pool, f"branch_preserve_{branch_k}", 500, k_by)

    for name, rows in {
        "rrf_sweep.csv": rrf_rows,
        "adaptive_depth.csv": adaptive_rows,
        "branch_preserve.csv": branch_rows,
    }.items():
        with (output_dir / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_pool(best_rrf, output_dir / "sparse_rrf_k20_top10000.json")
    write_json(
        output_dir / "summary.json",
        {
            "claims": len(claims),
            "bm25_pool": args.bm25_pool,
            "char_pool": args.char_pool,
            "rrf_rows": rrf_rows,
            "adaptive_rows": adaptive_rows,
            "branch_rows": branch_rows,
        },
    )
    print(f"Wrote {output_dir}")
    for row in adaptive_rows:
        print(
            f"{row['method']} macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"union={row['candidate_union']} avg_depth={row['avg_depth']:.1f}"
        )


if __name__ == "__main__":
    main()
