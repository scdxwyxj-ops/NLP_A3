import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from a3_factcheck.retrieval.query_views import (
    claim_query_views,
    entity_quantity_view,
    keyword_view,
    negation_comparison_view,
    number_context_view,
)
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf
from experiments.retrieval.run_round13_colab_generator import (
    limit_items,
    query_bm25_pool,
    subset_evidence_for_claims,
)


SMART_QUOTES = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "’": "'",
    "‘": "'",
    "\u00a0": " ",
})

CLIMATE_ALIASES = [
    (re.compile(r"\bco\s*2\b|\bco₂\b", re.IGNORECASE), "co2 carbon dioxide co 2"),
    (re.compile(r"\bcarbon\s+dioxide\b", re.IGNORECASE), "co2 carbon dioxide"),
    (re.compile(r"\bghgs?\b", re.IGNORECASE), "ghg greenhouse gas"),
    (re.compile(r"\bgreenhouse\s+gases?\b", re.IGNORECASE), "greenhouse gas ghg"),
    (re.compile(r"\bglobal\s+warming\b", re.IGNORECASE), "global warming climate change warming"),
    (re.compile(r"\bclimate\s+change\b", re.IGNORECASE), "climate change global warming"),
    (re.compile(r"\bsea[-\s]?level\b", re.IGNORECASE), "sea level sea-level sealevel"),
    (re.compile(r"\bipcc\b", re.IGNORECASE), "ipcc intergovernmental panel climate change"),
    (re.compile(r"\bel\s+ni(?:n|ñ)o\b", re.IGNORECASE), "el nino enso"),
    (re.compile(r"\bla\s+ni(?:n|ñ)a\b", re.IGNORECASE), "la nina enso"),
]


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


def clean_text(text):
    text = text.translate(SMART_QUOTES)
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def alias_text(text):
    cleaned = clean_text(text)
    aliases = []
    for pattern, expansion in CLIMATE_ALIASES:
        if pattern.search(cleaned):
            aliases.append(expansion)
    for value, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*(%|per\s*cent|percent|°\s*c|celsius|ppm|ppb|mm|cm|tonnes?|tons?)",
        cleaned,
        flags=re.IGNORECASE,
    ):
        unit_norm = re.sub(r"\s+", " ", unit.lower())
        aliases.append(f"{value} {unit_norm}")
        if "%" in unit_norm or "cent" in unit_norm:
            aliases.append(f"{value}% {value} percent {value} per cent")
        if "°" in unit_norm or "celsius" in unit_norm:
            aliases.append(f"{value} celsius {value} degree celsius {value} C")
    if not aliases:
        return cleaned
    return f"{cleaned} {' '.join(aliases)}"


def quote_views(text):
    views = []
    for match in re.finditer(r'"([^"]{12,})"', text):
        views.append(match.group(1))
    bracketless = re.sub(r"\[[^\]]+\]", " ", text)
    if bracketless != text:
        views.append(bracketless)
    ellipsis_parts = [part.strip(" .") for part in text.split("...") if len(part.split()) >= 4]
    views.extend(ellipsis_parts[:2])
    return views


def expanded_views(claim_text, max_views):
    cleaned = clean_text(claim_text)
    candidates = [
        alias_text(cleaned),
        entity_quantity_view(cleaned),
        number_context_view(cleaned),
        keyword_view(alias_text(cleaned)),
        negation_comparison_view(cleaned),
        *claim_query_views(alias_text(cleaned), max_views=4),
        *quote_views(cleaned),
    ]
    views = []
    seen = set()
    for candidate in candidates:
        candidate = alias_text(candidate)
        key = candidate.lower()
        if len(candidate.split()) < 2 or key in seen:
            continue
        views.append(candidate)
        seen.add(key)
        if len(views) >= max_views:
            break
    return views or [alias_text(cleaned)]


def pseudo_claims(claims, max_views):
    pseudo = {}
    mapping = {}
    for claim_id, claim in claims.items():
        for idx, view in enumerate(expanded_views(claim["claim_text"], max_views)):
            pseudo_id = f"{claim_id}__view{idx}"
            pseudo[pseudo_id] = {**claim, "claim_text": view}
            mapping[pseudo_id] = claim_id
    return pseudo, mapping


def merge_pseudo_pool(claims, pseudo_pool, mapping, top_k, rrf_k):
    scores = {claim_id: defaultdict(float) for claim_id in claims}
    for pseudo_id, candidates in pseudo_pool.items():
        claim_id = mapping[pseudo_id]
        for candidate in candidates[:top_k]:
            scores[claim_id][candidate.evidence_id] += 1.0 / (rrf_k + candidate.rank)
    pool = {}
    for claim_id, evidence_scores in scores.items():
        ranked = sorted(evidence_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        pool[claim_id] = [
            Candidate(claim_id=claim_id, evidence_id=evidence_id, rank=rank, score=float(score))
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return pool


def candidate_union_size(pool, top_k):
    seen = set()
    for candidates in pool.values():
        for candidate in candidates[:top_k]:
            seen.add(candidate.evidence_id)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(
        description="Audit sparse query-view and normalization retrieval over saved/dev data."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--bm25-pool", default="")
    parser.add_argument("--char-pool", default="")
    parser.add_argument("--view-pool", default="")
    parser.add_argument("--output-dir", default="outputs/round14/s18_sparse_query_views")
    parser.add_argument("--pool-top-k", type=int, default=5000)
    parser.add_argument("--top-k-values", default="500,1000,2000,3000,5000")
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--smoke-evidence-limit", type=int, default=0)
    parser.add_argument("--smoke-claims", type=int, default=0)
    args = parser.parse_args()

    claims = limit_items(load_json(args.claims), args.smoke_claims)
    evidence = subset_evidence_for_claims(load_json(args.evidence), {}, claims, args.smoke_evidence_limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.view_pool:
        view_bm25 = load_candidate_pool(args.view_pool)
        pseudo = {}
        print(f"loaded view pool from {args.view_pool}", flush=True)
    else:
        pseudo, mapping = pseudo_claims(claims, args.max_views)
        print(f"expanded claims {len(claims)} -> {len(pseudo)} pseudo views", flush=True)
        pseudo_bm25 = query_bm25_pool(pseudo, evidence, args.pool_top_k, k1=1.5, b=0.75)
        view_bm25 = merge_pseudo_pool(claims, pseudo_bm25, mapping, args.pool_top_k, args.rrf_k)
        write_pool(view_bm25, output_dir / f"view_bm25_top{args.pool_top_k}.json")

    pools = {"view_bm25": view_bm25}
    if args.bm25_pool:
        pools["bm25"] = load_candidate_pool(args.bm25_pool)
    if args.char_pool:
        pools["char"] = load_candidate_pool(args.char_pool)
    if "bm25" in pools and "char" in pools:
        pools["baseline_rrf"] = merge_rrf(
            "baseline_rrf", [pools["bm25"], pools["char"]], top_k=args.pool_top_k, rrf_k=args.rrf_k
        ).pool
    if "bm25" in pools and "char" in pools and "view_bm25" in pools:
        pools["rrf_plus_view_bm25"] = merge_rrf(
            "rrf_plus_view_bm25",
            [pools["bm25"], pools["char"], pools["view_bm25"]],
            top_k=args.pool_top_k,
            rrf_k=args.rrf_k,
        ).pool
        pools["rrf_plus_view_light"] = merge_rrf(
            "rrf_plus_view_light",
            [pools["bm25"], pools["char"], pools["bm25"], pools["char"], pools["view_bm25"]],
            top_k=args.pool_top_k,
            rrf_k=args.rrf_k,
        ).pool
        pools["rrf_plus_view_strong"] = merge_rrf(
            "rrf_plus_view_strong",
            [pools["bm25"], pools["char"], pools["view_bm25"], pools["view_bm25"]],
            top_k=args.pool_top_k,
            rrf_k=args.rrf_k,
        ).pool

    rows = []
    for method, pool in pools.items():
        for retained_k in parse_ints(args.top_k_values):
            row = evaluate_pool(claims, pool, method, retained_k, 0.0)
            row["candidate_union"] = candidate_union_size(pool, retained_k)
            rows.append(row)
    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "summary.json",
        {
            "claims": len(claims),
            "pseudo_views": len(pseudo),
            "view_pool": args.view_pool,
            "pool_top_k": args.pool_top_k,
            "max_views": args.max_views,
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
