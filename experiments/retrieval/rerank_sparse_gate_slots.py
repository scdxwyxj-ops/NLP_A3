import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate
from a3_factcheck.retrieval.query_views import STOPWORDS
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, merge_rrf
from experiments.retrieval.run_round13_colab_generator import (
    limit_items,
    log_stage,
    query_bm25_pool,
    query_char_tfidf_pool,
    subset_evidence_for_claims,
)
from experiments.retrieval.train_recall_compressor import pool_index, reciprocal_rank


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'%-]*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
CAP_RE = re.compile(r"\b[A-Z][A-Za-z0-9'%-]*(?:\s+[A-Z][A-Za-z0-9'%-]*){0,3}\b")
NEGATION_TERMS = {"no", "not", "never", "none", "without", "neither", "nor", "cannot"}
COMPARISON_TERMS = {
    "above",
    "after",
    "before",
    "below",
    "cooler",
    "decrease",
    "decline",
    "faster",
    "greater",
    "higher",
    "increase",
    "largest",
    "less",
    "lower",
    "more",
    "over",
    "slower",
    "smaller",
    "under",
    "warmer",
}

EXPANSIONS = {
    "acidification": {"acidify", "acidic"},
    "acquired": {"acquire", "buy", "bought", "purchase", "purchased", "takeover", "merger"},
    "acquire": {"acquired", "buy", "bought", "purchase", "purchased", "takeover", "merger"},
    "australia": {"australian"},
    "australian": {"australia"},
    "carbon": {"co2", "dioxide"},
    "ceo": {"chief", "executive"},
    "chief": {"ceo", "executive"},
    "climate": {"warming"},
    "co2": {"carbon", "dioxide"},
    "cool": {"cooling", "cooled"},
    "coronavirus": {"covid", "covid19"},
    "covid": {"coronavirus", "covid19"},
    "dioxide": {"co2", "carbon"},
    "emission": {"emissions", "emit", "emitted", "emits"},
    "emissions": {"emission", "emit", "emitted", "emits"},
    "film": {"movie"},
    "global": {"worldwide"},
    "greenhouse": {"ghg"},
    "ice": {"glacier", "glaciers"},
    "movie": {"film"},
    "temperature": {"temperatures", "warming", "warm"},
    "temperatures": {"temperature", "warming", "warm"},
    "united": {"us", "usa", "america", "states"},
    "warming": {"warm", "warmer", "temperature", "climate"},
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


def simple_variants(token):
    variants = {token}
    if token.endswith("'s"):
        variants.add(token[:-2])
    if len(token) > 4 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    if len(token) > 4 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s"):
        variants.add(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        variants.add(token[:-3])
        variants.add(token[:-3] + "e")
    if len(token) > 4 and token.endswith("ed"):
        variants.add(token[:-2])
        variants.add(token[:-1])
    return {item for item in variants if len(item) > 2}


def core_tokens(text):
    tokens = []
    for token in TOKEN_RE.findall(text):
        lower = token.lower().strip("'")
        if len(lower) < 3 or lower in STOPWORDS:
            continue
        tokens.append(lower)
    return tokens


def expanded_terms(tokens, max_expansions_per_token):
    terms = set()
    for token in tokens:
        variants = simple_variants(token)
        variants.update(EXPANSIONS.get(token, set()))
        for variant in sorted(variants)[: 1 + max_expansions_per_token]:
            if len(variant) > 2 and variant not in STOPWORDS:
                terms.add(variant)
    return terms


@lru_cache(maxsize=250_000)
def text_slots(text):
    lowered = text.lower()
    tokens = [token.lower().strip("'") for token in TOKEN_RE.findall(text)]
    token_set = {token for token in tokens if len(token) > 2}
    positions = defaultdict(list)
    for idx, token in enumerate(tokens):
        if len(token) > 2:
            positions[token].append(idx)
    caps = {match.group(0).lower() for match in CAP_RE.finditer(text)}
    cap_tokens = set()
    for cap in caps:
        cap_tokens.update(core_tokens(cap))
    return {
        "lowered": lowered,
        "tokens": token_set,
        "positions": dict(positions),
        "caps": caps,
        "cap_tokens": cap_tokens,
        "numbers": {item.lower() for item in NUMBER_RE.findall(text)},
        "years": set(YEAR_RE.findall(text)),
        "negation": {token for token in tokens if token in NEGATION_TERMS},
        "comparison": {token for token in tokens if token in COMPARISON_TERMS},
    }


def claim_profile(claim_text, max_expansions_per_token):
    tokens = core_tokens(claim_text)
    slots = text_slots(claim_text)
    slots = dict(slots)
    slots["core"] = set(tokens)
    slots["expanded"] = expanded_terms(tokens, max_expansions_per_token)
    return slots


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def coverage(terms, evidence_slots):
    if not terms:
        return 0.0
    hits = 0
    tokens = evidence_slots["tokens"]
    lowered = evidence_slots["lowered"]
    for term in terms:
        if " " in term:
            hits += int(term in lowered)
        else:
            hits += int(term in tokens)
    return hits / len(terms)


def proximity_score(claim_slots, evidence_slots, window):
    entity_terms = claim_slots["cap_tokens"]
    key_terms = claim_slots["core"] | claim_slots["expanded"]
    if not entity_terms or not key_terms:
        return 0.0
    positions = evidence_slots["positions"]
    entity_positions = []
    key_positions = []
    for term in entity_terms:
        entity_positions.extend(positions.get(term, []))
    for term in key_terms:
        key_positions.extend(positions.get(term, []))
    if not entity_positions or not key_positions:
        return 0.0
    best = min(abs(left - right) for left in entity_positions for right in key_positions)
    return max(0.0, 1.0 - (best / max(1, window))) if best <= window else 0.0


def mismatch_penalty(claim_slots, evidence_slots):
    penalty = 0.0
    if claim_slots["years"] and evidence_slots["years"] and not (
        claim_slots["years"] & evidence_slots["years"]
    ):
        penalty += 1.0
    if claim_slots["numbers"] and evidence_slots["numbers"] and not (
        claim_slots["numbers"] & evidence_slots["numbers"]
    ):
        penalty += 0.7
    if bool(claim_slots["negation"]) != bool(evidence_slots["negation"]):
        penalty += 0.4
    if claim_slots["comparison"] and evidence_slots["comparison"] and not (
        claim_slots["comparison"] & evidence_slots["comparison"]
    ):
        penalty += 0.3
    return penalty


def slot_score(claim_slots, evidence_slots, candidate, source_indexes, claim_id, evidence_id, args):
    bm25_item = source_indexes["bm25"].get(claim_id, {}).get(evidence_id)
    char_item = source_indexes["char"].get(claim_id, {}).get(evidence_id)
    bm25_rr = reciprocal_rank(None if bm25_item is None else bm25_item[0])
    char_rr = reciprocal_rank(None if char_item is None else char_item[0])
    rrf_rank_score = 1.0 - min(candidate.rank, args.max_candidates) / max(
        args.max_candidates, 1
    )

    cov = coverage(claim_slots["expanded"], evidence_slots)
    core_cov = coverage(claim_slots["core"], evidence_slots)
    entity_j = jaccard(claim_slots["caps"], evidence_slots["caps"])
    year_j = jaccard(claim_slots["years"], evidence_slots["years"])
    number_j = jaccard(claim_slots["numbers"], evidence_slots["numbers"])
    comparison_j = jaccard(claim_slots["comparison"], evidence_slots["comparison"])
    proximity = proximity_score(claim_slots, evidence_slots, args.proximity_window)
    mismatch = mismatch_penalty(claim_slots, evidence_slots)
    anchor_hits = (
        len(claim_slots["caps"] & evidence_slots["caps"])
        + len(claim_slots["years"] & evidence_slots["years"])
        + len(claim_slots["numbers"] & evidence_slots["numbers"])
    )

    return (
        args.w_rrf_rank * rrf_rank_score
        + args.w_bm25_rr * bm25_rr
        + args.w_char_rr * char_rr
        + args.w_core_coverage * core_cov
        + args.w_expanded_coverage * cov
        + args.w_entity * entity_j
        + args.w_year * year_j
        + args.w_number * number_j
        + args.w_comparison * comparison_j
        + args.w_proximity * proximity
        + args.w_anchor_hits * min(anchor_hits, 3)
        - args.w_mismatch * mismatch
    )


def rerank_with_slots(claims, evidence, gate_pool, source_pools, args):
    source_indexes = {name: pool_index(pool) for name, pool in source_pools.items()}
    result = {}
    for idx, (claim_id, claim) in enumerate(claims.items()):
        profile = claim_profile(claim["claim_text"], args.max_expansions_per_token)
        scored = []
        for candidate in gate_pool.get(claim_id, [])[: args.max_candidates]:
            evidence_text = evidence.get(candidate.evidence_id, "")
            evidence_slots = text_slots(evidence_text)
            score = slot_score(
                profile,
                evidence_slots,
                candidate,
                source_indexes,
                claim_id,
                candidate.evidence_id,
                args,
            )
            scored.append((candidate.evidence_id, float(score)))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[: args.output_k]
        result[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_id,
                rank=rank,
                score=score,
            )
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
        if idx and idx % 25 == 0:
            log_stage(f"slot reranked {idx}/{len(claims)} claims")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Rerank a large sparse gate with lightweight query expansion and slot matching."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--output-dir", default="outputs/round14_slot_sparse_rerank")
    parser.add_argument("--gate-top-k", type=int, default=12_000)
    parser.add_argument("--output-k", type=int, default=2_000)
    parser.add_argument("--top-k-values", default="500,1000,2000")
    parser.add_argument("--rrf-k", type=int, default=500)
    parser.add_argument("--query-char-max-claim-fanout", type=int, default=64)
    parser.add_argument("--query-char-max-df-ratio", type=float, default=0.25)
    parser.add_argument("--max-candidates", type=int, default=12_000)
    parser.add_argument("--max-expansions-per-token", type=int, default=3)
    parser.add_argument("--proximity-window", type=int, default=16)
    parser.add_argument("--smoke-evidence-limit", type=int, default=0)
    parser.add_argument("--smoke-claims", type=int, default=0)
    parser.add_argument("--write-pools", action="store_true")
    parser.add_argument("--w-rrf-rank", type=float, default=1.2)
    parser.add_argument("--w-bm25-rr", type=float, default=35.0)
    parser.add_argument("--w-char-rr", type=float, default=35.0)
    parser.add_argument("--w-core-coverage", type=float, default=1.2)
    parser.add_argument("--w-expanded-coverage", type=float, default=0.8)
    parser.add_argument("--w-entity", type=float, default=0.9)
    parser.add_argument("--w-year", type=float, default=0.8)
    parser.add_argument("--w-number", type=float, default=0.7)
    parser.add_argument("--w-comparison", type=float, default=0.3)
    parser.add_argument("--w-proximity", type=float, default=0.5)
    parser.add_argument("--w-anchor-hits", type=float, default=0.15)
    parser.add_argument("--w-mismatch", type=float, default=0.35)
    args = parser.parse_args()

    claims = limit_items(load_json(args.claims), args.smoke_claims)
    evidence = subset_evidence_for_claims(
        load_json(args.evidence),
        {},
        claims,
        args.smoke_evidence_limit,
    )
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    gate_top_k = min(args.gate_top_k, len(evidence))
    args.max_candidates = min(args.max_candidates, gate_top_k)
    args.output_k = min(args.output_k, args.max_candidates)
    top_k_values = sorted(set(min(k, args.output_k) for k in parse_ints(args.top_k_values)))

    log_stage(f"building BM25 top{gate_top_k}")
    bm25 = query_bm25_pool(claims, evidence, gate_top_k, k1=1.5, b=0.75)
    log_stage(f"building char top{gate_top_k}")
    char = query_char_tfidf_pool(
        claims,
        evidence,
        gate_top_k,
        max_claim_fanout=args.query_char_max_claim_fanout,
        max_df_ratio=args.query_char_max_df_ratio,
    )
    log_stage(f"building sparse RRF top{gate_top_k}")
    gate = merge_rrf("sparse_gate_bm25_char", [bm25, char], gate_top_k, args.rrf_k).pool

    log_stage("slot reranking sparse gate")
    slot_pool = rerank_with_slots(
        claims=claims,
        evidence=evidence,
        gate_pool=gate,
        source_pools={"bm25": bm25, "char": char},
        args=args,
    )
    fused = merge_rrf(
        "sparse_gate_plus_slots",
        [gate, slot_pool],
        args.output_k,
        args.rrf_k,
    ).pool

    rows = []
    for name, pool in {
        "sparse_gate_rrf": gate,
        "slot_only": slot_pool,
        "sparse_gate_plus_slots": fused,
    }.items():
        for retained_k in top_k_values:
            rows.append(evaluate_pool(claims, pool, name, retained_k, 0.0))

    summary_path = output_dir / "candidate_recall_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.write_pools:
        write_pool(gate, candidate_dir / f"sparse_gate_top{gate_top_k}.json")
        write_pool(slot_pool, candidate_dir / f"slot_only_top{args.output_k}.json")
        write_pool(fused, candidate_dir / f"sparse_gate_plus_slots_top{args.output_k}.json")

    write_json(output_dir / "summary.json", {"args": vars(args)})
    print(f"Wrote {summary_path}")
    for row in rows:
        print(
            f"{row['method']} k={row['retained_k']} "
            f"macro={row['macro_recall']:.4f} "
            f"nei={row['not_enough_info_macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f}"
        )


if __name__ == "__main__":
    main()
