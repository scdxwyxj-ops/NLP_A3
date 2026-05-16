#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.tools.common import find_forbidden_tokens, load_json, write_json

DEFAULT_CLAIMS_PATH = Path("data/dev-claims.json")
DEFAULT_EVIDENCE_PATH = Path("data/evidence.json")
DEFAULT_CE_POOL_PATH = Path(
    "round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/"
    "dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json"
)

DEFAULT_HAND_POOLS: tuple[Path, ...] = (
    Path("round18/outputs/o_sparse/o_s8_hand_feature_ranker/"
         "dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json"),
    Path("round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_top1000/"
         "dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_top1000_top500_candidates.json"),
    Path("round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_rrf_top1000/"
         "dev_full_dev_o_s10_wide_hand_feature_bm25_char_rrf_top1000_top500_candidates.json"),
    Path("round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_union_upper_top1000/"
         "dev_full_dev_o_s10_wide_hand_feature_bm25_char_union_upper_top1000_top500_candidates.json"),
    Path("round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_pos16/"
         "dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_pos16_top500_candidates.json"),
    Path("round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_train500/"
         "dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_train500_top500_candidates.json"),
)

DEFAULT_EVAL_KS = [1, 3, 5, 10, 64, 500]
DEFAULT_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_TIE_BREAKERS = (
    "min_source_rank",
    "ce_rank_first",
    "hand_rank_first",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round18 bounded CE+hand-feature score fusion sweep for top-k evidence ranking."
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_CLAIMS_PATH,
        help="Claims split JSON.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
        help="Evidence corpus JSON.",
    )
    parser.add_argument(
        "--ce-pool",
        type=Path,
        default=DEFAULT_CE_POOL_PATH,
        help="Candidate pool with CE top500 scoring.",
    )
    parser.add_argument(
        "--hand-pool",
        type=Path,
        action="append",
        default=None,
        help=(
            "Path to hand-feature top500 candidate pool. Can be passed multiple times. "
            "If omitted, uses built-in O-S8/O-S10 candidate sets."
        ),
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Keep this many evidence entries per claim after fusion.",
    )
    parser.add_argument(
        "--eval-k",
        default=",".join(str(v) for v in DEFAULT_EVAL_KS),
        help="Comma-separated recall-k values.",
    )
    parser.add_argument(
        "--weight-grid",
        default=",".join(str(v) for v in DEFAULT_WEIGHTS),
        help="Comma-separated CE weights (hand weight = 1 - CE weight).",
    )
    parser.add_argument(
        "--tie-breakers",
        default=",".join(DEFAULT_TIE_BREAKERS),
        help="Comma-separated tie-break modes: min_source_rank, ce_rank_first, hand_rank_first.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/reports/top3_score_fusion_spark"),
        help="Directory for experiment summary artifacts.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("eval-k must include at least one integer.")
    return sorted(set(values))


def parse_weights(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if not 0.0 <= val <= 1.0:
            raise ValueError("Weight values must be in [0, 1].")
        values.append(round(val, 4))
    if not values:
        raise ValueError("weight-grid must include at least one value.")
    return sorted(set(values))


def parse_tie_breakers(raw: str) -> list[str]:
    values = []
    for item in raw.split(","):
        val = item.strip()
        if not val:
            continue
        if val not in {"min_source_rank", "ce_rank_first", "hand_rank_first"}:
            raise ValueError(f"Unsupported tie-break mode: {val}")
        values.append(val)
    if not values:
        raise ValueError("tie-breakers must include at least one value.")
    return values


def coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_pool(
    path: Path,
    claims: dict[str, Any],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate pool: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Candidate pool must be a dict: {path}")

    parsed: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        entries = payload.get(claim_id, [])
        if not isinstance(entries, list):
            continue

        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for position, item in enumerate(entries, start=1):
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            rank = coerce_int(item.get("rank", 0))
            if rank <= 0:
                rank = position
            score = coerce_float(item.get("score", 0.0))
            reranker_score = coerce_float(item.get("reranker_score", score))
            source_score = coerce_float(
                item.get("source_score", item.get("reranker_score", score))
            )
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "rank": rank,
                    "score": score,
                    "reranker_score": reranker_score,
                    "source_score": source_score,
                }
            )
            if len(rows) >= candidate_k:
                break
        if rows:
            parsed[claim_id] = rows
    return parsed


def minmax_normalized(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if math.isclose(max_value, min_value):
        return {key: 0.0 for key in values}
    span = max_value - min_value
    return {key: (value - min_value) / span for key, value in values.items()}


def rank_key(
    score: float,
    tie_break: str,
    trace: dict[str, dict[str, Any]],
    evidence_id: str,
) -> tuple:
    ce_rank = trace.get("ce", {}).get("rank", 10**9)
    hand_rank = trace.get("hand", {}).get("rank", 10**9)
    if tie_break == "ce_rank_first":
        primary = ce_rank
        secondary = hand_rank
    elif tie_break == "hand_rank_first":
        primary = hand_rank
        secondary = ce_rank
    else:
        primary = min(ce_rank, hand_rank)
        secondary = max(ce_rank, hand_rank)
    return (-score, primary, secondary, evidence_id)


def build_fused_pool(
    ce_pool: dict[str, list[dict[str, Any]]],
    hand_pool: dict[str, list[dict[str, Any]]],
    claim_id: str,
    ce_weight: float,
    hand_weight: float,
    tie_break: str,
    candidate_k: int,
) -> list[dict[str, Any]]:
    ce_rows = ce_pool.get(claim_id, [])
    hand_rows = hand_pool.get(claim_id, [])

    ce_map: dict[str, dict[str, float]] = {
        row["evidence_id"]: {
            "rank": row["rank"],
            "source_score": float(row["source_score"]),
            "fusion_score": float(row["reranker_score"]),
        }
        for row in ce_rows
    }
    hand_map: dict[str, dict[str, float]] = {
        row["evidence_id"]: {
            "rank": row["rank"],
            "source_score": float(row["source_score"]),
            "fusion_score": float(row["source_score"]),
        }
        for row in hand_rows
    }

    ce_norm = minmax_normalized({eid: row["fusion_score"] for eid, row in ce_map.items()})
    hand_norm = minmax_normalized({eid: row["fusion_score"] for eid, row in hand_map.items()})

    fused_trace: dict[str, dict[str, Any]] = {}
    for evidence_id in set(ce_map) | set(hand_map):
        row = {}
        if evidence_id in ce_map:
            ce_raw = ce_map[evidence_id]
            ce_norm_score = ce_norm.get(evidence_id, 0.0)
            row["ce"] = {
                "rank": ce_raw["rank"],
                "raw_score": ce_raw["fusion_score"],
                "norm_score": ce_norm_score,
                "weight": ce_weight,
                "source_score": ce_raw["source_score"],
            }
        if evidence_id in hand_map:
            hand_raw = hand_map[evidence_id]
            hand_norm_score = hand_norm.get(evidence_id, 0.0)
            row["hand"] = {
                "rank": hand_raw["rank"],
                "raw_score": hand_raw["source_score"],
                "norm_score": hand_norm_score,
                "weight": hand_weight,
            }
        ce_component = row.get("ce", {}).get("norm_score", 0.0) * ce_weight
        hand_component = row.get("hand", {}).get("norm_score", 0.0) * hand_weight
        fused_trace[evidence_id] = {
            **row,
            "fused_score": ce_component + hand_component,
        }

    ranked = sorted(
        fused_trace.items(),
        key=lambda item: rank_key(item[1].get("fused_score", 0.0), tie_break, item[1], item[0]),
    )

    output: list[dict[str, Any]] = []
    for evidence_id, trace in ranked[:candidate_k]:
        source_breakdown = []
        if "ce" in trace:
            source_breakdown.append(
                {
                    "source": "ce",
                    "source_rank": trace["ce"]["rank"],
                    "source_score": trace["ce"]["raw_score"],
                    "weight": trace["ce"]["weight"],
                    "normalized_source_score": trace["ce"]["norm_score"],
                }
            )
        if "hand" in trace:
            source_breakdown.append(
                {
                    "source": "hand-feature",
                    "source_rank": trace["hand"]["rank"],
                    "source_score": trace["hand"]["raw_score"],
                    "weight": trace["hand"]["weight"],
                    "normalized_source_score": trace["hand"]["norm_score"],
                }
            )
        output.append(
            {
                "evidence_id": evidence_id,
                "score": trace["fused_score"],
                "source_breakdown": source_breakdown,
                "source_count": len(source_breakdown),
            }
        )
    return output


def evaluate_recall_at_k(
    claims: dict[str, Any],
    pool: dict[str, list[dict[str, Any]]],
    ks: list[int],
) -> dict[str, float]:
    ks = sorted(set(ks))
    rows = {
        "claims_with_evidence": 0,
        "evaluated_ks": ks,
    }
    claim_recall_totals: dict[int, list[float]] = {k: [] for k in ks}
    total_hits = {k: {"tp": 0.0, "gold": 0.0, "claims_hit": 0} for k in ks}
    claim_count = 0
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claim_count += 1
        ranked = [row["evidence_id"] for row in pool.get(claim_id, [])]
        for k in ks:
            predicted = set(ranked[:k])
            tp = len(gold & predicted)
            total_hits[k]["tp"] += tp
            total_hits[k]["gold"] += float(len(gold))
            recall = tp / float(len(gold))
            claim_recall_totals[k].append(recall)
            if tp > 0:
                total_hits[k]["claims_hit"] += 1

    rows["claims_with_evidence"] = claim_count
    rows["avg_candidate_count"] = (
        float(sum(len(v) for v in pool.values()) / claim_count) if claim_count else 0.0
    )
    rows["union_candidates"] = len({row["evidence_id"] for rows_ in pool.values() for row in rows_})

    for k in ks:
        rows[f"macro_recall_at_{k}"] = float(
            sum(claim_recall_totals[k]) / len(claim_recall_totals[k])
        ) if claim_recall_totals[k] else 0.0
        tp = float(total_hits[k]["tp"])
        gold_total = float(total_hits[k]["gold"])
        rows[f"micro_recall_at_{k}"] = tp / gold_total if gold_total else 0.0
        rows[f"hit_any_at_{k}"] = total_hits[k]["claims_hit"] / claim_count if claim_count else 0.0

    return rows


def load_claims(path: Path) -> dict[str, Any]:
    claims = load_json(path)
    if not isinstance(claims, dict):
        raise TypeError(f"Claims file must be a dict: {path}")
    return claims


def build_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    eval_ks = parse_k_list(args.eval_k)
    weights = parse_weights(args.weight_grid)
    tie_breakers = parse_tie_breakers(args.tie_breakers)

    if args.hand_pool:
        hand_pools = [Path(p) for p in args.hand_pool]
    else:
        hand_pools = list(DEFAULT_HAND_POOLS)

    for required in (args.claims, args.evidence, args.ce_pool, *hand_pools):
        if not required.exists():
            raise SystemExit(f"Missing required input: {required}")

    if args.candidate_k < max(eval_ks):
        raise ValueError("candidate-k must be >= max eval-k.")

    claims = load_claims(args.claims)
    ce_pool = parse_pool(args.ce_pool, claims, args.candidate_k)
    hand_pool_data = {
        str(path): parse_pool(path, claims, args.candidate_k)
        for path in hand_pools
    }

    records: list[dict[str, Any]] = []
    output_dir = args.output_dir
    build_output_dir(output_dir)

    for hand_path, hand_pool in hand_pool_data.items():
        for ce_weight in weights:
            hand_weight = 1.0 - ce_weight
            for tie_break in tie_breakers:
                variant_rows: dict[str, list[dict[str, Any]]] = {}
                for claim_id in claims:
                    variant_rows[claim_id] = build_fused_pool(
                        ce_pool=ce_pool,
                        hand_pool=hand_pool,
                        claim_id=claim_id,
                        ce_weight=ce_weight,
                        hand_weight=hand_weight,
                        tie_break=tie_break,
                        candidate_k=args.candidate_k,
                    )
                metrics = evaluate_recall_at_k(claims, variant_rows, eval_ks)

                variant_id = (
                    f"ce_{ce_weight:.2f}_hand_{hand_weight:.2f}_"
                    f"tie-{tie_break.replace('_', '-')}_"
                    f"hand_{Path(hand_path).name}"
                )
                row = {
                    "variant_id": variant_id,
                    "hand_pool": hand_path,
                    "ce_pool": str(args.ce_pool),
                    "ce_weight": ce_weight,
                    "hand_weight": hand_weight,
                    "tie_break": tie_break,
                    "candidate_k": args.candidate_k,
                    "status": "diagnostic-only",
                }
                for k in eval_ks:
                    row[f"macro_recall@{k}"] = metrics.get(f"macro_recall_at_{k}", 0.0)
                    row[f"micro_recall@{k}"] = metrics.get(f"micro_recall_at_{k}", 0.0)
                    row[f"hit_any@{k}"] = metrics.get(f"hit_any_at_{k}", 0.0)
                row["avg_candidate_count"] = metrics.get("avg_candidate_count", 0.0)
                row["union_candidates"] = metrics.get("union_candidates", 0)
                records.append(row)
    # Submission evidence uses the first three evidence items, so the diagnostic
    # sweep ranks variants by top-3 recall first. Top-10/top-64/top-500 are only
    # tie-breaks to avoid choosing a narrow improvement that damages nearby cuts.
    def score_for_ranking(row: dict[str, Any]) -> tuple:
        return (
            row.get("macro_recall@3", 0.0),
            row.get("macro_recall@10", 0.0),
            row.get("macro_recall@64", 0.0),
            row.get("macro_recall@500", 0.0),
        )

    best = sorted(records, key=score_for_ranking, reverse=True)[0]
    best_path = output_dir / f"top3_score_fusion_best_{Path(best['hand_pool']).stem}_{best['tie_break']}_ce{best['ce_weight']:.2f}_top500_candidates.json"
    # rewrite best candidate set for easy adoption
    hand_path_key = best["hand_pool"]
    ce_weight = best["ce_weight"]
    hand_weight = best["hand_weight"]
    tie_break = best["tie_break"]

    best_pool = {
        claim_id: build_fused_pool(
            ce_pool=ce_pool,
            hand_pool=hand_pool_data[hand_path_key],
            claim_id=claim_id,
            ce_weight=ce_weight,
            hand_weight=hand_weight,
            tie_break=tie_break,
            candidate_k=args.candidate_k,
        )
        for claim_id in claims
    }
    best_metrics = {key: val for key, val in best.items() if key.startswith(("macro_recall@", "micro_recall@", "hit_any@"))}
    best_metrics.update(
        {
            "status": best["status"],
            "ce_weight": best["ce_weight"],
            "hand_weight": best["hand_weight"],
            "tie_break": best["tie_break"],
            "hand_pool": best["hand_pool"],
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
        }
    )

    write_json(best_path, best_pool)
    write_json(
        output_dir / "top3_score_fusion_best_metrics.json",
        best_metrics,
    )
    write_json(
        output_dir / "top3_score_fusion_summary.json",
        {
            "selected_hand_pool": best["hand_pool"],
            "selection_criteria": "argmax [macro@3, macro@10, macro@64, macro@500]",
            "best_variant": {
                "variant_id": best["variant_id"],
                "ce_weight": best["ce_weight"],
                "hand_weight": best["hand_weight"],
                "tie_break": best["tie_break"],
                "status": best["status"],
            },
            "best_metrics": {
                "macro@1": best["macro_recall@1"],
                "macro@3": best["macro_recall@3"],
                "macro@5": best["macro_recall@5"],
                "macro@10": best["macro_recall@10"],
                "macro@64": best["macro_recall@64"],
                "macro@500": best["macro_recall@500"],
            },
            "records": records,
        },
    )

    summary_csv = output_dir / "top3_score_fusion_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["variant_id", "hand_pool", "ce_pool", "ce_weight", "hand_weight", "tie_break"]
        fieldnames.extend(["candidate_k", "status", "avg_candidate_count", "union_candidates"])
        for k in eval_ks:
            fieldnames.extend([f"macro_recall@{k}", f"micro_recall@{k}", f"hit_any@{k}"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(records, key=lambda item: score_for_ranking(item), reverse=True):
            writer.writerow(row)

    forbidden_hits = find_forbidden_tokens(
        [str(args.claims), str(args.evidence), str(args.ce_pool)] + list(hand_pool_data.keys()) + [str(output_dir)]
    )

    write_json(
        output_dir / "top3_score_fusion_report.json",
        {
            "leakage_risk": "low",
            "status": "diagnostic-only",
            "forbidden_hits": forbidden_hits,
            "run_inputs": {
                "claims": str(args.claims),
                "evidence": str(args.evidence),
                "ce_pool": str(args.ce_pool),
                "hand_pools": list(hand_pool_data.keys()),
                "weights": weights,
                "tie_breakers": tie_breakers,
            },
            "run_outputs": {
                "summary_csv": str(summary_csv),
                "summary_json": str(output_dir / "top3_score_fusion_summary.json"),
                "best_candidates": str(best_path),
            },
            "best_variant": best,
        },
    )

    print(f"wrote: {summary_csv}")
    print(f"wrote: {output_dir / 'top3_score_fusion_summary.json'}")
    print(f"wrote: {best_path}")


if __name__ == "__main__":
    main()
