from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from collections import defaultdict
from typing import Optional

from round18.tools.common import (
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


@dataclass(frozen=True)
class FusionConfig:
    variant_id: str
    label: str
    mode: str
    weights: dict[str, float]
    rrf_k: int
    status: str = "strict-candidate"


DEFAULT_SOURCE_PATHS = {
    "bm25": Path(
        "round18/outputs/o_sparse/o_s1_lexical_index_experiments/"
        "dev_full_bm25_dev_bm25_top500_candidates.json"
    ),
    "char": Path(
        "round18/outputs/o_sparse/o_s6_char_tfidf/"
        "dev_full_dev_o_s6_char_tfidf_tfidf_char_top500_candidates.json"
    ),
    "structured": Path(
        "round18/outputs/o_sparse/o_s2_structured/"
        "dev_full_dev_decomposed_candidates.json"
    ),
    "prf": Path("round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json"),
}
ALL_SOURCES = ("bm25", "char", "structured", "prf")
RRF_K_DEFAULT = 60

FIXED_VARIANTS: tuple[FusionConfig, ...] = (
    FusionConfig(
        variant_id="strict_rrf_equal_bm25_char",
        label="RRF(BM25+char equal)",
        mode="rrf",
        weights={"bm25": 1.0, "char": 1.0},
        rrf_k=RRF_K_DEFAULT,
        status="strict-candidate",
    ),
    FusionConfig(
        variant_id="strict_rrf_equal_bm25_char_structured_prf",
        label="RRF(BM25+char+structured+prf equal)",
        mode="rrf",
        weights={"bm25": 1.0, "char": 1.0, "structured": 1.0, "prf": 1.0},
        rrf_k=RRF_K_DEFAULT,
        status="strict-candidate",
    ),
    FusionConfig(
        variant_id="strict_rrf_char_heavy",
        label="RRF(char-heavy)",
        mode="rrf",
        weights={"bm25": 1.0, "char": 1.5, "structured": 1.0, "prf": 1.0},
        rrf_k=RRF_K_DEFAULT,
        status="strict-candidate",
    ),
    FusionConfig(
        variant_id="strict_rrf_bm25_heavy",
        label="RRF(bm25-heavy)",
        mode="rrf",
        weights={"bm25": 1.5, "char": 1.0, "structured": 1.0, "prf": 1.0},
        rrf_k=RRF_K_DEFAULT,
        status="strict-candidate",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-S7 plain leaf sparse fusion over independent leaf candidate pools."
        )
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/dev-claims.json"),
        help="Claims split input (dev-only in this worker).",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("data/evidence.json"),
        help="Evidence corpus file for scope checks.",
    )
    parser.add_argument(
        "--bm25-pool",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["bm25"],
        help="Current-run O-S1 BM25 leaf pool path.",
    )
    parser.add_argument(
        "--char-pool",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["char"],
        help="Current-run O-S6 char TF-IDF leaf pool path.",
    )
    parser.add_argument(
        "--structured-pool",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["structured"],
        help="Optional current-run O-S2 structured leaf pool path.",
    )
    parser.add_argument(
        "--prf-pool",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["prf"],
        help="Optional current-run O-S3 PRF leaf pool path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s7_plain_leaf_fusion"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-id",
        default="o_s7_plain_leaf_fusion",
        help="Run identifier used in output filenames.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s7_plain_leaf_fusion/run_manifest.json"),
        help="Manifest output path.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Top-k candidates to write for each claim.",
    )
    parser.add_argument(
        "--eval-k",
        default="100,500",
        help="Comma-separated recall-k values.",
    )
    parser.add_argument(
        "--diagnostic-rrf-k-grid",
        default="40,60,80",
        help="RRF-k values used for diagnostic grid.",
    )
    parser.add_argument(
        "--diagnostic-weight-grid",
        default="0.8,1.0,1.3",
        help="Source-weight grid used for diagnostic grid.",
    )
    parser.add_argument(
        "--enable-diagnostic",
        action="store_true",
        default=True,
        help="Include dev-only diagnostic candidate search (marked diagnostic-only).",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s7_plain_leaf_fusion/run_record.json"),
        help="Execution record JSON path.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Reproducibility marker.",
    )
    parser.add_argument(
        "--stage",
        default="o_s7_plain_leaf_fusion",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("eval-k must contain at least one integer.")
    return sorted(set(values))


def parse_float_list(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("weight grid must contain at least one float.")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("rrf-k grid must contain at least one integer.")
    return values


def _infer_split_name(path: Path) -> str:
    if str(path) == "data/train-claims.json":
        return "train"
    if str(path) == "data/dev-claims.json":
        return "dev"
    if str(path) == "data/test-claims-unlabelled.json":
        return "test"
    return "claims"


def _enforce_allowed_inputs(
    claims: Path,
    evidence: Path,
    bm25_pool: Path,
    char_pool: Path,
    structured_pool: Path,
    prf_pool: Path,
) -> None:
    if claims != Path("data/dev-claims.json"):
        raise SystemExit(
            "O-S7 is scoped to dev only: claims must be data/dev-claims.json."
        )
    if evidence != Path("data/evidence.json"):
        raise SystemExit("O-S7 requires data/evidence.json.")
    if bm25_pool != DEFAULT_SOURCE_PATHS["bm25"]:
        raise SystemExit(
            "O-S7 reads current-run O-S1 only: "
            f"{DEFAULT_SOURCE_PATHS['bm25']}."
        )
    if char_pool != DEFAULT_SOURCE_PATHS["char"]:
        raise SystemExit(
            "O-S7 reads current-run O-S6 only: "
            f"{DEFAULT_SOURCE_PATHS['char']}."
        )
    if structured_pool != DEFAULT_SOURCE_PATHS["structured"]:
        raise SystemExit(
            "O-S7 reads current-run O-S2 only: "
            f"{DEFAULT_SOURCE_PATHS['structured']}."
        )
    if prf_pool != DEFAULT_SOURCE_PATHS["prf"]:
        raise SystemExit(
            "O-S7 reads current-run O-S3 only: "
            f"{DEFAULT_SOURCE_PATHS['prf']}."
        )


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def parse_candidate_pool(
    path: Path,
    source_name: str,
    candidate_k: int,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, int]]:
    if not path.exists():
        return {}, { "claims_seen": 0, "items_seen": 0, "claims_loaded": 0}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate pool is not a dict: {path}")

    pool: dict[str, dict[str, dict[str, float]]] = {}
    evidence_count = 0
    claims_seen = 0
    for claim_id, entries in payload.items():
        claims_seen += 1
        if not isinstance(entries, list):
            continue
        rows: list[tuple[int, str, float | None, int]] = []
        for idx, raw_entry in enumerate(entries):
            if not isinstance(raw_entry, dict):
                continue
            evidence_id = str(raw_entry.get("evidence_id", "")).strip()
            if not evidence_id:
                continue
            rank = _coerce_int(raw_entry.get("rank"))
            if rank <= 0:
                rank = idx + 1
            score = _coerce_float(raw_entry.get("score"))
            rows.append((rank, evidence_id, score, idx))
            if source_name in {"bm25", "char", "structured", "prf"}:
                evidence_count += 1

        rows.sort(key=lambda row: (row[0], row[3], row[1]))
        by_evidence: dict[str, tuple[int, float | None]] = {}
        for rank, evidence_id, score, _ in rows:
            if evidence_id in by_evidence:
                continue
            by_evidence[evidence_id] = (rank, score if score is not None else 0.0)
            if candidate_k > 0 and len(by_evidence) >= candidate_k:
                break
        if by_evidence:
            pool[claim_id] = {
                evidence_id: {"rank": rank, "score": score}
                for evidence_id, (rank, score) in by_evidence.items()
            }
    claims_loaded = len(pool)
    return pool, {
        "claims_seen": claims_seen,
        "items_seen": evidence_count,
        "claims_loaded": claims_loaded,
    }


def build_fused_candidates_for_claim(
    claim_id: str,
    pools: dict[str, dict[str, dict[str, dict[str, float]]]],
    candidate_k: int,
    config: FusionConfig,
    source_weights: dict[str, float],
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    trace: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_name in ALL_SOURCES:
        weight = float(source_weights.get(source_name, 0.0))
        if weight <= 0:
            continue
        source_pool = pools.get(source_name, {}).get(claim_id, {})
        for evidence_id, entry in source_pool.items():
            rank = int(entry.get("rank", 0))
            if rank <= 0:
                continue
            source_score = entry.get("score")
            if config.mode == "rrf":
                contribution = weight / (float(config.rrf_k) + float(rank))
            else:
                contribution = weight / (float(rank) + 1e-12)

            fused_scores[evidence_id] = fused_scores.get(evidence_id, 0.0) + contribution
            trace[evidence_id].append(
                {
                    "source": source_name,
                    "rank": rank,
                    "weight": weight,
                    "source_score": float(source_score) if isinstance(source_score, (int, float)) else None,
                    "contribution": float(contribution),
                }
            )

    ranked = sorted(
        ((evidence_id, score) for evidence_id, score in fused_scores.items()),
        key=lambda row: (-row[1], row[0]),
    )
    candidates: list[dict[str, Any]] = []
    for rank_idx, (evidence_id, score) in enumerate(ranked[:candidate_k], start=1):
        source_rows = sorted(trace[evidence_id], key=lambda row: row["source"])
        candidates.append(
            {
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "rank": rank_idx,
                "score": float(score),
                "sources": [row["source"] for row in source_rows],
                "source_count": len(source_rows),
                "source_breakdown": source_rows,
            }
        )
    return candidates


def build_fused_pool(
    claims: dict[str, Any],
    pools: dict[str, dict[str, dict[str, dict[str, float]]]],
    config: FusionConfig,
    candidate_k: int,
    source_weights: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    fused: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        fused[claim_id] = build_fused_candidates_for_claim(
            claim_id=claim_id,
            pools=pools,
            candidate_k=candidate_k,
            config=config,
            source_weights=source_weights,
        )
    return fused


def _effective_weights(
    requested_weights: dict[str, float],
    active_sources: set[str],
) -> dict[str, float]:
    effective: dict[str, float] = {}
    for source_name in requested_weights:
        if source_name in active_sources:
            weight = float(requested_weights[source_name])
            if weight > 0:
                effective[source_name] = weight
    return effective


def evaluate_recall_at_k(
    claims: dict[str, Any],
    pool: dict[str, list[dict[str, Any]]],
    eval_ks: list[int],
) -> dict[str, Any]:
    ks = sorted(set(int(k) for k in eval_ks if int(k) > 0))
    per_k = {k: {"tp": 0, "gold": 0, "claims_hit": 0, "recalls": []} for k in ks}
    label_totals: dict[str, int] = {}
    label_hits: dict[int, dict[str, int]] = {k: {} for k in ks}

    claims_with_evidence = 0
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claims_with_evidence += 1
        ranked_ids = [candidate["evidence_id"] for candidate in pool.get(claim_id, [])]
        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0) + 1

        for k in ks:
            predicted = set(ranked_ids[:k])
            tp = len(gold.intersection(predicted))
            per_k[k]["tp"] += tp
            per_k[k]["gold"] += len(gold)
            recall = tp / len(gold) if gold else 0.0
            per_k[k]["recalls"].append(recall)
            if tp > 0:
                per_k[k]["claims_hit"] += 1
                label_hits[k][label] = label_hits[k].get(label, 0) + 1

    rows: dict[str, Any] = {
        "claims_with_evidence": claims_with_evidence,
        "evaluated_ks": ks,
        "avg_candidate_count": 0.0,
        "union_candidates": 0,
    }
    total_candidates = sum(len(values) for values in pool.values())
    rows["avg_candidate_count"] = float(total_candidates / claims_with_evidence) if claims_with_evidence else 0.0
    rows["union_candidates"] = len({candidate["evidence_id"] for candidates in pool.values() for candidate in candidates})

    for k in ks:
        recall_rows = per_k[k]["recalls"]
        rows[f"macro_recall_at_{k}"] = float(sum(recall_rows) / len(recall_rows)) if recall_rows else 0.0
        total_tp = float(per_k[k]["tp"])
        total_gold = float(per_k[k]["gold"])
        rows[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        rows[f"hit_any_at_{k}"] = per_k[k]["claims_hit"] / claims_with_evidence if claims_with_evidence else 0.0
        for label, total in label_totals.items():
            if total <= 0:
                continue
            rows[f"{label.lower()}_hit_any_at_{k}"] = label_hits[k].get(label, 0) / total

    return rows


def build_comparison_row(
    variant_id: str,
    config: FusionConfig,
    metrics: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "variant": variant_id,
        "label": config.label,
        "status": status,
        "mode": config.mode,
        "rrf_k": config.rrf_k,
        "bm25_weight": config.weights.get("bm25", 0.0),
        "char_weight": config.weights.get("char", 0.0),
        "structured_weight": config.weights.get("structured", 0.0),
        "prf_weight": config.weights.get("prf", 0.0),
        "macro_recall@100": metrics.get("macro_recall_at_100", 0.0),
        "macro_recall@500": metrics.get("macro_recall_at_500", 0.0),
        "micro_recall@100": metrics.get("micro_recall_at_100", 0.0),
        "micro_recall@500": metrics.get("micro_recall_at_500", 0.0),
        "hit_any@100": metrics.get("hit_any_at_100", 0.0),
        "hit_any@500": metrics.get("hit_any_at_500", 0.0),
    }


def _score_row_for_selection(metrics: dict[str, Any], config: FusionConfig) -> tuple:
    return (
        metrics.get("micro_recall_at_500", 0.0),
        metrics.get("macro_recall_at_500", 0.0),
        metrics.get("hit_any_at_500", 0.0),
        metrics.get("micro_recall_at_100", 0.0),
        metrics.get("macro_recall_at_100", 0.0),
        -config.rrf_k,
    )


def select_diagnostic_winner(
    scored_rows: list[tuple],
) -> tuple[Any, dict[str, Any], Optional[dict[str, Any]]]:
    scored_rows = [
        ( _score_row_for_selection(metrics, config), config, metrics, pool )
        for config, metrics, pool in scored_rows
    ]
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    if not scored_rows:
        return (
            FusionConfig(
                variant_id="diagnostic_none",
                label="No diagnostic candidate",
                mode="rrf",
                weights={"bm25": 1.0, "char": 1.0},
                rrf_k=RRF_K_DEFAULT,
                status="diagnostic-only",
            ),
            {},
            None,
        )
    _, config, metrics, pool = scored_rows[0]
    return config, metrics, pool


def build_data_flow(
    split_name: str,
    claims: dict[str, Any],
    evidence_count: int,
    args: argparse.Namespace,
    comparison_path: Path,
    candidate_count: int,
    metrics_path: Path,
    data_flow_path: Path,
    eval_ks: list[int],
    fixed_variants: list[str],
    diagnostic_info: dict[str, Any],
    source_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "pipeline": "O-S7 plain leaf-tree RRF fusion (parallel leafs only; no nested fusion input).",
        "split": split_name,
        "counts": {
            "claims_in_file": len(claims),
            "evidence_in_file": evidence_count,
        },
        "inputs": {
            "claims": str(args.claims),
            "evidence": str(args.evidence),
            "bm25_pool": str(args.bm25_pool),
            "char_pool": str(args.char_pool),
            "structured_pool": str(args.structured_pool),
            "prf_pool": str(args.prf_pool),
        },
        "source_metadata": source_meta,
        "outputs": {
            "comparison_table": str(comparison_path),
            "candidate_pool_count": candidate_count,
            "candidate_pool_type": "leaf_plain",
            "all_metrics": str(metrics_path),
            "data_flow_report": str(data_flow_path),
            "run_record": str(args.record_path),
            "manifest": str(args.manifest),
        },
        "processing": {
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "fusion_mode": "rrf",
            "fixed_variants": fixed_variants,
            "diagnostic": diagnostic_info,
        },
        "notes": [
            "All sources are treated as leaf inputs and fused directly.",
            "Diagnostic variants are marked diagnostic-only and do not replace strict variants.",
        ],
    }


def build_source_status(
    source_paths: dict[str, Path],
    source_pools: dict[str, dict[str, dict[str, dict[str, float]]]],
    source_meta: dict[str, dict[str, int]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    source_enabled: dict[str, bool] = {}
    source_status: dict[str, Any] = {}
    for source_name in ALL_SOURCES:
        enabled = bool(source_pools.get(source_name))
        source_enabled[source_name] = enabled
        source_status[source_name] = {
            "enabled": enabled,
            "path": str(source_paths[source_name]),
            "claims_seen": source_meta[source_name]["claims_seen"],
            "claims_loaded": source_meta[source_name]["claims_loaded"],
            "pool_items_seen": source_meta[source_name]["items_seen"],
        }
    return source_enabled, source_status


def build_record_input_files(
    claims: Path,
    evidence: Path,
    source_paths: dict[str, Path],
    source_enabled: dict[str, bool],
) -> list[dict[str, Any]]:
    split_name = _infer_split_name(claims)
    records = [
        {
            "path": str(claims),
            "sha256": sha256_file(claims),
            "split": split_name,
            "labels_used": split_name in {"train", "dev"},
        },
        {
            "path": str(evidence),
            "sha256": sha256_file(evidence),
            "split": "evidence",
            "labels_used": False,
        },
    ]
    for source_name in ("bm25", "char", "structured", "prf"):
        if source_enabled[source_name]:
            path = source_paths[source_name]
            records.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "split": "current_run_artifact",
                    "labels_used": False,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_command = "python round18/experiments/o_sparse/o_s7_plain_leaf_fusion/run_o_s7_plain_leaf_fusion.py"

    _enforce_allowed_inputs(
        claims=args.claims,
        evidence=args.evidence,
        bm25_pool=args.bm25_pool,
        char_pool=args.char_pool,
        structured_pool=args.structured_pool,
        prf_pool=args.prf_pool,
    )

    eval_ks = parse_k_list(args.eval_k)
    max_eval_k = max(eval_ks)
    if args.candidate_k < max_eval_k:
        args.candidate_k = max_eval_k

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    if not isinstance(claims, dict):
        raise SystemExit("Claims input must be a dict keyed by claim id.")
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence input must be a dict keyed by evidence id.")

    if not args.bm25_pool.exists():
        raise SystemExit(f"Missing BM25 source pool: {args.bm25_pool}")
    if not args.char_pool.exists():
        raise SystemExit(f"Missing char source pool: {args.char_pool}")

    source_paths = {
        "bm25": args.bm25_pool,
        "char": args.char_pool,
        "structured": args.structured_pool,
        "prf": args.prf_pool,
    }

    bm25_pool, bm25_meta = parse_candidate_pool(args.bm25_pool, "bm25", args.candidate_k)
    char_pool, char_meta = parse_candidate_pool(args.char_pool, "char", args.candidate_k)
    if source_paths["structured"] != DEFAULT_SOURCE_PATHS["structured"] or not source_paths["structured"].exists():
        structured_pool = {}
        structured_meta = {"claims_seen": 0, "claims_loaded": 0, "items_seen": 0}
    else:
        structured_pool, structured_meta = parse_candidate_pool(source_paths["structured"], "structured", args.candidate_k)
    if source_paths["prf"] != DEFAULT_SOURCE_PATHS["prf"] or not source_paths["prf"].exists():
        prf_pool = {}
        prf_meta = {"claims_seen": 0, "claims_loaded": 0, "items_seen": 0}
    else:
        prf_pool, prf_meta = parse_candidate_pool(source_paths["prf"], "prf", args.candidate_k)

    source_pools: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "bm25": bm25_pool,
        "char": char_pool,
        "structured": structured_pool,
        "prf": prf_pool,
    }
    source_meta: dict[str, dict[str, int]] = {
        "bm25": bm25_meta,
        "char": char_meta,
        "structured": structured_meta,
        "prf": prf_meta,
    }
    source_enabled, source_status = build_source_status(source_paths, source_pools, source_meta)
    active_sources = [name for name, enabled in source_enabled.items() if enabled]
    if not active_sources:
        raise SystemExit("No source pools could be loaded.")

    split_name = _infer_split_name(args.claims)
    split_file_suffix = "dev_full_dev" if split_name == "dev" else f"{split_name}_full"

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.claims),
            str(args.evidence),
            str(args.bm25_pool),
            str(args.char_pool),
            str(args.structured_pool),
            str(args.prf_pool),
            str(args.output_dir),
            str(args.manifest),
        ]
    )
    if forbidden_hits and any(token in {"outputs/round", "teacher", "checkpoint"} for token in forbidden_hits):
        raise SystemExit(f"Forbidden token(s) detected: {forbidden_hits}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[str] = []
    all_variant_results: list[dict[str, Any]] = []
    candidate_files_written = 0

    source_claim_coverage = {name: len(pool) for name, pool in source_pools.items()}

    for config in FIXED_VARIANTS:
        active_weights = _effective_weights(config.weights, set(active_sources))
        if not active_weights:
            continue
        fused_pool = build_fused_pool(
            claims=claims,
            pools=source_pools,
            config=config,
            candidate_k=args.candidate_k,
            source_weights=active_weights,
        )
        metrics = evaluate_recall_at_k(claims=claims, pool=fused_pool, eval_ks=eval_ks)
        metrics.update(
            {
                "fusion_variant": config.variant_id,
                "fusion_label": config.label,
                "fusion_mode": config.mode,
                "fusion_status": config.status,
                "fusion_weights": active_weights,
                "rrf_k": config.rrf_k,
                "candidate_top_k": args.candidate_k,
                "claims_input_count": len(claims),
                "claims_with_evidence": metrics.get("claims_with_evidence", 0),
                "source_claim_coverage": source_claim_coverage,
                "source_items_available": {
                    source_name: meta["items_seen"] for source_name, meta in source_meta.items()
                },
                "source_claims_loaded": {
                    source_name: meta["claims_loaded"] for source_name, meta in source_meta.items()
                },
                "evidence_items_available": len(evidence),
                "split": split_name,
                "smoke": False,
            }
        )

        candidate_path = (
            args.output_dir
            / f"{split_file_suffix}_{args.run_id}_{config.variant_id}_top{args.candidate_k}_candidates.json"
        )
        metrics_path = (
            args.output_dir
            / f"{split_file_suffix}_{args.run_id}_{config.variant_id}_metrics.json"
        )
        write_json(candidate_path, fused_pool)
        write_json(metrics_path, metrics)
        all_variant_results.append(
            {
                "config": config,
                "metrics": metrics,
                "pool": fused_pool,
                "paths": {"candidate": str(candidate_path), "metrics": str(metrics_path)},
                "status": config.status,
            }
        )
        output_files.extend([str(candidate_path), str(metrics_path)])
        candidate_files_written += 1

    diagnostic_best = None
    if args.enable_diagnostic:
        diagnostic_rows = []
        rrf_ks = parse_int_list(args.diagnostic_rrf_k_grid)
        weight_grid = parse_float_list(args.diagnostic_weight_grid)
        active_source_list = [name for name in ALL_SOURCES if source_enabled[name]]
        for weights_combo in product(weight_grid, repeat=len(active_source_list)):
            variant_weights = dict(zip(active_source_list, weights_combo))
            if not any(weight > 0 for weight in variant_weights.values()):
                continue
            for rrf_k in rrf_ks:
                config = FusionConfig(
                    variant_id="diagnostic_rrf_" + "_".join(
                        f"{name}{weight:g}" for name, weight in variant_weights.items()
                    ) + f"_k{rrf_k}",
                    label=f"Diagnostic RRF(k={rrf_k}, " +
                          "w=[" + ",".join(f"{variant_weights[name]}" for name in active_source_list) + "])",
                    mode="rrf",
                    weights=variant_weights,
                    rrf_k=rrf_k,
                    status="diagnostic-only",
                )
                fused_pool = build_fused_pool(
                    claims=claims,
                    pools=source_pools,
                    config=config,
                    candidate_k=args.candidate_k,
                    source_weights=variant_weights,
                )
                metrics = evaluate_recall_at_k(claims=claims, pool=fused_pool, eval_ks=eval_ks)
                diagnostic_rows.append((config, metrics, fused_pool))

        if diagnostic_rows:
            diagnostic_best = select_diagnostic_winner(diagnostic_rows)
            diag_config, diag_metrics, diag_pool = diagnostic_best
            if diag_pool is None:
                diagnostic_best = None
            else:
                diag_metrics.update(
                    {
                        "fusion_variant": diag_config.variant_id,
                        "fusion_label": diag_config.label,
                        "fusion_mode": diag_config.mode,
                        "fusion_status": "diagnostic-only",
                        "fusion_weights": diag_config.weights,
                        "rrf_k": diag_config.rrf_k,
                        "candidate_top_k": args.candidate_k,
                        "claims_input_count": len(claims),
                        "source_items_available": {
                            source_name: meta["items_seen"] for source_name, meta in source_meta.items()
                        },
                        "evidence_items_available": len(evidence),
                        "diagnostic_grid": {
                            "active_sources": active_source_list,
                            "rrf_k_grid": parse_int_list(args.diagnostic_rrf_k_grid),
                            "weight_grid": parse_float_list(args.diagnostic_weight_grid),
                        },
                        "split": split_name,
                        "smoke": False,
                    }
                )
                diag_candidate_path = (
                    args.output_dir
                    / f"{split_file_suffix}_{args.run_id}_diagnostic_best_top{args.candidate_k}_candidates.json"
                )
                diag_metrics_path = (
                    args.output_dir
                    / f"{split_file_suffix}_{args.run_id}_diagnostic_best_metrics.json"
                )
                diag_selection_path = (
                    args.output_dir
                    / f"{split_file_suffix}_{args.run_id}_diagnostic_selection.json"
                )
                write_json(
                    diag_candidate_path,
                    diag_pool,
                )
                write_json(diag_metrics_path, diag_metrics)
                write_json(
                    diag_selection_path,
                    {
                        "diagnostic_mode": "dev-only-selection",
                        "selection_status": "diagnostic-only",
                        "selection_rule": "argmax over [micro@500, macro@500, hit_any@500, micro@100, macro@100, -rrf_k]",
                        "best_variant": {
                            "variant_id": diag_config.variant_id,
                            "label": diag_config.label,
                            "rrf_k": diag_config.rrf_k,
                            "weights": diag_config.weights,
                            "candidate_file": str(diag_candidate_path),
                            "metric_file": str(diag_metrics_path),
                            "status": "diagnostic-only",
                        },
                        "weight_grid": parse_float_list(args.diagnostic_weight_grid),
                        "rrf_k_grid": parse_int_list(args.diagnostic_rrf_k_grid),
                        "active_sources": active_source_list,
                        "diagnostic_candidates_evaluated": len(diagnostic_rows),
                    },
                )
                output_files.extend([str(diag_candidate_path), str(diag_metrics_path), str(diag_selection_path)])
                all_variant_results.append(
                    {
                        "config": diag_config,
                        "metrics": diag_metrics,
                        "pool": diag_pool,
                        "paths": {
                            "candidate": str(diag_candidate_path),
                            "metrics": str(diag_metrics_path),
                            "selection": str(diag_selection_path),
                        },
                        "status": "diagnostic-only",
                    }
                )

    comparison_rows = [
        build_comparison_row(
            result["config"].variant_id,
            result["config"],
            result["metrics"],
            result["status"],
        )
        for result in all_variant_results
    ]
    comparison_rows.sort(
        key=lambda row: (
            row["micro_recall@500"],
            row["macro_recall@500"],
            row["hit_any@500"],
            row["micro_recall@100"],
            row["macro_recall@100"],
            row["status"] == "strict-candidate",
        ),
        reverse=True,
    )

    comparison_csv_path = (
        args.output_dir / f"{split_file_suffix}_{args.run_id}_comparison_table.csv"
    )
    comparison_json_path = (
        args.output_dir / f"{split_file_suffix}_{args.run_id}_comparison_table.json"
    )
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(comparison_rows[0].keys())
            if comparison_rows
            else [
                "variant",
                "label",
                "status",
                "mode",
                "rrf_k",
                "bm25_weight",
                "char_weight",
                "structured_weight",
                "prf_weight",
                "macro_recall@100",
                "macro_recall@500",
                "micro_recall@100",
                "micro_recall@500",
                "hit_any@100",
                "hit_any@500",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    write_json(comparison_json_path, comparison_rows)
    output_files.extend([str(comparison_csv_path), str(comparison_json_path)])

    all_metrics = {
        "split": split_name,
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "fixed_variants": [row["variant"] for row in comparison_rows if row["status"] == "strict-candidate"],
        "diagnostic_variant": diagnostic_best[0].variant_id if diagnostic_best else "none",
        "diagnostic_enabled": args.enable_diagnostic,
        "diagnostic_candidates_evaluated": (
            len(diagnostic_rows) if args.enable_diagnostic else 0
        ),
        "comparison_rows": comparison_rows,
        "command": base_command,
    }
    all_metrics_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_metrics.json"
    write_json(all_metrics_path, all_metrics)
    output_files.append(str(all_metrics_path))

    # Data-flow report
    diagnostic_payload = {
        "enabled": args.enable_diagnostic,
        "best_variant": diagnostic_best[0].variant_id if diagnostic_best else None,
        "candidates_evaluated": len(diagnostic_rows) if args.enable_diagnostic and "diagnostic_rows" in locals() else 0,
        "rrf_k_grid": parse_int_list(args.diagnostic_rrf_k_grid),
        "weight_grid": parse_float_list(args.diagnostic_weight_grid),
        "active_sources": active_sources,
    }
    data_flow_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_data_flow_report.json"
    data_flow = build_data_flow(
        split_name=split_name,
        claims=claims,
        evidence_count=len(evidence),
        args=args,
        comparison_path=comparison_csv_path,
        candidate_count=candidate_files_written,
        metrics_path=all_metrics_path,
        data_flow_path=data_flow_path,
        eval_ks=eval_ks,
        fixed_variants=[cfg.variant_id for cfg in FIXED_VARIANTS],
        diagnostic_info=diagnostic_payload,
        source_meta=source_status,
    )
    write_json(data_flow_path, data_flow)
    output_files.append(str(data_flow_path))

    command = (
        f"{base_command} --claims {args.claims} --evidence {args.evidence} "
        f"--bm25-pool {args.bm25_pool} --char-pool {args.char_pool} "
        f"--structured-pool {args.structured_pool} --prf-pool {args.prf_pool} "
        f"--candidate-k {args.candidate_k} --eval-k {','.join(map(str, eval_ks))} "
        f"--run-id {args.run_id} --output-dir {args.output_dir} --stage {args.stage} "
        f"--diagnostic-rrf-k-grid {','.join(map(str, parse_int_list(args.diagnostic_rrf_k_grid)))} "
        f"--diagnostic-weight-grid {','.join(map(str, parse_float_list(args.diagnostic_weight_grid)))}"
    )

    input_records = build_record_input_files(
        claims=args.claims,
        evidence=args.evidence,
        source_paths=source_paths,
        source_enabled=source_enabled,
    )

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate",
        mode="STRICT",
        stage=args.stage,
        command=command,
        working_directory=Path.cwd(),
        config_path=str(args.manifest),
        config_hash="",
        random_seed=args.random_seed,
        cv_seed=None,
    )
    manifest["input_files"] = input_records
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan against forbidden strict tokens passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = output_files + [str(args.manifest), str(args.record_path)]
    manifest["metrics"] = {
        "all_metrics_file": str(all_metrics_path),
        "comparison_table": str(comparison_csv_path),
        "comparison_json": str(comparison_json_path),
        "diagnostic_mode": "dev-only",
        "diagnostic_selection_status": "diagnostic-only",
        "comparison_rows": comparison_rows,
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Loaded current-run leaf pools from BM25, char TF-IDF, and optional structured/PRF pools, "
        "fused them as parallel leaves using RRF-only scoring, wrote plain per-claim candidate pools "
        "for fixed and diagnostic variants, and produced recall metrics."
    )
    manifest["split_isolation_summary"] = (
        "Restricted to dev split and evidence corpus only; labels are read from dev only."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low"
    manifest["notes"] = (
        "Diagnostic search is explicitly marked diagnostic-only and is not promoted to strict status."
    )
    write_json(args.manifest, manifest)
    output_files.append(str(args.manifest))

    record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT",
        "status": "strict-candidate",
        "split": split_name,
        "command": command,
        "command_args": {key: str(value) for key, value in vars(args).items()},
        "claims_count": len(claims),
        "evidence_count": len(evidence),
        "strict_variant_count": len([v for v in all_variant_results if v["status"] == "strict-candidate"]),
        "diagnostic_enabled": args.enable_diagnostic,
        "diagnostic_selected": diagnostic_best[0].variant_id if diagnostic_best else None,
        "diagnostic_candidates_evaluated": (
            len(diagnostic_rows) if args.enable_diagnostic and "diagnostic_rows" in locals() else 0
        ),
        "forbidden_hits": forbidden_hits,
        "files_written": output_files + [str(args.record_path)],
    }
    write_json(args.record_path, record)
    output_files.append(str(args.record_path))

    print(f"Wrote comparison table: {comparison_csv_path}")
    for row in comparison_rows:
        print(
            f"{row['variant']} "
            f"mR@100={row['macro_recall@100']:.4f} "
            f"mR@500={row['macro_recall@500']:.4f} "
            f"miR@500={row['micro_recall@500']:.4f} "
            f"hit@500={row['hit_any@500']:.4f} "
            f"status={row['status']}"
        )
    if diagnostic_best:
        print(f"Diagnostic winner: {diagnostic_best[0].variant_id} (diagnostic-only)")


if __name__ == "__main__":
    main()
