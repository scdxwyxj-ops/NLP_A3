from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    rrf_k: int | None = None
    status: str = "strict-candidate"


DEFAULT_SOURCE_PATHS = {
    "bm25": Path("round18/outputs/o_sparse/o_s1_lexical_index_experiments/dev_full_bm25_dev_bm25_top500_candidates.json"),
    "structured": Path("round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposed_candidates.json"),
    "prf": Path("round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json"),
}

STRICT_SOURCES = ("bm25", "structured", "prf")
STRICT_VARIANTS: tuple[FusionConfig, ...] = (
    FusionConfig(
        variant_id="strict_rrf_equal",
        label="RRF(equal weights)",
        mode="rrf",
        weights={"bm25": 1.0, "structured": 1.0, "prf": 1.0},
        rrf_k=60,
        status="strict-candidate",
    ),
    FusionConfig(
        variant_id="strict_rrf_bm25_heavy",
        label="RRF(BM25 up-weighted)",
        mode="rrf",
        weights={"bm25": 1.5, "structured": 1.0, "prf": 1.0},
        rrf_k=60,
        status="strict-candidate",
    ),
    FusionConfig(
        variant_id="strict_rrf_prf_heavy",
        label="RRF(PRF up-weighted)",
        mode="rrf",
        weights={"bm25": 1.0, "structured": 1.0, "prf": 1.5},
        rrf_k=60,
        status="strict-candidate",
    ),
    FusionConfig(
        variant_id="strict_combsum_inv_rank",
        label="CombSUM(inv-rank)",
        mode="combsum_inv_rank",
        weights={"bm25": 1.0, "structured": 1.0, "prf": 1.0},
        status="strict-candidate",
    ),
    FusionConfig(
        variant_id="strict_combsum_score_norm",
        label="CombSUM(score norm)",
        mode="combsum_score_norm",
        weights={"bm25": 1.0, "structured": 1.0, "prf": 1.0},
        status="strict-candidate",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-S4 sparse fusion over O-S1/O-S2/O-S3 candidate pools."
        )
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/dev-claims.json"),
        help="Claims split file (dev-only for this worker scope).",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("data/evidence.json"),
        help="Evidence corpus file for raw-data isolation checks.",
    )
    parser.add_argument(
        "--bm25-pool",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["bm25"],
        help="O-S1 BM25 candidate pool path.",
    )
    parser.add_argument(
        "--structured-pool",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["structured"],
        help="O-S2 structured decomposition pool path.",
    )
    parser.add_argument(
        "--prf-pool",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["prf"],
        help="O-S3 PRF candidate pool path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s4_fusion"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-id",
        default="o_s4_fusion",
        help="Run identifier used in output filenames.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s4_fusion/run_manifest.json"),
        help="Manifest output path.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Top-k fused candidate list for each claim.",
    )
    parser.add_argument(
        "--eval-k",
        default="100,500",
        help="Comma-separated recall-k values.",
    )
    parser.add_argument(
        "--diagnostic-rrf-k-grid",
        default="40,60,80",
        help="Comma-separated RRF denominators for dev diagnostic search.",
    )
    parser.add_argument(
        "--diagnostic-weight-grid",
        default="0.8,1.0,1.3",
        help="Comma-separated source weights for dev diagnostic search.",
    )
    parser.add_argument(
        "--enable-diagnostic",
        action="store_true",
        default=True,
        help="Include a dev diagnostics run to pick best-fused variant (diagnostic-only).",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s4_fusion/run_record.json"),
        help="Run record output path.",
    )
    parser.add_argument("--random-seed", type=int, default=1337, help="Reproducibility marker.")
    parser.add_argument(
        "--stage",
        default="o_s4_fusion",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("eval-k must include at least one integer.")
    return sorted(set(values))


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("weight grid must include at least one float.")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("rrf-k grid must include at least one integer.")
    return values


def _infer_split_name(path: Path) -> str:
    if str(path) == "data/train-claims.json":
        return "train"
    if str(path) == "data/dev-claims.json":
        return "dev"
    if str(path) == "data/test-claims-unlabelled.json":
        return "test"
    return "claims"


def _enforce_allowed_inputs(claims: Path, evidence: Path, bm25_pool: Path, structured_pool: Path, prf_pool: Path) -> None:
    if claims != Path("data/dev-claims.json"):
        raise SystemExit(
            "O-S4 in this scope is defined for dev split only: "
            "claims must be data/dev-claims.json."
        )
    if evidence != Path("data/evidence.json"):
        raise SystemExit("O-S4 requires data/evidence.json for scope compliance.")
    if bm25_pool != DEFAULT_SOURCE_PATHS["bm25"]:
        raise SystemExit(
            "Only the current-run O-S1 candidate pool is allowed for O-S4: "
            f"{DEFAULT_SOURCE_PATHS['bm25']}."
        )
    if structured_pool != DEFAULT_SOURCE_PATHS["structured"]:
        raise SystemExit(
            "Only the current-run O-S2 candidate pool is allowed for O-S4: "
            f"{DEFAULT_SOURCE_PATHS['structured']}."
        )
    if prf_pool != DEFAULT_SOURCE_PATHS["prf"]:
        raise SystemExit(
            "Only the current-run O-S3 candidate pool is allowed for O-S4: "
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


def parse_candidate_pool(path: Path, source_name: str, candidate_k: int) -> tuple[dict[str, dict[str, dict[str, float]]], int]:
    if not path.exists():
        raise SystemExit(f"Missing candidate pool: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate pool is not a dict: {path}")

    pool: dict[str, dict[str, dict[str, float]]] = {}
    for claim_id, entries in payload.items():
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
        rows.sort(key=lambda row: (row[0], row[3], row[1]))
        by_evidence: dict[str, tuple[int, float | None]] = {}
        for rank, evidence_id, score, _ in rows:
            if evidence_id in by_evidence:
                continue
            by_evidence[evidence_id] = (rank, score)
            if len(by_evidence) >= candidate_k > 0:
                break
        if not by_evidence:
            continue
        pool[claim_id] = {
            evidence_id: {"rank": rank, "score": score if score is not None else 0.0}
            for evidence_id, (rank, score) in by_evidence.items()
        }
    return pool, len(payload)


def build_score_normalization(
    pools: dict[str, dict[str, dict[str, dict[str, float]]]],
    source: str,
    candidate_k: int,
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for claim_id, candidates in pools.get(source, {}).items():
        scores = {
            evidence_id: entry["score"]
            for evidence_id, entry in candidates.items()
            if isinstance(entry.get("score"), (int, float))
        }
        if not scores:
            normalized[claim_id] = {}
            continue
        values = list(scores.values())
        score_min = min(values)
        score_max = max(values)
        denom = score_max - score_min
        if denom == 0:
            normalized[claim_id] = {eid: 1.0 for eid in scores}
        else:
            normalized[claim_id] = {
                eid: (value - score_min) / denom
                for eid, value in scores.items()
            }
    for claim_id in pools[source]:
        normalized.setdefault(claim_id, {})
    return normalized


def fuse_candidates_for_claim(
    claim_id: str,
    pools: dict[str, dict[str, dict[str, dict[str, float]]]],
    score_norms: dict[str, dict[str, dict[str, float]]],
    config: FusionConfig,
    candidate_k: int,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    source_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_name in STRICT_SOURCES:
        pool = pools.get(source_name, {}).get(claim_id, {})
        if not pool:
            continue
        weight = float(config.weights.get(source_name, 1.0))
        if weight <= 0:
            continue

        norm_lookup = score_norms.get(source_name, {}).get(claim_id, {})

        for evidence_id, entry in pool.items():
            rank = int(entry.get("rank", 0) or 0)
            if rank <= 0:
                continue
            raw_score = entry.get("score")
            if config.mode == "rrf":
                denom = float(config.rrf_k or 60) + float(rank)
                contribution = weight / denom
            elif config.mode == "combsum_inv_rank":
                contribution = weight / float(rank)
            elif config.mode == "combsum_score_norm":
                if isinstance(raw_score, (int, float)) and score_norms and evidence_id in norm_lookup:
                    contrib_score = float(norm_lookup[evidence_id])
                else:
                    contrib_score = 1.0 / float(rank)
                contribution = weight * contrib_score
            else:
                denom = float(config.rrf_k or 60) + float(rank)
                contribution = weight / denom

            fused_scores[evidence_id] = fused_scores.get(evidence_id, 0.0) + contribution
            source_trace[evidence_id].append(
                {
                    "source": source_name,
                    "rank": rank,
                    "weight": weight,
                    "source_score": float(raw_score) if isinstance(raw_score, (int, float)) else None,
                    "contribution": float(contribution),
                }
            )

    ranked = sorted(
        ((eid, score) for eid, score in fused_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )

    candidates: list[dict[str, Any]] = []
    for rank_idx, (evidence_id, score) in enumerate(ranked[:candidate_k], start=1):
        trace_entries = sorted(
            source_trace[evidence_id],
            key=lambda row: row["source"],
        )
        candidates.append(
            {
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "rank": rank_idx,
                "score": float(score),
                "source_count": len(trace_entries),
                "sources": [entry["source"] for entry in trace_entries],
                "source_breakdown": trace_entries,
                "fusion_mode": config.mode,
            }
        )
    return candidates


def build_fused_pool(
    claims: dict[str, Any],
    pools: dict[str, dict[str, dict[str, dict[str, float]]]],
    config: FusionConfig,
    candidate_k: int,
    score_norms: dict[str, dict[str, dict[str, float]]],
) -> dict[str, list[dict[str, Any]]]:
    fused: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        fused[claim_id] = fuse_candidates_for_claim(
            claim_id=claim_id,
            pools=pools,
            score_norms=score_norms,
            config=config,
            candidate_k=candidate_k,
        )
    return fused


def evaluate_recall_at_k(
    claims: dict[str, Any],
    pool: dict[str, list[dict[str, Any]]],
    eval_ks: list[int],
) -> dict[str, Any]:
    ks = sorted(set(int(k) for k in eval_ks if int(k) > 0))
    rows: dict[str, Any] = {
        "claims_with_evidence": 0,
        "evaluated_ks": ks,
        "avg_candidate_count": 0.0,
        "union_candidates": 0,
    }
    if not claims:
        return rows

    per_k_hits = {k: {"tp": 0, "gold": 0, "claims_hit": 0, "recalls": []} for k in ks}
    label_totals: dict[str, int] = {}
    label_hits: dict[int, dict[str, int]] = {k: {} for k in ks}
    claim_recall_count = 0

    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claim_recall_count += 1
        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0) + 1
        ranked = [candidate["evidence_id"] for candidate in pool.get(claim_id, [])]
        for k in ks:
            predicted = set(ranked[:k])
            tp = len(gold.intersection(predicted))
            per_k_hits[k]["tp"] += tp
            per_k_hits[k]["gold"] += len(gold)
            recall = tp / len(gold) if gold else 0.0
            per_k_hits[k]["recalls"].append(recall)
            if tp > 0:
                per_k_hits[k]["claims_hit"] += 1
                if label not in label_hits[k]:
                    label_hits[k][label] = 0
                label_hits[k][label] += 1

    total_candidates = sum(len(v) for v in pool.values())
    rows["claims_with_evidence"] = claim_recall_count
    rows["avg_candidate_count"] = float(total_candidates / claim_recall_count) if claim_recall_count else 0.0
    rows["union_candidates"] = len({cand["evidence_id"] for values in pool.values() for cand in values})

    for k in ks:
        recalls = per_k_hits[k]["recalls"]
        rows[f"macro_recall_at_{k}"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
        total_tp = float(per_k_hits[k]["tp"])
        total_gold = float(per_k_hits[k]["gold"])
        rows[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        rows[f"hit_any_at_{k}"] = per_k_hits[k]["claims_hit"] / claim_recall_count if claim_recall_count else 0.0
        for label in sorted(label_totals):
            if label_totals[label] == 0:
                continue
            hits = label_hits[k].get(label, 0)
            rows[f"{label.lower()}_hit_any_at_{k}"] = hits / label_totals[label]
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
        "rrf_k": config.rrf_k if config.mode.startswith("rrf") else None,
        "bm25_weight": config.weights.get("bm25", 0.0),
        "structured_weight": config.weights.get("structured", 0.0),
        "prf_weight": config.weights.get("prf", 0.0),
        "macro_recall@100": metrics.get("macro_recall_at_100", 0.0),
        "macro_recall@500": metrics.get("macro_recall_at_500", 0.0),
        "micro_recall@100": metrics.get("micro_recall_at_100", 0.0),
        "micro_recall@500": metrics.get("micro_recall_at_500", 0.0),
        "hit_any@100": metrics.get("hit_any_at_100", 0.0),
        "hit_any@500": metrics.get("hit_any_at_500", 0.0),
    }


def select_best_diagnostic_variant(
    rows: list[tuple],
) -> tuple[Any, Any, Any]:
    scored_rows = []
    for config, metrics, pool in rows:
        score = (
            metrics.get("micro_recall_at_500", 0.0),
            metrics.get("macro_recall_at_500", 0.0),
            metrics.get("hit_any_at_500", 0.0),
            metrics.get("micro_recall_at_100", 0.0),
            metrics.get("macro_recall_at_100", 0.0),
            -(config.rrf_k or 0),
        )
        scored_rows.append((score, config, metrics, pool))
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    if not scored_rows:
        return FusionConfig("diagnostic_none", "No diagnostic candidate", "rrf", {"bm25": 1.0, "structured": 1.0, "prf": 1.0}), {}, None
    return scored_rows[0][1], scored_rows[0][2], scored_rows[0][3]


def _build_input_records(args: argparse.Namespace, claims_path: Path, evidence_path: Path) -> list[dict[str, Any]]:
    split = _infer_split_name(claims_path)
    records = [
        {
            "path": str(claims_path),
            "sha256": sha256_file(claims_path),
            "split": split,
            "labels_used": split in {"train", "dev"},
        },
        {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "split": "evidence",
            "labels_used": False,
        },
        {
            "path": str(DEFAULT_SOURCE_PATHS["bm25"]),
            "sha256": sha256_file(DEFAULT_SOURCE_PATHS["bm25"]),
            "split": "current_run_artifact",
            "labels_used": False,
        },
        {
            "path": str(DEFAULT_SOURCE_PATHS["structured"]),
            "sha256": sha256_file(DEFAULT_SOURCE_PATHS["structured"]),
            "split": "current_run_artifact",
            "labels_used": False,
        },
        {
            "path": str(DEFAULT_SOURCE_PATHS["prf"]),
            "sha256": sha256_file(DEFAULT_SOURCE_PATHS["prf"]),
            "split": "current_run_artifact",
            "labels_used": False,
        },
    ]
    if str(args.bm25_pool) != str(DEFAULT_SOURCE_PATHS["bm25"]):
        records.append(
            {
                "path": str(args.bm25_pool),
                "sha256": sha256_file(args.bm25_pool),
                "split": "override_candidate",
                "labels_used": False,
            }
        )
    if str(args.structured_pool) != str(DEFAULT_SOURCE_PATHS["structured"]):
        records.append(
            {
                "path": str(args.structured_pool),
                "sha256": sha256_file(args.structured_pool),
                "split": "override_candidate",
                "labels_used": False,
            }
        )
    if str(args.prf_pool) != str(DEFAULT_SOURCE_PATHS["prf"]):
        records.append(
            {
                "path": str(args.prf_pool),
                "sha256": sha256_file(args.prf_pool),
                "split": "override_candidate",
                "labels_used": False,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_command = "python round18/experiments/o_sparse/o_s4_fusion/run_o_s4_fusion.py"

    _enforce_allowed_inputs(
        claims=args.claims,
        evidence=args.evidence,
        bm25_pool=args.bm25_pool,
        structured_pool=args.structured_pool,
        prf_pool=args.prf_pool,
    )

    eval_ks = parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        args.candidate_k = max(eval_ks)

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")

    for source_path in (args.bm25_pool, args.structured_pool, args.prf_pool):
        if not source_path.exists():
            raise SystemExit(f"Missing source pool: {source_path}")

    claims = load_json(args.claims)
    if not isinstance(claims, dict):
        raise SystemExit("Claims must be a JSON dict keyed by claim id.")
    evidence = load_json(args.evidence)
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence must be a JSON dict keyed by evidence id.")

    bm25_pool, bm25_loaded = parse_candidate_pool(args.bm25_pool, STRICT_SOURCES[0], args.candidate_k)
    structured_pool, structured_loaded = parse_candidate_pool(args.structured_pool, STRICT_SOURCES[1], args.candidate_k)
    prf_pool, prf_loaded = parse_candidate_pool(args.prf_pool, STRICT_SOURCES[2], args.candidate_k)

    source_pools = {
        "bm25": bm25_pool,
        "structured": structured_pool,
        "prf": prf_pool,
    }
    score_norms = {
        "bm25": build_score_normalization(source_pools, "bm25", args.candidate_k),
        "structured": build_score_normalization(source_pools, "structured", args.candidate_k),
        "prf": build_score_normalization(source_pools, "prf", args.candidate_k),
    }

    split_name = _infer_split_name(args.claims)
    split_file_suffix = "dev_full_dev" if split_name == "dev" else f"{split_name}_full"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_variant_results: list[dict[str, Any]] = []
    files_written: list[str] = []

    for config in STRICT_VARIANTS:
        fused_pool = build_fused_pool(
            claims=claims,
            pools=source_pools,
            config=config,
            candidate_k=args.candidate_k,
            score_norms=score_norms,
        )
        metrics = evaluate_recall_at_k(claims=claims, pool=fused_pool, eval_ks=eval_ks)
        metrics["fusion_variant"] = config.variant_id
        metrics["fusion_label"] = config.label
        metrics["fusion_mode"] = config.mode
        metrics["fusion_status"] = config.status
        metrics["fusion_weights"] = config.weights
        metrics["rrf_k"] = config.rrf_k
        metrics["candidate_top_k"] = args.candidate_k
        metrics["claims_input_count"] = len(claims)
        metrics["claims_with_source"] = sum(1 for claim_id in claims if (
            claim_id in bm25_pool or claim_id in structured_pool or claim_id in prf_pool
        ))
        metrics["source_claim_coverage"] = {
            "bm25": len(bm25_pool),
            "structured": len(structured_pool),
            "prf": len(prf_pool),
        }
        metrics["source_items_available"] = {
            "bm25": bm25_loaded,
            "structured": structured_loaded,
            "prf": prf_loaded,
        }
        metrics["evidence_items_available"] = len(evidence)
        metrics["smoke"] = False

        candidate_path = (
            args.output_dir
            / f"{split_file_suffix}_{args.run_id}_{config.variant_id}_top{args.candidate_k}_candidates.json"
        )
        metrics_path = (
            args.output_dir / f"{split_file_suffix}_{args.run_id}_{config.variant_id}_metrics.json"
        )
        write_json(candidate_path, fused_pool)
        write_json(metrics_path, metrics)
        files_written.extend([str(candidate_path), str(metrics_path)])

        all_variant_results.append(
            {
                "config": config,
                "metrics": metrics,
                "pool": fused_pool,
                "paths": {
                    "candidate": str(candidate_path),
                    "metrics": str(metrics_path),
                },
                "status": config.status,
            }
        )

    diagnostic_best: tuple[Any, Any, Any] | None = None
    if args.enable_diagnostic:
        diagnostic_pool_rows: list[tuple] = []
        weights = parse_float_list(args.diagnostic_weight_grid)
        rrf_ks = parse_int_list(args.diagnostic_rrf_k_grid)
        for bm25_w, structured_w, prf_w, rrf_k in [
            (w1, w2, w3, k)
            for w1 in weights
            for w2 in weights
            for w3 in weights
            for k in rrf_ks
        ]:
            if bm25_w <= 0 and structured_w <= 0 and prf_w <= 0:
                continue
            config = FusionConfig(
                variant_id=f"diagnostic_rrf_k{rrf_k}_w{bm25_w:.1f}_{structured_w:.1f}_{prf_w:.1f}",
                label=f"Diagnostic RRF(k={rrf_k}, w=[{bm25_w:.1f},{structured_w:.1f},{prf_w:.1f}])",
                mode="rrf",
                weights={"bm25": bm25_w, "structured": structured_w, "prf": prf_w},
                rrf_k=rrf_k,
                status="diagnostic-only",
            )
            fused_pool = build_fused_pool(
                claims=claims,
                pools=source_pools,
                config=config,
                candidate_k=args.candidate_k,
                score_norms=score_norms,
            )
            metrics = evaluate_recall_at_k(claims=claims, pool=fused_pool, eval_ks=eval_ks)
            diagnostic_pool_rows.append((config, metrics, fused_pool))
        diagnostic_best = select_best_diagnostic_variant(diagnostic_pool_rows)
        if diagnostic_best[0].variant_id != "diagnostic_none":
            diag_config, diag_metrics, diag_pool = diagnostic_best
            diag_metrics["fusion_variant"] = diag_config.variant_id
            diag_metrics["fusion_label"] = diag_config.label
            diag_metrics["fusion_mode"] = diag_config.mode
            diag_metrics["fusion_status"] = "diagnostic-only"
            diag_metrics["fusion_weights"] = diag_config.weights
            diag_metrics["rrf_k"] = diag_config.rrf_k
            diag_metrics["candidate_top_k"] = args.candidate_k
            diag_metrics["diagnostic_grid"] = {
                "weight_grid": parse_float_list(args.diagnostic_weight_grid),
                "rrf_k_grid": parse_int_list(args.diagnostic_rrf_k_grid),
            }
            diag_metrics["source_items_available"] = {
                "bm25": bm25_loaded,
                "structured": structured_loaded,
                "prf": prf_loaded,
            }
            diag_metrics["evidence_items_available"] = len(evidence)
            candidate_path = (
                args.output_dir
                / f"{split_file_suffix}_{args.run_id}_diagnostic_best_top{args.candidate_k}_candidates.json"
            )
            metrics_path = (
                args.output_dir
                / f"{split_file_suffix}_{args.run_id}_diagnostic_best_metrics.json"
            )
            selection_path = (
                args.output_dir / f"{split_file_suffix}_{args.run_id}_diagnostic_selection.json"
            )
            write_json(candidate_path, diag_pool)
            write_json(metrics_path, diag_metrics)
            write_json(
                selection_path,
                {
                    "diagnostic_mode": "dev-only-selection",
                    "selection_status": "diagnostic-only",
                    "selection_rule": "argmax over [micro@500, macro@500, hit_any@500, micro@100, macro@100, -rrf_k]",
                    "best_variant": {
                        "variant_id": diag_config.variant_id,
                        "label": diag_config.label,
                        "rrf_k": diag_config.rrf_k,
                        "weights": diag_config.weights,
                        "status": "diagnostic-only",
                        "candidate_file": str(candidate_path),
                        "metric_file": str(metrics_path),
                    },
                    "weight_grid": parse_float_list(args.diagnostic_weight_grid),
                    "rrf_k_grid": parse_int_list(args.diagnostic_rrf_k_grid),
                    "diagnostic_candidates_evaluated": len(diagnostic_pool_rows),
                },
            )
            files_written.extend([str(candidate_path), str(metrics_path), str(selection_path)])
            all_variant_results.append(
                {
                    "config": diag_config,
                    "metrics": diag_metrics,
                    "pool": diag_pool,
                    "paths": {
                        "candidate": str(candidate_path),
                        "metrics": str(metrics_path),
                        "selection": str(selection_path),
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
            row["macro_recall@500"],
            row["micro_recall@500"],
            row["hit_any@500"],
            row["macro_recall@100"],
            row["micro_recall@100"],
            row["status"] == "strict-candidate",
        ),
        reverse=True,
    )

    comparison_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_comparison_table.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()) if comparison_rows else [
            "variant",
            "label",
            "status",
            "mode",
            "rrf_k",
            "bm25_weight",
            "structured_weight",
            "prf_weight",
            "macro_recall@100",
            "macro_recall@500",
            "micro_recall@100",
            "micro_recall@500",
            "hit_any@100",
            "hit_any@500",
        ])
        writer.writeheader()
        writer.writerows(comparison_rows)

    all_metrics = {
        "split": split_name,
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "diagnostic_enabled": args.enable_diagnostic,
        "fixed_variants": [
            result["config"].variant_id for result in all_variant_results if result["status"] == "strict-candidate"
        ],
        "diagnostic_variant": diagnostic_best[0].variant_id if diagnostic_best else "none",
        "comparison_rows": comparison_rows,
        "random_seed": args.random_seed,
        "claims_file": str(args.claims),
        "evidence_file": str(args.evidence),
        "source_files": {
            "bm25": str(args.bm25_pool),
            "structured": str(args.structured_pool),
            "prf": str(args.prf_pool),
        },
    }
    all_metrics_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_metrics.json"
    write_json(all_metrics_path, all_metrics)
    files_written.append(str(all_metrics_path))

    comparison_json_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_comparison_table.json"
    write_json(comparison_json_path, comparison_rows)
    files_written.append(str(comparison_json_path))

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.claims),
            str(args.evidence),
            str(args.bm25_pool),
            str(args.structured_pool),
            str(args.prf_pool),
            str(args.output_dir),
            str(args.manifest),
        ]
    )

    data_flow_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_data_flow_report.json"
    data_flow = {
        "pipeline": "O-S1 + O-S2 + O-S3 sparse candidate fusion",
        "split": split_name,
        "inputs": {
            "claims": str(args.claims),
            "evidence": str(args.evidence),
            "bm25_pool": str(args.bm25_pool),
            "structured_pool": str(args.structured_pool),
            "prf_pool": str(args.prf_pool),
        },
        "input_file_hashes": {
            item["path"]: item["sha256"]
            for item in _build_input_records(args, args.claims, args.evidence)
            if item["path"] in {
                str(args.claims),
                str(args.evidence),
                str(DEFAULT_SOURCE_PATHS["bm25"]),
                str(DEFAULT_SOURCE_PATHS["structured"]),
                str(DEFAULT_SOURCE_PATHS["prf"]),
            }
        },
        "processing": {
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "fusion_modes": ["rrf", "combsum_inv_rank", "combsum_score_norm"],
            "fixed_variants": [variant.variant_id for variant in STRICT_VARIANTS],
            "diagnostic": {
                "enabled": args.enable_diagnostic,
                "weight_grid": parse_float_list(args.diagnostic_weight_grid),
                "rrf_k_grid": parse_int_list(args.diagnostic_rrf_k_grid),
            },
            "selected_diagnostic_variant": diagnostic_best[0].variant_id if diagnostic_best else None,
        },
        "outputs": {
            "candidate_pools_written": len([r for r in all_variant_results if r["status"] == "strict-candidate"]),
            "comparison_table": str(comparison_path),
            "all_metrics": str(all_metrics_path),
        },
        "notes": [
            "All fixed-weight strict variants are strict-candidate outputs.",
            "Dev diagnostic winner is explicitly labeled diagnostic-only; no dev-tuned candidate is promoted as strict.",
        ],
    }
    write_json(data_flow_path, data_flow)
    files_written.append(str(data_flow_path))

    command = (
        f"{base_command} --claims {args.claims} --evidence {args.evidence} "
        f"--bm25-pool {args.bm25_pool} --structured-pool {args.structured_pool} "
        f"--prf-pool {args.prf_pool} --candidate-k {args.candidate_k} --eval-k {','.join(map(str, eval_ks))} "
        f"--diagnostic-rrf-k-grid {','.join(map(str, parse_int_list(args.diagnostic_rrf_k_grid)))} "
        f"--diagnostic-weight-grid {','.join(map(str, parse_float_list(args.diagnostic_weight_grid)))} "
        f"--run-id {args.run_id} --output-dir {args.output_dir} --stage {args.stage}"
    ).strip()

    command_args = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            command_args[key] = str(value)
        else:
            command_args[key] = value
    record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT",
        "status": "strict-candidate",
        "split": split_name,
        "command": command,
        "command_args": command_args,
        "claims_count": len(claims),
        "evidence_count": len(evidence),
        "strict_variant_count": len(STRICT_VARIANTS),
        "diagnostic_enabled": args.enable_diagnostic,
        "diagnostic_candidates_evaluated": (
            diagnostic_best is not None and diagnostic_best[0].variant_id != "diagnostic_none"
        ),
        "forbidden_hits": forbidden_hits,
        "files_written": files_written + [str(args.record_path), str(args.manifest)],
    }
    write_json(args.record_path, record)
    files_written.append(str(args.record_path))
    files_written.append(str(comparison_path))

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
    manifest["input_files"] = _build_input_records(args, args.claims, args.evidence)
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan against forbidden strict tokens passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = files_written + [str(args.manifest)]
    manifest["metrics"] = {
        "all_metrics_file": str(all_metrics_path),
        "comparison_table": str(comparison_path),
        "comparison_json": str(comparison_json_path),
        "data_flow_report": str(data_flow_path),
        "diagnostic_mode": "dev-only",
        "diagnostic_selection_status": "diagnostic-only",
        "comparison_rows": comparison_rows,
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Loaded allowed raw claim/evidence JSON and the fixed O-S1/O-S2/O-S3 current-run candidate pools, "
        "fused per-claim candidates with RRF/weighted-RRF/CombSUM style methods, wrote strict and diagnostic pools, "
        "and reported recall metrics at requested k."
    )
    manifest["split_isolation_summary"] = (
        "This run is restricted to the dev claim split and does not read train/test labels for model selection. "
        "The diagnostic winner is explicitly not promoted to strict status because it is selected from dev tuning."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low"
    manifest["notes"] = (
        "Fixed-weight strict variants are strict-candidate outputs. "
        "Best diagnostic variant (if enabled) is marked diagnostic-only."
    )
    write_json(args.manifest, manifest)
    files_written.append(str(args.manifest))

    print(f"Wrote comparison table: {comparison_path}")
    print(f"Wrote manifest: {args.manifest}")
    for variant_name in [result["config"].variant_id for result in all_variant_results]:
        print(f"Computed: {variant_name}")
    for row in comparison_rows:
        print(
            f"{row['variant']} "
            f"mR@100={row['macro_recall@100']:.4f} mR@500={row['macro_recall@500']:.4f} "
            f"xR@500={row['micro_recall@500']:.4f} hit@500={row['hit_any@500']:.4f} status={row['status']}"
        )

    if diagnostic_best and diagnostic_best[0].variant_id != "diagnostic_none":
        print(
            f"Diagnostic winner: {diagnostic_best[0].variant_id} "
            f"(macro@500={diagnostic_best[1].get('macro_recall_at_500', 0.0):.6f}, "
            f"micro@500={diagnostic_best[1].get('micro_recall_at_500', 0.0):.6f}) "
            "marked diagnostic-only."
        )


if __name__ == "__main__":
    main()
