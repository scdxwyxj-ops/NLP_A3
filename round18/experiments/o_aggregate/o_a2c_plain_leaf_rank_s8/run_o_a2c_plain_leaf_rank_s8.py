from __future__ import annotations

import argparse
import csv
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import sys

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

DEFAULT_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_SPARSE_POOL = Path(
    "round18/outputs/o_sparse/o_s8_hand_feature_ranker/"
    "dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json"
)
DEFAULT_CE_POOL = Path(
    "round18/outputs/o_dense/o_d3b_cross_encoder_s7/"
    "dev_full_dev_full_dev_top64_s7_strict_top64_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8")
DEFAULT_MANIFEST = Path("round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/run_manifest.json")
DEFAULT_RECORD_PATH = Path("round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/run_record.json")
DEFAULT_RUN_ID = "o_a2c_plain_leaf_rank_s8"

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    label: str
    mode: str
    status: str = "strict-candidate"
    mmr_top_k: int | None = None
    mmr_lambda: float = 0.8
    sparse_weight: float = 1.0
    ce_weight: float = 1.0
    rrf_k: int = 60


STRICT_VARIANTS: tuple[VariantConfig, ...] = (
    VariantConfig(
        variant_id="strict_sparse_only",
        label="Sparse-only baseline",
        mode="sparse_only",
    ),
    VariantConfig(
        variant_id="strict_ce64_sparse_backfill",
        label="CE top64 + sparse backfill top500",
        mode="ce_backfill",
    ),
    VariantConfig(
        variant_id="strict_rrf_sparse_ce",
        label="RRF sparse + CE",
        mode="rrf",
        sparse_weight=1.0,
        ce_weight=1.0,
        rrf_k=60,
    ),
    VariantConfig(
        variant_id="strict_rrf_sparse_ce_mmr_top3",
        label="RRF sparse+CE + MMR top3",
        mode="rrf",
        sparse_weight=1.0,
        ce_weight=1.0,
        rrf_k=60,
        mmr_top_k=3,
        mmr_lambda=0.7,
    ),
    VariantConfig(
        variant_id="strict_rrf_sparse_ce_mmr_top10",
        label="RRF sparse+CE + MMR top10",
        mode="rrf",
        sparse_weight=1.0,
        ce_weight=1.0,
        rrf_k=60,
        mmr_top_k=10,
        mmr_lambda=0.7,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-A2c plain leaf-flat ranking over O-S8 hand-feature sparse and O-D3b CE candidate leaves "
            "with fixed-rule variants."
        )
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_CLAIMS,
        help="Claims split file (strictly data/dev-claims.json for this worker).",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence corpus path.",
    )
    parser.add_argument(
        "--sparse-pool",
        type=Path,
        default=DEFAULT_SPARSE_POOL,
        help="Current-run strict sparse pool used as the backfill universe.",
    )
    parser.add_argument(
        "--ce-pool",
        type=Path,
        default=DEFAULT_CE_POOL,
        help="Cross-encoder top64 pool (strict source).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for candidate/metric artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run identifier for filenames.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest output path.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=DEFAULT_RECORD_PATH,
        help="Run record output path.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Target candidate size per claim.",
    )
    parser.add_argument(
        "--eval-k",
        default="1,3,5,10,100,500",
        help="Comma-separated recall-k list.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF denominator used in sparse+CE fusion.",
    )
    parser.add_argument(
        "--disable-mmr-variants",
        action="store_true",
        help="Skip the optional fixed MMR(top3/top10) strict variants.",
    )
    parser.add_argument(
        "--enable-diagnostic",
        action="store_true",
        default=True,
        help="Run dev diagnostic best selection and write diagnostic selection artifacts.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Reproducibility marker.",
    )
    parser.add_argument(
        "--stage",
        default="o_a2c_plain_leaf_rank_s8",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--eval-k must include at least one integer")
    return sorted(set(values))


def _infer_split_name(path: Path) -> str:
    if str(path) == "data/dev-claims.json":
        return "dev"
    if str(path) == "data/train-claims.json":
        return "train"
    if str(path) == "data/test-claims-unlabelled.json":
        return "test"
    return "claims"


def _enforce_strict_inputs(claims: Path, evidence: Path, sparse_pool: Path, ce_pool: Path) -> None:
    if claims != DEFAULT_CLAIMS:
        raise SystemExit("O-A2c is defined for dev-only: claims must be data/dev-claims.json.")
    if evidence != DEFAULT_EVIDENCE:
        raise SystemExit("O-A2c requires data/evidence.json as evidence input.")
    if sparse_pool != DEFAULT_SPARSE_POOL:
        raise SystemExit(
            "O-A2c requires current-run strict O-S8 hand-feature sparse pool at: "
            f"{DEFAULT_SPARSE_POOL}"
        )
    if ce_pool != DEFAULT_CE_POOL:
        raise SystemExit(
            "O-A2c requires current-run O-D3b cross-encoder top64 pool at: "
            f"{DEFAULT_CE_POOL}"
        )


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_pool(
    path: Path,
    allowed_claims: dict[str, Any],
    candidate_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    if not path.exists():
        raise SystemExit(f"Missing candidate pool file: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate pool must be dict keyed by claim id: {path}")

    parsed: dict[str, list[dict[str, Any]]] = {}
    stats = {
        "missing_claims": 0,
        "missing_evidence_id": 0,
        "candidate_entries": 0,
    }

    for claim_id, entries in payload.items():
        if claim_id not in allowed_claims:
            continue
        if not isinstance(entries, list):
            continue

        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for position, item in enumerate(entries, start=1):
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id:
                stats["missing_evidence_id"] += 1
                continue
            if evidence_id in seen:
                continue
            seen.add(evidence_id)

            rank = _coerce_int(item.get("rank"))
            if rank <= 0:
                rank = position
            score = _coerce_float(item.get("score"))
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "rank": rank,
                    "score": score,
                    "source_payload": item,
                }
            )
            if len(rows) >= candidate_k:
                break

        if rows:
            parsed[claim_id] = rows
            stats["candidate_entries"] += len(rows)
        else:
            stats["missing_claims"] += 1

    return parsed, stats


def _build_source_maps(
    sparse_pool: dict[str, list[dict[str, Any]]],
    ce_pool: dict[str, list[dict[str, Any]]],
    claim_id: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    sparse_map: dict[str, dict[str, float]] = {}
    ce_map: dict[str, dict[str, float]] = {}
    for row in sparse_pool.get(claim_id, []):
        sparse_map[row["evidence_id"]] = {
            "rank": int(row["rank"]),
            "score": float(row.get("score", 0.0)),
        }
    for row in ce_pool.get(claim_id, []):
        ce_map[row["evidence_id"]] = {
            "rank": int(row["rank"]),
            "score": float(row.get("score", 0.0)),
        }
    return sparse_map, ce_map


def _merge_with_backfill(
    sparse_map: dict[str, dict[str, float]],
    ce_map: dict[str, dict[str, float]],
    candidate_k: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for evidence_id in ce_map:
        row = {
            "evidence_id": evidence_id,
            "sources": ["ce", "sparse"],
            "source_breakdown": [
                {
                    "source": "ce",
                    "source_rank": ce_map[evidence_id]["rank"],
                    "source_score": ce_map[evidence_id]["score"],
                    "weight": 1.0,
                    "contribution": 1.0,
                },
                {
                    "source": "sparse",
                    "source_rank": sparse_map[evidence_id]["rank"],
                    "source_score": sparse_map[evidence_id]["score"],
                    "weight": 1.0,
                    "contribution": 1.0,
                },
            ]
            if evidence_id in sparse_map
            else [
                {
                    "source": "ce",
                    "source_rank": ce_map[evidence_id]["rank"],
                    "source_score": ce_map[evidence_id]["score"],
                    "weight": 1.0,
                    "contribution": 1.0,
                }
            ],
        }
        row["score"] = 1.0 + (1.0 / max(1.0, float(ce_map[evidence_id]["rank"])) )
        row["source_count"] = len(row["source_breakdown"])
        row["sources"] = sorted(set([entry["source"] for entry in row["source_breakdown"]]))
        output.append(row)
        seen.add(evidence_id)
        if len(output) >= candidate_k:
            return output

    for evidence_id in sparse_map:
        if evidence_id in seen:
            continue
        row = {
            "evidence_id": evidence_id,
            "sources": ["sparse"],
            "source_breakdown": [
                {
                    "source": "sparse",
                    "source_rank": sparse_map[evidence_id]["rank"],
                    "source_score": sparse_map[evidence_id]["score"],
                    "weight": 1.0,
                    "contribution": 1.0,
                }
            ],
        }
        row["score"] = 0.01 * (1.0 / max(1.0, float(sparse_map[evidence_id]["rank"])) )
        row["source_count"] = 1
        output.append(row)
        if len(output) >= candidate_k:
            break

    for row in output:
        row["sources"] = sorted(set(row["sources"]))
    return output


def build_sparse_only_pool(
    claims: dict[str, Any],
    sparse_pool: dict[str, list[dict[str, Any]]],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    ranked_pool: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        ranked: list[dict[str, Any]] = []
        for row in sparse_pool.get(claim_id, [])[:candidate_k]:
            ranked.append(
                {
                    "evidence_id": row["evidence_id"],
                    "score": row.get("score", 0.0),
                    "sources": ["sparse"],
                    "source_breakdown": [
                        {
                            "source": "sparse",
                            "source_rank": row["rank"],
                            "source_score": row.get("score", 0.0),
                            "weight": 1.0,
                            "contribution": 1.0,
                        }
                    ],
                    "source_count": 1,
                }
            )
        ranked_pool[claim_id] = ranked
    return ranked_pool


def build_ce_backfill_pool(
    claims: dict[str, Any],
    sparse_pool: dict[str, list[dict[str, Any]]],
    ce_pool: dict[str, list[dict[str, Any]]],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    ranked_pool: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        sparse_map, ce_map = _build_source_maps(sparse_pool, ce_pool, claim_id)
        merged = _merge_with_backfill(sparse_map, ce_map, candidate_k)
        ranked_pool[claim_id] = merged
    return ranked_pool


def build_rrf_pool(
    claims: dict[str, Any],
    sparse_pool: dict[str, list[dict[str, Any]]],
    ce_pool: dict[str, list[dict[str, Any]]],
    candidate_k: int,
    sparse_weight: float = 1.0,
    ce_weight: float = 1.0,
    rrf_k: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    ranked_pool: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        sparse_map, ce_map = _build_source_maps(sparse_pool, ce_pool, claim_id)
        scores: dict[str, float] = {}
        traces: dict[str, list[dict[str, Any]]] = {}

        for evidence_id, entry in sparse_map.items():
            rank = max(1, int(entry["rank"]))
            contrib = float(sparse_weight) / (float(rrf_k) + float(rank))
            scores[evidence_id] = scores.get(evidence_id, 0.0) + contrib
            traces.setdefault(evidence_id, []).append(
                {
                    "source": "sparse",
                    "source_rank": rank,
                    "source_score": float(entry.get("score", 0.0)),
                    "weight": float(sparse_weight),
                    "contribution": float(contrib),
                }
            )

        for evidence_id, entry in ce_map.items():
            rank = max(1, int(entry["rank"]))
            contrib = float(ce_weight) / (float(rrf_k) + float(rank))
            scores[evidence_id] = scores.get(evidence_id, 0.0) + contrib
            traces.setdefault(evidence_id, []).append(
                {
                    "source": "ce",
                    "source_rank": rank,
                    "source_score": float(entry.get("score", 0.0)),
                    "weight": float(ce_weight),
                    "contribution": float(contrib),
                }
            )

        sorted_evidence = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:candidate_k]

        ranked: list[dict[str, Any]] = []
        for evidence_id, score in sorted_evidence:
            row_traces = sorted(
                traces.get(evidence_id, []),
                key=lambda entry: (entry["source"] != "ce", entry["source"]),
            )
            ranked.append(
                {
                    "evidence_id": evidence_id,
                    "score": float(score),
                    "sources": sorted({entry["source"] for entry in row_traces}),
                    "source_breakdown": row_traces,
                    "source_count": len(row_traces),
                }
            )

        ranked_pool[claim_id] = ranked

    return ranked_pool


def tokenize(text: str) -> tuple[str, ...]:
    if not text:
        return tuple()
    return tuple(token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1)


def jaccard_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    set_left = set(left)
    set_right = set(right)
    intersection = len(set_left & set_right)
    union = len(set_left | set_right)
    return intersection / union if union else 0.0


def apply_mmr(
    rows: list[dict[str, Any]],
    top_k: int,
    lambda_score: float,
    token_lookup: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    if top_k <= 1 or not rows:
        return rows
    selected: list[dict[str, Any]] = []
    remaining = list(rows)
    while remaining and len(selected) < top_k:
        best = -1.0e12
        best_index = 0
        for idx, candidate in enumerate(remaining):
            evidence_id = candidate["evidence_id"]
            tokens = token_lookup.get(evidence_id, ())
            redundancy = 0.0
            for picked in selected:
                picked_tokens = token_lookup.get(picked["evidence_id"], ())
                redundancy = max(redundancy, jaccard_similarity(tokens, picked_tokens))
            value = lambda_score * float(candidate["score"]) - (1.0 - lambda_score) * redundancy
            value += 1.0e-9 / max(1, int(candidate.get("source_breakdown", [{}])[0].get("source_rank", 1)))
            if value > best:
                best = value
                best_index = idx
        selected.append(remaining.pop(best_index))

    return selected + remaining


def apply_rankings(
    ranked_pool: dict[str, list[dict[str, Any]]],
    eval_config: VariantConfig,
    claim_keys: list[str],
    token_lookup: dict[str, tuple[str, ...]],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claim_keys:
        rows = list(ranked_pool.get(claim_id, []))
        if eval_config.mmr_top_k:
            rows = apply_mmr(rows, eval_config.mmr_top_k, eval_config.mmr_lambda, token_lookup)

        final_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(rows[:candidate_k], start=1):
            final_row = dict(row)
            final_row.update(
                {
                    "claim_id": claim_id,
                    "rank": rank,
                    "fusion_mode": eval_config.mode,
                    "fusion_variant": eval_config.variant_id,
                    "rrf_k": eval_config.rrf_k if eval_config.mode == "rrf" else None,
                }
            )
            if eval_config.mmr_top_k:
                final_row["mmr_top_k"] = eval_config.mmr_top_k
                final_row["mmr_lambda"] = eval_config.mmr_lambda
            final_rows.append(final_row)
        output[claim_id] = final_rows
    return output


def evaluate_recall_at_k(
    claims: dict[str, Any],
    pool: dict[str, list[dict[str, Any]]],
    eval_ks: list[int],
) -> dict[str, Any]:
    ks = sorted(set(eval_ks))
    rows = {
        "claims_with_evidence": 0,
        "evaluated_ks": ks,
        "avg_candidate_count": 0.0,
        "union_candidates": 0,
    }
    if not claims:
        return rows

    per_k_hits = {k: {"tp": 0.0, "gold": 0.0, "claims_hit": 0} for k in ks}
    claim_recall_totals: dict[int, list[float]] = {k: [] for k in ks}
    label_totals: dict[str, int] = {}
    label_hits: dict[int, dict[str, int]] = {k: {} for k in ks}
    claim_count = 0

    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claim_count += 1
        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0) + 1
        ranked = [row["evidence_id"] for row in pool.get(claim_id, [])]

        for k in ks:
            predicted = set(ranked[:k])
            tp = float(len(gold.intersection(predicted)))
            per_k_hits[k]["tp"] += tp
            per_k_hits[k]["gold"] += float(len(gold))
            recall = tp / len(gold) if gold else 0.0
            claim_recall_totals[k].append(recall)
            if tp > 0:
                per_k_hits[k]["claims_hit"] += 1
                label_hits[k][label] = label_hits[k].get(label, 0) + 1

    total_candidates = sum(len(v) for v in pool.values())
    rows["claims_with_evidence"] = claim_count
    rows["avg_candidate_count"] = float(total_candidates / claim_count) if claim_count else 0.0
    rows["union_candidates"] = len({cand["evidence_id"] for claims_rows in pool.values() for cand in claims_rows})

    for k in ks:
        recalls = claim_recall_totals[k]
        rows[f"macro_recall_at_{k}"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
        total_tp = float(per_k_hits[k]["tp"])
        total_gold = float(per_k_hits[k]["gold"])
        rows[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        rows[f"hit_any_at_{k}"] = per_k_hits[k]["claims_hit"] / claim_count if claim_count else 0.0
        for label in sorted(label_totals):
            if label_totals[label] <= 0:
                continue
            rows[f"{label.lower()}_hit_any_at_{k}"] = label_hits[k].get(label, 0) / label_totals[label]

    return rows


def build_comparison_row(
    config: VariantConfig,
    metrics: dict[str, Any],
    eval_ks: list[int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": config.variant_id,
        "label": config.label,
        "status": config.status,
        "mode": config.mode,
        "rrf_k": config.rrf_k if config.mode == "rrf" else None,
        "sparse_weight": config.sparse_weight,
        "ce_weight": config.ce_weight,
        "mmr_top_k": config.mmr_top_k,
        "mmr_lambda": config.mmr_lambda if config.mmr_top_k else None,
    }
    for k in eval_ks:
        row[f"macro_recall@{k}"] = metrics.get(f"macro_recall_at_{k}", 0.0)
        row[f"micro_recall@{k}"] = metrics.get(f"micro_recall_at_{k}", 0.0)
        row[f"hit_any@{k}"] = metrics.get(f"hit_any_at_{k}", 0.0)
    return row


def select_best_diagnostic_variant(
    rows,
):
    if not rows:
        return (
            VariantConfig("diagnostic_none", "No diagnostic candidate", "diagnostic", status="diagnostic-only"),
            {},
            None,
        )

    scored_rows = []
    for config, metrics, pool in rows:
        score = (
            metrics.get("micro_recall_at_500", 0.0),
            metrics.get("macro_recall_at_500", 0.0),
            metrics.get("hit_any_at_500", 0.0),
            metrics.get("micro_recall_at_100", 0.0),
            metrics.get("macro_recall_at_100", 0.0),
            -(config.rrf_k if config.rrf_k else 0),
        )
        scored_rows.append((score, config, metrics, pool))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    _, best_config, best_metrics, best_pool = scored_rows[0]
    return best_config, best_metrics, best_pool


def build_input_records(args: argparse.Namespace, claims_path: Path, evidence_path: Path) -> list[dict[str, Any]]:
    split = _infer_split_name(claims_path)
    return [
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
            "path": str(DEFAULT_SPARSE_POOL),
            "sha256": sha256_file(DEFAULT_SPARSE_POOL),
            "split": "current_run_artifact",
            "labels_used": False,
        },
        {
            "path": str(DEFAULT_CE_POOL),
            "sha256": sha256_file(DEFAULT_CE_POOL),
            "split": "current_run_artifact",
            "labels_used": False,
        },
    ]


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_command = "python round18/experiments/o_aggregate/o_a2c_plain_leaf_rank_s8/run_o_a2c_plain_leaf_rank_s8.py"

    _enforce_strict_inputs(args.claims, args.evidence, args.sparse_pool, args.ce_pool)

    eval_ks = parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        args.candidate_k = max(eval_ks)
    if args.candidate_k <= 0:
        raise SystemExit("candidate-k must be > 0")

    for path in (args.claims, args.evidence, args.sparse_pool, args.ce_pool):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    claims = load_json(args.claims)
    if not isinstance(claims, dict):
        raise SystemExit("Claims file must be a JSON dict keyed by claim id.")

    evidence = load_json(args.evidence)
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence file must be a JSON dict keyed by evidence id.")

    sparse_pool, sparse_stats = _parse_pool(args.sparse_pool, claims, args.candidate_k)
    ce_pool, ce_stats = _parse_pool(args.ce_pool, claims, 64)

    # Precompute candidate text tokens for the optional fixed MMR code path.
    tokenized_evidence: dict[str, tuple[str, ...]] = {}
    candidate_evidence_ids: set[str] = set()
    for pool in (sparse_pool, ce_pool):
        for rows in pool.values():
            for row in rows:
                candidate_evidence_ids.add(row["evidence_id"])
    for evidence_id in candidate_evidence_ids:
        tokenized_evidence[evidence_id] = tokenize(str(evidence.get(evidence_id, "")))

    split_name = _infer_split_name(args.claims)
    split_file_suffix = "dev_full_dev" if split_name == "dev" else f"{split_name}_full"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_variant_rows: list[tuple[VariantConfig, dict[str, Any], dict[str, list[dict[str, Any]]]]] = []
    all_outputs: list[str] = []
    comparison_rows: list[dict[str, Any]] = []
    files_written: list[str] = []

    variant_configs = list(STRICT_VARIANTS)
    if args.disable_mmr_variants:
        variant_configs = [cfg for cfg in variant_configs if not cfg.variant_id.endswith("_mmr_top3") and not cfg.variant_id.endswith("_mmr_top10")]

    claim_order = list(claims.keys())

    for config in variant_configs:
        if config.mode == "sparse_only":
            base_pool = build_sparse_only_pool(claims, sparse_pool, args.candidate_k)
        elif config.mode == "ce_backfill":
            base_pool = build_ce_backfill_pool(
                claims=claims,
                sparse_pool=sparse_pool,
                ce_pool=ce_pool,
                candidate_k=args.candidate_k,
            )
        elif config.mode == "rrf":
            base_pool = build_rrf_pool(
                claims=claims,
                sparse_pool=sparse_pool,
                ce_pool=ce_pool,
                candidate_k=args.candidate_k,
                sparse_weight=config.sparse_weight,
                ce_weight=config.ce_weight,
                rrf_k=args.rrf_k,
            )
        else:
            raise SystemExit(f"Unsupported mode: {config.mode}")

        ranked_pool = apply_rankings(
            ranked_pool=base_pool,
            eval_config=config,
            claim_keys=claim_order,
            token_lookup=tokenized_evidence,
            candidate_k=args.candidate_k,
        )

        metrics = evaluate_recall_at_k(claims=claims, pool=ranked_pool, eval_ks=eval_ks)
        metrics["fusion_variant"] = config.variant_id
        metrics["fusion_label"] = config.label
        metrics["fusion_mode"] = config.mode
        metrics["fusion_status"] = config.status
        metrics["candidate_top_k"] = args.candidate_k
        metrics["split"] = split_name
        metrics["claims_input_count"] = len(claims)
        metrics["union_candidates"] = len({
            row["evidence_id"] for rows in ranked_pool.values() for row in rows
        })

        candidate_path = (
            args.output_dir
            / f"{split_file_suffix}_{args.run_id}_{config.variant_id}_top{args.candidate_k}_candidates.json"
        )
        metrics_path = (
            args.output_dir
            / f"{split_file_suffix}_{args.run_id}_{config.variant_id}_metrics.json"
        )

        write_json(candidate_path, ranked_pool)
        write_json(metrics_path, metrics)
        all_outputs.extend([str(candidate_path), str(metrics_path)])

        comparison_rows.append(
            build_comparison_row(config=config, metrics=metrics, eval_ks=eval_ks)
        )
        all_variant_rows.append((config, metrics, ranked_pool))

    comparison_rows.sort(
        key=lambda row: (
            row.get("macro_recall@500", 0.0),
            row.get("micro_recall@500", 0.0),
            row.get("hit_any@500", 0.0),
            row.get("macro_recall@100", 0.0),
            row.get("micro_recall@100", 0.0),
            row.get("status") == "strict-candidate",
            - (row.get("mmr_top_k", 0) or 0),
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
            "sparse_weight",
            "ce_weight",
            "mmr_top_k",
            "mmr_lambda",
        ])
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_json_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_comparison_table.json"
    write_json(comparison_json_path, comparison_rows)

    files_written.extend([str(comparison_path), str(comparison_json_path)])

    selected_diagnostic = None
    selection_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_diagnostic_selection.json"
    if args.enable_diagnostic and all_variant_rows:
        selected_diagnostic = select_best_diagnostic_variant(all_variant_rows)
        if selected_diagnostic[0].variant_id != "diagnostic_none":
            diag_config, diag_metrics, diag_pool = selected_diagnostic
            diag_metrics = dict(diag_metrics)
            diag_metrics["fusion_variant"] = diag_config.variant_id
            diag_metrics["fusion_label"] = diag_config.label
            diag_metrics["fusion_mode"] = diag_config.mode
            diag_metrics["fusion_status"] = "diagnostic-only"
            diag_metrics["fusion_selection_rule"] = "dev-only fixed-rule winner"
            candidate_path = (
                args.output_dir
                / f"{split_file_suffix}_{args.run_id}_diagnostic_best_top{args.candidate_k}_candidates.json"
            )
            metrics_path = (
                args.output_dir
                / f"{split_file_suffix}_{args.run_id}_diagnostic_best_metrics.json"
            )
            if diag_pool is not None:
                write_json(candidate_path, diag_pool)
                files_written.extend([str(candidate_path), str(metrics_path)])
                write_json(metrics_path, diag_metrics)
            write_json(
                selection_path,
                {
                    "selection_mode": "dev-only-fixed-rule",
                    "selection_status": "diagnostic-only",
                    "selection_rule": "argmax over [micro@500, macro@500, hit_any@500, micro@100, macro@100, -rrf_k]",
                    "best_variant": {
                        "variant_id": diag_config.variant_id,
                        "label": diag_config.label,
                        "mode": diag_config.mode,
                        "status": "diagnostic-only",
                        "candidate_file": str(candidate_path),
                        "metric_file": str(metrics_path),
                    },
                    "candidate_k": args.candidate_k,
                    "eval_k": eval_ks,
                    "rrf_k": args.rrf_k,
                    "diagnostic_candidates_evaluated": len(all_variant_rows),
                },
            )
            files_written.append(str(selection_path))

    all_metrics = {
        "split": split_name,
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "diagnostic_enabled": args.enable_diagnostic,
        "diagnostic_variant": selected_diagnostic[0].variant_id if selected_diagnostic else "none",
        "fixed_variants": [
            config.variant_id for config, _, _ in all_variant_rows if config.status == "strict-candidate"
        ],
        "comparison_rows": comparison_rows,
        "claims_file": str(args.claims),
        "evidence_file": str(args.evidence),
        "source_files": {
            "sparse": str(args.sparse_pool),
            "ce": str(args.ce_pool),
        },
        "source_claim_coverage": {
            "sparse": len(sparse_pool),
            "ce": len(ce_pool),
        },
        "source_candidate_counts": {
            "sparse": sparse_stats["candidate_entries"],
            "ce": ce_stats["candidate_entries"],
        },
        "random_seed": args.random_seed,
    }
    all_metrics_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_metrics.json"
    write_json(all_metrics_path, all_metrics)
    files_written.append(str(all_metrics_path))

    data_flow_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_data_flow_report.json"
    data_flow = {
        "pipeline": "Round18 leaf-flat calibrated rank over strict sparse+CE leafs with fixed fusion rules.",
        "split": split_name,
        "inputs": {
            "claims": str(args.claims),
            "evidence": str(args.evidence),
            "sparse_pool": str(args.sparse_pool),
            "ce_pool": str(args.ce_pool),
        },
        "input_file_hashes": {
            path: sha256_file(Path(path))
            for path in {
                str(args.claims),
                str(args.evidence),
                str(args.sparse_pool),
                str(args.ce_pool),
            }
        },
        "processing": {
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "rrf_k": args.rrf_k,
            "mmr_enabled": not args.disable_mmr_variants,
            "candidate_sources": [
                "sparse",
                "cross_encoder",
            ],
            "selection": {
                "type": "diagnostic-dev-best",
                "diagnostic_enabled": args.enable_diagnostic,
                "selected_variant": selected_diagnostic[0].variant_id if selected_diagnostic else None,
            },
        },
        "outputs": {
            "candidate_files": len(variant_configs),
            "comparison_table": str(comparison_path),
            "all_metrics": str(all_metrics_path),
            "data_flow_report": str(data_flow_path),
        },
        "notes": [
            "Fixed variants are strict-candidate outputs.",
            "Any dev-selected winner is written as diagnostic-only and is not promoted as strict.",
            "Optional MMR variants are deterministic and fixed to top-k greedy diversity." if not args.disable_mmr_variants else "Optional MMR variants were disabled.",
        ],
    }
    write_json(data_flow_path, data_flow)
    files_written.append(str(data_flow_path))

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.claims),
            str(args.evidence),
            str(args.sparse_pool),
            str(args.ce_pool),
            str(args.output_dir),
            str(args.manifest),
        ]
    )

    command = (
        f"{base_command} --claims {args.claims} --evidence {args.evidence} "
        f"--sparse-pool {args.sparse_pool} --ce-pool {args.ce_pool} "
        f"--output-dir {args.output_dir} --run-id {args.run_id} --candidate-k {args.candidate_k} "
        f"--eval-k {','.join(map(str, eval_ks))} --rrf-k {args.rrf_k} "
        f"--stage {args.stage}"
    )
    if args.disable_mmr_variants:
        command += " --disable-mmr-variants"

    command_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    run_record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT",
        "status": "strict-candidate",
        "split": split_name,
        "command": command,
        "command_args": command_args,
        "claims_count": len(claims),
        "evidence_count": len(evidence),
        "diagnostic_enabled": args.enable_diagnostic,
        "diagnostic_selected_variant": selected_diagnostic[0].variant_id if selected_diagnostic else "none",
        "forbidden_hits": forbidden_hits,
        "files_written": files_written + [str(args.record_path), str(args.manifest)],
    }
    write_json(args.record_path, run_record)

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
    manifest["input_files"] = build_input_records(args, args.claims, args.evidence)
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan against forbidden strict tokens passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = [
        *all_outputs,
        str(all_metrics_path),
        str(comparison_path),
        str(comparison_json_path),
        str(data_flow_path),
        str(args.record_path),
        str(args.manifest),
    ]
    manifest["metrics"] = {
        "all_metrics_file": str(all_metrics_path),
        "comparison_table": str(comparison_path),
        "comparison_json": str(comparison_json_path),
        "data_flow_report": str(data_flow_path),
        "comparison_rows": comparison_rows,
        "diagnostic_selection": str(selection_path) if args.enable_diagnostic else "",
        "diagnostic_mode": "dev-only" if args.enable_diagnostic else "disabled",
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Loaded strict dev claims/evidence and current-run sparse+CE(+fallback) candidate pools, "
        "built fixed leaf-flat variants (sparse-only, CE-priority backfill, RRF sparse+CE, optional MMR), "
        "computed recall metrics at requested k, and wrote pool/metrics/comparison artifacts."
    )
    manifest["split_isolation_summary"] = (
        "This run is restricted to dev claims and fixed run artifacts, with no training-time tuning "
        "used to construct candidate pools."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low"
    manifest["notes"] = (
        "Fixed predeclared variants are strict-candidate. Dev-selected best is marked diagnostic-only."
    )
    write_json(args.manifest, manifest)

    files_written = [
        *all_outputs,
        str(all_metrics_path),
        str(comparison_path),
        str(comparison_json_path),
        str(data_flow_path),
        str(args.record_path),
        str(args.manifest),
    ]
    if args.enable_diagnostic:
        files_written.extend([str(selection_path)])

    print(f"Wrote comparison table: {comparison_path}")
    for row in comparison_rows:
        macro_500 = row.get("macro_recall@500", 0.0)
        micro_500 = row.get("micro_recall@500", 0.0)
        hit_500 = row.get("hit_any@500", 0.0)
        print(
            f"{row['variant']} mR@500={macro_500:.4f} xR@500={micro_500:.4f} "
            f"hit@500={hit_500:.4f} status={row['status']}"
        )

    if selected_diagnostic and selected_diagnostic[0].variant_id != "diagnostic_none":
        print(f"Diagnostic winner (dev-only): {selected_diagnostic[0].variant_id}")


if __name__ == "__main__":
    main()
