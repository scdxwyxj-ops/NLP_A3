#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from round18.tools.common import find_forbidden_tokens, load_json, write_json


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_DEV_CE_POOL = Path(
    "round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/"
    "dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json"
)
DEFAULT_DEV_HAND_POOL = Path(
    "round18/outputs/o_sparse/o_s8_hand_feature_ranker/"
    "dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/reports/ce_hand_feature_fusion_kfold")

DEFAULT_EVAL_K = "3,10,64,500"
DEFAULT_CE_WEIGHTS = "0.0,0.25,0.5,0.75,1.0"
DEFAULT_TIE_BREAKERS = "min_source_rank,ce_rank_first,hand_rank_first"
DEFAULT_RANKER_C = "0.25,0.5,1.0"


@dataclass(frozen=True)
class CandidateRow:
    evidence_id: str
    score: float
    rank: int
    features: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 bounded CE + hand-feature fusion and tune via train-only CV, "
            "with dev confirmation and diagnostic smoke mode."
        )
    )
    parser.add_argument(
        "--train-claims",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS,
        help="Train split claims.",
    )
    parser.add_argument(
        "--dev-claims",
        type=Path,
        default=DEFAULT_DEV_CLAIMS,
        help="Dev split claims.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence corpus JSON (for input provenance).",
    )
    parser.add_argument(
        "--train-ce-pool",
        type=Path,
        default=None,
        help="Train CE top500/ranked candidate JSON (required for strict mode).",
    )
    parser.add_argument(
        "--train-hand-pool",
        type=Path,
        default=None,
        help="Train hand-feature/f-r candidate JSON (required for strict mode).",
    )
    parser.add_argument(
        "--dev-ce-pool",
        type=Path,
        default=DEFAULT_DEV_CE_POOL,
        help="Dev CE top500/ranked candidate JSON.",
    )
    parser.add_argument(
        "--dev-hand-pool",
        type=Path,
        default=DEFAULT_DEV_HAND_POOL,
        help="Dev hand-feature/f-r candidate JSON.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Per-claim bound on final ranked candidate list.",
    )
    parser.add_argument(
        "--eval-k",
        default=DEFAULT_EVAL_K,
        help="Comma-separated evaluation ks; primary is @64.",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "diagnostic"),
        default="strict",
        help="strict: use train-only k-fold then dev confirmation; diagnostic: smoke-only.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force diagnostic run on dev subset only.",
    )
    parser.add_argument(
        "--smoke-claims",
        type=int,
        default=64,
        help="Max claims to use in smoke mode.",
    )
    parser.add_argument(
        "--tune-mode",
        choices=("fusion", "shallow-ranker"),
        default="fusion",
        help="Tune score fusion or a simple shallow ranker.",
    )
    parser.add_argument(
        "--ce-weight-grid",
        default=DEFAULT_CE_WEIGHTS,
        help="Grid for CE weight in fusion mode.",
    )
    parser.add_argument(
        "--tie-breakers",
        default=DEFAULT_TIE_BREAKERS,
        help="Tie-breakers for equal fused scores.",
    )
    parser.add_argument(
        "--ranker-c-grid",
        default=DEFAULT_RANKER_C,
        help="Grid for LogisticRegression C in shallow-ranker mode.",
    )
    parser.add_argument("--k-folds", type=int, default=5, help="CV folds for strict tuning.")
    parser.add_argument(
        "--k-fold-seed",
        type=int,
        default=2026,
        help="Random seed for fold splitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary JSON/CSV + report.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("ce_hand_feature_fusion_kfold_summary.json"),
        help="Summary JSON filename in output-dir.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("ce_hand_feature_fusion_kfold_summary.csv"),
        help="Summary CSV filename in output-dir.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("ce_hand_feature_fusion_kfold_report.md"),
        help="Report filename in output-dir.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("eval-k must contain at least one integer.")
    return sorted(set(values))


def parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise argparse.ArgumentTypeError("CE weights must be in [0, 1].")
        values.append(round(value, 4))
    if not values:
        raise argparse.ArgumentTypeError("No valid CE weights supplied.")
    return sorted(set(values))


def parse_ranker_c_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("ranker C values must be > 0.")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("No valid ranker C values supplied.")
    return sorted(set(values))


def parse_tie_breakers(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"min_source_rank", "ce_rank_first", "hand_rank_first"}
    for item in values:
        if item not in valid:
            raise argparse.ArgumentTypeError(f"Unsupported tie-breaker: {item}")
    return values


def _coerce_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _coerce_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _load_claims(path: Path) -> dict[str, Any]:
    claims = load_json(path)
    if not isinstance(claims, dict):
        raise SystemExit(f"Claims file must be dict: {path}")
    return claims


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if math.isclose(min_value, max_value):
        return {key: 0.0 for key in values}
    span = max_value - min_value
    return {key: (value - min_value) / span for key, value in values.items()}


def parse_candidate_pool(
    path: Path,
    claims: dict[str, Any],
    candidate_k: int,
) -> dict[str, dict[str, CandidateRow]]:
    if not path.exists():
        raise SystemExit(f"Missing candidate pool: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate pool must be object: {path}")

    parsed: dict[str, dict[str, CandidateRow]] = {}
    for claim_id in claims:
        rows = payload.get(claim_id, [])
        if not isinstance(rows, list):
            continue
        seen: set[str] = set()
        candidates: dict[str, CandidateRow] = {}
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            rank = _coerce_int(row.get("rank"), fallback=position)
            score = _coerce_float(row.get("source_score", row.get("reranker_score", row.get("score", 0.0))))
            feature_block: dict[str, float] = {}
            for key, value in row.items():
                if key in {
                    "evidence_id",
                    "claim_id",
                    "score",
                    "rank",
                    "source_rank",
                    "source_count",
                    "source_mode",
                    "reranker_score",
                    "reranker_mode",
                    "fusion_mode",
                    "fusion_variant",
                    "rrf_k",
                    "model_name",
                    "source_score",
                }:
                    continue
                numeric = None
                try:
                    numeric = float(value)
                except Exception:
                    continue
                feature_block[key] = numeric

            candidates[evidence_id] = CandidateRow(
                evidence_id=evidence_id,
                score=score,
                rank=max(1, rank),
                features=feature_block,
            )
            if len(candidates) >= candidate_k:
                break
        if candidates:
            parsed[claim_id] = candidates
    return parsed


def make_fusion_rows_for_claim(
    ce_rows: dict[str, CandidateRow],
    hand_rows: dict[str, CandidateRow],
    ce_weight: float,
    hand_weight: float,
    tie_break: str,
    candidate_k: int,
) -> list[dict[str, Any]]:
    union = set(ce_rows) | set(hand_rows)
    if not union:
        return []

    ce_scores = {eid: row.score for eid, row in ce_rows.items()}
    hand_scores = {eid: row.score for eid, row in hand_rows.items()}
    ce_norm = _minmax(ce_scores)
    hand_norm = _minmax(hand_scores)

    merged: list[dict[str, Any]] = []
    for evidence_id in union:
        ce_row = ce_rows.get(evidence_id)
        hand_row = hand_rows.get(evidence_id)
        ce_norm_score = ce_norm.get(evidence_id, 0.0)
        hand_norm_score = hand_norm.get(evidence_id, 0.0)
        fused_score = ce_weight * ce_norm_score + hand_weight * hand_norm_score

        ce_rank = ce_row.rank if ce_row else 10**9
        hand_rank = hand_row.rank if hand_row else 10**9
        if tie_break == "ce_rank_first":
            primary, secondary = ce_rank, hand_rank
        elif tie_break == "hand_rank_first":
            primary, secondary = hand_rank, ce_rank
        else:
            primary, secondary = min(ce_rank, hand_rank), max(ce_rank, hand_rank)

        source_breakdown: list[dict[str, Any]] = []
        if ce_row is not None:
            source_breakdown.append(
                {
                    "source": "ce",
                    "source_rank": ce_row.rank,
                    "source_score": ce_row.score,
                    "normalized_score": ce_norm_score,
                    "weight": ce_weight,
                }
            )
        if hand_row is not None:
            source_breakdown.append(
                {
                    "source": "hand-feature",
                    "source_rank": hand_row.rank,
                    "source_score": hand_row.score,
                    "normalized_score": hand_norm_score,
                    "weight": hand_weight,
                }
            )

        merged.append(
            {
                "evidence_id": evidence_id,
                "score": fused_score,
                "fusion_score": fused_score,
                "source_breakdown": source_breakdown,
                "source_count": len(source_breakdown),
                "ce_rank": ce_rank,
                "hand_rank": hand_rank,
                "tie_break_primary": primary,
                "tie_break_secondary": secondary,
            }
        )

    merged.sort(
        key=lambda row: (-row["score"], row["tie_break_primary"], row["tie_break_secondary"], row["evidence_id"])
    )
    for rank, row in enumerate(merged[:candidate_k], start=1):
        row["rank"] = rank
    return merged[:candidate_k]


def fused_rankers_for_claims(
    claim_ids: set[str] | None,
    ce_pool: dict[str, dict[str, CandidateRow]],
    hand_pool: dict[str, dict[str, CandidateRow]],
    ce_weight: float,
    tie_break: str,
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    effective_claims = claim_ids if claim_ids is not None else set(ce_pool) | set(hand_pool)
    fused: dict[str, list[dict[str, Any]]] = {}
    for claim_id in effective_claims:
        fused[claim_id] = make_fusion_rows_for_claim(
            ce_rows=ce_pool.get(claim_id, {}),
            hand_rows=hand_pool.get(claim_id, {}),
            ce_weight=ce_weight,
            hand_weight=1.0 - ce_weight,
            tie_break=tie_break,
            candidate_k=candidate_k,
        )
    return fused


def evaluate_recall_at_k(
    claims: dict[str, Any],
    ranked: dict[str, list[dict[str, Any]]],
    ks: list[int],
    claim_subset: set[str] | None = None,
) -> dict[str, float]:
    ks = sorted(set(ks))
    metrics = {f"macro_recall@{k}": 0.0 for k in ks}
    micro: dict[int, float] = {k: 0.0 for k in ks}
    recall_hits: dict[int, float] = {k: 0.0 for k in ks}
    hit_any: dict[int, int] = {k: 0 for k in ks}
    claims_seen = 0

    if claim_subset is None:
        claim_ids = claims
    else:
        claim_ids = claim_subset

    for claim_id in sorted(claim_ids):
        claim = claims.get(claim_id)
        if not claim:
            continue
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claims_seen += 1
        predicted = [row["evidence_id"] for row in ranked.get(claim_id, [])]
        for k in ks:
            preds = set(predicted[:k])
            tp = len(preds & gold)
            recall = tp / float(len(gold))
            metrics[f"macro_recall@{k}"] += recall
            micro[k] += float(tp)
            recall_hits[k] += float(len(gold))
            if tp > 0:
                hit_any[k] += 1

    if claims_seen:
        for k in ks:
            metrics[f"macro_recall@{k}"] /= float(claims_seen)
            metrics[f"micro_recall@{k}"] = micro[k] / float(recall_hits[k]) if recall_hits[k] else 0.0
            metrics[f"hit_any@{k}"] = hit_any[k] / float(claims_seen)
    else:
        for k in ks:
            metrics[f"micro_recall@{k}"] = 0.0
            metrics[f"hit_any@{k}"] = 0.0

    metrics["claims_with_evidence"] = claims_seen
    metrics["avg_candidate_count"] = (
        float(sum(len(v) for v in ranked.values())) / float(claims_seen)
        if claims_seen
        else 0.0
    )
    metrics["union_candidates"] = len({row["evidence_id"] for rows in ranked.values() for row in rows})
    return metrics


def split_folds(claim_ids: list[str], n_folds: int, seed: int) -> list[tuple[list[str], list[str]]]:
    if n_folds < 2:
        raise ValueError("k-folds must be >=2.")
    if len(claim_ids) < n_folds:
        raise ValueError(f"Cannot create {n_folds} folds from only {len(claim_ids)} claims.")
    items = list(claim_ids)
    random.Random(seed).shuffle(items)
    buckets = [items[i::n_folds] for i in range(n_folds)]
    splits: list[tuple[list[str], list[str]]] = []
    for i in range(n_folds):
        valid = set(buckets[i])
        train = [cid for cid in items if cid not in valid]
        splits.append((train, list(buckets[i])))
    return splits


def build_shallow_examples(
    claims: dict[str, Any],
    ce_pool: dict[str, dict[str, CandidateRow]],
    hand_pool: dict[str, dict[str, CandidateRow]],
    claim_subset: set[str],
) -> dict[str, list[dict[str, float]]]:
    examples: dict[str, list[dict[str, float]]] = {}
    for claim_id in claim_subset:
        claim = claims.get(claim_id)
        if not claim:
            continue
        gold = set(claim.get("evidences", []))
        ce_rows = ce_pool.get(claim_id, {})
        hand_rows = hand_pool.get(claim_id, {})
        union = sorted(set(ce_rows) | set(hand_rows), key=lambda eid: (min(
            ce_rows.get(eid, CandidateRow(eid, 0.0, 10**9, {})).rank,
            hand_rows.get(eid, CandidateRow(eid, 0.0, 10**9, {})).rank,
        ), eid))

        rows: list[dict[str, float]] = []
        for evidence_id in union:
            ce_row = ce_rows.get(evidence_id)
            hand_row = hand_rows.get(evidence_id)
            ce_present = 1.0 if ce_row else 0.0
            hand_present = 1.0 if hand_row else 0.0
            ce_rank = ce_row.rank if ce_row else 10**9
            hand_rank = hand_row.rank if hand_row else 10**9

            feature_row: dict[str, float] = {
                "label": 1.0 if evidence_id in gold else 0.0,
                "ce_present": ce_present,
                "hand_present": hand_present,
                "ce_rank": float(ce_rank),
                "hand_rank": float(hand_rank),
                "ce_inv_rank": 1.0 / (1.0 + ce_rank),
                "hand_inv_rank": 1.0 / (1.0 + hand_rank),
                "ce_score": ce_row.score if ce_row else 0.0,
                "hand_score": hand_row.score if hand_row else 0.0,
            }
            if ce_row:
                for key, value in ce_row.features.items():
                    feature_row[f"ce_{key}"] = value
            if hand_row:
                for key, value in hand_row.features.items():
                    feature_row[f"hand_{key}"] = value
            feature_row["evidence_id"] = evidence_id
            rows.append(feature_row)
        examples[claim_id] = rows
    return examples


def rank_with_shallow_model(
    train_examples: dict[str, list[dict[str, float]]],
    val_examples: dict[str, list[dict[str, float]]],
    feature_names: list[str],
    c_value: float,
    candidate_k: int,
) -> tuple[bool, dict[str, list[dict[str, Any]]], str]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        return False, {}, f"missing sklearn dependency: {exc}"

    try:
        import numpy as np
    except Exception as exc:
        return False, {}, f"missing numpy dependency: {exc}"

    X_train: list[list[float]] = []
    y_train: list[int] = []
    for rows in train_examples.values():
        for row in rows:
            X_train.append([row.get(feature, 0.0) for feature in feature_names])
            y_train.append(int(row["label"]))

    if not X_train:
        return False, {}, "no train rows"
    if len(set(y_train)) < 2:
        return False, {}, "single-class train split (all labels identical)"

    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
    )
    model.fit(np.array(X_train, dtype=float), np.array(y_train, dtype=int))

    ranked: dict[str, list[dict[str, Any]]] = {}
    for claim_id, rows in val_examples.items():
        if not rows:
            ranked[claim_id] = []
            continue
        X_val = np.array([[row.get(feature, 0.0) for feature in feature_names] for row in rows], dtype=float)
        scores = model.predict_proba(X_val)[:, 1]
        scored = sorted(
            zip(rows, scores),
            key=lambda item: (item[1], item[0].get("evidence_id", "")),
            reverse=True,
        )
        ranked_rows: list[dict[str, Any]] = []
        for rank, (row, score) in enumerate(scored, start=1):
            ranked_rows.append(
                {
                    "evidence_id": row["evidence_id"],
                    "score": float(score),
                    "shallow_score": float(score),
                    "label": row["label"],
                    "ce_score": row.get("ce_score", 0.0),
                    "hand_score": row.get("hand_score", 0.0),
                    "source_count": 2,
                    "source_breakdown": [
                        {"source": "ce", "weight": 0.0, "source_score": row.get("ce_score", 0.0)},
                        {"source": "hand-feature", "weight": 0.0, "source_score": row.get("hand_score", 0.0)},
                    ],
                    "rank": rank,
                }
            )
            if rank >= candidate_k:
                break
        ranked[claim_id] = ranked_rows
    return True, ranked, ""


def gather_feature_names(examples: dict[str, list[dict[str, float]]]) -> list[str]:
    feature_names: set[str] = set()
    for rows in examples.values():
        for row in rows:
            for key in row.keys():
                if key in {"label", "evidence_id"}:
                    continue
                feature_names.add(key)
    names = sorted(feature_names)
    if not names:
        raise SystemExit("No numeric shallow features could be extracted from train candidates.")
    return names


def _score_by_cv(
    fold_metrics: list[dict[str, float]],
    eval_ks: list[int],
    prefix: str,
) -> dict[str, float]:
    if not fold_metrics:
        return {f"{prefix}_macro_recall@{k}": 0.0 for k in eval_ks} | {
            f"{prefix}_micro_recall@{k}": 0.0 for k in eval_ks
        } | {f"{prefix}_hit_any@{k}": 0.0 for k in eval_ks}
    out: dict[str, float] = {}
    for k in eval_ks:
        out[f"{prefix}_macro_recall@{k}"] = sum(
            item[f"macro_recall@{k}"] for item in fold_metrics
        ) / float(len(fold_metrics))
        out[f"{prefix}_micro_recall@{k}"] = sum(
            item[f"micro_recall@{k}"] for item in fold_metrics
        ) / float(len(fold_metrics))
        out[f"{prefix}_hit_any@{k}"] = sum(
            item[f"hit_any@{k}"] for item in fold_metrics
        ) / float(len(fold_metrics))
    return out


def run_fusion_tuning(
    tuning_claims: dict[str, Any],
    tuning_ce: dict[str, dict[str, CandidateRow]],
    tuning_hand: dict[str, dict[str, CandidateRow]],
    eval_claims: dict[str, Any],
    eval_ce: dict[str, dict[str, CandidateRow]],
    eval_hand: dict[str, dict[str, CandidateRow]],
    eval_ks: list[int],
    ce_weight_grid: list[float],
    tie_breakers: list[str],
    k_folds: int,
    seed: int,
    candidate_k: int,
    status: str,
    strict_mode: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not tuning_claims:
        raise SystemExit("No tuning claims were provided.")

    tuning_ids = list(tuning_claims.keys())
    folds = split_folds(tuning_ids, n_folds=k_folds, seed=seed)
    rows: list[dict[str, Any]] = []
    for ce_weight in ce_weight_grid:
        for tie_break in tie_breakers:
            fold_metrics: list[dict[str, float]] = []
            for train_ids, valid_ids in folds:
                valid_set = set(valid_ids)
                candidate_pool = fused_rankers_for_claims(
                    claim_ids=valid_set,
                    ce_pool=tuning_ce,
                    hand_pool=tuning_hand,
                    ce_weight=ce_weight,
                    tie_break=tie_break,
                    candidate_k=candidate_k,
                )
                _ = train_ids  # reserved for parity with tune-only split
                metrics = evaluate_recall_at_k(
                    claims=tuning_claims,
                    ranked=candidate_pool,
                    ks=eval_ks,
                    claim_subset=valid_set,
                )
                fold_metrics.append(metrics)

            row: dict[str, Any] = {
                "tune_mode": "fusion",
                "ce_weight": ce_weight,
                "hand_weight": 1.0 - ce_weight,
                "tie_break": tie_break,
                "selection_basis": "train_cv" if strict_mode else "diagnostic_cv",
                "status": status,
                "fold_count": len(fold_metrics),
                "candidate_k": candidate_k,
            }
            row.update(_score_by_cv(fold_metrics, eval_ks, "cv"))
            rows.append(row)

    if not rows:
        raise SystemExit("No fusion candidates were evaluated.")

    rank_key = lambda row: (
        row.get("cv_macro_recall@64", 0.0),
        row.get("cv_macro_recall@3", 0.0),
        row.get("cv_macro_recall@10", 0.0),
        row.get("cv_macro_recall@500", 0.0),
    )
    best = sorted(rows, key=rank_key, reverse=True)[0]

    # final confirmation / diagnostic eval is always on eval_claims using eval pools.
    final_pool = fused_rankers_for_claims(
        claim_ids=set(eval_claims.keys()),
        ce_pool=eval_ce,
        hand_pool=eval_hand,
        ce_weight=best["ce_weight"],
        tie_break=best["tie_break"],
        candidate_k=candidate_k,
    )
    final_metrics = evaluate_recall_at_k(eval_claims, final_pool, eval_ks)
    for k in eval_ks:
        best[f"dev_macro_recall@{k}"] = final_metrics[f"macro_recall@{k}"]
        best[f"dev_micro_recall@{k}"] = final_metrics[f"micro_recall@{k}"]
        best[f"dev_hit_any@{k}"] = final_metrics[f"hit_any@{k}"]
    best["selection_basis"] = "train_cv" if strict_mode else "diagnostic_cv"
    return rows, best, final_pool


def run_shallow_tuning(
    tuning_claims: dict[str, Any],
    tuning_ce: dict[str, dict[str, CandidateRow]],
    tuning_hand: dict[str, dict[str, CandidateRow]],
    eval_claims: dict[str, Any],
    eval_ce: dict[str, dict[str, CandidateRow]],
    eval_hand: dict[str, dict[str, CandidateRow]],
    eval_ks: list[int],
    c_grid: list[float],
    k_folds: int,
    seed: int,
    candidate_k: int,
    status: str,
    strict_mode: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    tuning_examples = build_shallow_examples(tuning_claims, tuning_ce, tuning_hand, set(tuning_claims))
    if not tuning_examples:
        raise SystemExit("No shallow examples from tuning claims.")

    feature_names = gather_feature_names(tuning_examples)
    tuning_ids = list(tuning_claims.keys())
    if len(tuning_ids) < k_folds:
        raise SystemExit(f"Cannot create {k_folds} folds from {len(tuning_ids)} tuning claims.")
    folds = split_folds(tuning_ids, n_folds=k_folds, seed=seed)
    rows: list[dict[str, Any]] = []

    for c_value in c_grid:
        fold_metrics: list[dict[str, float]] = []
        for train_ids, valid_ids in folds:
            valid_set = set(valid_ids)
            train_set = set(train_ids)
            train_examples = {cid: tuning_examples[cid] for cid in train_set if cid in tuning_examples}
            valid_examples = {cid: tuning_examples[cid] for cid in valid_set if cid in tuning_examples}
            ok, ranked, err = rank_with_shallow_model(
                train_examples=train_examples,
                val_examples=valid_examples,
                feature_names=feature_names,
                c_value=c_value,
                candidate_k=candidate_k,
            )
            if not ok:
                continue
            metrics = evaluate_recall_at_k(
                claims=tuning_claims,
                ranked=ranked,
                ks=eval_ks,
                claim_subset=valid_set,
            )
            fold_metrics.append(metrics)

        row: dict[str, Any] = {
            "tune_mode": "shallow-ranker",
            "model_c": c_value,
            "selection_basis": "train_cv" if strict_mode else "diagnostic_cv",
            "status": status,
            "fold_count": len(fold_metrics),
            "candidate_k": candidate_k,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        }
        row.update(_score_by_cv(fold_metrics, eval_ks, "cv"))
        rows.append(row)

    rows = [row for row in rows if row.get("fold_count", 0) > 0]
    if not rows:
        raise SystemExit("No shallow-ranker candidates could be evaluated.")

    rank_key = lambda row: (
        row.get("cv_macro_recall@64", 0.0),
        row.get("cv_macro_recall@3", 0.0),
        row.get("cv_macro_recall@10", 0.0),
        row.get("cv_macro_recall@500", 0.0),
    )
    best = sorted(rows, key=rank_key, reverse=True)[0]

    eval_pool = build_shallow_examples(eval_claims, eval_ce, eval_hand, set(eval_claims))
    if not eval_pool:
        raise SystemExit("No evaluation examples to rank for final shallow confirmation.")
    ok, final_ranked, err = rank_with_shallow_model(
        train_examples=tuning_examples,
        val_examples=eval_pool,
        feature_names=feature_names,
        c_value=best["model_c"],
        candidate_k=candidate_k,
    )
    if not ok:
        raise SystemExit(f"Failed final shallow fit on evaluation set: {err}")

    final_metrics = evaluate_recall_at_k(eval_claims, final_ranked, eval_ks)
    for k in eval_ks:
        best[f"dev_macro_recall@{k}"] = final_metrics[f"macro_recall@{k}"]
        best[f"dev_micro_recall@{k}"] = final_metrics[f"micro_recall@{k}"]
        best[f"dev_hit_any@{k}"] = final_metrics[f"hit_any@{k}"]
    best["selection_basis"] = "train_cv" if strict_mode else "diagnostic_cv"
    return rows, best, final_ranked


def write_summary_csv(rows: list[dict[str, Any]], path: Path, eval_ks: list[int]) -> None:
    if not rows:
        return
    key_set: set[str] = set()
    for row in rows:
        key_set.update(row.keys())
    ordered_keys: list[str] = []
    for key in ("tune_mode", "status", "selection_basis", "ce_weight", "hand_weight", "model_c", "tie_break", "candidate_k", "feature_count", "fold_count"):
        if key in key_set:
            ordered_keys.append(key)
    for k in sorted(eval_ks):
        for prefix in ("cv_macro_recall@", "cv_micro_recall@", "cv_hit_any@", "dev_macro_recall@", "dev_micro_recall@", "dev_hit_any@"):
            full = f"{prefix}{k}"
            if full in key_set and full not in ordered_keys:
                ordered_keys.append(full)
    ordered_keys.extend(sorted(key for key in key_set if key not in ordered_keys))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_report(
    report_path: Path,
    args: argparse.Namespace,
    strict_possible: bool,
    eval_ks: list[int],
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    best_status: str,
    summary_json_path: Path,
    summary_csv_path: Path,
    command: str,
    forbidden_hits: dict[str, list[str]],
) -> None:
    run_lines = [
        "# CE + Hand-Feature Fusion / Shallow-Ranker Tuning",
        "",
        f"- Run mode: `{'strict' if strict_possible else 'diagnostic-only'}`",
        f"- Tune mode: `{args.tune_mode}`",
        f"- Candidate budget: `top{args.candidate_k}`",
        f"- Eval ks: `{eval_ks}`",
        f"- Selection policy: train-only k-fold then dev once" if strict_possible else "- Selection policy: diagnostic-only (dev artifacts only)",
        "",
        "## Inputs",
        f"- Train claims: `{args.train_claims}`",
        f"- Dev claims: `{args.dev_claims}`",
        f"- Evidence: `{args.evidence}`",
        f"- Train CE pool: `{args.train_ce_pool}`",
        f"- Train hand pool: `{args.train_hand_pool}`",
        f"- Dev CE pool: `{args.dev_ce_pool}`",
        f"- Dev hand pool: `{args.dev_hand_pool}`",
        "",
        "## Command",
        "```bash",
        command,
        "```",
        "",
        "## Forbidden-token scan",
    ]
    if forbidden_hits:
        run_lines.append("- Forbidden markers found in inputs:")
        for token, hits in sorted(forbidden_hits.items()):
            run_lines.append(f"  - `{token}`: {len(hits)} hit(s)")
    else:
        run_lines.append("- No forbidden tokens in scanned command/path fields.")

    run_lines.extend(
        [
            "",
            "## Status discipline",
            f"- Strict-vs-diagnostic marker: `{best_status}`",
            "- Hyperparameters are selected from train-only k-fold only in strict mode.",
            "- Dev-only selected hyperparameters are **not** promoted when strict mode is enabled.",
        ]
    )

    if strict_possible:
        run_lines.extend(
            [
                "",
                "## Best strict selection and dev confirmation",
                f"- Selected config: `{best.get('tune_mode')}`",
                f"- Selection basis: `{best.get('selection_basis')}`",
                f"- CE weight: `{best.get('ce_weight', 'n/a')}`",
                f"- Hand weight: `{best.get('hand_weight', 'n/a')}`",
                f"- Tie-break: `{best.get('tie_break', 'n/a')}`",
                f"- `model_c` (if shallow): `{best.get('model_c', 'n/a')}`",
                f"- CV macro@3: `{best.get('cv_macro_recall@3', 0.0):.6f}`",
                f"- CV macro@10: `{best.get('cv_macro_recall@10', 0.0):.6f}`",
                f"- CV macro@64: `{best.get('cv_macro_recall@64', 0.0):.6f}`",
                f"- CV macro@500: `{best.get('cv_macro_recall@500', 0.0):.6f}`",
                f"- Dev macro@3: `{best.get('dev_macro_recall@3', 0.0):.6f}`",
                f"- Dev macro@10: `{best.get('dev_macro_recall@10', 0.0):.6f}`",
                f"- Dev macro@64: `{best.get('dev_macro_recall@64', 0.0):.6f}`",
                f"- Dev macro@500: `{best.get('dev_macro_recall@500', 0.0):.6f}`",
            ]
        )
    else:
        run_lines.extend(
            [
                "",
                "## Diagnostic run only",
                f"- `--smoke` was used: run is bounded to `{args.smoke_claims}` dev claims.",
                f"- Best diagnostic config: `{best.get('tune_mode')}`",
                f"- CV macro@64: `{best.get('cv_macro_recall@64', 0.0):.6f}`",
                f"- CV macro@3: `{best.get('cv_macro_recall@3', 0.0):.6f}`",
                f"- CV macro@10: `{best.get('cv_macro_recall@10', 0.0):.6f}`",
                f"- CV macro@500: `{best.get('cv_macro_recall@500', 0.0):.6f}`",
            ]
        )

    if best_status != "strict-candidate":
        run_lines.extend(
            [
                "",
                "## Dev-only hyperparameter note",
                "- This run cannot be used for strict submission selection; status is diagnostic-only.",
                "- Use `--mode strict --train-ce-pool ... --train-hand-pool ...` once artifacts exist.",
            ]
        )

    run_lines.extend(
        [
            "",
            "## Outputs",
            f"- Summary JSON: `{summary_json_path}`",
            f"- Summary CSV: `{summary_csv_path}`",
            "",
            "## Top 10 candidates",
        ]
    )
    selected_key = "cv_macro_recall@64"
    table_rows = sorted(rows, key=lambda row: row.get(selected_key, 0.0), reverse=True)[:10]
    run_lines.append("| rank | tune_mode | status | ce_weight | hand_weight | model_c | tie_break | cv@64 | dev@64 |")
    run_lines.append("|---:|---|---|---:|---:|---:|---|---:|---:|")
    for idx, row in enumerate(table_rows, start=1):
        run_lines.append(
            "| " + " | ".join(
                [
                    str(idx),
                    str(row.get("tune_mode")),
                    str(row.get("status")),
                    str(row.get("ce_weight", "")),
                    str(row.get("hand_weight", "")),
                    str(row.get("model_c", "")),
                    str(row.get("tie_break", "")),
                    f"{row.get('cv_macro_recall@64', 0.0):.4f}",
                    f"{row.get('dev_macro_recall@64', row.get('macro_recall@64', 0.0)):.4f}",
                ]
            ) + " |"
        )
    report_path.write_text("\n".join(run_lines) + "\n", encoding="utf-8")


def select_claim_subset(claims: dict[str, Any], max_count: int) -> dict[str, Any]:
    if max_count <= 0 or max_count >= len(claims):
        return claims
    selected = {}
    for claim_id in list(claims.keys())[:max_count]:
        selected[claim_id] = claims[claim_id]
    return selected


def main() -> None:
    args = parse_args()
    eval_ks = parse_k_list(args.eval_k)
    if args.candidate_k < max(eval_ks):
        raise SystemExit("--candidate-k must be >= max eval-k.")

    run_mode = "diagnostic" if args.smoke else args.mode
    strict_requested = run_mode == "strict"
    strict_possible = False
    if strict_requested:
        if args.train_ce_pool is None or args.train_hand_pool is None:
            raise SystemExit(
                "Strict mode needs both --train-ce-pool and --train-hand-pool. "
                "Pass --smoke for a diagnostic run from dev artifacts."
            )
        if not args.train_ce_pool.exists() or not args.train_hand_pool.exists():
            raise SystemExit(
                "Strict train artifacts are missing. Use --smoke for diagnostic mode "
                "or provide existing train CE/hand candidate pools."
            )
        strict_possible = True

    train_claims = _load_claims(args.train_claims) if strict_possible else {}
    dev_claims = _load_claims(args.dev_claims)
    _ = _load_claims(args.evidence)
    if strict_possible:
        ce_train = parse_candidate_pool(args.train_ce_pool, train_claims, args.candidate_k)
        hand_train = parse_candidate_pool(args.train_hand_pool, train_claims, args.candidate_k)
        if not ce_train or not hand_train:
            raise SystemExit("Strict mode requires non-empty train candidate pools.")
    else:
        ce_train = {}
        hand_train = {}

    ce_dev = parse_candidate_pool(args.dev_ce_pool, dev_claims, args.candidate_k)
    hand_dev = parse_candidate_pool(args.dev_hand_pool, dev_claims, args.candidate_k)
    if not ce_dev or not hand_dev:
        raise SystemExit("Dev candidate pools could not be parsed or are empty.")

    command = " ".join(sys.argv)
    ce_weight_grid = parse_float_grid(args.ce_weight_grid)
    tie_breakers = parse_tie_breakers(args.tie_breakers)
    ranker_c_grid = parse_ranker_c_grid(args.ranker_c_grid)

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.evidence),
            str(args.dev_ce_pool),
            str(args.dev_hand_pool),
            str(args.output_dir),
            command,
        ]
        + ([] if not strict_possible else [str(args.train_ce_pool), str(args.train_hand_pool)])
    )

    if run_mode == "diagnostic":
        tuning_claims = select_claim_subset(dev_claims, args.smoke_claims)
        eval_claims = tuning_claims
        eval_ce = ce_dev
        eval_hand = hand_dev
        status = "diagnostic-only"
        if len(tuning_claims) < 2:
            raise SystemExit("Diagnostic mode needs at least 2 claims to run fold tuning.")
    else:
        tuning_claims = train_claims
        eval_claims = dev_claims
        eval_ce = ce_dev
        eval_hand = hand_dev
        status = "strict-candidate"

    if args.tune_mode == "fusion":
        rows, best, best_candidates = run_fusion_tuning(
            tuning_claims=tuning_claims,
            tuning_ce=ce_train if strict_possible else ce_dev,
            tuning_hand=hand_train if strict_possible else hand_dev,
            eval_claims=eval_claims,
            eval_ce=eval_ce,
            eval_hand=eval_hand,
            eval_ks=eval_ks,
            ce_weight_grid=ce_weight_grid,
            tie_breakers=tie_breakers,
            k_folds=args.k_folds if strict_possible else max(2, min(args.k_folds, len(tuning_claims)) if len(tuning_claims) >= 2 else 2),
            seed=args.k_fold_seed,
            candidate_k=args.candidate_k,
            status=status,
            strict_mode=strict_possible,
        )
    else:
        if not strict_possible:
            if len(tuning_claims) < 2:
                raise SystemExit("Diagnostic mode needs at least 2 claims.")
        rows, best, best_candidates = run_shallow_tuning(
            tuning_claims=tuning_claims,
            tuning_ce=ce_train if strict_possible else ce_dev,
            tuning_hand=hand_train if strict_possible else hand_dev,
            eval_claims=eval_claims,
            eval_ce=eval_ce,
            eval_hand=eval_hand,
            eval_ks=eval_ks,
            c_grid=ranker_c_grid,
            k_folds=args.k_folds if strict_possible else max(2, min(args.k_folds, len(tuning_claims))),
            seed=args.k_fold_seed,
            candidate_k=args.candidate_k,
            status=status,
            strict_mode=strict_possible,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / args.summary_json
    summary_csv = args.output_dir / args.summary_csv
    report_path = args.output_dir / args.report
    best_candidates_path = args.output_dir / f"best_{args.tune_mode}_{run_mode}_dev_top{args.candidate_k}.json"

    for row in rows:
        row["run_mode"] = run_mode
        row["eval_k"] = eval_ks
        row["candidate_k"] = args.candidate_k
        row["status"] = status if row.get("status") in {"strict-candidate", "diagnostic-only"} else status

    summary_payload = {
        "run_mode": run_mode,
        "strict_mode": strict_requested,
        "strict_run_possible": strict_possible,
        "tune_mode": args.tune_mode,
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "command": command,
        "smoke": args.smoke,
        "smoke_claim_limit": args.smoke_claims if run_mode == "diagnostic" else 0,
        "inputs": {
            "train_claims": str(args.train_claims),
            "dev_claims": str(args.dev_claims),
            "evidence": str(args.evidence),
            "train_ce_pool": str(args.train_ce_pool) if args.train_ce_pool else None,
            "train_hand_pool": str(args.train_hand_pool) if args.train_hand_pool else None,
            "dev_ce_pool": str(args.dev_ce_pool),
            "dev_hand_pool": str(args.dev_hand_pool),
        },
        "tuning_claims_count": len(tuning_claims),
        "eval_claims_count": len(eval_claims),
        "best_candidates_file": str(best_candidates_path),
        "best_selection": best,
        "rows": rows,
        "forbidden_input_scan": forbidden_hits,
    }
    write_json(summary_json, summary_payload)
    write_summary_csv(rows, summary_csv, eval_ks)
    write_json(best_candidates_path, best_candidates)

    best_status = best.get("status", status)
    make_report(
        report_path=report_path,
        args=args,
        strict_possible=strict_possible,
        eval_ks=eval_ks,
        rows=rows,
        best=best,
        best_status=best_status,
        summary_json_path=summary_json,
        summary_csv_path=summary_csv,
        command=command,
        forbidden_hits=forbidden_hits,
    )

    print(f"wrote {summary_json}")
    print(f"wrote {summary_csv}")
    print(f"wrote {best_candidates_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
