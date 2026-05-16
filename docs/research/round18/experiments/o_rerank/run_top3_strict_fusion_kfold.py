#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.tools.common import find_forbidden_tokens, load_json, write_json  # noqa: E402


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_TRAIN_CE = Path(
    "round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/"
    "dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json"
)
DEFAULT_DEV_CE = Path(
    "round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/"
    "dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json"
)
DEFAULT_TRAIN_EMBEDDING = Path(
    "round18/outputs/o_dense/o_d1x_embedding_bm25_char_train_top500/"
    "train_full_train_o_d1x_embedding_bm25_char_train_top500_strict_top500_candidates.json"
)
DEFAULT_DEV_EMBEDDING = Path(
    "round18/outputs/o_dense/o_d1x_embedding_bm25_char_dev_top500/"
    "dev_full_dev_o_d1x_embedding_bm25_char_dev_top500_strict_top500_candidates.json"
)
DEFAULT_TRAIN_HAND = Path(
    "round18/outputs/o_sparse/o_s8_hand_feature_ranker/"
    "train_full_o_s8_hand_feature_ranker_rrf_bm25_char_top500_pool.json"
)
DEFAULT_DEV_HAND = Path(
    "round18/outputs/o_sparse/o_s8_hand_feature_ranker/"
    "dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_rerank/top3_strict_fusion_kfold")
DEFAULT_REPORT_DIR = Path("round18/reports/top3_strict_fusion_kfold")
DEFAULT_RUN_ID = "top3_strict_fusion_kfold"
DEFAULT_EVAL_K = "1,3,5,10,64,500"


@dataclass(frozen=True)
class CandidateItem:
    evidence_id: str
    source_rank: int
    score: float
    rank: int
    raw_score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-kfold top3-focused fusion over CE / embedding / hand pools.\n"
            "Strict mode: train split k-fold for variant selection, dev once for confirmation."
        )
    )
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-ce-pool", type=Path, default=DEFAULT_TRAIN_CE)
    parser.add_argument("--dev-ce-pool", type=Path, default=DEFAULT_DEV_CE)
    parser.add_argument("--train-embedding-pool", type=Path, default=DEFAULT_TRAIN_EMBEDDING)
    parser.add_argument("--dev-embedding-pool", type=Path, default=DEFAULT_DEV_EMBEDDING)
    parser.add_argument("--train-hand-pool", type=Path, default=DEFAULT_TRAIN_HAND)
    parser.add_argument("--dev-hand-pool", type=Path, default=DEFAULT_DEV_HAND)
    parser.add_argument("--candidate-k", type=int, default=500, help="Top-k per claim to rerank.")
    parser.add_argument(
        "--eval-k",
        default=DEFAULT_EVAL_K,
        help="Comma-separated recall cutoffs. Focus is k=1/3/5/10.",
    )
    parser.add_argument("--k-folds", type=int, default=5, help="Train k-fold count in strict mode.")
    parser.add_argument("--fold-seed", type=int, default=2026, help="RNG seed for strict split.")
    parser.add_argument("--ce-weight-grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--embedding-weight-grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--shallow-weight-grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--rank-prior-grid", default="0.0,0.25,0.5,1.0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Diagnostic smoke run. Uses a claim subset and skips strict CV.",
    )
    parser.add_argument(
        "--smoke-claims",
        type=int,
        default=32,
        help="Subset size in smoke mode.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--eval-k must contain at least one integer.")
    return sorted(set(values))


def parse_weight_grid(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0:
            raise argparse.ArgumentTypeError(f"{name} must be >= 0.")
        values.append(round(value, 6))
    if not values:
        raise argparse.ArgumentTypeError(f"{name} has no valid values.")
    return sorted(set(values))


def coerce_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def coerce_int(value: Any, fallback: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def load_claims(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Claims file must be a JSON object: {path}")
    return payload


def parse_ce_pool(path: Path, claims: dict[str, Any], candidate_k: int) -> dict[str, dict[str, CandidateItem]]:
    if not path.exists():
        raise SystemExit(f"Missing CE pool: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"CE pool must be a JSON object: {path}")
    parsed: dict[str, dict[str, CandidateItem]] = {}
    for claim_id in claims:
        rows = payload.get(claim_id, [])
        if not isinstance(rows, list):
            continue
        seen = set[str]()
        entries: dict[str, CandidateItem] = {}
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            rank = coerce_int(row.get("rank", idx), idx)
            source_rank = coerce_int(row.get("source_rank", rank), rank)
            score = coerce_float(row.get("reranker_score", row.get("score", 0.0)))
            source_score = coerce_float(row.get("source_score", score), score)
            entries[evidence_id] = CandidateItem(
                evidence_id=evidence_id,
                source_rank=source_rank,
                score=score,
                rank=rank,
                raw_score=source_score,
            )
            if len(entries) >= candidate_k:
                break
        if entries:
            parsed[claim_id] = entries
    return parsed


def parse_embedding_pool(
    path: Path, claims: dict[str, Any], candidate_k: int
) -> dict[str, dict[str, CandidateItem]]:
    if not path.exists():
        raise SystemExit(f"Missing embedding pool: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Embedding pool must be a JSON object: {path}")
    parsed: dict[str, dict[str, CandidateItem]] = {}
    for claim_id in claims:
        rows = payload.get(claim_id, [])
        if not isinstance(rows, list):
            continue
        seen = set[str]()
        entries: dict[str, CandidateItem] = {}
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            rank = coerce_int(row.get("embedding_rank", row.get("rank", idx)), idx)
            source_rank = coerce_int(row.get("source_rank", rank), rank)
            score = coerce_float(row.get("embedding_score", row.get("reranker_score", row.get("score", 0.0))))
            source_score = coerce_float(row.get("source_score", score), score)
            entries[evidence_id] = CandidateItem(
                evidence_id=evidence_id,
                source_rank=source_rank,
                score=score,
                rank=rank,
                raw_score=source_score,
            )
            if len(entries) >= candidate_k:
                break
        if entries:
            parsed[claim_id] = entries
    return parsed


def parse_hand_pool(
    path: Path, claims: dict[str, Any], candidate_k: int
) -> dict[str, dict[str, CandidateItem]]:
    if not path.exists():
        raise SystemExit(f"Missing hand pool: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Hand pool must be a JSON object: {path}")
    parsed: dict[str, dict[str, CandidateItem]] = {}
    for claim_id in claims:
        rows = payload.get(claim_id, [])
        if not isinstance(rows, list):
            continue
        seen = set[str]()
        entries: dict[str, CandidateItem] = {}
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            rank = coerce_int(row.get("rank", idx), idx)
            source_rank = coerce_int(row.get("source_rank", rank), rank)
            score = coerce_float(row.get("score", 0.0))
            source_score = coerce_float(row.get("source_score", score), score)
            entries[evidence_id] = CandidateItem(
                evidence_id=evidence_id,
                source_rank=source_rank,
                score=score,
                rank=rank,
                raw_score=source_score,
            )
            if len(entries) >= candidate_k:
                break
        if entries:
            parsed[claim_id] = entries
    return parsed


def split_folds(claim_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    if n_folds < 2:
        raise ValueError("--k-folds must be >= 2.")
    if len(claim_ids) < n_folds:
        raise ValueError(f"Cannot split {len(claim_ids)} claims into {n_folds} folds.")
    rng = np.random.default_rng(seed)
    shuffled = list(claim_ids)
    rng.shuffle(shuffled)
    return [shuffled[idx::n_folds] for idx in range(n_folds)]


def minmax_normalized(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if math.isclose(lo, hi):
        return {key: 0.0 for key in values}
    span = hi - lo
    return {key: (value - lo) / span for key, value in values.items()}


def score_candidate_rows(
    claim_id: str,
    variant: str,
    ce_pool: dict[str, dict[str, CandidateItem]],
    embedding_pool: dict[str, dict[str, CandidateItem]],
    hand_pool: dict[str, dict[str, CandidateItem]],
    ce_weight: float,
    embedding_weight: float,
    hand_weight: float,
    prior_weight: float,
    candidate_k: int,
) -> list[dict[str, Any]]:
    ce_rows = ce_pool.get(claim_id, {})
    embedding_rows = embedding_pool.get(claim_id, {})
    hand_rows = hand_pool.get(claim_id, {})

    if variant == "ce_only":
        sources = {"ce": ce_rows}
    elif variant == "embedding_only":
        sources = {"embedding": embedding_rows}
    elif variant == "ce_plus_shallow":
        sources = {"ce": ce_rows, "shallow": hand_rows}
    elif variant == "embedding_plus_shallow":
        sources = {"embedding": embedding_rows, "shallow": hand_rows}
    else:
        raise ValueError(f"Unknown variant: {variant}")

    available_ids = set().union(*[set(v.keys()) for v in sources.values()]) if sources else set()
    if not available_ids:
        return []

    total_w = ce_weight + embedding_weight + hand_weight + prior_weight
    if total_w <= 0.0:
        return []

    ce_norm = minmax_normalized({eid: row.score for eid, row in ce_rows.items()})
    emb_norm = minmax_normalized({eid: row.score for eid, row in embedding_rows.items()})
    hand_norm = minmax_normalized({eid: row.score for eid, row in hand_rows.items()})

    prior_vals: dict[str, float] = {}
    for evidence_id in available_ids:
        source_ranks: list[int] = []
        if "ce" in sources and evidence_id in ce_rows:
            source_ranks.append(ce_rows[evidence_id].source_rank)
        if "embedding" in sources and evidence_id in embedding_rows:
            source_ranks.append(embedding_rows[evidence_id].source_rank)
        if "shallow" in sources and evidence_id in hand_rows:
            source_ranks.append(hand_rows[evidence_id].source_rank)
        if source_ranks:
            prior_vals[evidence_id] = 1.0 / (60.0 + min(source_ranks))
    prior_norm = minmax_normalized(prior_vals)

    ce_ratio = ce_weight / total_w
    emb_ratio = embedding_weight / total_w
    hand_ratio = hand_weight / total_w
    prior_ratio = prior_weight / total_w

    rows: list[dict[str, Any]] = []
    for evidence_id in available_ids:
        candidate_rows: list[dict[str, Any]] = []
        source_ranks: list[int] = []
        score = 0.0

        if "ce" in sources and evidence_id in ce_rows:
            source_ranks.append(ce_rows[evidence_id].source_rank)
            ce_v = ce_norm.get(evidence_id, 0.0)
            score += ce_ratio * ce_v
            candidate_rows.append(
                {
                    "source": "ce",
                    "source_rank": ce_rows[evidence_id].source_rank,
                    "raw_score": float(ce_rows[evidence_id].score),
                    "normalized_score": ce_v,
                    "weight": ce_ratio,
                }
            )

        if "embedding" in sources and evidence_id in embedding_rows:
            source_ranks.append(embedding_rows[evidence_id].source_rank)
            emb_v = emb_norm.get(evidence_id, 0.0)
            score += emb_ratio * emb_v
            candidate_rows.append(
                {
                    "source": "embedding",
                    "source_rank": embedding_rows[evidence_id].source_rank,
                    "raw_score": float(embedding_rows[evidence_id].score),
                    "normalized_score": emb_v,
                    "weight": emb_ratio,
                }
            )

        if "shallow" in sources and evidence_id in hand_rows:
            source_ranks.append(hand_rows[evidence_id].source_rank)
            hand_v = hand_norm.get(evidence_id, 0.0)
            score += hand_ratio * hand_v
            candidate_rows.append(
                {
                    "source": "shallow",
                    "source_rank": hand_rows[evidence_id].source_rank,
                    "raw_score": float(hand_rows[evidence_id].score),
                    "normalized_score": hand_v,
                    "weight": hand_ratio,
                }
            )

        prior_v = prior_norm.get(evidence_id, 0.0)
        score += prior_ratio * prior_v
        candidate_rows.append(
            {
                "source": "source_rank_prior",
                "source_rank": min(source_ranks) if source_ranks else 10**9,
                "raw_score": prior_v,
                "normalized_score": prior_v,
                "weight": prior_ratio,
            }
        )

        rows.append(
            {
                "evidence_id": evidence_id,
                "score": float(score),
                "fusion_score": float(score),
                "source_count": len(candidate_rows),
                "source_rank": min(source_ranks) if source_ranks else 10**9,
                "source_breakdown": candidate_rows,
                "reranker_variant": variant,
                "ce_weight": float(ce_ratio),
                "embedding_weight": float(emb_ratio),
                "hand_weight": float(hand_ratio),
                "rank_prior_weight": float(prior_ratio),
            }
        )

    rows.sort(key=lambda item: (-item["fusion_score"], item["source_rank"], item["evidence_id"]))
    return [item | {"rank": idx} for idx, item in enumerate(rows[:candidate_k], start=1)]


def rerank_claims(
    claim_ids: list[str],
    ce_pool: dict[str, dict[str, CandidateItem]],
    embedding_pool: dict[str, dict[str, CandidateItem]],
    hand_pool: dict[str, dict[str, CandidateItem]],
    variant: str,
    ce_weight: float,
    embedding_weight: float,
    hand_weight: float,
    prior_weight: float,
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    ranked: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claim_ids:
        ranked[claim_id] = score_candidate_rows(
            claim_id=claim_id,
            variant=variant,
            ce_pool=ce_pool,
            embedding_pool=embedding_pool,
            hand_pool=hand_pool,
            ce_weight=ce_weight,
            embedding_weight=embedding_weight,
            hand_weight=hand_weight,
            prior_weight=prior_weight,
            candidate_k=candidate_k,
        )
    return ranked


def evaluate_recall_at_k(
    claims: dict[str, Any],
    ranked: dict[str, list[dict[str, Any]]],
    ks: list[int],
    claim_subset: set[str] | None = None,
) -> dict[str, float]:
    ks = sorted(set(ks))
    totals = {k: {"macro": [], "tp": 0.0, "gold": 0.0, "hit_any": 0} for k in ks}
    claim_ids = sorted(claim_subset) if claim_subset is not None else sorted(claims)
    claims_with_gold = 0
    for claim_id in claim_ids:
        claim = claims.get(claim_id)
        if not claim:
            continue
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claims_with_gold += 1
        predicted = [row["evidence_id"] for row in ranked.get(claim_id, [])]
        for k in ks:
            hit = set(predicted[:k]) & gold
            tp = len(hit)
            totals[k]["macro"].append(tp / float(len(gold)))
            totals[k]["tp"] += float(tp)
            totals[k]["gold"] += float(len(gold))
            if tp > 0:
                totals[k]["hit_any"] += 1
    out = {
        "claims_with_evidence": float(claims_with_gold),
        "avg_candidate_count": float(sum(len(v) for v in ranked.values()) / claims_with_gold)
        if claims_with_gold
        else 0.0,
    }
    for k in ks:
        out[f"macro_recall@{k}"] = (
            float(sum(totals[k]["macro"]) / len(totals[k]["macro"])) if totals[k]["macro"] else 0.0
        )
        out[f"micro_recall@{k}"] = totals[k]["tp"] / totals[k]["gold"] if totals[k]["gold"] else 0.0
        out[f"hit_any@{k}"] = totals[k]["hit_any"] / float(claims_with_gold) if claims_with_gold else 0.0
    return out


def zero_cv_metrics(eval_ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in eval_ks:
        out[f"cv_macro_recall@{k}"] = 0.0
        out[f"cv_micro_recall@{k}"] = 0.0
        out[f"cv_hit_any@{k}"] = 0.0
    out["cv_claims_seen_avg"] = 0.0
    return out


def cv_metrics_for_variant(
    variant: str,
    eval_claims: dict[str, Any],
    folds: list[list[str]],
    ce_pool: dict[str, dict[str, CandidateItem]],
    embedding_pool: dict[str, dict[str, CandidateItem]],
    hand_pool: dict[str, dict[str, CandidateItem]],
    ce_weight: float,
    embedding_weight: float,
    hand_weight: float,
    prior_weight: float,
    eval_ks: list[int],
    candidate_k: int,
) -> dict[str, float]:
    fold_metrics: list[dict[str, float]] = []
    for valid_ids in folds:
        ranked = rerank_claims(
            claim_ids=valid_ids,
            ce_pool=ce_pool,
            embedding_pool=embedding_pool,
            hand_pool=hand_pool,
            variant=variant,
            ce_weight=ce_weight,
            embedding_weight=embedding_weight,
            hand_weight=hand_weight,
            prior_weight=prior_weight,
            candidate_k=candidate_k,
        )
        fold_metrics.append(evaluate_recall_at_k(eval_claims, ranked, eval_ks, set(valid_ids)))
    out: dict[str, float] = {}
    for k in eval_ks:
        out[f"cv_macro_recall@{k}"] = float(sum(metric[f"macro_recall@{k}"] for metric in fold_metrics) / len(fold_metrics))
        out[f"cv_micro_recall@{k}"] = float(sum(metric[f"micro_recall@{k}"] for metric in fold_metrics) / len(fold_metrics))
        out[f"cv_hit_any@{k}"] = float(sum(metric[f"hit_any@{k}"] for metric in fold_metrics) / len(fold_metrics))
    out["cv_claims_seen_avg"] = float(
        sum(metric.get("claims_with_evidence", 0.0) for metric in fold_metrics) / len(fold_metrics)
    )
    return out


def run_variants_for_mode(
    strict: bool,
    tuning_claims: dict[str, Any],
    eval_claims: dict[str, Any],
    tuning_ce_pool: dict[str, dict[str, CandidateItem]],
    tuning_embedding_pool: dict[str, dict[str, CandidateItem]],
    tuning_hand_pool: dict[str, dict[str, CandidateItem]],
    eval_ce_pool: dict[str, dict[str, CandidateItem]],
    eval_embedding_pool: dict[str, dict[str, CandidateItem]],
    eval_hand_pool: dict[str, dict[str, CandidateItem]],
    ce_weights: list[float],
    emb_weights: list[float],
    hand_weights: list[float],
    prior_weights: list[float],
    eval_ks: list[int],
    k_folds: int,
    fold_seed: int,
    candidate_k: int,
) -> list[dict[str, Any]]:
    variants = ["ce_only", "embedding_only", "ce_plus_shallow", "embedding_plus_shallow"]
    tuning_claim_ids = sorted(tuning_claims)
    folds = split_folds(tuning_claim_ids, k_folds, fold_seed) if strict else []

    eval_claim_ids = sorted(eval_claims)
    rows: list[dict[str, Any]] = []

    for variant in variants:
        if variant == "ce_only":
            for ce_weight in ce_weights:
                for prior_weight in prior_weights:
                    if ce_weight == 0.0 and prior_weight == 0.0:
                        continue
                    cv_stats = (
                        cv_metrics_for_variant(
                            variant=variant,
                            eval_claims=tuning_claims,
                            folds=folds,
                            ce_pool=tuning_ce_pool,
                            embedding_pool=tuning_embedding_pool,
                            hand_pool=tuning_hand_pool,
                            ce_weight=ce_weight,
                            embedding_weight=0.0,
                            hand_weight=0.0,
                            prior_weight=prior_weight,
                            eval_ks=eval_ks,
                            candidate_k=candidate_k,
                        )
                        if strict
                        else zero_cv_metrics(eval_ks)
                    )
                    dev_ranked = rerank_claims(
                        claim_ids=eval_claim_ids,
                        ce_pool=eval_ce_pool,
                        embedding_pool=eval_embedding_pool,
                        hand_pool=eval_hand_pool,
                        variant=variant,
                        ce_weight=ce_weight,
                        embedding_weight=0.0,
                        hand_weight=0.0,
                        prior_weight=prior_weight,
                        candidate_k=candidate_k,
                    )
                    dev_metrics = evaluate_recall_at_k(eval_claims, dev_ranked, eval_ks)
                    row = {
                        "variant": variant,
                        "candidate_k": candidate_k,
                        "status": "strict-candidate" if strict else "diagnostic-only",
                        "selection_basis": "train_kfold" if strict else "diagnostic",
                        "ce_weight": float(ce_weight),
                        "embedding_weight": 0.0,
                        "hand_weight": 0.0,
                        "rank_prior_weight": float(prior_weight),
                    }
                    row.update(cv_stats)
                    for k in eval_ks:
                        row[f"dev_macro_recall@{k}"] = dev_metrics[f"macro_recall@{k}"]
                        row[f"dev_micro_recall@{k}"] = dev_metrics[f"micro_recall@{k}"]
                        row[f"dev_hit_any@{k}"] = dev_metrics[f"hit_any@{k}"]
                    row["candidate_count_avg"] = dev_metrics["avg_candidate_count"]
                    rows.append(row)

        elif variant == "embedding_only":
            for emb_weight in emb_weights:
                for prior_weight in prior_weights:
                    if emb_weight == 0.0 and prior_weight == 0.0:
                        continue
                    cv_stats = (
                        cv_metrics_for_variant(
                            variant=variant,
                            eval_claims=tuning_claims,
                            folds=folds,
                            ce_pool=tuning_ce_pool,
                            embedding_pool=tuning_embedding_pool,
                            hand_pool=tuning_hand_pool,
                            ce_weight=0.0,
                            embedding_weight=emb_weight,
                            hand_weight=0.0,
                            prior_weight=prior_weight,
                            eval_ks=eval_ks,
                            candidate_k=candidate_k,
                        )
                        if strict
                        else zero_cv_metrics(eval_ks)
                    )
                    dev_ranked = rerank_claims(
                        claim_ids=eval_claim_ids,
                        ce_pool=eval_ce_pool,
                        embedding_pool=eval_embedding_pool,
                        hand_pool=eval_hand_pool,
                        variant=variant,
                        ce_weight=0.0,
                        embedding_weight=emb_weight,
                        hand_weight=0.0,
                        prior_weight=prior_weight,
                        candidate_k=candidate_k,
                    )
                    dev_metrics = evaluate_recall_at_k(eval_claims, dev_ranked, eval_ks)
                    row = {
                        "variant": variant,
                        "candidate_k": candidate_k,
                        "status": "strict-candidate" if strict else "diagnostic-only",
                        "selection_basis": "train_kfold" if strict else "diagnostic",
                        "ce_weight": 0.0,
                        "embedding_weight": float(emb_weight),
                        "hand_weight": 0.0,
                        "rank_prior_weight": float(prior_weight),
                    }
                    row.update(cv_stats)
                    for k in eval_ks:
                        row[f"dev_macro_recall@{k}"] = dev_metrics[f"macro_recall@{k}"]
                        row[f"dev_micro_recall@{k}"] = dev_metrics[f"micro_recall@{k}"]
                        row[f"dev_hit_any@{k}"] = dev_metrics[f"hit_any@{k}"]
                    row["candidate_count_avg"] = dev_metrics["avg_candidate_count"]
                    rows.append(row)

        elif variant == "ce_plus_shallow":
            for ce_weight in ce_weights:
                for hand_weight in hand_weights:
                    for prior_weight in prior_weights:
                        if ce_weight == 0.0 and hand_weight == 0.0 and prior_weight == 0.0:
                            continue
                        cv_stats = (
                            cv_metrics_for_variant(
                                variant=variant,
                                eval_claims=tuning_claims,
                                folds=folds,
                                ce_pool=tuning_ce_pool,
                                embedding_pool=tuning_embedding_pool,
                                hand_pool=tuning_hand_pool,
                                ce_weight=ce_weight,
                                embedding_weight=0.0,
                                hand_weight=hand_weight,
                                prior_weight=prior_weight,
                                eval_ks=eval_ks,
                                candidate_k=candidate_k,
                            )
                            if strict
                            else zero_cv_metrics(eval_ks)
                        )
                        dev_ranked = rerank_claims(
                            claim_ids=eval_claim_ids,
                            ce_pool=eval_ce_pool,
                            embedding_pool=eval_embedding_pool,
                            hand_pool=eval_hand_pool,
                            variant=variant,
                            ce_weight=ce_weight,
                            embedding_weight=0.0,
                            hand_weight=hand_weight,
                            prior_weight=prior_weight,
                            candidate_k=candidate_k,
                        )
                        dev_metrics = evaluate_recall_at_k(eval_claims, dev_ranked, eval_ks)
                        row = {
                            "variant": variant,
                            "candidate_k": candidate_k,
                            "status": "strict-candidate" if strict else "diagnostic-only",
                            "selection_basis": "train_kfold" if strict else "diagnostic",
                            "ce_weight": float(ce_weight),
                            "embedding_weight": 0.0,
                            "hand_weight": float(hand_weight),
                            "rank_prior_weight": float(prior_weight),
                        }
                        row.update(cv_stats)
                        for k in eval_ks:
                            row[f"dev_macro_recall@{k}"] = dev_metrics[f"macro_recall@{k}"]
                            row[f"dev_micro_recall@{k}"] = dev_metrics[f"micro_recall@{k}"]
                            row[f"dev_hit_any@{k}"] = dev_metrics[f"hit_any@{k}"]
                        row["candidate_count_avg"] = dev_metrics["avg_candidate_count"]
                        rows.append(row)

        else:
            for emb_weight in emb_weights:
                for hand_weight in hand_weights:
                    for prior_weight in prior_weights:
                        if emb_weight == 0.0 and hand_weight == 0.0 and prior_weight == 0.0:
                            continue
                        cv_stats = (
                            cv_metrics_for_variant(
                                variant=variant,
                                eval_claims=tuning_claims,
                                folds=folds,
                                ce_pool=tuning_ce_pool,
                                embedding_pool=tuning_embedding_pool,
                                hand_pool=tuning_hand_pool,
                                ce_weight=0.0,
                                embedding_weight=emb_weight,
                                hand_weight=hand_weight,
                                prior_weight=prior_weight,
                                eval_ks=eval_ks,
                                candidate_k=candidate_k,
                            )
                            if strict
                            else zero_cv_metrics(eval_ks)
                        )
                        dev_ranked = rerank_claims(
                            claim_ids=eval_claim_ids,
                            ce_pool=eval_ce_pool,
                            embedding_pool=eval_embedding_pool,
                            hand_pool=eval_hand_pool,
                            variant=variant,
                            ce_weight=0.0,
                            embedding_weight=emb_weight,
                            hand_weight=hand_weight,
                            prior_weight=prior_weight,
                            candidate_k=candidate_k,
                        )
                        dev_metrics = evaluate_recall_at_k(eval_claims, dev_ranked, eval_ks)
                        row = {
                            "variant": variant,
                            "candidate_k": candidate_k,
                            "status": "strict-candidate" if strict else "diagnostic-only",
                            "selection_basis": "train_kfold" if strict else "diagnostic",
                            "ce_weight": 0.0,
                            "embedding_weight": float(emb_weight),
                            "hand_weight": float(hand_weight),
                            "rank_prior_weight": float(prior_weight),
                        }
                        row.update(cv_stats)
                        for k in eval_ks:
                            row[f"dev_macro_recall@{k}"] = dev_metrics[f"macro_recall@{k}"]
                            row[f"dev_micro_recall@{k}"] = dev_metrics[f"micro_recall@{k}"]
                            row[f"dev_hit_any@{k}"] = dev_metrics[f"hit_any@{k}"]
                        row["candidate_count_avg"] = dev_metrics["avg_candidate_count"]
                        rows.append(row)

    return rows


def score_for_selection(row: dict[str, Any], strict: bool) -> tuple[float, ...]:
    if strict:
        return (
            row.get("cv_macro_recall@1", 0.0),
            row.get("cv_macro_recall@3", 0.0),
            row.get("cv_macro_recall@5", 0.0),
            row.get("cv_macro_recall@10", 0.0),
            row.get("cv_macro_recall@64", 0.0),
            row.get("cv_macro_recall@500", 0.0),
        )
    return (
        row.get("dev_macro_recall@1", 0.0),
        row.get("dev_macro_recall@3", 0.0),
        row.get("dev_macro_recall@5", 0.0),
        row.get("dev_macro_recall@10", 0.0),
        row.get("dev_macro_recall@64", 0.0),
        row.get("dev_macro_recall@500", 0.0),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    strict: bool,
    elapsed: float,
    forbidden_hits: dict[str, list[str]],
) -> None:
    lines = [
        "# Top3 Strict Fusion K-Fold Reranker",
        "",
        "## Policy",
        f"- Mode: `{'strict-candidate' if strict else 'diagnostic'}`",
        "- Variants tested: `ce_only`, `embedding_only`, `ce_plus_shallow`, `embedding_plus_shallow`.",
        "- Evaluation targets include R@1, R@3, R@5, R@10 and extended cuts R@64, R@500.",
        f"- Candidate budget: `top{args.candidate_k}`.",
        f"- Eval cuts: `{parse_k_list(args.eval_k)}`.",
        "",
        "## Inputs",
        f"- Train claims: `{args.train_claims}`",
        f"- Dev claims: `{args.dev_claims}`",
        f"- Train CE pool: `{args.train_ce_pool}`",
        f"- Dev CE pool: `{args.dev_ce_pool}`",
        f"- Train embedding pool: `{args.train_embedding_pool}`",
        f"- Dev embedding pool: `{args.dev_embedding_pool}`",
        f"- Train hand pool: `{args.train_hand_pool}`",
        f"- Dev hand pool: `{args.dev_hand_pool}`",
        "",
        "## Selected Variant",
        f"- Variant: `{best['variant']}`",
        f"- CV@1: `{best.get('cv_macro_recall@1', 0.0):.6f}`",
        f"- CV@3: `{best.get('cv_macro_recall@3', 0.0):.6f}`",
        f"- CV@5: `{best.get('cv_macro_recall@5', 0.0):.6f}`",
        f"- CV@10: `{best.get('cv_macro_recall@10', 0.0):.6f}`",
        f"- Dev@1: `{best.get('dev_macro_recall@1', 0.0):.6f}`",
        f"- Dev@3: `{best.get('dev_macro_recall@3', 0.0):.6f}`",
        f"- Dev@5: `{best.get('dev_macro_recall@5', 0.0):.6f}`",
        f"- Dev@10: `{best.get('dev_macro_recall@10', 0.0):.6f}`",
        "",
        "## Top Records by Selection Score",
        "| variant | ce_w | emb_w | hand_w | prior_w | cv@1 | cv@3 | cv@5 | cv@10 | dev@1 | dev@3 | dev@5 | dev@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[: min(len(rows), 15)]:
        lines.append(
            f"| {row['variant']} | {row.get('ce_weight', 0.0):.3f} | {row.get('embedding_weight', 0.0):.3f} | "
            f"{row.get('hand_weight', 0.0):.3f} | {row.get('rank_prior_weight', 0.0):.3f} | "
            f"{row.get('cv_macro_recall@1', 0.0):.4f} | {row.get('cv_macro_recall@3', 0.0):.4f} | "
            f"{row.get('cv_macro_recall@5', 0.0):.4f} | {row.get('cv_macro_recall@10', 0.0):.4f} | "
            f"{row.get('dev_macro_recall@1', 0.0):.4f} | {row.get('dev_macro_recall@3', 0.0):.4f} | "
            f"{row.get('dev_macro_recall@5', 0.0):.4f} | {row.get('dev_macro_recall@10', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Forbidden-token scan",
            "- Passed" if not forbidden_hits else f"- Found: {json.dumps(forbidden_hits, ensure_ascii=False)}",
            "",
            "## Risk notes",
            f"- Strict mode selected: `{strict}`.",
            "- CE-only and CE+shallow depend on train CE pool coverage; if train CE claims are sparse, fusion scores can drift toward prior-only behavior.",
            "- Source-rank prior is additional unsupervised signal (`1/(60+min_source_rank)` normalized per claim).",
            "- Diagnostic mode only reports smoke/dev-subset metrics and should not be promoted as final selection.",
            "",
            "## Runtime",
            f"- Wall seconds: `{elapsed:.3f}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    eval_ks = parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        raise SystemExit("--candidate-k must be >= max eval-k.")
    if args.smoke:
        if args.smoke_claims <= 0:
            raise SystemExit("--smoke-claims must be > 0.")

    for required in (
        args.train_claims,
        args.dev_claims,
        args.evidence,
        args.train_ce_pool,
        args.dev_ce_pool,
        args.train_embedding_pool,
        args.dev_embedding_pool,
        args.train_hand_pool,
        args.dev_hand_pool,
    ):
        if not required.exists():
            raise SystemExit(f"Missing required input: {required}")

    train_claims = load_claims(args.train_claims)
    dev_claims_all = load_claims(args.dev_claims)
    _ = load_json(args.evidence)

    ce_weights = parse_weight_grid(args.ce_weight_grid, "ce-weight-grid")
    emb_weights = parse_weight_grid(args.embedding_weight_grid, "embedding-weight-grid")
    hand_weights = parse_weight_grid(args.shallow_weight_grid, "shallow-weight-grid")
    prior_weights = parse_weight_grid(args.rank_prior_grid, "rank-prior-grid")

    strict_mode = not args.smoke
    if strict_mode:
        tuning_claims = train_claims
        tuning_ce_pool = parse_ce_pool(args.train_ce_pool, tuning_claims, args.candidate_k)
        tuning_embedding_pool = parse_embedding_pool(args.train_embedding_pool, tuning_claims, args.candidate_k)
        tuning_hand_pool = parse_hand_pool(args.train_hand_pool, tuning_claims, args.candidate_k)
    else:
        tuning_claims = dev_claims_all
        tuning_ce_pool = parse_ce_pool(args.dev_ce_pool, tuning_claims, args.candidate_k)
        tuning_embedding_pool = parse_embedding_pool(args.dev_embedding_pool, tuning_claims, args.candidate_k)
        tuning_hand_pool = parse_hand_pool(args.dev_hand_pool, tuning_claims, args.candidate_k)

    if args.smoke:
        smoke_ids = sorted(dev_claims_all)[: args.smoke_claims]
        if not smoke_ids:
            raise SystemExit("Smoke mode has zero claims selected from dev split.")
        eval_claims = {claim_id: dev_claims_all[claim_id] for claim_id in smoke_ids}
    else:
        eval_claims = dev_claims_all

    eval_ce_pool = parse_ce_pool(args.dev_ce_pool, eval_claims if args.smoke else dev_claims_all, args.candidate_k)
    eval_embedding_pool = parse_embedding_pool(
        args.dev_embedding_pool, eval_claims if args.smoke else dev_claims_all, args.candidate_k
    )
    eval_hand_pool = parse_hand_pool(
        args.dev_hand_pool, eval_claims if args.smoke else dev_claims_all, args.candidate_k
    )

    rows = run_variants_for_mode(
        strict=strict_mode,
        tuning_claims=tuning_claims,
        eval_claims=eval_claims,
        tuning_ce_pool=tuning_ce_pool,
        tuning_embedding_pool=tuning_embedding_pool,
        tuning_hand_pool=tuning_hand_pool,
        eval_ce_pool=eval_ce_pool,
        eval_embedding_pool=eval_embedding_pool,
        eval_hand_pool=eval_hand_pool,
        ce_weights=ce_weights,
        emb_weights=emb_weights,
        hand_weights=hand_weights,
        prior_weights=prior_weights,
        eval_ks=eval_ks,
        k_folds=args.k_folds,
        fold_seed=args.fold_seed,
        candidate_k=args.candidate_k,
    )

    rows.sort(key=lambda row: score_for_selection(row, strict_mode), reverse=True)
    best = rows[0]

    if strict_mode:
        final_eval_claims = dev_claims_all
        final_eval_claim_ids = sorted(final_eval_claims)
    else:
        final_eval_claims = eval_claims
        final_eval_claim_ids = sorted(final_eval_claims)

    best_ranked = rerank_claims(
        claim_ids=final_eval_claim_ids,
        ce_pool=eval_ce_pool if not strict_mode else parse_ce_pool(args.dev_ce_pool, final_eval_claims, args.candidate_k),
        embedding_pool=(
            eval_embedding_pool
            if not strict_mode
            else parse_embedding_pool(args.dev_embedding_pool, final_eval_claims, args.candidate_k)
        ),
        hand_pool=eval_hand_pool if not strict_mode else parse_hand_pool(args.dev_hand_pool, final_eval_claims, args.candidate_k),
        variant=best["variant"],
        ce_weight=float(best.get("ce_weight", 0.0)),
        embedding_weight=float(best.get("embedding_weight", 0.0)),
        hand_weight=float(best.get("hand_weight", 0.0)),
        prior_weight=float(best.get("rank_prior_weight", 0.0)),
        candidate_k=args.candidate_k,
    )

    final_eval_metrics = evaluate_recall_at_k(final_eval_claims, best_ranked, eval_ks)
    best["dev_macro_recall@1"] = final_eval_metrics["macro_recall@1"] if eval_ks else best.get("dev_macro_recall@1", 0.0)
    best["dev_macro_recall@3"] = final_eval_metrics["macro_recall@3"] if eval_ks else best.get("dev_macro_recall@3", 0.0)
    best["dev_macro_recall@5"] = final_eval_metrics["macro_recall@5"] if eval_ks else best.get("dev_macro_recall@5", 0.0)
    best["dev_macro_recall@10"] = final_eval_metrics["macro_recall@10"] if eval_ks else best.get("dev_macro_recall@10", 0.0)
    best["candidate_count_avg"] = final_eval_metrics["avg_candidate_count"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    summary_json = args.output_dir / f"{args.run_id}_summary.json"
    summary_csv = args.output_dir / f"{args.run_id}_summary.csv"
    best_candidates_path = args.output_dir / f"{args.run_id}_best_{'smoke_' if args.smoke else ''}dev_top{args.candidate_k}_candidates.json"
    report_path = args.report_dir / f"{args.run_id}_report.md"

    train_pool_coverage = {
        "tuning_claims": len(tuning_claims),
        "tuning_claims_with_ce": len(tuning_ce_pool),
        "tuning_claims_with_embedding": len(tuning_embedding_pool),
        "tuning_claims_with_hand": len(tuning_hand_pool),
    }
    strict_risk = {
        "smoke": args.smoke,
        "tuning_claims_from_train": strict_mode,
        "train_ce_coverage_for_tuning": train_pool_coverage["tuning_claims_with_ce"] > 0,
        "train_embedding_coverage_for_tuning": train_pool_coverage["tuning_claims_with_embedding"] > 0,
        "train_hand_coverage_for_tuning": train_pool_coverage["tuning_claims_with_hand"] > 0,
        "train_ce_path": str(args.train_ce_pool),
    }

    write_json(best_candidates_path, best_ranked)
    write_json(
        summary_json,
        {
            "run_id": args.run_id,
            "status": "strict-candidate" if strict_mode else "diagnostic-only",
            "selection_basis": "train_kfold" if strict_mode else "diagnostic",
            "primary_metric": "macro_recall@3",
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "strict_mode": strict_mode,
            "strict_risk": strict_risk,
            "inputs": {
                "train_claims": str(args.train_claims),
                "dev_claims": str(args.dev_claims),
                "train_ce_pool": str(args.train_ce_pool),
                "train_embedding_pool": str(args.train_embedding_pool),
                "train_hand_pool": str(args.train_hand_pool),
                "dev_ce_pool": str(args.dev_ce_pool),
                "dev_embedding_pool": str(args.dev_embedding_pool),
                "dev_hand_pool": str(args.dev_hand_pool),
            },
            "weights": {
                "ce_grid": ce_weights,
                "embedding_grid": emb_weights,
                "shallow_grid": hand_weights,
                "prior_grid": prior_weights,
            },
            "best_variant": best,
            "best_candidates_file": str(best_candidates_path),
            "rows": rows,
            "runtime_seconds": round(time.perf_counter() - start, 3),
            "forbidden_hits": find_forbidden_tokens(
                [str(p) for p in (args.train_claims, args.dev_claims, args.evidence, args.train_ce_pool, args.dev_ce_pool,
                                  args.train_embedding_pool, args.dev_embedding_pool, args.train_hand_pool, args.dev_hand_pool,
                                  args.output_dir, args.report_dir)]
            ),
        },
    )
    write_csv(rows=rows, path=summary_csv)

    make_report(
        path=report_path,
        args=args,
        rows=rows,
        best=best,
        strict=strict_mode,
        elapsed=time.perf_counter() - start,
        forbidden_hits=find_forbidden_tokens(
            [
                str(p) for p in (args.train_claims, args.dev_claims, args.evidence, args.train_ce_pool, args.dev_ce_pool,
                                 args.train_embedding_pool, args.dev_embedding_pool, args.train_hand_pool, args.dev_hand_pool,
                                 args.output_dir, args.report_dir, " ".join(sys.argv))
            ]
        ),
    )

    print(f"wrote {summary_json}")
    print(f"wrote {summary_csv}")
    print(f"wrote {best_candidates_path}")
    print(f"wrote {report_path}")
    print(
        f"best: variant={best['variant']} cv@3={best.get('cv_macro_recall@3', 0.0):.6f} "
        f"dev@3={best.get('dev_macro_recall@3', 0.0):.6f}"
    )


if __name__ == "__main__":
    main()
