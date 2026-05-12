#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.tools.common import find_forbidden_tokens, load_json, write_json  # noqa: E402


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_TRAIN_SPARSE = Path(
    "round18/outputs/o_sparse/o_s9_union_gate_bm25_char_train/"
    "train_full_train_o_s9_union_gate_bm25_char_rrf_top1000_candidates.json"
)
DEFAULT_DEV_SPARSE = Path(
    "round18/outputs/o_sparse/o_s9_union_gate_bm25_char_dev/"
    "dev_full_dev_o_s9_union_gate_bm25_char_rrf_top1000_candidates.json"
)
DEFAULT_TRAIN_CE = Path(
    "round18/outputs/o_dense/o_d3x_bm25_char_train_top500/"
    "train_full_train_o_d3x_bm25_char_train_top500_strict_top500_candidates.json"
)
DEFAULT_DEV_CE = Path(
    "round18/outputs/o_dense/o_d3x_bm25_char_dev_top500/"
    "dev_full_dev_o_d3x_bm25_char_dev_top500_strict_top500_candidates.json"
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
DEFAULT_OUTPUT_DIR = Path("round18/reports/top3_train_kfold_fusion_with_hand")
DEFAULT_RUN_ID = "top3_train_kfold_fusion_with_hand"
DEFAULT_EVAL_K = "1,3,5,10,64,100,500"


COMPONENTS = (
    "ce_score",
    "embedding_score",
    "hand_score",
    "source_rank",
    "ce_rank",
    "embedding_rank",
    "hand_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only top3 evidence fusion over CE, embedding, and source-rank leaves. "
            "Weights are selected on train claim folds; dev is confirmation only."
        )
    )
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--train-sparse", type=Path, default=DEFAULT_TRAIN_SPARSE)
    parser.add_argument("--dev-sparse", type=Path, default=DEFAULT_DEV_SPARSE)
    parser.add_argument("--train-ce", type=Path, default=DEFAULT_TRAIN_CE)
    parser.add_argument("--dev-ce", type=Path, default=DEFAULT_DEV_CE)
    parser.add_argument("--train-embedding", type=Path, default=DEFAULT_TRAIN_EMBEDDING)
    parser.add_argument("--dev-embedding", type=Path, default=DEFAULT_DEV_EMBEDDING)
    parser.add_argument("--train-hand", type=Path, default=DEFAULT_TRAIN_HAND)
    parser.add_argument("--dev-hand", type=Path, default=DEFAULT_DEV_HAND)
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--eval-k", default=DEFAULT_EVAL_K)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1818)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--eval-k must contain at least one integer.")
    return sorted(set(values))


def coerce_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def coerce_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def load_claims(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Claims must be a JSON object: {path}")
    return payload


def parse_pool(path: Path, claims: dict[str, Any], candidate_k: int, score_name: str) -> dict[str, dict[str, dict[str, float]]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate pool must be a JSON object: {path}")
    parsed: dict[str, dict[str, dict[str, float]]] = {}
    for claim_id in claims:
        rows = payload.get(claim_id, [])
        if not isinstance(rows, list):
            continue
        seen: set[str] = set()
        claim_rows: dict[str, dict[str, float]] = {}
        for pos, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            rank = coerce_int(row.get("rank"), pos)
            score = coerce_float(
                row.get(
                    score_name,
                    row.get("reranker_score", row.get("score", row.get("source_score", 0.0))),
                )
            )
            source_rank = coerce_int(row.get("source_rank"), rank)
            source_score = coerce_float(row.get("source_score", row.get("score", 0.0)))
            claim_rows[evidence_id] = {
                "rank": float(rank if rank > 0 else pos),
                "score": float(score),
                "source_rank": float(source_rank if source_rank > 0 else pos),
                "source_score": float(source_score),
            }
            if len(claim_rows) >= candidate_k:
                break
        parsed[claim_id] = claim_rows
    return parsed


def minmax_by_claim(values: dict[str, float]) -> dict[str, float]:
    present = list(values.values())
    if not present:
        return {}
    lo = min(present)
    hi = max(present)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def build_features(
    claims: dict[str, Any],
    sparse: dict[str, dict[str, dict[str, float]]],
    ce: dict[str, dict[str, dict[str, float]]],
    embedding: dict[str, dict[str, dict[str, float]]],
    hand: dict[str, dict[str, dict[str, float]]],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    by_claim: dict[str, list[dict[str, Any]]] = {}
    missing_rank = float(candidate_k + 1)
    for claim_id in claims:
        evidence_ids = (
            set(sparse.get(claim_id, {}))
            | set(ce.get(claim_id, {}))
            | set(embedding.get(claim_id, {}))
            | set(hand.get(claim_id, {}))
        )
        ce_scores = {eid: ce.get(claim_id, {}).get(eid, {}).get("score", 0.0) for eid in evidence_ids if eid in ce.get(claim_id, {})}
        emb_scores = {
            eid: embedding.get(claim_id, {}).get(eid, {}).get("score", 0.0)
            for eid in evidence_ids
            if eid in embedding.get(claim_id, {})
        }
        hand_scores = {
            eid: hand.get(claim_id, {}).get(eid, {}).get("score", 0.0)
            for eid in evidence_ids
            if eid in hand.get(claim_id, {})
        }
        sparse_scores = {
            eid: sparse.get(claim_id, {}).get(eid, {}).get("source_score", 0.0)
            for eid in evidence_ids
            if eid in sparse.get(claim_id, {})
        }
        ce_norm = minmax_by_claim(ce_scores)
        emb_norm = minmax_by_claim(emb_scores)
        hand_norm = minmax_by_claim(hand_scores)
        sparse_norm = minmax_by_claim(sparse_scores)
        rows: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            sparse_meta = sparse.get(claim_id, {}).get(evidence_id, {})
            ce_meta = ce.get(claim_id, {}).get(evidence_id, {})
            emb_meta = embedding.get(claim_id, {}).get(evidence_id, {})
            hand_meta = hand.get(claim_id, {}).get(evidence_id, {})
            source_rank = sparse_meta.get("rank", ce_meta.get("source_rank", emb_meta.get("source_rank", missing_rank)))
            ce_rank = ce_meta.get("rank", missing_rank)
            emb_rank = emb_meta.get("rank", missing_rank)
            hand_rank = hand_meta.get("rank", missing_rank)
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "features": {
                        "ce_score": float(ce_norm.get(evidence_id, 0.0)),
                        "embedding_score": float(emb_norm.get(evidence_id, 0.0)),
                        "hand_score": float(hand_norm.get(evidence_id, 0.0)),
                        "source_score": float(sparse_norm.get(evidence_id, 0.0)),
                        "source_rank": 1.0 / (60.0 + float(source_rank)),
                        "ce_rank": 1.0 / (60.0 + float(ce_rank)),
                        "embedding_rank": 1.0 / (60.0 + float(emb_rank)),
                        "hand_rank": 1.0 / (60.0 + float(hand_rank)),
                    },
                    "trace": {
                        "source_rank": int(source_rank),
                        "ce_rank": int(ce_rank),
                        "embedding_rank": int(emb_rank),
                        "hand_rank": int(hand_rank),
                        "raw_ce_score": float(ce_meta.get("score", 0.0)),
                        "raw_embedding_score": float(emb_meta.get("score", 0.0)),
                        "raw_hand_score": float(hand_meta.get("score", 0.0)),
                    },
                }
            )
        by_claim[claim_id] = rows
    return by_claim


def split_folds(claim_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(claim_ids)
    rng.shuffle(shuffled)
    return [shuffled[idx::n_folds] for idx in range(n_folds)]


def weight_grid(total: int = 4) -> list[dict[str, float]]:
    weights: list[dict[str, float]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            raw = prefix + [remaining]
            if sum(raw) == 0:
                return
            weights.append({name: value / float(total) for name, value in zip(COMPONENTS, raw)})
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], total, len(COMPONENTS))
    weights.extend(
        [
            {"ce_score": 1.0},
            {"embedding_score": 1.0},
            {"hand_score": 1.0},
            {"source_rank": 1.0},
            {"ce_rank": 1.0},
            {"embedding_rank": 1.0},
            {"hand_rank": 1.0},
        ]
    )
    unique: dict[tuple[float, ...], dict[str, float]] = {}
    for row in weights:
        key = tuple(row.get(component, 0.0) for component in COMPONENTS)
        unique[key] = {component: row.get(component, 0.0) for component in COMPONENTS}
    return list(unique.values())


def score_rows(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = sum(float(weights.get(name, 0.0)) * float(row["features"].get(name, 0.0)) for name in COMPONENTS)
        out = {
            "evidence_id": row["evidence_id"],
            "score": float(score),
            "fusion_score": float(score),
            "component_features": row["features"],
            "trace": row["trace"],
        }
        scored.append(out)
    scored.sort(
        key=lambda item: (
            -item["fusion_score"],
            item["trace"].get("ce_rank", 10**9),
            item["trace"].get("hand_rank", 10**9),
            item["trace"].get("source_rank", 10**9),
            item["trace"].get("embedding_rank", 10**9),
            item["evidence_id"],
        )
    )
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank
        item["fusion_weights"] = weights
    return scored


def rank_all(features: dict[str, list[dict[str, Any]]], weights: dict[str, float], candidate_k: int) -> dict[str, list[dict[str, Any]]]:
    return {claim_id: score_rows(rows, weights)[:candidate_k] for claim_id, rows in features.items()}


def evaluate(
    claims: dict[str, Any],
    ranked: dict[str, list[dict[str, Any]]],
    ks: list[int],
    claim_subset: set[str] | None = None,
) -> dict[str, float]:
    claim_ids = sorted(claim_subset) if claim_subset is not None else sorted(claims)
    out: dict[str, float] = {"claims_with_evidence": 0.0}
    per_k = {
        k: {"macro": [], "tp": 0.0, "gold": 0.0, "hit": 0.0, "f": []}
        for k in ks
    }
    for claim_id in claim_ids:
        claim = claims.get(claim_id)
        if not claim:
            continue
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        out["claims_with_evidence"] += 1.0
        ids = [row["evidence_id"] for row in ranked.get(claim_id, [])]
        for k in ks:
            pred_list = ids[:k]
            pred = set(pred_list)
            inter = gold & pred
            tp = float(len(inter))
            precision = tp / float(len(pred_list)) if pred_list else 0.0
            recall = tp / float(len(gold)) if gold else 0.0
            f_score = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            per_k[k]["macro"].append(recall)
            per_k[k]["tp"] += tp
            per_k[k]["gold"] += float(len(gold))
            per_k[k]["hit"] += 1.0 if tp > 0 else 0.0
            per_k[k]["f"].append(f_score)
    count = out["claims_with_evidence"]
    for k in ks:
        macro_vals = per_k[k]["macro"]
        f_vals = per_k[k]["f"]
        out[f"macro_recall@{k}"] = float(sum(macro_vals) / len(macro_vals)) if macro_vals else 0.0
        out[f"micro_recall@{k}"] = per_k[k]["tp"] / per_k[k]["gold"] if per_k[k]["gold"] else 0.0
        out[f"hit_any@{k}"] = per_k[k]["hit"] / count if count else 0.0
        out[f"evidence_f@{k}"] = float(sum(f_vals) / len(f_vals)) if f_vals else 0.0
    return out


def evaluate_config_cv(
    weights: dict[str, float],
    train_claims: dict[str, Any],
    train_features: dict[str, list[dict[str, Any]]],
    folds: list[list[str]],
    eval_ks: list[int],
    candidate_k: int,
) -> dict[str, float]:
    fold_metrics: list[dict[str, float]] = []
    for valid_ids in folds:
        valid_set = set(valid_ids)
        ranked = rank_all(
            {claim_id: train_features[claim_id] for claim_id in valid_set if claim_id in train_features},
            weights,
            candidate_k,
        )
        fold_metrics.append(evaluate(train_claims, ranked, eval_ks, valid_set))
    out: dict[str, float] = {}
    for key in fold_metrics[0]:
        out[f"cv_{key}"] = float(sum(row[key] for row in fold_metrics) / len(fold_metrics))
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    eval_ks = parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        raise SystemExit("--candidate-k must be >= max eval-k.")
    if args.k_folds < 2:
        raise SystemExit("--k-folds must be >= 2.")
    for path in (
        args.train_claims,
        args.dev_claims,
        args.train_sparse,
        args.dev_sparse,
        args.train_ce,
        args.dev_ce,
        args.train_embedding,
        args.dev_embedding,
        args.train_hand,
        args.dev_hand,
    ):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    train_claims = load_claims(args.train_claims)
    dev_claims = load_claims(args.dev_claims)
    train_sparse = parse_pool(args.train_sparse, train_claims, args.candidate_k, "source_score")
    dev_sparse = parse_pool(args.dev_sparse, dev_claims, args.candidate_k, "source_score")
    train_ce = parse_pool(args.train_ce, train_claims, args.candidate_k, "reranker_score")
    dev_ce = parse_pool(args.dev_ce, dev_claims, args.candidate_k, "reranker_score")
    train_embedding = parse_pool(args.train_embedding, train_claims, args.candidate_k, "embedding_score")
    dev_embedding = parse_pool(args.dev_embedding, dev_claims, args.candidate_k, "embedding_score")
    train_hand = parse_pool(args.train_hand, train_claims, args.candidate_k, "score")
    dev_hand = parse_pool(args.dev_hand, dev_claims, args.candidate_k, "score")

    train_features = build_features(train_claims, train_sparse, train_ce, train_embedding, train_hand, args.candidate_k)
    dev_features = build_features(dev_claims, dev_sparse, dev_ce, dev_embedding, dev_hand, args.candidate_k)
    folds = split_folds(list(train_features), args.k_folds, args.fold_seed)

    rows: list[dict[str, Any]] = []
    for weights in weight_grid():
        row: dict[str, Any] = {
            "status": "strict-candidate",
            "selection_basis": "train_kfold",
            **{f"w_{name}": value for name, value in weights.items()},
        }
        row.update(evaluate_config_cv(weights, train_claims, train_features, folds, eval_ks, args.candidate_k))
        dev_ranked = rank_all(dev_features, weights, args.candidate_k)
        dev_metrics = evaluate(dev_claims, dev_ranked, eval_ks)
        row.update({f"dev_{key}": value for key, value in dev_metrics.items()})
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get("cv_evidence_f@3", 0.0),
            row.get("cv_macro_recall@3", 0.0),
            row.get("cv_evidence_f@5", 0.0),
            row.get("cv_macro_recall@10", 0.0),
        ),
        reverse=True,
    )
    best = rows[0]
    best_weights = {name: float(best[f"w_{name}"]) for name in COMPONENTS}
    best_train_ranked = rank_all(train_features, best_weights, args.candidate_k)
    best_dev_ranked = rank_all(dev_features, best_weights, args.candidate_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / f"{args.run_id}_summary.json"
    summary_csv = args.output_dir / f"{args.run_id}_grid.csv"
    best_train_candidates_path = args.output_dir / f"{args.run_id}_best_train_top{args.candidate_k}_candidates.json"
    best_dev_candidates_path = args.output_dir / f"{args.run_id}_best_dev_top{args.candidate_k}_candidates.json"
    report_path = args.output_dir / f"{args.run_id}_report.md"

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.train_sparse),
            str(args.dev_sparse),
            str(args.train_ce),
            str(args.dev_ce),
            str(args.train_embedding),
            str(args.dev_embedding),
            str(args.train_hand),
            str(args.dev_hand),
            str(args.output_dir),
            " ".join(sys.argv),
        ]
    )

    write_json(best_train_candidates_path, best_train_ranked)
    write_json(best_dev_candidates_path, best_dev_ranked)
    write_csv(rows, summary_csv)
    payload = {
        "run_id": args.run_id,
        "status": "strict-candidate",
        "selection_basis": "train_kfold",
        "primary_metric": "evidence_f@3",
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "components": list(COMPONENTS),
        "inputs": {
            "train_sparse": str(args.train_sparse),
            "dev_sparse": str(args.dev_sparse),
            "train_ce": str(args.train_ce),
            "dev_ce": str(args.dev_ce),
            "train_embedding": str(args.train_embedding),
            "dev_embedding": str(args.dev_embedding),
            "train_hand": str(args.train_hand),
            "dev_hand": str(args.dev_hand),
        },
        "best": best,
        "best_weights": best_weights,
        "best_train_candidates_file": str(best_train_candidates_path),
        "best_dev_candidates_file": str(best_dev_candidates_path),
        "top10": rows[:10],
        "forbidden_input_scan": forbidden_hits,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    write_json(summary_json, payload)

    lines = [
        "# Top3 Train-KFold Fusion",
        "",
        "Goal: select final evidence top3 ranking weights from train folds, then confirm once on dev.",
        "",
        "## Selected Weights",
        "",
        "| component | weight |",
        "|---|---:|",
    ]
    for name, value in best_weights.items():
        lines.append(f"| {name} | {value:.2f} |")
    lines.extend(
        [
            "",
            "## Best Metrics",
            "",
            f"- CV evidence F@3: `{best.get('cv_evidence_f@3', 0.0):.6f}`",
            f"- CV macro recall@3: `{best.get('cv_macro_recall@3', 0.0):.6f}`",
            f"- Dev evidence F@3: `{best.get('dev_evidence_f@3', 0.0):.6f}`",
            f"- Dev macro recall@3: `{best.get('dev_macro_recall@3', 0.0):.6f}`",
            f"- Dev macro recall@10: `{best.get('dev_macro_recall@10', 0.0):.6f}`",
            f"- Dev macro recall@64: `{best.get('dev_macro_recall@64', 0.0):.6f}`",
            "",
            "## Strict Notes",
            "",
            "- Weight selection uses train-claim folds only.",
            "- Dev labels are used only for confirmation metrics.",
            "- The grid was designed after prior diagnostic work, so this should be reported as train-selected recovery, not as a blind preregistered leaderboard result.",
            "- No forbidden path markers found." if not forbidden_hits else f"- Forbidden hits: `{forbidden_hits}`",
            "",
            "## Top Configurations",
            "",
            "| ce_score | embedding_score | hand_score | source_rank | ce_rank | embedding_rank | hand_rank | cv F@3 | cv R@3 | dev F@3 | dev R@3 | dev R@10 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows[:10]:
        lines.append(
            f"| {row['w_ce_score']:.2f} | {row['w_embedding_score']:.2f} | "
            f"{row['w_hand_score']:.2f} | {row['w_source_rank']:.2f} | "
            f"{row['w_ce_rank']:.2f} | {row['w_embedding_rank']:.2f} | {row['w_hand_rank']:.2f} | "
            f"{row.get('cv_evidence_f@3', 0.0):.4f} | {row.get('cv_macro_recall@3', 0.0):.4f} | "
            f"{row.get('dev_evidence_f@3', 0.0):.4f} | {row.get('dev_macro_recall@3', 0.0):.4f} | "
            f"{row.get('dev_macro_recall@10', 0.0):.4f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {summary_json}")
    print(f"wrote {summary_csv}")
    print(f"wrote {best_train_candidates_path}")
    print(f"wrote {best_dev_candidates_path}")
    print(f"wrote {report_path}")
    print(
        "best "
        f"cv_F@3={best.get('cv_evidence_f@3', 0.0):.6f} "
        f"dev_F@3={best.get('dev_evidence_f@3', 0.0):.6f} "
        f"dev_R@3={best.get('dev_macro_recall@3', 0.0):.6f}"
    )


if __name__ == "__main__":
    main()
