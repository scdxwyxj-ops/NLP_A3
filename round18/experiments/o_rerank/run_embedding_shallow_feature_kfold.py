#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # The NLP conda env currently lacks lightgbm, so keep the experiment runnable.
    from lightgbm import LGBMRanker

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - depends on local environment
    LGBMRanker = None  # type: ignore[assignment]
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier
    sys.modules.setdefault(
        "lightgbm",
        types.SimpleNamespace(LGBMRanker=object),
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rerank.round16_branch_a_requirement_selector import (  # noqa: E402
    FEATURES as SHALLOW_FEATURES,
    make_features,
    prepare_claim_profiles,
    text_profile,
)
from round18.tools.common import find_forbidden_tokens, load_json, write_json  # noqa: E402


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_TRAIN_EMBEDDING = Path(
    "round18/outputs/o_dense/o_d1x_embedding_bm25_char_train_top500/"
    "train_full_train_o_d1x_embedding_bm25_char_train_top500_strict_top500_candidates.json"
)
DEFAULT_DEV_EMBEDDING = Path(
    "round18/outputs/o_dense/o_d1x_embedding_bm25_char_dev_top500/"
    "dev_full_dev_o_d1x_embedding_bm25_char_dev_top500_strict_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/reports/embedding_shallow_feature_kfold")
DEFAULT_RUN_ID = "embedding_shallow_feature_kfold"
DEFAULT_EVAL_K = "1,3,5,10,64,100,500"


@dataclass(frozen=True)
class Example:
    claim_id: str
    evidence_id: str
    label: int
    features: dict[str, float]
    source_rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only k-fold test for whether explicit shallow matching features "
            "supplement MiniLM embedding inner-product scores in top64 context selection."
        )
    )
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-embedding-pool", type=Path, default=DEFAULT_TRAIN_EMBEDDING)
    parser.add_argument("--dev-embedding-pool", type=Path, default=DEFAULT_DEV_EMBEDDING)
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--eval-k", default=DEFAULT_EVAL_K)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1818)
    parser.add_argument("--random-seed", type=int, default=1821)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
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


def parse_embedding_pool(
    path: Path,
    claims: dict[str, Any],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise SystemExit(f"Missing embedding pool: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Embedding pool must be a JSON object: {path}")

    parsed: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        rows = payload.get(claim_id, [])
        if not isinstance(rows, list):
            continue
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            selected.append(
                {
                    "evidence_id": evidence_id,
                    "embedding_rank": coerce_int(row.get("embedding_rank", row.get("rank")), position),
                    "embedding_score": coerce_float(
                        row.get("embedding_score", row.get("reranker_score", row.get("score", 0.0)))
                    ),
                    "source_rank": coerce_int(row.get("source_rank"), position),
                    "source_score": coerce_float(row.get("source_score", 0.0)),
                }
            )
            if len(selected) >= candidate_k:
                break
        if selected:
            parsed[claim_id] = selected
    return parsed


def build_examples(
    claims: dict[str, Any],
    evidence: dict[str, str],
    embedding_pool: dict[str, list[dict[str, Any]]],
) -> dict[str, list[Example]]:
    claim_profiles = prepare_claim_profiles(claims)
    evidence_cache: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[Example]] = {}
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        claim_profile = claim_profiles[claim_id]
        rows: list[Example] = []
        for item in embedding_pool.get(claim_id, []):
            evidence_id = item["evidence_id"]
            if evidence_id not in evidence:
                continue
            if evidence_id not in evidence_cache:
                evidence_cache[evidence_id] = text_profile(evidence[evidence_id])
            shallow = make_features(
                claim_profile=claim_profile,
                evidence_profile=evidence_cache[evidence_id],
                source_rank=item["source_rank"],
                source_score=item["source_score"],
                extra_ranks={},
            )
            features = {name: float(shallow.get(name, 0.0)) for name in SHALLOW_FEATURES}
            features.update(
                {
                    "embedding_score": float(item["embedding_score"]),
                    "embedding_rank": float(item["embedding_rank"]),
                    "embedding_rr": 1.0 / (60.0 + float(item["embedding_rank"])),
                }
            )
            rows.append(
                Example(
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    label=1 if evidence_id in gold else 0,
                    features=features,
                    source_rank=int(item["source_rank"]),
                )
            )
        examples[claim_id] = rows
    return examples


def factual_feature_names() -> list[str]:
    return [
        name
        for name in SHALLOW_FEATURES
        if name.startswith("claim_key")
        or "content" in name
        or "entity" in name
        or "number" in name
        or "year" in name
        or "negation" in name
        or "relation" in name
    ]


def feature_names_for_variant(variant: str) -> list[str]:
    embedding = ["embedding_score", "embedding_rank", "embedding_rr"]
    if variant == "embedding_only":
        return ["embedding_score"]
    if variant == "shallow_only":
        return list(SHALLOW_FEATURES)
    if variant == "embedding_plus_factual":
        return embedding + factual_feature_names()
    if variant == "embedding_plus_shallow":
        return embedding + list(SHALLOW_FEATURES)
    raise ValueError(f"Unknown variant: {variant}")


def split_folds(claim_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    if n_folds < 2:
        raise ValueError("--k-folds must be >= 2.")
    if len(claim_ids) < n_folds:
        raise ValueError(f"Cannot split {len(claim_ids)} claims into {n_folds} folds.")
    rng = np.random.default_rng(seed)
    shuffled = list(claim_ids)
    rng.shuffle(shuffled)
    return [shuffled[idx::n_folds] for idx in range(n_folds)]


def matrix_for_examples(examples: list[Example], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[example.features.get(name, 0.0) for name in feature_names] for example in examples],
        dtype=np.float32,
    )


def train_ranker(
    train_examples_by_claim: dict[str, list[Example]],
    feature_names: list[str],
    args: argparse.Namespace,
) -> Any:
    ordered_claims = [cid for cid in sorted(train_examples_by_claim) if train_examples_by_claim[cid]]
    rows = [example for cid in ordered_claims for example in train_examples_by_claim[cid]]
    groups = [len(train_examples_by_claim[cid]) for cid in ordered_claims]
    y = np.asarray([example.label for example in rows], dtype=np.int32)
    x = matrix_for_examples(rows, feature_names)
    weights = np.ones(len(y), dtype=np.float32)
    weights[y == 1] = float(args.positive_weight)
    if HAS_LIGHTGBM and LGBMRanker is not None:
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=12,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
        ranker.fit(x, y, group=groups, sample_weight=weights)
        return ranker

    classifier = HistGradientBoostingClassifier(
        max_iter=args.n_estimators,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.num_leaves,
        l2_regularization=0.0,
        random_state=args.random_seed,
    )
    classifier.fit(x, y, sample_weight=weights)
    return classifier


def predict_scores(ranker: Any, x: np.ndarray) -> np.ndarray:
    if HAS_LIGHTGBM and LGBMRanker is not None and isinstance(ranker, LGBMRanker):
        return np.asarray(ranker.predict(x), dtype=np.float32)
    if hasattr(ranker, "predict_proba"):
        probabilities = ranker.predict_proba(x)
        if probabilities.shape[1] == 1:
            return probabilities[:, 0].astype(np.float32)
        return probabilities[:, 1].astype(np.float32)
    return np.asarray(ranker.predict(x), dtype=np.float32)


def rank_examples(
    examples_by_claim: dict[str, list[Example]],
    variant: str,
    ranker: Any | None,
    feature_names: list[str],
    candidate_k: int,
) -> dict[str, list[dict[str, Any]]]:
    ranked: dict[str, list[dict[str, Any]]] = {}
    for claim_id, examples in examples_by_claim.items():
        if not examples:
            ranked[claim_id] = []
            continue
        if variant == "embedding_only":
            scores = np.asarray([example.features["embedding_score"] for example in examples], dtype=np.float32)
        else:
            if ranker is None:
                raise ValueError("Non-embedding variant requires a trained ranker.")
            scores = predict_scores(ranker, matrix_for_examples(examples, feature_names))
        scored = sorted(
            zip(examples, scores),
            key=lambda item: (
                -float(item[1]),
                item[0].features.get("embedding_rank", 10**9),
                item[0].evidence_id,
            ),
        )
        out: list[dict[str, Any]] = []
        for rank, (example, score) in enumerate(scored[:candidate_k], start=1):
            out.append(
                {
                    "evidence_id": example.evidence_id,
                    "rank": rank,
                    "score": float(score),
                    "embedding_score": float(example.features.get("embedding_score", 0.0)),
                    "embedding_rank": int(example.features.get("embedding_rank", 0.0)),
                    "source_rank": example.source_rank,
                    "reranker_variant": variant,
                }
            )
        ranked[claim_id] = out
    return ranked


def evaluate_recall_at_k(
    claims: dict[str, Any],
    ranked: dict[str, list[dict[str, Any]]],
    ks: list[int],
    claim_subset: set[str] | None = None,
) -> dict[str, float]:
    totals = {k: {"macro": [], "tp": 0.0, "gold": 0.0, "hit_any": 0} for k in ks}
    claim_ids = sorted(claim_subset) if claim_subset is not None else sorted(claims)
    count = 0
    for claim_id in claim_ids:
        claim = claims.get(claim_id)
        if not claim:
            continue
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        count += 1
        ids = [row["evidence_id"] for row in ranked.get(claim_id, [])]
        for k in ks:
            predicted = set(ids[:k])
            tp = len(predicted & gold)
            totals[k]["macro"].append(tp / float(len(gold)))
            totals[k]["tp"] += float(tp)
            totals[k]["gold"] += float(len(gold))
            if tp > 0:
                totals[k]["hit_any"] += 1
    metrics: dict[str, float] = {
        "claims_with_evidence": float(count),
        "avg_candidate_count": float(sum(len(v) for v in ranked.values()) / count) if count else 0.0,
    }
    for k in ks:
        vals = totals[k]["macro"]
        metrics[f"macro_recall@{k}"] = float(sum(vals) / len(vals)) if vals else 0.0
        metrics[f"micro_recall@{k}"] = totals[k]["tp"] / totals[k]["gold"] if totals[k]["gold"] else 0.0
        metrics[f"hit_any@{k}"] = totals[k]["hit_any"] / float(count) if count else 0.0
    return metrics


def cv_metrics_for_variant(
    variant: str,
    train_claims: dict[str, Any],
    train_examples: dict[str, list[Example]],
    folds: list[list[str]],
    eval_ks: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    feature_names = feature_names_for_variant(variant)
    fold_metrics: list[dict[str, float]] = []
    all_ids = set(train_examples)
    for valid_ids in folds:
        valid_set = set(valid_ids)
        train_set = all_ids - valid_set
        valid_examples = {cid: train_examples[cid] for cid in valid_set if cid in train_examples}
        if variant == "embedding_only":
            ranker = None
        else:
            ranker = train_ranker(
                {cid: train_examples[cid] for cid in train_set if cid in train_examples},
                feature_names,
                args,
            )
        ranked = rank_examples(valid_examples, variant, ranker, feature_names, args.candidate_k)
        fold_metrics.append(evaluate_recall_at_k(train_claims, ranked, eval_ks, valid_set))
    out: dict[str, float] = {}
    for k in eval_ks:
        out[f"cv_macro_recall@{k}"] = float(
            sum(m[f"macro_recall@{k}"] for m in fold_metrics) / len(fold_metrics)
        )
        out[f"cv_micro_recall@{k}"] = float(
            sum(m[f"micro_recall@{k}"] for m in fold_metrics) / len(fold_metrics)
        )
        out[f"cv_hit_any@{k}"] = float(
            sum(m[f"hit_any@{k}"] for m in fold_metrics) / len(fold_metrics)
        )
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    embedding_dev: dict[str, float],
    forbidden_hits: dict[str, list[str]],
    elapsed: float,
) -> None:
    delta = best.get("dev_macro_recall@64", 0.0) - embedding_dev.get("macro_recall@64", 0.0)
    lines = [
        "# Embedding + Shallow Feature K-Fold Reranker",
        "",
        "Goal: test whether explicit shallow matching features complement MiniLM embedding inner-product scores for top64 context selection.",
        "",
        "## Strict Policy",
        "- Variant selection uses train-claim k-fold CV only.",
        "- Dev is used once for confirmation after the selected variant is fixed.",
        "- Primary metric: `macro_recall@64`.",
        "",
        "## Inputs",
        f"- Train embedding pool: `{args.train_embedding_pool}`",
        f"- Dev embedding pool: `{args.dev_embedding_pool}`",
        f"- Candidate budget: `top{args.candidate_k}`",
        f"- K folds: `{args.k_folds}`",
        "",
        "## Selected Variant",
        f"- Variant: `{best['variant']}`",
        f"- CV macro@64: `{best.get('cv_macro_recall@64', 0.0):.6f}`",
        f"- Dev macro@64: `{best.get('dev_macro_recall@64', 0.0):.6f}`",
        f"- Embedding-only dev macro@64: `{embedding_dev.get('macro_recall@64', 0.0):.6f}`",
        f"- Dev delta vs embedding-only: `{delta:+.6f}`",
        "",
        "## Leakage Scan",
        "- No forbidden path markers found." if not forbidden_hits else f"- Forbidden hits: `{forbidden_hits}`",
        "",
        "## Variant Table",
        "| variant | cv@3 | cv@10 | cv@64 | dev@3 | dev@10 | dev@64 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | "
            f"{row.get('cv_macro_recall@3', 0.0):.4f} | "
            f"{row.get('cv_macro_recall@10', 0.0):.4f} | "
            f"{row.get('cv_macro_recall@64', 0.0):.4f} | "
            f"{row.get('dev_macro_recall@3', 0.0):.4f} | "
            f"{row.get('dev_macro_recall@10', 0.0):.4f} | "
            f"{row.get('dev_macro_recall@64', 0.0):.4f} |"
        )
    lines.extend(["", "## Runtime", f"- Wall seconds: `{elapsed:.3f}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    eval_ks = parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        raise SystemExit("--candidate-k must be >= max eval-k.")
    for required in (
        args.train_claims,
        args.dev_claims,
        args.evidence,
        args.train_embedding_pool,
        args.dev_embedding_pool,
    ):
        if not required.exists():
            raise SystemExit(f"Missing required input: {required}")

    train_claims = load_claims(args.train_claims)
    dev_claims = load_claims(args.dev_claims)
    evidence = load_json(args.evidence)
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence must be a JSON object.")

    train_embedding = parse_embedding_pool(args.train_embedding_pool, train_claims, args.candidate_k)
    dev_embedding = parse_embedding_pool(args.dev_embedding_pool, dev_claims, args.candidate_k)
    train_examples = build_examples(train_claims, evidence, train_embedding)
    dev_examples = build_examples(dev_claims, evidence, dev_embedding)
    folds = split_folds(list(train_examples), args.k_folds, args.fold_seed)

    variants = ["embedding_only", "shallow_only", "embedding_plus_factual", "embedding_plus_shallow"]
    rows: list[dict[str, Any]] = []
    dev_ranked_by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in variants:
        row: dict[str, Any] = {
            "variant": variant,
            "status": "strict-candidate",
            "selection_basis": "train_kfold",
            "candidate_k": args.candidate_k,
            "feature_count": len(feature_names_for_variant(variant)),
            "learned_ranker_family": "lightgbm_lambdarank"
            if HAS_LIGHTGBM
            else "sklearn_hist_gradient_boosting_classifier",
        }
        row.update(cv_metrics_for_variant(variant, train_claims, train_examples, folds, eval_ks, args))
        feature_names = feature_names_for_variant(variant)
        ranker = None if variant == "embedding_only" else train_ranker(train_examples, feature_names, args)
        dev_ranked = rank_examples(dev_examples, variant, ranker, feature_names, args.candidate_k)
        dev_ranked_by_variant[variant] = dev_ranked
        dev_metrics = evaluate_recall_at_k(dev_claims, dev_ranked, eval_ks)
        for k in eval_ks:
            row[f"dev_macro_recall@{k}"] = dev_metrics[f"macro_recall@{k}"]
            row[f"dev_micro_recall@{k}"] = dev_metrics[f"micro_recall@{k}"]
            row[f"dev_hit_any@{k}"] = dev_metrics[f"hit_any@{k}"]
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get("cv_macro_recall@64", 0.0),
            row.get("cv_macro_recall@10", 0.0),
            row.get("cv_macro_recall@3", 0.0),
            row.get("cv_macro_recall@500", 0.0),
        ),
        reverse=True,
    )
    best = rows[0]
    embedding_dev_metrics = next(row for row in rows if row["variant"] == "embedding_only")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / f"{args.run_id}_summary.json"
    summary_csv = args.output_dir / f"{args.run_id}_summary.csv"
    best_candidates_path = args.output_dir / f"{args.run_id}_best_dev_top{args.candidate_k}_candidates.json"
    report_path = args.output_dir / f"{args.run_id}_report.md"

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.evidence),
            str(args.train_embedding_pool),
            str(args.dev_embedding_pool),
            str(args.output_dir),
            " ".join(sys.argv),
        ]
    )

    write_json(best_candidates_path, dev_ranked_by_variant[best["variant"]])
    payload = {
        "run_id": args.run_id,
        "status": "strict-candidate",
        "selection_basis": "train_kfold",
        "primary_metric": "macro_recall@64",
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "inputs": {
            "train_claims": str(args.train_claims),
            "dev_claims": str(args.dev_claims),
            "evidence": str(args.evidence),
            "train_embedding_pool": str(args.train_embedding_pool),
            "dev_embedding_pool": str(args.dev_embedding_pool),
        },
        "learned_ranker_family": "lightgbm_lambdarank"
        if HAS_LIGHTGBM
        else "sklearn_hist_gradient_boosting_classifier",
        "best_variant": best,
        "rows": rows,
        "best_candidates_file": str(best_candidates_path),
        "forbidden_input_scan": forbidden_hits,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    write_json(summary_json, payload)
    write_csv(rows, summary_csv)
    make_report(
        path=report_path,
        args=args,
        rows=rows,
        best=best,
        embedding_dev={
            key.replace("dev_", ""): val
            for key, val in embedding_dev_metrics.items()
            if key.startswith("dev_")
        },
        forbidden_hits=forbidden_hits,
        elapsed=time.perf_counter() - start,
    )

    print(f"wrote {summary_json}")
    print(f"wrote {summary_csv}")
    print(f"wrote {best_candidates_path}")
    print(f"wrote {report_path}")
    print(
        f"best_variant={best['variant']} "
        f"cv@64={best.get('cv_macro_recall@64', float('nan')):.6f} "
        f"dev@64={best.get('dev_macro_recall@64', float('nan')):.6f}"
    )


if __name__ == "__main__":
    main()
