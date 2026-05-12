#!/usr/bin/env python
"""Run fixed-candidate robustness checks for the final Worker G decision.

Targets:
  - Candidate: k=5 TF-IDF + shallow side features
    (max_features=30000, ngram=1-2, C=1.0)
  - Baseline: k=20 TF-IDF only
    (max_features=60000, ngram=1-2, C=4.0, O-C4-like)

No dev-based selection is performed. Train-only repeated holdout (required seeds)
is used; 5-fold CV is optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve()
for parent in REPO_ROOT.parents:
    if (parent / "round18" / "tools").exists():
        REPO_ROOT = parent
        break
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from round18.experiments.o_classifier.o_c5_comparison.worker_f_trainonly_selection.run_worker_f_trainonly_selection import (  # noqa: E402
    LABEL_ORDER,
    SIDE_FEATURE_NAMES,
    _build_contexts,
    _build_predictions_payload,
    _build_profile_cache,
    _build_shallow_matrix,
    _collapse_gate,
    _load_inputs,
    _rows_from_contexts,
    _train_and_evaluate,
    _write_confusion_csv,
)
from round18.experiments.o_classifier.o_c5_comparison.worker_f_trainonly_selection.run_worker_f_trainonly_selection import (
    _ensure_family_contract,
)
from round18.experiments.o_classifier.o_c5_comparison.worker_f_trainonly_selection.run_worker_f_trainonly_selection import manifest_base
from round18.tools.common import find_forbidden_tokens, sha256_file, write_json


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_TRAIN_POOL = Path(
    "round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json"
)
DEFAULT_DEV_POOL = Path(
    "round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json"
)
DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/o_c5_comparison/worker_g_final_candidate")
DEFAULT_REPORT_PATH = Path("round18/reports/o_c5_classification_comparison/worker_g_final_candidate_report.md")
DEFAULT_RUN_ID = "worker_g_final_candidate"


@dataclass(frozen=True)
class Candidate:
    tag: str
    model_id: str
    use_shallow: bool
    context_k: int
    tfidf_max_features: int
    tfidf_ngram_range: tuple[int, int]
    C: float
    n_side_features: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-candidate robustness checks (train-only)."
    )
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--dev-pool", type=Path, default=DEFAULT_DEV_POOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
    parser.add_argument("--selection-seeds", type=str, default="1337,2027,42,7,99")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--evidence-token-budget", type=int, default=0)
    parser.add_argument("--no-dev-check", action="store_true")
    return parser.parse_args()


def _parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise argparse.ArgumentTypeError("selection-seeds needs at least one integer.")
    return seeds


def _build_context_payload(
    claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    pool: dict[str, Any],
    context_k: int,
    family: str,
    evidence_token_budget: int,
    claim_profiles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    contexts, _diag = _build_contexts(
        claims,
        evidence=evidence,
        pool=pool,
        top_k=context_k,
        source_family=family,
    )
    claim_ids, texts, labels, context_items = _rows_from_contexts(
        claims,
        contexts,
        evidence_token_budget=evidence_token_budget,
    )

    side_matrix = None
    if claim_profiles is not None:
        side_matrix, _ = _build_shallow_matrix(
            claim_ids=claim_ids,
            claim_profiles=claim_profiles,
            claim_context_items=context_items,
            evidence=evidence,
        )

    payload = {
        "context_k": context_k,
        "claim_ids": claim_ids,
        "texts": texts,
        "labels": labels,
        "contexts": contexts,
        "side_matrix": side_matrix,
    }
    return payload


def _build_candidate_payloads(
    train_claims: dict[str, dict[str, Any]],
    dev_claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    train_pool: dict[str, list[dict[str, Any]]],
    dev_pool: dict[str, list[dict[str, Any]]],
    train_family: str,
    dev_family: str,
    args: argparse.Namespace,
    candidates: dict[str, Candidate],
) -> dict[str, dict[str, Any]]:
    train_profiles = _build_profile_cache(train_claims)
    dev_profiles = _build_profile_cache(dev_claims)

    payloads: dict[str, dict[str, Any]] = {}
    for tag, spec in candidates.items():
        train_payload = _build_context_payload(
            claims=train_claims,
            evidence=evidence,
            pool=train_pool,
            context_k=spec.context_k,
            family=train_family,
            evidence_token_budget=args.evidence_token_budget,
            claim_profiles=train_profiles if spec.use_shallow else None,
        )
        dev_payload = _build_context_payload(
            claims=dev_claims,
            evidence=evidence,
            pool=dev_pool,
            context_k=spec.context_k,
            family=dev_family,
            evidence_token_budget=args.evidence_token_budget,
            claim_profiles=dev_profiles if spec.use_shallow else None,
        )

        payloads[tag] = {
            "spec": spec,
            "train": train_payload,
            "dev": dev_payload,
        }
    return payloads


def _split_plan_holdout(labels: list[str], seeds: list[int], fraction: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    indices = np.arange(len(labels))
    y = np.asarray(labels, dtype=object)
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        try:
            train_idx, val_idx = next(splitter.split(indices, y))
        except ValueError:
            continue
        rows.append((f"holdout_seed_{seed}", np.asarray(train_idx), np.asarray(val_idx)))
    return rows


def _split_plan_cv(labels: list[str], folds: int, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if folds < 2 or folds > len(labels):
        return []
    indices = np.arange(len(labels))
    y = np.asarray(labels, dtype=object)
    selector = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [
        (f"cv_fold_{idx}", np.asarray(train_idx), np.asarray(val_idx))
        for idx, (train_idx, val_idx) in enumerate(selector.split(indices, y), start=1)
    ]


def _eval_on_split(
    payload: dict[str, Any],
    split_name: str,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    random_seed: int,
    n_jobs: int,
    collapse_threshold: float,
) -> dict[str, Any]:
    spec = payload["spec"]
    train_payload = payload["train"]
    model_args = argparse.Namespace(random_seed=random_seed, n_jobs=n_jobs)
    result, y_pred, y_prob = _train_and_evaluate(
        model_id="tfidf_with_shallow" if spec.use_shallow else "tfidf_only",
        train_texts=[train_payload["texts"][i] for i in tr_idx],
        y_train=[train_payload["labels"][i] for i in tr_idx],
        dev_texts=[train_payload["texts"][i] for i in va_idx],
        y_dev=[train_payload["labels"][i] for i in va_idx],
        tfidf_config={
            "max_features": spec.tfidf_max_features,
            "ngram_range": tuple(spec.tfidf_ngram_range),
        },
        c_val=spec.C,
        args=model_args,
        train_side=train_payload["side_matrix"][tr_idx] if spec.use_shallow else None,
        dev_side=train_payload["side_matrix"][va_idx] if spec.use_shallow else None,
    )

    return {
        "split": split_name,
        "candidate": spec.tag,
        "macro_f1": float(result["macro_f1"]),
        "accuracy": float(result["accuracy"]),
        "macro_recall": float(result["macro_recall"]),
        "top_class_share": float(result["top_class_share"]),
        "per_class_recall": {k: float(v) for k, v in result["per_class_recall"].items()},
        "prediction_histogram": result["prediction_histogram"],
        "collapse_gate": _collapse_gate(result["prediction_histogram"], collapse_threshold),
    }


def _summarize(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": int(len(rows))}
    for name in metric_names:
        vals = [float(row[name]) for row in rows if row.get(name) is not None]
        if not vals:
            summary[f"{name}_mean"] = 0.0
            summary[f"{name}_std"] = 0.0
            continue
        summary[f"{name}_mean"] = float(np.mean(vals))
        summary[f"{name}_std"] = float(np.std(vals))
    for label in LABEL_ORDER:
        vals = [float(r.get("per_class_recall", {}).get(label, 0.0)) for r in rows]
        summary[f"per_class_recall_{label}_mean"] = float(np.mean(vals)) if vals else 0.0
        summary[f"per_class_recall_{label}_std"] = float(np.std(vals)) if vals else 0.0
    summary["collapse_pass_rate"] = float(
        sum(1 for row in rows if row.get("collapse_gate", {}).get("status") == "passed")
        / max(1, len(rows))
    )
    return summary


def _run_mode(
    mode_name: str,
    split_plan: list[tuple[str, np.ndarray, np.ndarray]],
    candidate_payloads: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in candidate_payloads.values():
        for split_name, tr_idx, va_idx in split_plan:
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            rows.append(
                _eval_on_split(
                    payload=payload,
                    split_name=f"{mode_name}::{split_name}",
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    random_seed=args.random_seed,
                    n_jobs=args.n_jobs,
                    collapse_threshold=args.collapse_threshold,
                )
            )

    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], []).append(row)

    summary: dict[str, Any] = {
        tag: _summarize(lines, ["macro_f1", "accuracy", "macro_recall", "top_class_share"])
        for tag, lines in by_candidate.items()
    }

    deltas: list[dict[str, Any]] = []
    if "candidate" in by_candidate and "baseline" in by_candidate:
        base_map = {
            row["split"].split("::", 1)[1]: row
            for row in by_candidate["baseline"]
            if row.get("candidate") == "baseline"
        }
        for row in by_candidate["candidate"]:
            split_name = row["split"].split("::", 1)[1]
            base = base_map.get(split_name)
            if not base:
                continue
            deltas.append({
                "split": row["split"],
                "delta_macro_f1": float(row["macro_f1"] - base["macro_f1"]),
                "delta_accuracy": float(row["accuracy"] - base["accuracy"]),
                "delta_macro_recall": float(row["macro_recall"] - base["macro_recall"]),
                "delta_top_class_share": float(row["top_class_share"] - base["top_class_share"]),
            })

    return {
        "mode": mode_name,
        "rows": rows,
        "summary_by_candidate": summary,
        "paired_deltas": deltas,
        "paired_summary": _summarize(
            deltas,
            [
                "delta_macro_f1",
                "delta_accuracy",
                "delta_macro_recall",
                "delta_top_class_share",
            ],
        ),
    }


def _promote_signal(summary: dict[str, Any], deltas: list[dict[str, Any]]) -> tuple[str, str]:
    cand_sum = summary.get("candidate")
    base_sum = summary.get("baseline")
    if not cand_sum or not base_sum:
        return "provisional_negative", "缺少候选/基线任一分组，无法完成配对比较。"

    mean_delta = cand_sum["macro_f1_mean"] - base_sum["macro_f1_mean"]
    all_pass = (cand_sum["collapse_pass_rate"] == 1.0 and base_sum["collapse_pass_rate"] == 1.0)
    all_split_delta_nonnegative = all(row["delta_macro_f1"] >= 0 for row in deltas)
    if mean_delta > 0 and all_pass and all_split_delta_nonnegative:
        return (
            "strict_promotion",
            f"训练内重复抽样下候选平均提升 {mean_delta:+.6f}，且所有 split 均不低于基线。",
        )
    if mean_delta > 0 and all_pass:
        min_delta = min(row["delta_macro_f1"] for row in deltas) if deltas else 0.0
        return (
            "provisional_positive",
            f"候选平均提升 {mean_delta:+.6f}，但部分 split delta 为负（最差 {min_delta:+.6f}）。",
        )
    return "provisional_negative", f"候选平均 delta_macro-f1={mean_delta:+.6f}，不支持 strict promote。"


def _run_final_dev(
    tag: str,
    payload: dict[str, Any],
    candidate_lookup: dict[str, dict[str, Any]],
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    spec = payload["spec"]
    train_payload = payload["train"]
    dev_payload = payload["dev"]
    model_args = argparse.Namespace(random_seed=args.random_seed, n_jobs=args.n_jobs)
    metrics, y_pred, y_prob = _train_and_evaluate(
        model_id="tfidf_with_shallow" if spec.use_shallow else "tfidf_only",
        train_texts=train_payload["texts"],
        y_train=train_payload["labels"],
        dev_texts=dev_payload["texts"],
        y_dev=dev_payload["labels"],
        tfidf_config={
            "max_features": spec.tfidf_max_features,
            "ngram_range": tuple(spec.tfidf_ngram_range),
        },
        c_val=spec.C,
        args=model_args,
        train_side=train_payload["side_matrix"] if spec.use_shallow else None,
        dev_side=dev_payload["side_matrix"] if spec.use_shallow else None,
    )

    pred_path = args.output_root / f"{run_id}_{tag}_dev_predictions.json"
    metric_path = args.output_root / f"{run_id}_{tag}_dev_metrics.json"
    confusion_path = args.output_root / f"{run_id}_{tag}_dev_confusion_matrix.csv"
    report_path = args.output_root / f"{run_id}_{tag}_classification_report.txt"

    pred_payload = _build_predictions_payload(
        claim_ids=dev_payload["claim_ids"],
        claim_lookup=candidate_lookup,
        contexts=dev_payload["contexts"],
        y_pred=y_pred,
        y_prob=y_prob,
    )
    write_json(pred_path, pred_payload)
    write_json(metric_path, {
        **metrics,
        "collapse_gate": _collapse_gate(metrics["prediction_histogram"], args.collapse_threshold),
    })
    _write_confusion_csv(confusion_path, np.array(metrics["confusion_matrix"]), LABEL_ORDER)
    report_path.write_text(metrics["classification_report"], encoding="utf-8")

    return {
        "metrics": str(metric_path),
        "predictions": str(pred_path),
        "confusion_matrix": str(confusion_path),
        "classification_report": str(report_path),
        "raw_metrics": metrics,
    }


def _format_markdown_table_row(mode: str, label: str, payload: dict[str, Any]) -> str:
    return (
        f"| {mode} | {label} | "
        f"{payload['macro_f1_mean']:.4f} ± {payload['macro_f1_std']:.4f} | "
        f"{payload['accuracy_mean']:.4f} ± {payload['accuracy_std']:.4f} | "
        f"{payload['macro_recall_mean']:.4f} ± {payload['macro_recall_std']:.4f} | "
        f"{payload['top_class_share_mean']:.4f} ± {payload['top_class_share_std']:.4f} | "
        f"{payload['collapse_pass_rate']:.2%} |"
    )


def _write_report(
    report_path: Path,
    outcome: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = [
        "# Worker G Robustness Report",
        "",
        "## Command",
        "",
        f"`{' '.join(sys.argv)}`",
        "",
        "## 结论",
        "",
    ]

    if not args.no_dev_check:
        cand = outcome["final_dev"]["candidate"]["raw_metrics"]
        base = outcome["final_dev"]["baseline"]["raw_metrics"]
        delta = cand["macro_f1"] - base["macro_f1"]
        holdout = outcome["modes"]["repeated_holdout"]
        signal, rationale = _promote_signal(
            holdout["summary_by_candidate"],
            holdout["paired_deltas"],
        )
        lines.append(
            f"- 决策：{signal.replace('_', ' ')}。{rationale}"
        )
        lines.append(
            f"- Full dev (only confirmation): candidate macro-F1={cand['macro_f1']:.6f}, accuracy={cand['accuracy']:.6f}, "
            f"top-class-share={cand['top_class_share']:.4f}, collapse={cand['collapse_gate']['status']}; "
            f"baseline macro-F1={base['macro_f1']:.6f}, accuracy={base['accuracy']:.6f}, "
            f"top-class-share={base['top_class_share']:.4f}, collapse={base['collapse_gate']['status']}; "
            f"delta={delta:+.6f}"
        )
        lines.append(
            f"- 与 Dev 对比用途声明：本节仅用于复核，不能作为新的选择标准（未参与候选筛选）。"
        )

    lines.extend([
        "",
        f"## Train-only repeated holdout (seeds: {','.join(map(str, _parse_seed_list(args.selection_seeds)))})",
        "",
        "| mode | method | macro-F1 mean±std | accuracy mean±std | macro-recall mean±std | top-class-share mean±std | collapse pass rate |",
        "|---|---|---|---|---|---|---|",
    ])

    hold = outcome["modes"]["repeated_holdout"]
    cand_sum = hold["summary_by_candidate"].get("candidate", {})
    base_sum = hold["summary_by_candidate"].get("baseline", {})
    if cand_sum and base_sum:
        lines.append(_format_markdown_table_row("repeated_holdout", "k5 tfidf+shallow", cand_sum))
        lines.append(_format_markdown_table_row("repeated_holdout", "k20 tfidf-only baseline", base_sum))
        lines.append(
            "| repeated_holdout | paired deltas | "
            f"{hold['paired_summary']['delta_macro_f1_mean']:+.4f} ± {hold['paired_summary']['delta_macro_f1_std']:.4f} | "
            f"{hold['paired_summary']['delta_accuracy_mean']:+.4f} ± {hold['paired_summary']['delta_accuracy_std']:.4f} | "
            f"{hold['paired_summary']['delta_macro_recall_mean']:+.4f} ± {hold['paired_summary']['delta_macro_recall_std']:.4f} | "
            f"{hold['paired_summary']['delta_top_class_share_mean']:+.4f} ± {hold['paired_summary']['delta_top_class_share_std']:.4f} | "
            "N/A |"
        )

    if "cv" in outcome["modes"]:
        cv = outcome["modes"]["cv"]
        lines.append("")
        lines.append(f"## 5-Fold CV (k={args.cv_folds})")
        cand_sum = cv["summary_by_candidate"].get("candidate", {})
        base_sum = cv["summary_by_candidate"].get("baseline", {})
        if cand_sum and base_sum:
            lines.append(_format_markdown_table_row("cv", "k5 tfidf+shallow", cand_sum))
            lines.append(_format_markdown_table_row("cv", "k20 tfidf-only baseline", base_sum))
            lines.append(
                "| cv | paired deltas | "
                f"{cv['paired_summary']['delta_macro_f1_mean']:+.4f} ± {cv['paired_summary']['delta_macro_f1_std']:.4f} | "
                f"{cv['paired_summary']['delta_accuracy_mean']:+.4f} ± {cv['paired_summary']['delta_accuracy_std']:.4f} | "
                f"{cv['paired_summary']['delta_macro_recall_mean']:+.4f} ± {cv['paired_summary']['delta_macro_recall_std']:.4f} | "
                f"{cv['paired_summary']['delta_top_class_share_mean']:+.4f} ± {cv['paired_summary']['delta_top_class_share_std']:.4f} | "
                "N/A |"
            )

    lines.extend([
        "",
        "## Paired per-split deltas",
        "",
        "| split | Δmacro-F1 | Δaccuracy | Δmacro-recall | Δtop-class-share |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in outcome["modes"]["repeated_holdout"]["paired_deltas"]:
        lines.append(
            f"| {row['split']} | {row['delta_macro_f1']:+.4f} | {row['delta_accuracy']:+.4f} | "
            f"{row['delta_macro_recall']:+.4f} | {row['delta_top_class_share']:+.4f} |"
        )
    if "cv" in outcome["modes"]:
        for row in outcome["modes"]["cv"]["paired_deltas"]:
            lines.append(
                f"| {row['split']} | {row['delta_macro_f1']:+.4f} | {row['delta_accuracy']:+.4f} | "
                f"{row['delta_macro_recall']:+.4f} | {row['delta_top_class_share']:+.4f} |"
            )

    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- `train_robustness_results.json`: full train-only robustness payload",
        f"- `run_manifest.json` and `train_robustness_results.csv`",
        f"- dev confirmation files: `{args.run_id}_candidate_*` / `{args.run_id}_baseline_*`",
        "",
        "## 严格性说明",
        "",
        "- 所有 split 仅来自 train 数据。",
        "- dev 仅作为最终复核，不参与候选筛选。",
        "- collapse gate 同前：`top_class_share <= 0.70` 且四类均有非零召回。",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\\n".join(lines), encoding="utf-8")


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], field_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in field_names})


def main() -> None:
    start = time.perf_counter()
    args = _parse_args()
    args.command = " ".join(sys.argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seed_list(args.selection_seeds)
    forbidden = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.evidence),
            str(args.train_pool),
            str(args.dev_pool),
            str(args.output_root),
            str(args.report_path),
        ]
    )
    if forbidden:
        raise SystemExit(f"STRICT GUARD FAILED: {json.dumps(forbidden, sort_keys=True)}")

    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    train_family, dev_family = _ensure_family_contract(args.train_pool, args.dev_pool)

    if train_family != dev_family:
        raise SystemExit("Train/dev context family mismatch.")

    candidates = {
        "candidate": Candidate(
            tag="candidate",
            model_id="tfidf_plus_shallow",
            use_shallow=True,
            context_k=5,
            tfidf_max_features=30000,
            tfidf_ngram_range=(1, 2),
            C=1.0,
            n_side_features=len(SIDE_FEATURE_NAMES),
        ),
        "baseline": Candidate(
            tag="baseline",
            model_id="tfidf_only",
            use_shallow=False,
            context_k=20,
            tfidf_max_features=60000,
            tfidf_ngram_range=(1, 2),
            C=4.0,
            n_side_features=0,
        ),
    }

    payloads = _build_candidate_payloads(
        train_claims=train_claims,
        dev_claims=dev_claims,
        evidence=evidence,
        train_pool=train_pool,
        dev_pool=dev_pool,
        train_family=train_family,
        dev_family=dev_family,
        args=args,
        candidates=candidates,
    )

    repeated_plan = _split_plan_holdout(
        labels=payloads["candidate"]["train"]["labels"],
        seeds=seeds,
        fraction=args.train_val_fraction,
    )
    cv_plan = []
    if not args.skip_cv:
        cv_plan = _split_plan_cv(
            labels=payloads["candidate"]["train"]["labels"],
            folds=args.cv_folds,
            seed=args.random_seed,
        )

    outcome: dict[str, Any] = {
        "settings": {
            "selection_seeds": seeds,
            "train_val_fraction": args.train_val_fraction,
            "cv_folds": args.cv_folds,
            "skip_cv": args.skip_cv,
            "collapse_threshold": args.collapse_threshold,
            "no_dev_check": args.no_dev_check,
            "n_jobs": args.n_jobs,
        },
        "modes": {},
        "final_dev": {},
    }

    outcome["modes"]["repeated_holdout"] = _run_mode(
        "repeated_holdout",
        repeated_plan,
        payloads,
        args,
    )
    if cv_plan:
        outcome["modes"]["cv"] = _run_mode("cv", cv_plan, payloads, args)

    if not args.no_dev_check:
        for tag, payload in payloads.items():
            dev_res = _run_final_dev(
                tag=tag,
                payload=payload,
                candidate_lookup=dev_claims,
                run_id=args.run_id,
                args=args,
            )
            outcome["final_dev"][tag] = dev_res

    summary_path = args.output_root / "train_robustness_results.json"
    write_json(summary_path, outcome)

    csv_path = args.output_root / "train_robustness_results.csv"
    csv_rows: list[dict[str, Any]] = []
    for mode, payload in outcome["modes"].items():
        for row in payload["rows"]:
            flat_row = {
                "mode": mode,
                "split": row["split"],
                "candidate": row["candidate"],
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
                "macro_recall": row["macro_recall"],
                "top_class_share": row["top_class_share"],
                "collapse_gate_status": row["collapse_gate"].get("status"),
                "collapse_gate_share": row["collapse_gate"].get("max_class_share"),
                "collapse_gate_threshold": row["collapse_gate"].get("threshold"),
            }
            for label in LABEL_ORDER:
                flat_row[f"per_class_recall_{label}"] = row.get("per_class_recall", {}).get(label, 0.0)
            prediction_hist = row.get("prediction_histogram", {})
            for label in LABEL_ORDER:
                flat_row[f"pred_hist_{label}"] = prediction_hist.get(label, 0)
            flat_row["prediction_hist_total"] = sum(prediction_hist.values())
            csv_rows.append(flat_row)
    csv_fieldnames = [
        "mode",
        "split",
        "candidate",
        "macro_f1",
        "accuracy",
        "macro_recall",
        "top_class_share",
        "collapse_gate_status",
        "collapse_gate_share",
        "collapse_gate_threshold",
    ]
    for label in LABEL_ORDER:
        csv_fieldnames.append(f"per_class_recall_{label}")
    for label in LABEL_ORDER:
        csv_fieldnames.append(f"pred_hist_{label}")
    csv_fieldnames.append("prediction_hist_total")
    if csv_rows:
        _write_csv_rows(csv_path, csv_rows, csv_fieldnames)

    if not args.no_dev_check:
        _write_report(args.report_path, outcome, args)

    manifest = manifest_base(
        run_id=args.run_id,
        status="final_candidate_robustness",
        mode="STRICT",
        stage="o_c5_tfidf_side_final_candidate",
        command=args.command,
        working_directory=Path.cwd(),
        random_seed=args.random_seed,
        cv_seed=args.random_seed,
    )
    manifest["input_files"] = [
        {"path": str(args.train_claims), "sha256": sha256_file(args.train_claims)},
        {"path": str(args.dev_claims), "sha256": sha256_file(args.dev_claims)},
        {"path": str(args.evidence), "sha256": sha256_file(args.evidence)},
        {"path": str(args.train_pool), "sha256": sha256_file(args.train_pool)},
        {"path": str(args.dev_pool), "sha256": sha256_file(args.dev_pool)},
    ]
    manifest["forbidden_input_scan"]["passed"] = not bool(forbidden)
    manifest["forbidden_input_scan"]["notes"] = "pass"
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["runtime"]["device"] = "cpu"
    manifest["output_files"] = [
        str(summary_path),
        str(csv_path),
        str(args.report_path),
        str(args.output_root / "run_manifest.json"),
    ]
    if not args.no_dev_check:
        for tag, res in outcome["final_dev"].items():
            manifest["output_files"].extend(
                [str(res["metrics"]), str(res["predictions"]), str(res["confusion_matrix"]), str(res["classification_report"])]
            )
    manifest["data_flow_summary"] = "Train split repeated-holdout/CV only, then fixed final dev confirmation."
    manifest["split_isolation_summary"] = "Dev never used for candidate selection."
    write_json(args.output_root / "run_manifest.json", manifest)

    print(f"Summary: {summary_path}")
    print(f"Report: {args.report_path}")
    if not args.no_dev_check:
        print(
            "Final dev macro-F1 candidate="
            f"{outcome['final_dev']['candidate']['raw_metrics']['macro_f1']:.6f}, baseline="
            f"{outcome['final_dev']['baseline']['raw_metrics']['macro_f1']:.6f}"
        )


if __name__ == "__main__":
    main()
