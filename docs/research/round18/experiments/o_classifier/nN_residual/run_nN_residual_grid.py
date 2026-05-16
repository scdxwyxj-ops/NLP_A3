#!/usr/bin/env python
"""Generalized n/N residual classifier grid for strict top-N contexts.

Main branch: claim + top-n evidence.
Residual branch: ranks n+1..N evidence.

Selection uses train-holdout only. Dev is held out for confirmation and for a
diagnostic upper bound over the full grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler


def _resolve_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "round18" / "tools").exists():
            return candidate
    return Path(__file__).resolve().parents[5]


REPO_ROOT = _resolve_repo_root()
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from round18.experiments.o_classifier.o_c5_comparison.worker_a_tfidf_side.run_worker_a_tfidf_side import (  # noqa: E402
    ContextItem,
    DEFAULT_DEV_CLAIMS,
    DEFAULT_DEV_POOL,
    DEFAULT_EVIDENCE,
    DEFAULT_TRAIN_CLAIMS,
    DEFAULT_TRAIN_POOL,
    LABEL_ORDER,
    _build_contexts,
    _build_profile_cache,
    _build_shallow_matrix,
    _ensure_family_contract,
    _load_inputs,
    _train_holdout_indices,
)
from round18.tools.common import find_forbidden_tokens, load_json, manifest_base, sha256_file, write_json  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/nN_residual_grid")
DEFAULT_REPORT_PATH = Path("round18/reports/nN_residual_grid_report.md")
DEFAULT_TOP_N_GRID = "3,5,8,10"
DEFAULT_TOP_N_CAP_GRID = "10,15,20,32,64"
DEFAULT_TAIL_ALPHA_GRID = "0,0.05,0.1,0.25,0.5,1.0"
DEFAULT_MAX_FEATURES = 60000
DEFAULT_LOGREG_C = 1.0


@dataclass(frozen=True)
class GridSpec:
    top_n: int
    top_N: int
    tail_alpha: float

    @property
    def name(self) -> str:
        alpha = str(self.tail_alpha).replace(".", "p")
        return f"head{self.top_n}_tail{self.top_N}_a{alpha}"


@dataclass(frozen=True)
class BranchBundle:
    claim_ids: list[str]
    labels: list[str]
    head_texts: list[str]
    tail_texts: list[str]
    head_side: csr_matrix
    tail_side: csr_matrix


def _parse_int_grid(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("grid values must be > 0.")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value.")
    return sorted(set(values))


def _parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generalized n/N residual classifier grid.")
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--dev-pool", type=Path, default=DEFAULT_DEV_POOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--top-n-grid", default=DEFAULT_TOP_N_GRID)
    parser.add_argument("--top-N-grid", dest="top_N_grid", default=DEFAULT_TOP_N_CAP_GRID)
    parser.add_argument("--tail-alpha-grid", default=DEFAULT_TAIL_ALPHA_GRID)
    parser.add_argument("--tfidf-max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--logreg-c", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--evidence-token-budget", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def _context_text(claim_text: str, items: list[ContextItem], evidence_token_budget: int, include_claim: bool) -> str:
    blocks: list[str] = []
    if include_claim:
        blocks.append(f"CLAIM: {claim_text.strip()}")
    for item in items:
        text = item.text
        if evidence_token_budget > 0:
            text = " ".join(text.split()[:evidence_token_budget])
        blocks.append(f"EVIDENCE_{item.rank}: {text}")
    return "\n".join(blocks)


def _slice_items(items: list[ContextItem], top_n: int, top_N: int) -> tuple[list[ContextItem], list[ContextItem]]:
    if top_N <= top_n:
        raise SystemExit(f"Invalid grid pair: N must be > n, got n={top_n}, N={top_N}.")
    if len(items) < top_N:
        raise SystemExit(f"Insufficient context items: need {top_N}, got {len(items)}.")
    return items[:top_n], items[top_n:top_N]


def _build_bundle(
    claims: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    claim_profiles: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    top_n: int,
    top_N: int,
    evidence_token_budget: int,
) -> BranchBundle:
    claim_ids = list(claims.keys())
    head_texts: list[str] = []
    tail_texts: list[str] = []
    labels: list[str] = []
    head_items: dict[str, list[ContextItem]] = {}
    tail_items: dict[str, list[ContextItem]] = {}
    for claim_id in claim_ids:
        full_items = [ContextItem(**row) for row in contexts[claim_id]["classifier_evidence_context"]]
        head, tail = _slice_items(full_items, top_n=top_n, top_N=top_N)
        head_items[claim_id] = head
        tail_items[claim_id] = tail
        claim_text = claims[claim_id].get("claim_text", "")
        head_texts.append(_context_text(claim_text, head, evidence_token_budget, include_claim=True))
        tail_texts.append(_context_text("", tail, evidence_token_budget, include_claim=False))
        labels.append(str(claims[claim_id].get("claim_label")))

    head_side, _ = _build_shallow_matrix(claim_ids, claim_profiles, head_items, evidence)
    tail_side, _ = _build_shallow_matrix(claim_ids, claim_profiles, tail_items, evidence)
    return BranchBundle(
        claim_ids=claim_ids,
        labels=labels,
        head_texts=head_texts,
        tail_texts=tail_texts,
        head_side=head_side,
        tail_side=tail_side,
    )


def _subset_bundle(bundle: BranchBundle, indices: np.ndarray) -> BranchBundle:
    idx = indices.tolist()
    return BranchBundle(
        claim_ids=[bundle.claim_ids[i] for i in idx],
        labels=[bundle.labels[i] for i in idx],
        head_texts=[bundle.head_texts[i] for i in idx],
        tail_texts=[bundle.tail_texts[i] for i in idx],
        head_side=bundle.head_side[indices],
        tail_side=bundle.tail_side[indices],
    )


def _evaluate_bundle(bundle_train: BranchBundle, bundle_dev: BranchBundle, spec: GridSpec, args: argparse.Namespace) -> dict[str, Any]:
    head_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=int(args.tfidf_max_features),
        dtype=np.float32,
    )
    tail_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=int(args.tfidf_max_features),
        dtype=np.float32,
    )

    train_head_x = head_vec.fit_transform(bundle_train.head_texts)
    dev_head_x = head_vec.transform(bundle_dev.head_texts)
    train_tail_x = tail_vec.fit_transform(bundle_train.tail_texts) * float(spec.tail_alpha)
    dev_tail_x = tail_vec.transform(bundle_dev.tail_texts) * float(spec.tail_alpha)

    head_scaler = MaxAbsScaler()
    tail_scaler = MaxAbsScaler()
    train_head_side_x = head_scaler.fit_transform(bundle_train.head_side)
    dev_head_side_x = head_scaler.transform(bundle_dev.head_side)
    train_tail_side_x = tail_scaler.fit_transform(bundle_train.tail_side) * float(spec.tail_alpha)
    dev_tail_side_x = tail_scaler.transform(bundle_dev.tail_side) * float(spec.tail_alpha)

    x_train = hstack([train_head_x, train_tail_x, train_head_side_x, train_tail_side_x], format="csr")
    x_dev = hstack([dev_head_x, dev_tail_x, dev_head_side_x, dev_tail_side_x], format="csr")

    model = OneVsRestClassifier(
        LogisticRegression(
            C=float(args.logreg_c),
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=args.random_seed,
        ),
        n_jobs=1,
    )

    start = time.perf_counter()
    model.fit(x_train, bundle_train.labels)
    fit_seconds = time.perf_counter() - start

    y_pred = model.predict(x_dev).astype(str).tolist()
    accuracy = accuracy_score(bundle_dev.labels, y_pred)
    macro_f1 = f1_score(bundle_dev.labels, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    micro_f1 = f1_score(bundle_dev.labels, y_pred, labels=LABEL_ORDER, average="micro", zero_division=0)
    macro_recall = recall_score(bundle_dev.labels, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    hist = {label: int(sum(1 for item in y_pred if item == label)) for label in LABEL_ORDER}
    return {
        "name": spec.name,
        "top_n": int(spec.top_n),
        "top_N": int(spec.top_N),
        "tail_alpha": float(spec.tail_alpha),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_recall": float(macro_recall),
        "top_class_share": float(max(hist.values()) / max(1, len(y_pred))),
        "top_class": max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0],
        "prediction_histogram": hist,
        "confusion_matrix": confusion_matrix(bundle_dev.labels, y_pred, labels=LABEL_ORDER).astype(int).tolist(),
        "classification_report": classification_report(bundle_dev.labels, y_pred, labels=LABEL_ORDER, zero_division=0),
        "fit_seconds": float(fit_seconds),
        "head_word_dim": int(train_head_x.shape[1]),
        "tail_word_dim": int(train_tail_x.shape[1]),
        "head_side_dim": int(bundle_train.head_side.shape[1]),
        "tail_side_dim": int(bundle_train.tail_side.shape[1]),
        "train_rows": int(len(bundle_train.labels)),
        "dev_rows": int(len(bundle_dev.labels)),
    }


def _selection_score(result: dict[str, Any]) -> tuple[float, float, float]:
    return (float(result["macro_f1"]), float(result["accuracy"]), -float(result["top_class_share"]))


def _compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "name",
        "top_n",
        "top_N",
        "tail_alpha",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "macro_recall",
        "top_class_share",
        "top_class",
        "prediction_histogram",
        "confusion_matrix",
        "fit_seconds",
        "head_word_dim",
        "tail_word_dim",
        "head_side_dim",
        "tail_side_dim",
        "train_rows",
        "dev_rows",
    ]
    return {key: result[key] for key in keys}


def _pair_grid_results(
    selection_results: list[dict[str, Any]],
    dev_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate alpha-level rows into one auditable row per (n, N)."""

    dev_by_name = {row["name"]: row for row in dev_results}
    pairs = sorted({(int(row["top_n"]), int(row["top_N"])) for row in selection_results})
    rows: list[dict[str, Any]] = []
    for top_n, top_N in pairs:
        pair_selection = [
            row for row in selection_results if int(row["top_n"]) == top_n and int(row["top_N"]) == top_N
        ]
        pair_dev = [row for row in dev_results if int(row["top_n"]) == top_n and int(row["top_N"]) == top_N]
        if not pair_selection or not pair_dev:
            continue
        selected_train = max(
            pair_selection,
            key=lambda row: (float(row["macro_f1"]), float(row["accuracy"]), -float(row["top_class_share"])),
        )
        selected_dev = dev_by_name[selected_train["name"]]
        diagnostic_best = max(
            pair_dev,
            key=lambda row: (float(row["macro_f1"]), float(row["accuracy"]), -float(row["top_class_share"])),
        )
        rows.append(
            {
                "n": top_n,
                "N": top_N,
                "selected_name": selected_train["name"],
                "selected_tail_alpha": float(selected_train["tail_alpha"]),
                "train_holdout_macro_f1": float(selected_train["macro_f1"]),
                "train_holdout_accuracy": float(selected_train["accuracy"]),
                "train_selected_dev_macro_f1": float(selected_dev["macro_f1"]),
                "train_selected_dev_accuracy": float(selected_dev["accuracy"]),
                "train_selected_dev_top_class_share": float(selected_dev["top_class_share"]),
                "diagnostic_name": diagnostic_best["name"],
                "diagnostic_tail_alpha": float(diagnostic_best["tail_alpha"]),
                "diagnostic_dev_best_macro_f1": float(diagnostic_best["macro_f1"]),
                "diagnostic_dev_best_accuracy": float(diagnostic_best["accuracy"]),
                "diagnostic_dev_best_top_class_share": float(diagnostic_best["top_class_share"]),
                "diagnostic_status": "non-promoted",
            }
        )
    return rows


def _load_reference_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    payload = load_json(path)
    return {
        "status": "available",
        "path": str(path),
        "macro_f1": float(payload["macro_f1"]),
        "accuracy": float(payload["accuracy"]),
        "top_class_share": float(payload["top_class_share"]),
    }


def _reference_candidates() -> dict[str, dict[str, Any]]:
    return {
        "top5_shallow_dev_best": _load_reference_metrics(
            REPO_ROOT
            / "round18"
            / "outputs"
            / "o_classifier"
            / "o_c5_comparison"
            / "worker_a_tfidf_side"
            / "worker_a_tfidf_side_k5"
            / "tfidf_logreg_plus_side_mf30000_ngram1x2_c1p0_sf47_metrics.json"
        ),
        "k20_promoted_dev_best": _load_reference_metrics(
            REPO_ROOT
            / "round18"
            / "outputs"
            / "o_classifier"
            / "o_c4_ce_factual_context_k20_trainholdout"
            / "o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_metrics.json"
        ),
    }


def _format_ref_line(name: str, ref: dict[str, Any]) -> str:
    if ref.get("status") != "available":
        return f"- {name}: missing ({ref.get('path')})"
    return (
        f"- {name}: macro-F1 `{ref['macro_f1']:.6f}`, "
        f"accuracy `{ref['accuracy']:.6f}`, top class share `{ref['top_class_share']:.6f}`."
    )


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    args.command = " ".join(sys.argv)

    top_n_grid = _parse_int_grid(args.top_n_grid)
    top_N_grid = _parse_int_grid(args.top_N_grid)
    tail_alpha_grid = _parse_float_grid(args.tail_alpha_grid)
    specs = [
        GridSpec(top_n=top_n, top_N=top_N, tail_alpha=tail_alpha)
        for top_n in top_n_grid
        for top_N in top_N_grid
        for tail_alpha in tail_alpha_grid
        if top_N > top_n
    ]
    if not specs:
        raise SystemExit("No valid grid specs after enforcing N > n.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    forbidden_hits = find_forbidden_tokens(
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
    if forbidden_hits:
        raise SystemExit(
            "STRICT GUARD FAILED: forbidden token(s) detected\n"
            + json.dumps(forbidden_hits, sort_keys=True)
        )

    train_family, dev_family = _ensure_family_contract(args.train_pool, args.dev_pool)
    if train_family != dev_family:
        raise SystemExit("Train/dev context family mismatch in strict mode.")

    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    if not train_claims or not dev_claims:
        raise SystemExit("Input claim files must be non-empty.")

    max_N = max(spec.top_N for spec in specs)
    train_contexts, train_diag = _build_contexts(train_claims, evidence, train_pool, top_k=max_N, source_family=train_family)
    dev_contexts, dev_diag = _build_contexts(dev_claims, evidence, dev_pool, top_k=max_N, source_family=dev_family)

    train_profiles = _build_profile_cache(train_claims)
    dev_profiles = _build_profile_cache(dev_claims)

    train_full_cache: dict[tuple[int, int], BranchBundle] = {}
    dev_full_cache: dict[tuple[int, int], BranchBundle] = {}
    for top_n in top_n_grid:
        for top_N in top_N_grid:
            if top_N <= top_n:
                continue
            train_full_cache[(top_n, top_N)] = _build_bundle(
                train_claims,
                train_contexts,
                train_profiles,
                evidence,
                top_n=top_n,
                top_N=top_N,
                evidence_token_budget=args.evidence_token_budget,
            )
            dev_full_cache[(top_n, top_N)] = _build_bundle(
                dev_claims,
                dev_contexts,
                dev_profiles,
                evidence,
                top_n=top_n,
                top_N=top_N,
                evidence_token_budget=args.evidence_token_budget,
            )

    train_split = _train_holdout_indices(next(iter(train_full_cache.values())).labels, args)
    if train_split is None:
        raise SystemExit("Could not build train holdout split.")
    tr_idx, val_idx = train_split

    selection_results: list[dict[str, Any]] = []
    dev_results: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    for idx, spec in enumerate(specs, start=1):
        train_bundle = train_full_cache[(spec.top_n, spec.top_N)]
        dev_bundle = dev_full_cache[(spec.top_n, spec.top_N)]
        train_sub = _subset_bundle(train_bundle, tr_idx)
        val_sub = _subset_bundle(train_bundle, val_idx)

        selection_result = _evaluate_bundle(train_sub, val_sub, spec, args)
        dev_result = _evaluate_bundle(train_bundle, dev_bundle, spec, args)

        selection_results.append(
            {
                **_compact_metrics(selection_result),
                "selection_score": list(_selection_score(selection_result)),
            }
        )
        dev_results.append(
            {
                **_compact_metrics(dev_result),
                "diagnostic_score": list(_selection_score(dev_result)),
            }
        )
        all_results.append(
            {
                "name": spec.name,
                "top_n": int(spec.top_n),
                "top_N": int(spec.top_N),
                "tail_alpha": float(spec.tail_alpha),
                "train_holdout": _compact_metrics(selection_result),
                "dev": _compact_metrics(dev_result),
                "selection_score": list(_selection_score(selection_result)),
                "diagnostic_score": list(_selection_score(dev_result)),
            }
        )
        if idx % 20 == 0 or idx == len(specs):
            print(f"finished {idx}/{len(specs)} specs", flush=True)

    selection_results.sort(key=lambda row: (row["selection_score"][0], row["selection_score"][1], row["selection_score"][2]), reverse=True)
    dev_results.sort(key=lambda row: (row["diagnostic_score"][0], row["diagnostic_score"][1], row["diagnostic_score"][2]), reverse=True)
    all_results.sort(key=lambda row: (row["top_n"], row["top_N"], row["tail_alpha"]))
    grid_results = _pair_grid_results(selection_results, dev_results)

    selected_name = selection_results[0]["name"]
    selected_dev = next(row for row in dev_results if row["name"] == selected_name)
    best_diagnostic = dev_results[0]
    best_pair_strict = max(
        grid_results,
        key=lambda row: (
            float(row["train_selected_dev_macro_f1"]),
            float(row["train_selected_dev_accuracy"]),
            -float(row["train_selected_dev_top_class_share"]),
        ),
    )
    best_pair_diagnostic = max(
        grid_results,
        key=lambda row: (
            float(row["diagnostic_dev_best_macro_f1"]),
            float(row["diagnostic_dev_best_accuracy"]),
            -float(row["diagnostic_dev_best_top_class_share"]),
        ),
    )

    refs = _reference_candidates()

    payload = {
        "run_id": "nN_residual_grid",
        "top_n_grid": top_n_grid,
        "top_N_grid": top_N_grid,
        "tail_alpha_grid": tail_alpha_grid,
        "selection_rule": "train-holdout macro-F1, accuracy tie-break, lower top-class-share tie-break",
        "strict_selected_name": selected_name,
        "strict_selected_train_holdout": next(row for row in selection_results if row["name"] == selected_name),
        "strict_selected_dev": selected_dev,
        "best_strict": {
            "name": selected_name,
            "train_holdout": next(row for row in selection_results if row["name"] == selected_name),
            "dev": selected_dev,
        },
        "best_diagnostic": best_diagnostic,
        "best_pair_strict": best_pair_strict,
        "best_pair_diagnostic": best_pair_diagnostic,
        "top_train_selected": selection_results[:10],
        "top_dev_diagnostic": dev_results[:10],
        "grid_results": grid_results,
        "all_results": all_results,
        "references": refs,
        "strict_guard": {
            "family_match": train_family == dev_family,
            "train_family": train_family,
            "dev_family": dev_family,
            "forbidden_input_scan": {
                "passed": not bool(forbidden_hits),
                "notes": "Input path scan passed." if not forbidden_hits else str(forbidden_hits),
            },
            "train_holdout_available": True,
        },
        "context_diagnostics": {"train": train_diag, "dev": dev_diag},
        "manifest": manifest_base(
            run_id="nN_residual_grid",
            status="diagnostic_with_train_selected_result",
            mode="STRICT_SELECTION_PLUS_DIAGNOSTIC_DEV_BEST",
            stage="nN_residual_grid",
            command=args.command,
            working_directory=Path.cwd(),
            random_seed=args.random_seed,
        ),
        "runtime_seconds": float(time.perf_counter() - start),
    }

    payload["manifest"]["input_files"] = [
        {"path": str(args.train_claims), "sha256": sha256_file(args.train_claims), "split": "train", "labels_used": True},
        {"path": str(args.dev_claims), "sha256": sha256_file(args.dev_claims), "split": "dev", "labels_used": True},
        {"path": str(args.evidence), "sha256": sha256_file(args.evidence), "split": "evidence", "labels_used": False},
        {"path": str(args.train_pool), "sha256": sha256_file(args.train_pool), "split": "current_run_train_artifact", "labels_used": False},
        {"path": str(args.dev_pool), "sha256": sha256_file(args.dev_pool), "split": "current_run_dev_artifact", "labels_used": False},
    ]
    payload["manifest"]["forbidden_input_scan"] = {
        "passed": not bool(forbidden_hits),
        "notes": "Input path scan passed." if not forbidden_hits else str(forbidden_hits),
    }
    payload["manifest"]["output_files"] = [
        str(args.output_root / "nN_residual_grid_results.json"),
        str(args.report_path),
    ]
    payload["manifest"]["metrics"] = {
        "selection_rule": payload["selection_rule"],
        "train_family": train_family,
        "dev_family": dev_family,
        "strict_family_match": True,
        "train_holdout_fraction": float(args.train_val_fraction),
        "grid_top_n": top_n_grid,
        "grid_top_N": top_N_grid,
        "grid_tail_alpha": tail_alpha_grid,
        "max_context_k": int(max_N),
        "train_context_diag": train_diag,
        "dev_context_diag": dev_diag,
        "strict_selected_name": selected_name,
        "strict_selected_macro_f1": float(selected_dev["macro_f1"]),
        "strict_selected_accuracy": float(selected_dev["accuracy"]),
        "diagnostic_best_name": best_diagnostic["name"],
        "diagnostic_best_macro_f1": float(best_diagnostic["macro_f1"]),
        "diagnostic_best_accuracy": float(best_diagnostic["accuracy"]),
        "references": refs,
    }
    payload["manifest"]["runtime"]["wall_seconds"] = round(float(payload["runtime_seconds"]), 6)
    payload["manifest"]["runtime"]["device"] = "cpu"
    payload["manifest"]["data_flow_summary"] = (
        "Built strict family-matched train/dev contexts at the largest N in the grid, then "
        "evaluated head claim+top-n and tail rank n+1..N branches with separate TF-IDF "
        "and shallow feature pipelines."
    )
    payload["manifest"]["split_isolation_summary"] = (
        "Train-holdout selection used only the train split; dev split was reserved for confirmation "
        "and diagnostic upper-bounding."
    )

    out_path = args.output_root / "nN_residual_grid_results.json"
    write_json(out_path, payload)

    report_lines = [
        "# n/N Residual Classifier Grid",
        "",
        "Architecture: head branch uses claim + top-n evidence; residual branch uses ranks n+1..N evidence.",
        "Both branches use separate TF-IDF word 1-2 vectorizers and separate shallow feature scalers.",
        "",
        "## References",
        "",
        _format_ref_line("Top5 shallow dev-best", refs["top5_shallow_dev_best"]),
        _format_ref_line("k20 promoted dev-best", refs["k20_promoted_dev_best"]),
        "",
        "## Train-Holdout Selected Result",
        "",
        f"- selected config: `{selected_name}`",
        f"- train-holdout macro-F1: `{next(row for row in selection_results if row['name'] == selected_name)['macro_f1']:.6f}`",
        f"- dev macro-F1: `{selected_dev['macro_f1']:.6f}`",
        f"- dev accuracy: `{selected_dev['accuracy']:.6f}`",
        f"- tail alpha: `{selected_dev['tail_alpha']}`",
        f"- head/tail dims: `{selected_dev['head_word_dim']}` / `{selected_dev['tail_word_dim']}` word features, "
        f"`{selected_dev['head_side_dim']}` / `{selected_dev['tail_side_dim']}` side features",
        "",
        "## Best (n, N) By Train-Selected Dev Confirmation",
        "",
        f"- best cell: `n={best_pair_strict['n']}`, `N={best_pair_strict['N']}`",
        f"- selected config: `{best_pair_strict['selected_name']}`",
        f"- selected tail alpha: `{best_pair_strict['selected_tail_alpha']}`",
        f"- dev macro-F1: `{best_pair_strict['train_selected_dev_macro_f1']:.6f}`",
        "",
        "## Dev Diagnostic Best",
        "",
        f"- best config: `{best_diagnostic['name']}`",
        f"- dev macro-F1: `{best_diagnostic['macro_f1']:.6f}`",
        f"- dev accuracy: `{best_diagnostic['accuracy']:.6f}`",
        f"- tail alpha: `{best_diagnostic['tail_alpha']}`",
        "",
        "## Best (n, N) By Dev Diagnostic Upper Bound",
        "",
        f"- best diagnostic cell: `n={best_pair_diagnostic['n']}`, `N={best_pair_diagnostic['N']}`",
        f"- diagnostic config: `{best_pair_diagnostic['diagnostic_name']}`",
        f"- diagnostic tail alpha: `{best_pair_diagnostic['diagnostic_tail_alpha']}`",
        f"- dev macro-F1: `{best_pair_diagnostic['diagnostic_dev_best_macro_f1']:.6f}`",
        "- diagnostic values are non-promoted.",
        "",
        "## Top Train-Holdout Configs",
        "",
        "| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(selection_results[:10], start=1):
        report_lines.append(
            f"| {rank} | `{row['name']}` | {row['top_n']} | {row['top_N']} | {row['tail_alpha']:.2f} | "
            f"{row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['top_class_share']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Top Dev Diagnostic Configs",
            "",
            "| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(dev_results[:10], start=1):
        report_lines.append(
            f"| {rank} | `{row['name']}` | {row['top_n']} | {row['top_N']} | {row['tail_alpha']:.2f} | "
            f"{row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['top_class_share']:.6f} |"
        )
    report_lines.extend(["", f"Artifacts: `{out_path}`"])
    args.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "results": str(out_path),
                "report": str(args.report_path),
                "selected_name": selected_name,
                "selected_dev_macro_f1": float(selected_dev["macro_f1"]),
                "dev_best_macro_f1": float(best_diagnostic["macro_f1"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
