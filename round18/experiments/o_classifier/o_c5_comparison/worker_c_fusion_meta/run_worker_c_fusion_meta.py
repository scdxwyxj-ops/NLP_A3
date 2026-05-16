#!/usr/bin/env python
"""Round18 O-C5 Worker C: strict late-fusion + optional meta learner."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler

# Allow direct execution from repository subdirectories.
for p in Path(__file__).resolve().parents:
    if (p / "round18" / "tools").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round18.tools.common import (
    find_forbidden_tokens,
    manifest_base,
    sha256_file,
    write_json,
)


LABEL_ORDER = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


A_SCRIPT_PATH = Path(
    "round18/experiments/o_classifier/o_c5_comparison/"
    "worker_a_tfidf_side/run_worker_a_tfidf_side.py"
)
A_CONTEXT_K = 5
A_MAX_FEATURES = 30000
A_NGRAM_RANGE = [1, 2]
A_C_VALUE = 1.0


DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/o_c5_comparison/worker_c_fusion_meta")
DEFAULT_REPORT_ROOT = Path("round18/reports/o_c5_classification_comparison")
RUN_ID = "worker_c_fusion_meta"

DEFAULT_B_TRAIN_HOLDOUT_PROBA = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/"
    "worker_b_transformer_text_full_k20_k20_ml256_lr3em05_ep3_train_holdout_proba.json"
)
DEFAULT_B_TRAIN_HOLDOUT_PRED = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/"
    "worker_b_transformer_text_full_k20_k20_ml256_lr3em05_ep3_train_holdout_predictions.json"
)
DEFAULT_B_DEV_PROBA = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/"
    "worker_b_transformer_text_full_k20_k20_ml256_lr3em05_ep3_dev_proba.json"
)
DEFAULT_B_DEV_PRED = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/"
    "worker_b_transformer_text_full_k20_k20_ml256_lr3em05_ep3_dev_predictions.json"
)

DEFAULT_D_TRAIN_HOLDOUT_PROBA = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_d_embedding_classifier/"
    "worker_d_embedding_classifier_final_k20_embedding_only_train_holdout_proba.json"
)
DEFAULT_D_TRAIN_HOLDOUT_PRED = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_d_embedding_classifier/"
    "worker_d_embedding_classifier_final_k20_embedding_only_train_holdout_predictions.json"
)
DEFAULT_D_DEV_PROBA = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_d_embedding_classifier/"
    "worker_d_embedding_classifier_final_k20_embedding_only_dev_proba.json"
)
DEFAULT_D_DEV_PRED = Path(
    "round18/outputs/o_classifier/o_c5_comparison/worker_d_embedding_classifier/"
    "worker_d_embedding_classifier_final_k20_embedding_only_dev_predictions.json"
)

WEIGHT_GRID_STEP = 0.1
C_GRID_DEFAULT = "0.05,0.1,0.5,1.0,2.0,5.0"
BASELINE_MACRO_F1 = 0.495421
COLLAPSE_THRESHOLD = 0.70
TRAIN_HOLDOUT_FRACTION = 0.2
RANDOM_SEED = 1337


def _resolve_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "round18" / "tools").exists():
            return candidate
    return Path(__file__).resolve().parents[7]


def _add_repo_to_path() -> None:
    repo_root = _resolve_repo_root()
    for p in (repo_root, repo_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("worker_a_tfidf_side", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round18 O-C5 Worker C: strict/diagnostic fusion.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--weight-step", type=float, default=WEIGHT_GRID_STEP)
    parser.add_argument("--meta-c-grid", default=C_GRID_DEFAULT)
    parser.add_argument("--disable-meta", action="store_true")
    parser.add_argument("--baseline-macro-f1", type=float, default=BASELINE_MACRO_F1)
    parser.add_argument("--collapse-threshold", type=float, default=COLLAPSE_THRESHOLD)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--b-train-holdout-proba", type=Path, default=DEFAULT_B_TRAIN_HOLDOUT_PROBA)
    parser.add_argument("--b-train-holdout-pred", type=Path, default=DEFAULT_B_TRAIN_HOLDOUT_PRED)
    parser.add_argument("--b-dev-proba", type=Path, default=DEFAULT_B_DEV_PROBA)
    parser.add_argument("--b-dev-pred", type=Path, default=DEFAULT_B_DEV_PRED)
    parser.add_argument("--d-train-holdout-proba", type=Path, default=DEFAULT_D_TRAIN_HOLDOUT_PROBA)
    parser.add_argument("--d-train-holdout-pred", type=Path, default=DEFAULT_D_TRAIN_HOLDOUT_PRED)
    parser.add_argument("--d-dev-proba", type=Path, default=DEFAULT_D_DEV_PROBA)
    parser.add_argument("--d-dev-pred", type=Path, default=DEFAULT_D_DEV_PRED)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: float, denominator: float) -> float:
    return _safe_float(numerator) / _safe_float(denominator) if _safe_float(denominator) else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _write_csv_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(v) for v in row]])


def _parse_float_grid(raw: str) -> list[float]:
    vals: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise argparse.ArgumentTypeError("meta c-grid must not be empty")
    return vals


def _evaluate_predictions(y_true: list[str], y_pred: list[str], y_prob: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    row_totals = matrix.sum(axis=1).astype(float)
    per_class_map: dict[str, float] = {}
    for idx, label in enumerate(LABEL_ORDER):
        per_class_map[label] = _safe_div(float(matrix[idx, idx]), float(row_totals[idx]))
    hist = Counter(y_pred)
    pred_hist = {label: int(hist.get(label, 0)) for label in LABEL_ORDER}
    top_share = max(pred_hist.values()) / max(len(y_pred), 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=LABEL_ORDER, average="micro", zero_division=0)),
        "classification_report": report,
        "per_class_recall": dict(zip(LABEL_ORDER, [float(v) for v in per_class_recall])),
        "prediction_histogram": pred_hist,
        "top_class_share": float(top_share),
        "top_class": max(pred_hist.items(), key=lambda kv: (kv[1], kv[0]))[0],
        "confusion_matrix": matrix.tolist(),
        "proba": y_prob.tolist() if y_prob.size else [],
    }


def _collapse_gate(histogram: dict[str, int], threshold: float) -> dict[str, Any]:
    total = sum(histogram.get(label, 0) for label in LABEL_ORDER)
    if total <= 0:
        return {
            "status": "failed",
            "reason": "no predictions produced",
            "threshold": float(threshold),
            "max_class_share": 0.0,
            "collapsed_classes": [],
        }
    max_label, max_count = max(histogram.items(), key=lambda kv: (kv[1], kv[0]))
    share = max_count / total
    collapsed = [label for label, count in histogram.items() if _safe_div(count, total) > threshold]
    return {
        "status": "passed" if share <= threshold else "failed",
        "reason": (
            "passed" if share <= threshold else f"single-class collapse above threshold {threshold}: {max_label}"
        ),
        "max_class": max_label,
        "max_class_share": float(share),
        "threshold": float(threshold),
        "collapsed_classes": collapsed,
    }


def _acceptance_gate(metrics: dict[str, Any], threshold_f1: float, collapse_threshold: float) -> dict[str, Any]:
    failures: list[str] = []
    if any(int(metrics["prediction_histogram"].get(label, 0)) <= 0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_predictions")
    if any(_safe_float(metrics["per_class_recall"].get(label, 0.0)) <= 0.0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_recall")
    if _safe_float(metrics.get("top_class_share", 1.0)) > collapse_threshold:
        failures.append("max_class_share_above_threshold")
    if _safe_float(metrics.get("macro_f1", 0.0)) < threshold_f1:
        failures.append("macro_f1_below_baseline")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "thresholds": {
            "max_class_share": float(collapse_threshold),
            "min_macro_f1": float(threshold_f1),
            "require_nonzero_predictions_for_all_classes": True,
            "require_nonzero_recall_for_all_classes": True,
        },
    }


def _train_tfidf_plus_side_fixed(
    train_texts: list[str],
    y_train: list[str],
    dev_texts: list[str],
    y_dev: list[str],
    train_side: np.ndarray,
    dev_side: np.ndarray | None,
    c_value: float,
    random_state: int = 1337,
    split: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=tuple(A_NGRAM_RANGE),
        max_features=A_MAX_FEATURES,
        dtype=np.float32,
    )

    x_train_text_all = tfidf.fit_transform(train_texts)
    x_train_text = x_train_text_all
    x_dev_text = tfidf.transform(dev_texts)
    y_train_use = y_train
    y_dev_use = y_dev

    if split is not None:
        tr_idx, va_idx = split
        x_train_text = x_train_text_all[tr_idx]
        y_train_use = [y_train[i] for i in tr_idx.tolist()]
        x_dev_text = x_dev_text[va_idx]
        y_dev_use = [y_dev[i] for i in va_idx.tolist()]
        side_tr = train_side[tr_idx]
        side_va = train_side[va_idx]
    else:
        side_tr = train_side
        side_va = dev_side

    scaler = MaxAbsScaler()
    x_train_side = scaler.fit_transform(side_tr)
    x_dev_side = scaler.transform(side_va) if side_va is not None else side_tr
    x_train = hstack([x_train_text, x_train_side], format="csr")
    x_dev = hstack([x_dev_text, x_dev_side], format="csr")

    clf = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    fit_start = time.perf_counter()
    clf.fit(x_train, y_train_use)
    fit_seconds = time.perf_counter() - fit_start
    y_pred = clf.predict(x_dev).astype(str).tolist()
    y_prob = clf.predict_proba(x_dev).astype(float)
    metrics = _evaluate_predictions(y_dev_use, y_pred, y_prob)
    metrics["fit_seconds"] = float(fit_seconds)
    metrics["selection_trace"] = {
        "selection_mode": "manual_replay",
        "model_id": "tfidf_with_shallow",
        "tfidf_max_features": A_MAX_FEATURES,
        "tfidf_ngram_range": A_NGRAM_RANGE,
        "C": float(c_value),
        "feature_mode": "tfidf_with_shallow",
    }
    return metrics, y_pred, y_prob


def _claim_ids_from_predictions_path(path: Path) -> list[str]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid prediction payload: {path}")
    if "claim_ids" in payload and isinstance(payload["claim_ids"], list):
        return [str(v) for v in payload["claim_ids"]]
    if "predictions" in payload and isinstance(payload["predictions"], dict):
        return [str(v) for v in payload["predictions"].keys()]
    if payload and all(isinstance(k, str) for k in payload):
        # legacy style: keys are claim ids
        return [str(k) for k in payload.keys()]
    raise RuntimeError(f"Cannot infer claim ids from: {path}")


def _load_prob_payload(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid probability payload: {path}")
    claim_ids = [str(v) for v in payload.get("claim_ids", [])]
    probs = np.asarray(payload.get("probabilities", []), dtype=float)
    if probs.ndim != 2:
        raise RuntimeError(f"Probability payload matrix invalid: {path}")
    source_labels = [str(x) for x in payload.get("label_order", LABEL_ORDER)]
    if probs.shape[1] != len(source_labels):
        raise RuntimeError(f"Probability payload label dimension mismatch: {path}")
    return claim_ids, probs, source_labels


def _reorder_probabilities(probs: np.ndarray, source_labels: list[str], target_labels: list[str]) -> np.ndarray:
    if list(source_labels) == list(target_labels):
        return probs
    idx = [source_labels.index(lbl) for lbl in target_labels]
    return probs[:, idx]


def _align_by_ids(
    source_ids: list[str],
    source_probs: np.ndarray,
    target_ids: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    index_map = {str(cid): i for i, cid in enumerate(source_ids)}
    aligned: list[list[float]] = []
    missing_ids: list[str] = []
    for cid in target_ids:
        idx = index_map.get(str(cid))
        if idx is None:
            missing_ids.append(str(cid))
            aligned.append([0.0, 0.0, 0.0, 0.0])
        else:
            aligned.append([_safe_float(v) for v in source_probs[idx].tolist()])
    return np.asarray(aligned, dtype=float), {
        "target_size": len(target_ids),
        "source_size": int(source_probs.shape[0]),
        "aligned": len(missing_ids) == 0,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids[:20],
    }


def _parse_val_ids(ids: list[str]) -> tuple[bool, list[str]]:
    vals: list[str] = []
    for i, raw in enumerate(ids):
        m = re.match(r"^val-(\d+)$", str(raw))
        if m is None:
            return False, []
        vals.append(m.group(1))
    if vals != [str(i) for i in range(len(ids))]:
        return False, vals
    return True, vals


def _build_a_replay(run_id: str, output_root: Path, random_seed: int) -> dict[str, Any]:
    mod = _load_module(A_SCRIPT_PATH)
    saved_argv = list(sys.argv)
    sys.argv = [
        str(A_SCRIPT_PATH),
        "--context-ks",
        str(A_CONTEXT_K),
        "--run-tfidf-shallow",
        "--n-jobs",
        "1",
        "--train-val-fraction",
        str(TRAIN_HOLDOUT_FRACTION),
        "--random-seed",
        str(random_seed),
    ]
    try:
        a_args = mod._parse_args()
    finally:
        sys.argv = saved_argv
    a_args.command = f"python {A_SCRIPT_PATH}"
    a_args.context_ks = [A_CONTEXT_K]
    a_args.train_val_fraction = float(TRAIN_HOLDOUT_FRACTION)
    a_args.selection_mode = "train_holdout"
    a_args.context_k = A_CONTEXT_K

    train_claims, dev_claims, evidence, train_pool, dev_pool = mod._load_inputs(a_args)
    mod._ensure_family_contract(train_pool=a_args.train_pool, dev_pool=a_args.dev_pool)
    train_family = "o_ce_factual_context_for_classifier_alias"

    train_contexts, _ = mod._build_contexts(
        train_claims,
        evidence,
        train_pool,
        top_k=A_CONTEXT_K,
        source_family=train_family,
    )
    dev_contexts, _ = mod._build_contexts(
        dev_claims,
        evidence,
        dev_pool,
        top_k=A_CONTEXT_K,
        source_family=train_family,
    )

    train_ids, train_texts, y_train, train_side_rows = mod._rows_from_contexts(
        train_claims, train_contexts, evidence_token_budget=0
    )
    dev_ids, dev_texts, y_dev, dev_side_rows = mod._rows_from_contexts(
        dev_claims, dev_contexts, evidence_token_budget=0
    )

    train_side_matrix, _ = mod._build_shallow_matrix(
        train_ids,
        mod._build_profile_cache(train_claims),
        train_side_rows,
        evidence,
    )
    dev_side_matrix, _ = mod._build_shallow_matrix(
        dev_ids,
        mod._build_profile_cache(dev_claims),
        dev_side_rows,
        evidence,
    )

    split = mod._train_holdout_indices(y_train, a_args)
    if split is None:
        raise RuntimeError("A replay failed: cannot generate train-holdout split")
    train_idx, hold_idx = split
    holdout_ids = [str(train_ids[i]) for i in hold_idx.tolist()]
    holdout_labels = [y_train[i] for i in hold_idx.tolist()]

    hold_metrics, hold_pred, hold_probs = _train_tfidf_plus_side_fixed(
        train_texts=train_texts,
        y_train=y_train,
        dev_texts=train_texts,
        y_dev=y_train,
        train_side=train_side_matrix,
        dev_side=train_side_matrix,
        c_value=A_C_VALUE,
        random_state=random_seed,
        split=(train_idx, hold_idx),
    )

    dev_metrics, dev_pred, dev_probs = _train_tfidf_plus_side_fixed(
        train_texts=train_texts,
        y_train=y_train,
        dev_texts=dev_texts,
        y_dev=y_dev,
        train_side=train_side_matrix,
        dev_side=dev_side_matrix,
        c_value=A_C_VALUE,
        random_state=random_seed,
        split=None,
    )

    holdout_payload = mod._build_predictions_payload(
        holdout_ids,
        train_claims,
        train_contexts,
        hold_pred,
        hold_probs,
    )
    dev_payload = mod._build_predictions_payload(
        dev_ids,
        dev_claims,
        dev_contexts,
        dev_pred,
        dev_probs,
    )

    _write_json(
        output_root / f"{run_id}_a_k{A_CONTEXT_K}_train_holdout_proba.json",
        {
            "label_order": LABEL_ORDER,
            "claim_ids": holdout_ids,
            "probabilities": np.asarray(hold_probs, dtype=float).tolist(),
        },
    )
    _write_json(
        output_root / f"{run_id}_a_k{A_CONTEXT_K}_dev_proba.json",
        {
            "label_order": LABEL_ORDER,
            "claim_ids": [str(i) for i in dev_ids],
            "probabilities": np.asarray(dev_probs, dtype=float).tolist(),
        },
    )
    _write_json(
        output_root / f"{run_id}_a_k{A_CONTEXT_K}_train_holdout_predictions.json",
        holdout_payload,
    )
    _write_json(
        output_root / f"{run_id}_a_k{A_CONTEXT_K}_dev_predictions.json",
        dev_payload,
    )
    _write_json(
        output_root / f"{run_id}_a_k{A_CONTEXT_K}_train_holdout_metrics.json",
        hold_metrics,
    )
    _write_json(
        output_root / f"{run_id}_a_k{A_CONTEXT_K}_dev_metrics.json",
        dev_metrics,
    )

    return {
        "train_claims": train_claims,
        "dev_claims": dev_claims,
        "train_contexts": train_contexts,
        "dev_contexts": dev_contexts,
        "holdout_ids": holdout_ids,
        "holdout_labels": holdout_labels,
        "holdout_probs": np.asarray(hold_probs, dtype=float),
        "holdout_pred": [str(p) for p in hold_pred],
        "holdout_metrics": hold_metrics,
        "dev_ids": [str(i) for i in dev_ids],
        "dev_labels": [str(v) for v in y_dev],
        "dev_probs": np.asarray(dev_probs, dtype=float),
        "dev_pred": [str(p) for p in dev_pred],
        "dev_metrics": dev_metrics,
    }


def _load_b_payloads(args: argparse.Namespace, train_holdout_size: int, dev_size: int) -> dict[str, Any]:
    b_hold_ids, b_hold_probs, b_hold_src_labels = _load_prob_payload(args.b_train_holdout_proba)
    if b_hold_probs.shape[0] != train_holdout_size:
        raise RuntimeError("B holdout row count does not match A holdout size.")
    b_hold_ok, _ = _parse_val_ids(b_hold_ids)
    _load_json(args.b_train_holdout_pred)  # keep input availability check

    b_dev_prob_ids, b_dev_probs, b_dev_src_labels = _load_prob_payload(args.b_dev_proba)
    b_dev_pred_ids = _claim_ids_from_predictions_path(args.b_dev_pred)
    if b_dev_probs.shape[0] != dev_size:
        raise RuntimeError("B dev probability row count does not match A dev size.")
    if len(b_dev_pred_ids) != b_dev_probs.shape[0]:
        raise RuntimeError("B dev prediction count does not match B dev proba count.")
    # B dev payload has no claim_ids inside proba; use prediction order
    b_dev_probs = _reorder_probabilities(b_dev_probs, b_dev_src_labels, LABEL_ORDER)

    if len(b_hold_src_labels) != 0:
        b_hold_probs = _reorder_probabilities(b_hold_probs, b_hold_src_labels, LABEL_ORDER)

    if b_hold_ids == []:
        b_hold_ids = [f"val-{i}" for i in range(b_hold_probs.shape[0])]
    if b_dev_ids := b_dev_prob_ids:
        if len(b_dev_ids) != len(b_dev_pred_ids):
            raise RuntimeError("B dev proba ids (if present) and prediction ids mismatch.")
    else:
        b_dev_ids = b_dev_pred_ids

    return {
        "train_holdout": {"claim_ids": b_hold_ids, "probabilities": b_hold_probs, "val_placeholder": bool(b_hold_ok)},
        "dev": {"claim_ids": b_dev_ids, "probabilities": b_dev_probs, "prediction_claim_ids": b_dev_pred_ids},
        "holdout_val_placeholder_valid": b_hold_ok,
        "alignment": {
            "holdout_val_placeholder_count": len(b_hold_ids),
            "holdout_val_placeholder_valid": b_hold_ok,
        },
    }


def _load_d_payloads(args: argparse.Namespace, train_holdout_size: int, dev_size: int) -> dict[str, Any]:
    d_hold_ids, d_hold_probs, d_hold_src_labels = _load_prob_payload(args.d_train_holdout_proba)
    if d_hold_probs.shape[0] != train_holdout_size:
        raise RuntimeError("D holdout row count does not match A holdout size.")
    d_hold_probs = _reorder_probabilities(d_hold_probs, d_hold_src_labels, LABEL_ORDER)
    _load_json(args.d_train_holdout_pred)

    d_dev_ids, d_dev_probs, d_dev_src_labels = _load_prob_payload(args.d_dev_proba)
    if d_dev_probs.shape[0] != dev_size:
        raise RuntimeError("D dev row count does not match A dev size.")
    d_dev_probs = _reorder_probabilities(d_dev_probs, d_dev_src_labels, LABEL_ORDER)
    _load_json(args.d_dev_pred)

    return {
        "train_holdout": {"claim_ids": [str(x) for x in d_hold_ids], "probabilities": d_hold_probs},
        "dev": {"claim_ids": [str(x) for x in d_dev_ids], "probabilities": d_dev_probs},
    }


def _prepare_alignment(
    a_result: dict[str, Any],
    b_result: dict[str, Any],
    d_result: dict[str, Any],
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "a": {
            "holdout_size": len(a_result["holdout_ids"]),
            "dev_size": len(a_result["dev_ids"]),
        },
        "b": {
            "holdout_placeholder_mode": b_result["train_holdout"].get("val_placeholder", False),
            "holdout_placeholder_valid": b_result.get("holdout_val_placeholder_valid", False),
            "holdout_alignment": {},
            "dev": {},
        },
        "d": {
            "holdout_alignment": {},
            "dev": {},
        },
    }

    b_hold_ids = b_result["train_holdout"]["claim_ids"]
    b_hold_aligned = np.asarray(b_result["train_holdout"]["probabilities"], dtype=float)
    if (
        len(b_hold_ids) == len(a_result["holdout_ids"])
        and b_result["train_holdout"].get("val_placeholder", False) is False
    ):
        trace["b"]["holdout_alignment"] = {
            "strategy": "claim_id_aligned",
            "status": "passed",
            "missing_count": 0,
        }
        b_result["train_holdout"]["aligned_claim_ids"] = [str(i) for i in a_result["holdout_ids"]]
        b_result["train_holdout"]["aligned_probabilities"] = b_hold_aligned
    elif (
        len(b_hold_ids) == len(a_result["holdout_ids"])
        and b_result["train_holdout"].get("val_placeholder", False) is True
    ):
        trace["b"]["holdout_alignment"] = {
            "strategy": "val_index",
            "status": "failed_for_strict_claim_alignment",
            "missing_count": len(a_result["holdout_ids"]),
        }
        b_result["train_holdout"]["aligned_probabilities"] = b_hold_aligned[: len(a_result["holdout_ids"])]
        b_result["train_holdout"]["aligned_claim_ids"] = a_result["holdout_ids"]
    else:
        trace["b"]["holdout_alignment"] = {
            "strategy": "claim_id_mismatch",
            "status": "failed",
            "missing_count": len(a_result["holdout_ids"]),
        }
        b_result["train_holdout"]["aligned_probabilities"] = b_hold_aligned[: len(a_result["holdout_ids"])]
        b_result["train_holdout"]["aligned_claim_ids"] = a_result["holdout_ids"]

    d_hold_aligned, d_hold_trace = _align_by_ids(
        d_result["train_holdout"]["claim_ids"],
        d_result["train_holdout"]["probabilities"],
        a_result["holdout_ids"],
    )
    b_dev_aligned, b_dev_trace = _align_by_ids(
        b_result["dev"]["claim_ids"],
        b_result["dev"]["probabilities"],
        a_result["dev_ids"],
    )
    d_dev_aligned, d_dev_trace = _align_by_ids(
        d_result["dev"]["claim_ids"],
        d_result["dev"]["probabilities"],
        a_result["dev_ids"],
    )
    b_result["dev"]["aligned_claim_ids"] = [str(i) for i in a_result["dev_ids"]]
    b_result["dev"]["aligned_probabilities"] = b_dev_aligned
    d_result["train_holdout"]["aligned_claim_ids"] = [str(i) for i in a_result["holdout_ids"]]
    d_result["train_holdout"]["aligned_probabilities"] = d_hold_aligned
    d_result["dev"]["aligned_claim_ids"] = [str(i) for i in a_result["dev_ids"]]
    d_result["dev"]["aligned_probabilities"] = d_dev_aligned

    trace["b"]["dev"] = b_dev_trace
    trace["d"]["holdout_alignment"] = d_hold_trace
    trace["d"]["dev"] = d_dev_trace

    strict_possible = (
        len(b_result["train_holdout"]["claim_ids"]) == len(a_result["holdout_ids"])
        and b_result["train_holdout"].get("val_placeholder", False) is False
        and trace["d"]["holdout_alignment"].get("aligned", False)
        and trace["b"]["dev"].get("missing_count", len(a_result["dev_ids"])) == 0
        and trace["d"]["dev"].get("missing_count", len(a_result["dev_ids"])) == 0
    )
    trace["strict_possible"] = bool(strict_possible)
    return trace


def _weight_grid(step: float) -> list[tuple[float, float, float]]:
    if step <= 0.0:
        raise ValueError("weight-step must be >0.")
    values = np.arange(0.0, 1.0 + 1e-12, step)
    combos: list[tuple[float, float, float]] = []
    for wa in values:
        for wb in values:
            wc = 1.0 - wa - wb
            if wc < -1e-12:
                continue
            combos.append((float(round(wa, 10)), float(round(wb, 10)), float(round(wc, 10))))
    uniq: dict[tuple[float, float, float], tuple[float, float, float]] = {}
    for wa, wb, wc in combos:
        s = round(wa + wb + wc, 10)
        if abs(s - 1.0) <= 1e-6:
            uniq[(wa, wb, wc)] = (wa, wb, wc)
    return sorted(uniq.values())


def _selection_score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["macro_recall"]),
        float(metrics["accuracy"]),
        -float(metrics.get("top_class_share", 1.0)),
    )


def _run_late_fusion(
    a_probs_sel: np.ndarray,
    b_probs_sel: np.ndarray,
    d_probs_sel: np.ndarray,
    y_sel: list[str],
    a_probs_eval: np.ndarray,
    b_probs_eval: np.ndarray,
    d_probs_eval: np.ndarray,
    y_eval: list[str],
    selection_mode: str,
    weight_step: float,
    collapse_threshold: float,
) -> dict[str, Any]:
    if not (a_probs_sel.shape == b_probs_sel.shape == d_probs_sel.shape):
        raise ValueError("A/B/D holdout/dev probability shapes mismatch.")
    if not (a_probs_eval.shape == b_probs_eval.shape == d_probs_eval.shape):
        raise ValueError("A/B/D evaluation probability shapes mismatch.")
    if a_probs_sel.shape[1] != len(LABEL_ORDER):
        raise ValueError("Probability class dimension mismatch.")

    candidates = _weight_grid(weight_step)
    rows: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0, -1.0, 1.0)
    best: dict[str, Any] | None = None

    for wa, wb, wc in candidates:
        fused_sel = wa * a_probs_sel + wb * b_probs_sel + wc * d_probs_sel
        pred_idx_sel = np.argmax(fused_sel, axis=1).astype(int).tolist()
        pred_sel = [LABEL_ORDER[i] for i in pred_idx_sel]
        metrics_sel = _evaluate_predictions(y_sel, pred_sel, fused_sel)
        row: dict[str, Any] = {
            "weights": {"a": wa, "b": wb, "d": wc},
            "sum_weights": float(wa + wb + wc),
            "macro_f1": metrics_sel["macro_f1"],
            "macro_recall": metrics_sel["macro_recall"],
            "accuracy": metrics_sel["accuracy"],
            "top_class_share": metrics_sel["top_class_share"],
            "collapse_gate": _collapse_gate(metrics_sel["prediction_histogram"], collapse_threshold),
        }
        rows.append(row)
        score = _selection_score(metrics_sel)
        if score > best_score:
            fused_eval = wa * a_probs_eval + wb * b_probs_eval + wc * d_probs_eval
            pred_idx_eval = np.argmax(fused_eval, axis=1).astype(int).tolist()
            pred_eval = [LABEL_ORDER[i] for i in pred_idx_eval]
            metrics_eval = _evaluate_predictions(y_eval, pred_eval, fused_eval)
            best_score = score
            best = {
                "selection_mode": selection_mode,
                "selection_mode_note": "holdout" if selection_mode == "train_holdout" else "dev_only",
                "weights": {"a": wa, "b": wb, "d": wc},
                "selection_size": len(y_sel),
                "selection_metrics": metrics_sel,
                "selection_rows_count": len(candidates),
                "selection_rows": rows,
                "dev_metrics": metrics_eval,
                "dev_proba": fused_eval,
                "dev_pred": pred_eval,
                "dev_pred_claim_ids": [],
            }

    if best is None:
        raise RuntimeError("No fusion candidates evaluated.")
    return best


def _run_meta_logreg(
    a_train: np.ndarray,
    b_train: np.ndarray,
    d_train: np.ndarray,
    y_train: list[str],
    a_dev: np.ndarray,
    b_dev: np.ndarray,
    d_dev: np.ndarray,
    y_dev: list[str],
    c_grid: list[float],
    random_seed: int,
    collapse_threshold: float,
    baseline_f1: float,
) -> dict[str, Any]:
    if len(y_train) < 40:
        return {
            "status": "skipped_small_split",
            "reason": "train split too small for stratified CV",
        }

    X_train = np.concatenate([a_train, b_train, d_train], axis=1)
    X_dev = np.concatenate([a_dev, b_dev, d_dev], axis=1)
    y_idx = np.array([LABEL_ORDER.index(v) for v in y_train], dtype=int)
    y_dev_idx = [LABEL_ORDER.index(v) for v in y_dev]

    n_classes = len(np.unique(y_idx))
    n_folds = min(5, len(y_idx) // max(2, n_classes))
    if n_folds < 2 or n_classes < 2:
        return {
            "status": "skipped_few_classes_or_folds",
            "reason": "insufficient classes/folds for CV",
        }

    selector = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    best_cv = -1.0
    best_c = float(c_grid[0])
    cv_rows: list[dict[str, Any]] = []
    for c_value in c_grid:
        fold_scores: list[float] = []
        for train_idx, val_idx in selector.split(X_train, y_idx):
            mdl = LogisticRegression(
                C=float(c_value),
                max_iter=2000,
                n_jobs=1,
                multi_class="auto",
                class_weight="balanced",
            )
            mdl.fit(X_train[train_idx], y_idx[train_idx])
            pred_idx = mdl.predict(X_train[val_idx]).astype(int).tolist()
            truth = y_idx[val_idx].astype(int).tolist()
            fold_scores.append(
                f1_score(
                    truth,
                    pred_idx,
                    labels=list(range(len(LABEL_ORDER))),
                    average="macro",
                    zero_division=0,
                )
            )
        mean_score = float(np.mean(fold_scores))
        cv_rows.append({
            "c": float(c_value),
            "cv_macro_f1_mean": mean_score,
            "cv_macro_f1_std": float(np.std(fold_scores)),
            "cv_scores": [float(v) for v in fold_scores],
            "folds": len(fold_scores),
        })
        if mean_score > best_cv:
            best_cv = mean_score
            best_c = float(c_value)

    model = LogisticRegression(
        C=best_c,
        max_iter=2000,
        n_jobs=1,
        multi_class="auto",
        class_weight="balanced",
    )
    fit_t0 = time.perf_counter()
    model.fit(X_train, y_idx)
    fit_seconds = time.perf_counter() - fit_t0
    train_pred_idx = model.predict(X_train).astype(int).tolist()
    train_pred = [LABEL_ORDER[i] for i in train_pred_idx]
    train_metrics = _evaluate_predictions([LABEL_ORDER[i] for i in y_idx.tolist()], train_pred, model.predict_proba(X_train))

    dev_proba = model.predict_proba(X_dev)
    dev_pred_idx = np.argmax(dev_proba, axis=1).astype(int).tolist()
    dev_pred = [LABEL_ORDER[i] for i in dev_pred_idx]
    dev_metrics = _evaluate_predictions(y_dev, dev_pred, dev_proba)

    return {
        "method": "meta_logreg",
        "selection_mode": "train_holdout",
        "status": "ok",
        "best_c": best_c,
        "cv_best_score": best_cv,
        "cv_rows": cv_rows,
        "fit_seconds": float(fit_seconds),
        "train_metrics": train_metrics,
        "train_pred": train_pred,
        "train_proba": model.predict_proba(X_train),
        "dev_metrics": dev_metrics,
        "dev_proba": dev_proba,
        "dev_pred": dev_pred,
        "acceptance_gate": _acceptance_gate(dev_metrics, baseline_f1, collapse_threshold),
    }


def _build_fusion_payload(
    claim_ids: list[str],
    probs: np.ndarray,
    label_lookup: dict[str, str],
    label_probs: list[str],
    method_tag: str,
    claim_lookup: dict[str, dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pred_idx = np.argmax(probs, axis=1).astype(int).tolist() if probs.size else []
    payload: dict[str, Any] = {}
    for claim_id, p, idx in zip(claim_ids, probs.tolist(), pred_idx):
        pred_label = label_probs[idx]
        payload[str(claim_id)] = {
            "claim_label": label_lookup.get(str(claim_id), ""),
            "claim_text": claim_lookup.get(str(claim_id), {}).get("claim_text", ""),
            "evidences": evidence_lookup.get(str(claim_id), {}).get("final_evidence_candidates", []),
            "label_distribution": [{"label": lbl, "prob": float(v)} for lbl, v in zip(label_probs, p)],
            "pred_class": pred_label,
            "pred_class_probability": float(max(p)) if p else 0.0,
        }
    return payload


def _emit_method_outputs(
    output_root: Path,
    run_id: str,
    mode: str,
    method: str,
    claim_ids: list[str],
    probs: np.ndarray,
    claim_lookup: dict[str, dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
    truth: list[str],
    baseline_f1: float,
    collapse_threshold: float,
    meta: bool = False,
) -> dict[str, Any]:
    pred_idx = np.argmax(probs, axis=1).astype(int).tolist() if probs.size else []
    y_pred = [LABEL_ORDER[i] for i in pred_idx] if pred_idx else []
    metrics = _evaluate_predictions(truth, y_pred, probs, )
    metrics["acceptance_gate"] = _acceptance_gate(metrics, baseline_f1, collapse_threshold)
    metrics["collapse_gate"] = _collapse_gate(metrics["prediction_histogram"], collapse_threshold)

    pred_payload = _build_fusion_payload(
        claim_ids=claim_ids,
        probs=probs,
        label_lookup={cid: str(truth[i]) for i, cid in enumerate(claim_ids)},
        label_probs=LABEL_ORDER,
        method_tag=method,
        claim_lookup=claim_lookup,
        evidence_lookup=evidence_lookup,
    )

    suffix = "_meta" if meta else "_late"
    pred_path = output_root / f"{run_id}_{mode}_{method}{suffix}_dev_predictions.json"
    proba_path = output_root / f"{run_id}_{mode}_{method}{suffix}_dev_proba.json"
    metric_path = output_root / f"{run_id}_{mode}_{method}{suffix}_dev_metrics.json"
    report_path = output_root / f"{run_id}_{mode}_{method}{suffix}_dev_classification_report.txt"
    conf_path = output_root / f"{run_id}_{mode}_{method}{suffix}_dev_confusion_matrix.csv"

    _write_json(pred_path, pred_payload)
    _write_json(
        proba_path,
        {"label_order": LABEL_ORDER, "claim_ids": [str(cid) for cid in claim_ids], "probabilities": np.asarray(probs, dtype=float).tolist()},
    )
    _write_json(metric_path, metrics)
    report_path.write_text(metrics["classification_report"], encoding="utf-8")
    _write_csv_matrix(conf_path, np.asarray(metrics["confusion_matrix"], dtype=float), LABEL_ORDER)

    return {
        "predictions_path": str(pred_path),
        "proba_path": str(proba_path),
        "metrics_path": str(metric_path),
        "metrics": metrics,
        "prediction_count": len(claim_ids),
    }


def _write_report(
    path: Path,
    baseline_macro: float,
    strict_possible: bool,
    a_result: dict[str, Any],
    align: dict[str, Any],
    strict_result: dict[str, Any] | None,
    diagnostic_result: dict[str, Any] | None,
    elapsed: float,
) -> None:
    lines: list[str] = []
    lines.append("# O-C5 Worker C: Fusion & Meta Learner\n")
    lines.append(f"- baseline (A k5 TF-IDF + shallow macro-F1): `{baseline_macro:.6f}`\n")
    lines.append(f"- strict alignment possible: `{strict_possible}`\n")
    lines.append(f"- elapsed wall seconds: `{elapsed:.2f}`\n\n")
    lines.append("## A k5 selected config replay\n")
    lines.append(f"- holdout macro-F1: `{a_result['holdout_metrics'].get('macro_f1', 0.0):.6f}`\n")
    lines.append(f"- dev macro-F1: `{a_result['dev_metrics'].get('macro_f1', 0.0):.6f}`\n")
    lines.append("## claim_id alignment\n")
    lines.append(f"- B holdout uses placeholder `val-*` rows: `{align.get('b', {}).get('holdout_placeholder_mode', False)}`\n")
    lines.append(f"- B holdout placeholder mapping valid: `{align.get('b', {}).get('holdout_placeholder_valid', False)}`\n")
    lines.append(f"- B holdout alignment: `{align.get('b', {}).get('holdout_alignment', {}).get('status', 'n/a')}`\n")
    lines.append(f"- D holdout alignment missing count: `{align.get('d', {}).get('holdout_alignment', {}).get('missing_count', 'n/a')}`\n")
    lines.append(f"- D dev alignment missing count: `{align.get('d', {}).get('dev', {}).get('missing_count', 'n/a')}`\n")
    lines.append(f"- B dev alignment missing count: `{align.get('b', {}).get('dev', {}).get('missing_count', 'n/a')}`\n\n")

    lines.append("## strict vs diagnostic\n")
    lines.append("| mode | method | status | macro_f1 | accuracy | macro_recall | top_class_share | collapse | acceptance |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---|---|\n")

    for mode_name, payload in [
        ("strict", strict_result),
        ("diagnostic", diagnostic_result),
    ]:
        if payload is None:
            lines.append(f"| {mode_name} | - | skipped | 0.000000 | 0.000000 | 0.000000 | 0.0000 | - | skipped |\n")
            continue
        for method_key in ("late_fusion", "meta_logreg"):
            method_payload = payload.get(method_key, {})
            if not method_payload:
                continue
            meth_metrics = method_payload.get("dev_metrics") or method_payload.get("metrics")
            if not isinstance(meth_metrics, dict):
                continue
            status = method_payload.get("status", "ok")
            if "acceptance_gate" not in meth_metrics:
                status = meth_metrics.get("acceptance_gate", {}).get("status", status)
            lines.append(
                f"| {mode_name} | {method_key} | {status} | "
                f"{meth_metrics.get('macro_f1', 0.0):.6f} | {meth_metrics.get('accuracy', 0.0):.6f} | "
                f"{meth_metrics.get('macro_recall', 0.0):.6f} | "
                f"{meth_metrics.get('top_class_share', 0.0):.4f} | "
                f"{meth_metrics.get('collapse_gate', {}).get('status', 'n/a')} | "
                f"{meth_metrics.get('acceptance_gate', {}).get('status', 'n/a')} |\n"
            )

    lines.append("\n## collapse and per-class checks\n")
    lines.append("Each result includes `prediction_histogram`, `per_class_recall`, and `confusion_matrix` in per-method metric file.\n")

    if strict_result is not None:
        strict_late = strict_result.get("late_fusion", {}).get("dev_metrics", {}).get("macro_f1", 0.0)
        if strict_possible and _safe_float(strict_late) > baseline_macro:
            lines.append("\n- Strict candidate achieved: yes\n")
        elif strict_possible:
            lines.append("\n- Strict candidate achieved: no\n")
        else:
            lines.append("\n- Strict candidate achieved: no (alignment failed)\n")
    else:
        lines.append("\n- Strict candidate achieved: no (not run)\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def _as_scalar(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_as_scalar(item) for item in v]
    if isinstance(v, dict):
        return {key: _as_scalar(value) for key, value in v.items()}
    return v


def _compact_mode_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    compact: dict[str, Any] = {
        "mode": payload.get("mode"),
        "strict": payload.get("strict"),
    }
    for method_name in ("late_fusion", "meta_logreg"):
        method_payload = payload.get(method_name, {})
        if not method_payload:
            continue
        compact[method_name] = {
            "selection_mode": method_payload.get("selection_mode"),
            "weights": method_payload.get("weights"),
            "best_c": method_payload.get("best_c"),
            "cv_best_score": method_payload.get("cv_best_score"),
            "runtime_seconds": method_payload.get("runtime_seconds"),
        }
        if "dev_metrics" in method_payload and isinstance(method_payload["dev_metrics"], dict):
            compact[method_name]["dev_macro_f1"] = _as_scalar(method_payload["dev_metrics"].get("macro_f1"))
            compact[method_name]["dev_macro_recall"] = _as_scalar(method_payload["dev_metrics"].get("macro_recall"))
            compact[method_name]["dev_accuracy"] = _as_scalar(method_payload["dev_metrics"].get("accuracy"))
            compact[method_name]["dev_top_class_share"] = _as_scalar(method_payload["dev_metrics"].get("top_class_share"))
            compact[method_name]["status"] = method_payload.get("status", "ok")
        if "selection_metrics" in method_payload and isinstance(method_payload["selection_metrics"], dict):
            compact[method_name]["selection_macro_f1"] = _as_scalar(method_payload["selection_metrics"].get("macro_f1"))
        if "train_metrics" in method_payload and isinstance(method_payload["train_metrics"], dict):
            compact[method_name]["train_macro_f1"] = _as_scalar(method_payload["train_metrics"].get("macro_f1"))
    return compact


def main() -> None:
    _add_repo_to_path()
    args = _parse_args()
    args.output_root = Path(args.output_root)
    args.report_root = Path(args.report_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    forbidden = find_forbidden_tokens(
        [
            str(args.b_train_holdout_proba),
            str(args.b_train_holdout_pred),
            str(args.b_dev_proba),
            str(args.b_dev_pred),
            str(args.d_train_holdout_proba),
            str(args.d_train_holdout_pred),
            str(args.d_dev_proba),
            str(args.d_dev_pred),
            str(args.output_root),
        ]
    )
    if forbidden:
        raise RuntimeError(f"forbidden token check failed: {json.dumps(forbidden, sort_keys=True)}")

    start = time.perf_counter()
    run_id = args.run_id

    a_result = _build_a_replay(run_id=run_id, output_root=args.output_root, random_seed=args.random_seed)
    b_result = _load_b_payloads(args, train_holdout_size=len(a_result["holdout_ids"]), dev_size=len(a_result["dev_ids"]))
    d_result = _load_d_payloads(args, train_holdout_size=len(a_result["holdout_ids"]), dev_size=len(a_result["dev_ids"]))
    align = _prepare_alignment(a_result, b_result, d_result)
    strict_possible = bool(align.get("strict_possible", False))

    b_hold_aligned = b_result["train_holdout"]["aligned_probabilities"]
    d_hold_aligned = d_result["train_holdout"]["aligned_probabilities"]
    b_dev_aligned = b_result["dev"]["aligned_probabilities"]
    d_dev_aligned = d_result["dev"]["aligned_probabilities"]

    a_hold = a_result["holdout_probs"]
    a_dev = a_result["dev_probs"]
    dev_claim_ids = [str(i) for i in a_result["dev_ids"]]
    holdout_claim_ids = [str(i) for i in a_result["holdout_ids"]]

    results: dict[str, Any] = {}

    def _run_mode(mode: str, do_strict: bool) -> dict[str, Any]:
        mode_payload: dict[str, Any] = {"mode": mode, "strict": do_strict}
        if do_strict:
            a_sel, b_sel, d_sel = a_hold, b_hold_aligned, d_hold_aligned
            y_sel = [str(v) for v in a_result["holdout_labels"]]
            claim_sel = holdout_claim_ids
        else:
            a_sel, b_sel, d_sel = a_dev, b_dev_aligned, d_dev_aligned
            y_sel = [str(v) for v in a_result["dev_labels"]]
            claim_sel = dev_claim_ids

        t0 = time.perf_counter()
        late = _run_late_fusion(
            a_probs_sel=a_sel,
            b_probs_sel=b_sel,
            d_probs_sel=d_sel,
            y_sel=y_sel,
            a_probs_eval=a_dev,
            b_probs_eval=b_dev_aligned,
            d_probs_eval=d_dev_aligned,
            y_eval=[str(v) for v in a_result["dev_labels"]],
            selection_mode=("train_holdout" if do_strict else "dev"),
            weight_step=args.weight_step,
            collapse_threshold=args.collapse_threshold,
        )
        late["runtime_seconds"] = float(time.perf_counter() - t0)
        late["dev_claim_ids"] = dev_claim_ids
        late["dev_proba"] = np.asarray(late["dev_proba"], dtype=float)
        late["selection_rows_count"] = len(late.get("selection_rows", []))
        mode_payload["late_fusion"] = late

        late_out = _emit_method_outputs(
            output_root=args.output_root,
            run_id=run_id,
            mode=mode,
            method="late_fusion",
            claim_ids=dev_claim_ids,
            probs=np.asarray(late["dev_proba"], dtype=float),
            claim_lookup=a_result["dev_claims"],
            evidence_lookup=a_result["dev_contexts"],
            truth=[str(v) for v in a_result["dev_labels"]],
            baseline_f1=args.baseline_macro_f1,
            collapse_threshold=args.collapse_threshold,
        )
        late["output_paths"] = late_out

        if do_strict and not args.disable_meta:
            t1 = time.perf_counter()
            meta = _run_meta_logreg(
                a_train=a_sel,
                b_train=b_sel,
                d_train=d_sel,
                y_train=y_sel,
                a_dev=a_dev,
                b_dev=b_dev_aligned,
                d_dev=d_dev_aligned,
                y_dev=[str(v) for v in a_result["dev_labels"]],
                c_grid=_parse_float_grid(args.meta_c_grid),
                random_seed=args.random_seed,
                collapse_threshold=args.collapse_threshold,
                baseline_f1=args.baseline_macro_f1,
            )
            meta["runtime_seconds"] = float(time.perf_counter() - t1)
            if isinstance(meta.get("dev_proba"), np.ndarray):
                meta["dev_proba"] = np.asarray(meta["dev_proba"], dtype=float)
            if isinstance(meta.get("dev_proba"), list):
                meta["dev_proba"] = np.asarray(meta["dev_proba"], dtype=float)
            mode_payload["meta_logreg"] = meta

            if isinstance(meta.get("dev_proba"), np.ndarray):
                meta_out = _emit_method_outputs(
                    output_root=args.output_root,
                    run_id=run_id,
                    mode=mode,
                    method="meta_logreg",
                    claim_ids=dev_claim_ids,
                    probs=np.asarray(meta["dev_proba"], dtype=float),
                    claim_lookup=a_result["dev_claims"],
                    evidence_lookup=a_result["dev_contexts"],
                    truth=[str(v) for v in a_result["dev_labels"]],
                    baseline_f1=args.baseline_macro_f1,
                    collapse_threshold=args.collapse_threshold,
                    meta=True,
                )
                meta["output_paths"] = meta_out
        else:
            mode_payload["meta_logreg"] = {"status": "skipped_non_strict_or_disabled"}

        return mode_payload

    if strict_possible:
        results["strict"] = _run_mode("strict", True)
    else:
        # never run strict if alignment is not clean
        results["strict"] = None

    results["diagnostic"] = _run_mode("diagnostic", False)

    wall = float(time.perf_counter() - start)

    manifest = manifest_base(
        run_id=run_id,
        status="strict-candidate" if strict_possible else "diagnostic-only",
        mode="STRICT" if strict_possible else "DIAGNOSTIC",
        stage="o_c5_comparison",
        command=f"python {Path(__file__).name}",
        working_directory=Path.cwd(),
        random_seed=args.random_seed,
    )
    manifest["input_files"] = [
        {"path": str(A_SCRIPT_PATH), "sha256": "", "split": "code", "labels_used": True},
        {"path": str(args.b_train_holdout_proba), "sha256": sha256_file(args.b_train_holdout_proba), "split": "b_holdout", "labels_used": False},
        {"path": str(args.b_dev_proba), "sha256": sha256_file(args.b_dev_proba), "split": "b_dev", "labels_used": False},
        {"path": str(args.d_train_holdout_proba), "sha256": sha256_file(args.d_train_holdout_proba), "split": "d_holdout", "labels_used": False},
        {"path": str(args.d_dev_proba), "sha256": sha256_file(args.d_dev_proba), "split": "d_dev", "labels_used": False},
    ]
    manifest["output_files"] = [
        str(args.output_root / f"{run_id}_a_k{A_CONTEXT_K}_train_holdout_proba.json"),
        str(args.output_root / f"{run_id}_a_k{A_CONTEXT_K}_dev_proba.json"),
        str(args.output_root / f"{run_id}_a_k{A_CONTEXT_K}_train_holdout_predictions.json"),
        str(args.output_root / f"{run_id}_a_k{A_CONTEXT_K}_dev_predictions.json"),
        str(args.output_root / f"{run_id}_a_k{A_CONTEXT_K}_train_holdout_metrics.json"),
        str(args.output_root / f"{run_id}_a_k{A_CONTEXT_K}_dev_metrics.json"),
    ]
    if strict_possible:
        manifest["output_files"].extend(
            [
                str(args.output_root / f"{run_id}_strict_late_fusion_late_dev_predictions.json"),
                str(args.output_root / f"{run_id}_strict_late_fusion_late_dev_proba.json"),
            ]
        )
    manifest["output_files"].extend(
        [
            str(args.output_root / f"{run_id}_diagnostic_late_fusion_late_dev_predictions.json"),
            str(args.output_root / f"{run_id}_diagnostic_late_fusion_late_dev_proba.json"),
        ]
    )
    manifest["metrics"] = {
        "strict_alignment": align,
        "selection_trace": {
            "alignment": align,
            "results": {
                "strict": _compact_mode_payload(results.get("strict")),
                "diagnostic": _compact_mode_payload(results.get("diagnostic")),
            },
            "weight_step": args.weight_step,
            "baseline_macro_f1": args.baseline_macro_f1,
        },
    }
    manifest["runtime"]["wall_seconds"] = wall
    manifest["runtime"]["device"] = "cpu"
    manifest["split_isolation_summary"] = (
        "strict mode: weights/C tuned on train-holdout, final metrics on dev."
        if strict_possible
        else "strict mode disabled, diagnostic run only."
    )
    manifest["leakage_risk"] = "low" if strict_possible else "low_diagnostic"
    manifest["data_flow_summary"] = (
        "Replay A tfidf+side k5 with fixed config; align B/D probabilities to A claim_id splits; "
        "run strict late fusion/meta on holdout when possible, else diagnostic-only."
    )
    manifest_path = args.output_root / "run_manifest.json"
    _write_json(manifest_path, manifest)

    record = {
        "run_id": run_id,
        "stage": "o_c5_comparison",
        "status": manifest["status"],
        "strict_possible": strict_possible,
        "alignment": align,
        "results": {
            "strict": _compact_mode_payload(results.get("strict")),
            "diagnostic": _compact_mode_payload(results.get("diagnostic")),
        },
        "wall_seconds": wall,
    }
    record_path = args.output_root / "run_record.json"
    _write_json(record_path, record)

    selection_trace = {
        "run_id": run_id,
        "strict_alignment": strict_possible,
        "alignment": align,
        "weight_step": args.weight_step,
        "baseline_macro_f1": args.baseline_macro_f1,
        "results": {
            "strict": _compact_mode_payload(results.get("strict")),
            "diagnostic": _compact_mode_payload(results.get("diagnostic")),
        },
    }
    selection_trace_path = args.output_root / "run_selection_trace.json"
    _write_json(selection_trace_path, selection_trace)

    report_path = args.report_root / "worker_c_fusion_meta_report.md"
    _write_report(
        path=report_path,
        baseline_macro=args.baseline_macro_f1,
        strict_possible=strict_possible,
        a_result=a_result,
        align=align,
        strict_result=results.get("strict"),
        diagnostic_result=results.get("diagnostic"),
        elapsed=wall,
    )

    print(f"[worker_c_fusion_meta] done in {wall:.2f}s")
    if strict_possible:
        best_strict_f1 = results["strict"]["late_fusion"]["dev_metrics"]["macro_f1"]
        promoted = best_strict_f1 > args.baseline_macro_f1
    else:
        promoted = False
    print(f"[strict] possible: {strict_possible}, promoted: {promoted}")
    print(f"[outputs] manifest: {manifest_path}")
    print(f"[outputs] record: {record_path}")
    print(f"[outputs] report: {report_path}")


if __name__ == "__main__":
    main()
