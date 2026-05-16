#!/usr/bin/env python
"""Round18 O-C3 strict-context recovery experiment.

Builds strict train/dev classifier contexts from explicit candidate pools and runs
TF-IDF + Logistic Regression / TF-IDF + calibrated linear SVM variants.

No hidden fallback to raw-evidence retrieval is used.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC, SVC

REPO_ROOT = Path(__file__).resolve().parents[4]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from round18.tools.common import (
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_classifier/o_c3_strict_context")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "run_manifest.json"
DEFAULT_RECORD = DEFAULT_OUTPUT_DIR / "run_record.json"
DEFAULT_RUN_ID = "o_c3_strict_context"

LABEL_ORDER = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
DEFAULT_NGRAM_GRID = "1-2"
DEFAULT_MAX_FEATURES_GRID = "20000,50000"
DEFAULT_LOGREG_C_GRID = "0.5,1.0,2.0"
DEFAULT_SVM_C_GRID = "0.5,1.0,2.0"


@dataclass(frozen=True)
class ContextItem:
    evidence_id: str
    rank: int
    score: float
    source: str
    text: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict O-C3 context recovery using explicit pool artifacts."
    )
    parser.add_argument(
        "--train-claims",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS,
        help="Train split claims JSON.",
    )
    parser.add_argument(
        "--dev-claims",
        type=Path,
        default=DEFAULT_DEV_CLAIMS,
        help="Dev split claims JSON.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence corpus JSON.",
    )
    parser.add_argument(
        "--train-pool",
        type=Path,
        required=True,
        help="Explicit train context pool.",
    )
    parser.add_argument(
        "--dev-pool",
        type=Path,
        required=True,
        help="Explicit dev context pool.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id/prefix.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--train-context-k", type=int, default=20)
    parser.add_argument("--dev-context-k", type=int, default=20)
    parser.add_argument(
        "--evidence-token-budget",
        type=int,
        default=0,
        help="Per-evidence token budget in classifier text; 0 keeps full evidence text.",
    )
    parser.add_argument(
        "--clf-max-features-grid",
        default=DEFAULT_MAX_FEATURES_GRID,
        help="Comma-separated max_features values for TF-IDF.",
    )
    parser.add_argument(
        "--clf-ngram-grid",
        default=DEFAULT_NGRAM_GRID,
        help="Comma-separated ngram ranges like 1-2.",
    )
    parser.add_argument(
        "--logreg-c-grid",
        default=DEFAULT_LOGREG_C_GRID,
        help="Comma-separated C values for LogisticRegression.",
    )
    parser.add_argument(
        "--svm-c-grid",
        default=DEFAULT_SVM_C_GRID,
        help="Comma-separated C values for LinearSVC.",
    )
    parser.add_argument("--svm-cv", type=int, default=3)
    parser.add_argument("--run-logreg", action="store_true")
    parser.add_argument("--run-svm", action="store_true")
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    parser.add_argument("--min-macro-f1", type=float, default=0.3585558388189967)
    parser.add_argument("--selection-mode", choices=("train_holdout", "fixed_first", "dev_diagnostic"), default="train_holdout")
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--enable-diagnostic",
        action="store_true",
        help="Allow mixed train/dev pool family (diagnostic-only).",
    )
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def _parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            values.append(float(value))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid float grid value: {value}") from exc
    if not values:
        raise argparse.ArgumentTypeError("float grid must contain at least one value.")
    return values


def _parse_int_grid(raw: str) -> list[int]:
    values: list[int] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            values.append(int(value))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid int grid value: {value}") from exc
    if not values:
        raise argparse.ArgumentTypeError("int grid must contain at least one value.")
    return values


def _parse_ngram_grid(raw: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", value)
        if not m:
            raise argparse.ArgumentTypeError(
                f"Invalid ngram range '{value}' (use lo-hi, e.g., 1-2)."
            )
        lo = int(m.group(1))
        hi = int(m.group(2))
        if lo <= 0 or hi <= 0 or lo > hi:
            raise argparse.ArgumentTypeError(f"Invalid ngram range '{value}'.")
        ranges.append((lo, hi))
    if not ranges:
        raise argparse.ArgumentTypeError("ngram grid must contain at least one value.")
    return ranges


def _extract_pool_family(pool_path: Path) -> str:
    stem = pool_path.name.replace("-", "_")
    stem = stem.removesuffix(".json")
    stem = re.sub(r"_top\d+_candidates$", "", stem)
    stem = re.sub(r"_candidates$", "", stem)
    match = re.search(r"(?:^|_)(o_[a-z0-9_]+)(?:_|$)", stem)
    if not match:
        raise SystemExit(f"Cannot infer pool family from: {pool_path}")
    return match.group(1)


def _ensure_family_contract(
    train_pool: Path,
    dev_pool: Path,
    enable_diagnostic: bool,
) -> tuple[str, str, str]:
    train_family = _extract_pool_family(train_pool)
    dev_family = _extract_pool_family(dev_pool)
    if train_family != dev_family and not enable_diagnostic:
        raise SystemExit(
            "Strict context source mismatch: "
            f"{train_family!r} (train-pool) != {dev_family!r} (dev-pool). "
            "Use --enable-diagnostic to allow this in diagnostic-only mode."
        )
    mode = "diagnostic-only" if (train_family != dev_family and enable_diagnostic) else "strict-candidate"
    return train_family, dev_family, mode


def _build_context_items(
    rows: list[Any],
    evidence: dict[str, str],
    top_k: int,
    source_family: str,
) -> tuple[list[ContextItem], dict[str, int]]:
    items: list[ContextItem] = []
    diagnostics: dict[str, int] = {
        "missing_evidence_text": 0,
        "deduplicated": 0,
    }
    seen: set[str] = set()

    for pos, row in enumerate(rows):
        if len(items) >= top_k:
            break
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("evidence_id", "")).strip()
        if not evidence_id or evidence_id in seen:
            if evidence_id in seen:
                diagnostics["deduplicated"] += 1
            continue
        seen.add(evidence_id)

        raw_rank = row.get("rank")
        if isinstance(raw_rank, int) and raw_rank > 0:
            rank = int(raw_rank)
        else:
            rank = pos + 1
        raw_score = row.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        text = evidence.get(evidence_id, "")
        if not text:
            diagnostics["missing_evidence_text"] += 1
            text = ""

        items.append(
            ContextItem(
                evidence_id=evidence_id,
                rank=rank,
                score=score,
                source=str(row.get("source", source_family)),
                text=str(text),
            )
        )
    return items, diagnostics


def _build_claim_context_text(
    claim_text: str,
    evidence_items: list[ContextItem],
    evidence_token_budget: int = 0,
) -> str:
    blocks = [f"CLAIM: {claim_text.strip()}"]
    for item in evidence_items:
        if item.text:
            text = item.text
            if evidence_token_budget > 0:
                text = " ".join(text.split()[:evidence_token_budget])
            blocks.append(f"EVIDENCE_{item.rank}: {text}")
    return "\n".join(blocks)


def _build_contexts(
    claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    pool: dict[str, Any],
    top_k: int,
    source_family: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "missing_claims_in_pool": [],
        "insufficient_context_claims": [],
        "missing_evidence_text": 0,
        "total_candidates": 0,
    }

    for claim_id, claim in claims.items():
        if claim_id not in pool:
            diagnostics["missing_claims_in_pool"].append(claim_id)
            raise SystemExit(
                f"Hidden fallback disabled: missing claim in pool -> {claim_id}"
            )
        raw_rows = pool[claim_id]
        if not isinstance(raw_rows, list):
            raise SystemExit(f"Pool entry must be a list for {claim_id}: {type(raw_rows)}")
        items, row_diag = _build_context_items(
            raw_rows,
            evidence=evidence,
            top_k=top_k,
            source_family=source_family,
        )
        diagnostics["missing_evidence_text"] += row_diag["missing_evidence_text"]
        diagnostics["total_candidates"] += len(items)
        if len(items) < top_k:
            diagnostics["insufficient_context_claims"].append(claim_id)
            raise SystemExit(
                "Hidden fallback disabled: context too short for claim "
                f"{claim_id} (need {top_k}, got {len(items)})."
            )

        contexts[claim_id] = {
            "claim_id": claim_id,
            "claim_text": claim.get("claim_text", ""),
            "claim_label": claim.get("claim_label"),
            "context_source_family": source_family,
            "classifier_context_top_k": top_k,
            "final_evidence_candidates": [item.evidence_id for item in items],
            "classifier_evidence_context": [item.__dict__ for item in items],
            "context_item_count": len(items),
        }

    return contexts, diagnostics


def _rows_from_contexts(
    claims: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    evidence_token_budget: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    claim_ids = list(claims.keys())
    texts: list[str] = []
    labels: list[str] = []
    for claim_id in claim_ids:
        context = contexts[claim_id]
        items = [ContextItem(**row) for row in context["classifier_evidence_context"]]
        texts.append(
            _build_claim_context_text(
                claims[claim_id].get("claim_text", ""),
                items,
                evidence_token_budget=evidence_token_budget,
            )
        )
        labels.append(str(claims[claim_id].get("claim_label")))
    return claim_ids, texts, labels


def _validate_claim_labels(claims: dict[str, dict[str, Any]], split_name: str) -> None:
    unknown = set()
    for claim in claims.values():
        label = claim.get("claim_label")
        if label not in LABEL_ORDER:
            unknown.add(str(label))
    if unknown:
        raise SystemExit(
            f"{split_name} labels include unsupported classes: {sorted(unknown)}"
        )


def _load_inputs(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    for path in (args.train_claims, args.dev_claims, args.evidence, args.train_pool, args.dev_pool):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pool = load_json(args.train_pool)
    dev_pool = load_json(args.dev_pool)
    if not isinstance(train_claims, dict):
        raise SystemExit("train_claims must be a dict keyed by claim_id.")
    if not isinstance(dev_claims, dict):
        raise SystemExit("dev_claims must be a dict keyed by claim_id.")
    if not isinstance(evidence, dict):
        raise SystemExit("evidence must be a dict keyed by evidence_id.")
    if not isinstance(train_pool, dict):
        raise SystemExit("train-pool must be a dict keyed by claim_id.")
    if not isinstance(dev_pool, dict):
        raise SystemExit("dev-pool must be a dict keyed by claim_id.")

    _validate_claim_labels(train_claims, "train")
    _validate_claim_labels(dev_claims, "dev")
    return train_claims, dev_claims, evidence, train_pool, dev_pool


def _evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    y_prob: np.ndarray,
) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    micro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="micro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    per_class_recall: dict[str, float] = {}
    row_totals = matrix.sum(axis=1).astype(float)
    for idx, label in enumerate(LABEL_ORDER):
        recall = float(matrix[idx, idx]) / row_totals[idx] if row_totals[idx] > 0 else 0.0
        per_class_recall[label] = recall

    hist = {label: 0 for label in LABEL_ORDER}
    for pred in y_pred:
        hist[pred] = hist.get(pred, 0) + 1
    total_pred = max(len(y_pred), 1)
    max_share = max(hist.values()) / total_pred if total_pred else 0.0
    max_label = max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0]

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_recall": float(macro_recall),
        "classification_report": report,
        "per_class_recall": per_class_recall,
        "prediction_histogram": hist,
        "top_class_share": float(max_share),
        "top_class": max_label,
        "confusion_matrix": matrix.tolist(),
        "proba": y_prob.tolist(),
    }


def _collapse_gate(prediction_hist: dict[str, int], threshold: float) -> dict[str, Any]:
    total = sum(prediction_hist.values())
    if total == 0:
        return {
            "status": "failed",
            "reason": "no predictions produced",
            "threshold": float(threshold),
            "max_class_share": 0.0,
            "collapsed_classes": [],
        }
    max_class = max(prediction_hist.items(), key=lambda kv: (kv[1], kv[0]))
    share = max_class[1] / total
    passed = share <= threshold
    collapsed = [label for label, count in prediction_hist.items() if count / total > threshold]
    return {
        "status": "passed" if passed else "failed",
        "reason": (
            "passed"
            if passed
            else f"single-class collapse above threshold {threshold}: {max_class[0]}"
        ),
        "max_class": max_class[0],
        "max_class_share": float(share),
        "threshold": float(threshold),
        "collapsed_classes": collapsed,
    }


def _acceptance_gate(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    histogram = metrics.get("prediction_histogram", {})
    per_class_recall = metrics.get("per_class_recall", {})
    if any(int(histogram.get(label, 0)) <= 0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_predictions")
    if any(float(per_class_recall.get(label, 0.0)) <= 0.0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_recall")
    if float(metrics.get("top_class_share", 1.0)) > float(args.collapse_threshold):
        failures.append("max_class_share_above_threshold")
    if float(metrics.get("macro_f1", 0.0)) < float(args.min_macro_f1):
        failures.append("macro_f1_below_baseline")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "thresholds": {
            "max_class_share": float(args.collapse_threshold),
            "min_macro_f1": float(args.min_macro_f1),
            "require_nonzero_predictions_for_all_classes": True,
            "require_nonzero_recall_for_all_classes": True,
        },
    }


def _evidence_assignment_metrics(
    claims: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    claim_accuracy: float,
) -> dict[str, float]:
    assignment_scores: list[float] = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        predicted = set(contexts.get(claim_id, {}).get("final_evidence_candidates", []))
        if not gold:
            assignment_scores.append(0.0)
            continue
        overlap = gold & predicted
        precision = len(overlap) / len(predicted) if predicted else 0.0
        recall = len(overlap) / len(gold)
        if precision == 0.0 and recall == 0.0:
            assignment_scores.append(0.0)
        else:
            assignment_scores.append(2.0 * precision * recall / (precision + recall))
    assignment_f = float(np.mean(assignment_scores)) if assignment_scores else 0.0
    harmonic = (
        2.0 * claim_accuracy * assignment_f / (claim_accuracy + assignment_f)
        if (claim_accuracy + assignment_f) > 0.0
        else 0.0
    )
    return {
        "assignment_f": assignment_f,
        "harmonic_mean_F_A": float(harmonic),
    }


def _build_predictions_payload(
    claim_ids: list[str],
    claim_lookup: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    y_pred: list[str],
    y_prob: np.ndarray,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for claim_id, pred, probs in zip(claim_ids, y_pred, y_prob):
        label_dist = [
            {"label": label, "prob": float(prob)} for label, prob in zip(LABEL_ORDER, probs.tolist())
        ]
        payload[claim_id] = {
            "claim_text": claim_lookup[claim_id].get("claim_text", ""),
            "claim_label": pred,
            "evidences": contexts[claim_id].get("final_evidence_candidates", []),
            "label_distribution": label_dist,
            "pred_class_probability": float(max(probs)),
            "pred_class": pred,
        }
    return payload


def _resolve_svm_cv(labels: list[str], requested_cv: int) -> int | None:
    class_counts = Counter(labels)
    if not class_counts:
        return None
    if requested_cv < 2:
        return None

    min_count = min(class_counts.values())
    safe_cv = min(requested_cv, len(labels), min_count)
    if safe_cv < 2:
        return None
    return safe_cv


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_confusion_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(v) for v in row]])


def _variant_key(model_id: str, config: dict[str, Any]) -> str:
    return (
        f"{model_id}_mf{int(config['tfidf_max_features'])}"
        f"_ngram{config['tfidf_ngram_range'][0]}x{config['tfidf_ngram_range'][1]}"
        f"_c{str(config['C']).replace('.', 'p')}"
    )


def _train_and_evaluate(
    model_id: str,
    train_texts: list[str],
    y_train: list[str],
    dev_texts: list[str],
    y_dev: list[str],
    tfidf_config: dict[str, Any],
    model_config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=tuple(tfidf_config["ngram_range"]),
        max_features=int(tfidf_config["max_features"]),
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_dev = vectorizer.transform(dev_texts)

    svm_cv = None

    if model_id == "tfidf_logreg":
        estimator = LogisticRegression(
            C=float(model_config["C"]),
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    elif model_id == "linear_svm":
        svm_cv = _resolve_svm_cv(y_train, args.svm_cv)
        if svm_cv is None:
            estimator = SVC(
                kernel="linear",
                C=float(model_config["C"]),
                class_weight="balanced",
                probability=True,
                random_state=args.random_seed,
            )
        else:
            base = LinearSVC(
                C=float(model_config["C"]),
                class_weight="balanced",
                random_state=args.random_seed,
                dual=False,
            )
            estimator = CalibratedClassifierCV(
                estimator=base,
                cv=svm_cv,
                method="sigmoid",
            )
    else:
        raise ValueError(model_id)

    fit_start = time.perf_counter()
    estimator.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    y_pred = estimator.predict(x_dev).astype(str).tolist()
    y_prob = estimator.predict_proba(x_dev).astype(float)
    stats = _evaluate_predictions(y_dev, y_pred, y_prob)
    stats["fit_seconds"] = float(fit_seconds)
    stats["vectorizer_dim"] = int(x_train.shape[1])
    stats["classes_"] = [str(c) for c in estimator.classes_]
    stats["config"] = {
        "model_id": model_id,
        "tfidf_max_features": int(tfidf_config["max_features"]),
        "tfidf_ngram_range": list(tfidf_config["ngram_range"]),
        "C": float(model_config["C"]),
        "cv": svm_cv,
    }
    return stats, y_pred, y_prob


def _select_winner(
    results: list[dict[str, Any]],
    all_predicts: list[list[str]],
    all_probs: list[np.ndarray],
) -> tuple[int, dict[str, Any], list[str], np.ndarray]:
    best_idx = -1
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for idx, result in enumerate(results):
        score = _selection_score(result)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0:
        raise SystemExit("No classifier variant evaluated.")
    return best_idx, results[best_idx], all_predicts[best_idx], all_probs[best_idx]


def _selection_score(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result["macro_f1"]),
        float(result["micro_f1"]),
        float(result["macro_recall"]),
        -float(result["top_class_share"]),
    )


def _config_key(config: dict[str, Any]) -> tuple[str, int, tuple[int, int], float]:
    return (
        str(config["model_id"]),
        int(config["tfidf_max_features"]),
        tuple(int(x) for x in config["tfidf_ngram_range"]),
        float(config["C"]),
    )


def _train_holdout_indices(labels: list[str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args.train_val_fraction),
        random_state=args.random_seed,
    )
    indexes = np.arange(len(labels))
    try:
        train_idx, val_idx = next(splitter.split(indexes, labels))
    except ValueError:
        return None
    return train_idx, val_idx


def _subset(items: list[Any], indexes: np.ndarray) -> list[Any]:
    return [items[int(idx)] for idx in indexes]


def _grid_search(
    model_id: str,
    train_texts: list[str],
    y_train: list[str],
    dev_texts: list[str],
    y_dev: list[str],
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[str],
    np.ndarray,
    dict[str, Any],
]:
    if model_id == "tfidf_logreg":
        c_grid = _parse_float_grid(args.logreg_c_grid)
    elif model_id == "linear_svm":
        c_grid = _parse_float_grid(args.svm_c_grid)
    else:
        raise ValueError(model_id)
    max_features_grid = _parse_int_grid(args.clf_max_features_grid)
    ngram_grid = _parse_ngram_grid(args.clf_ngram_grid)
    configs: list[dict[str, Any]] = [
        {
            "model_id": model_id,
            "tfidf_max_features": int(max_features),
            "tfidf_ngram_range": list(ngram_range),
            "C": float(c_value),
        }
        for max_features in max_features_grid
        for ngram_range in ngram_grid
        for c_value in c_grid
    ]

    selection_meta: dict[str, Any] = {
        "selection_mode": args.selection_mode,
        "selected_by": "pre_registered_first_config",
        "train_val_fraction": float(args.train_val_fraction),
        "selection_results": [],
    }
    selected_config = configs[0]

    if args.selection_mode == "dev_diagnostic":
        if not args.enable_diagnostic:
            raise SystemExit("--selection-mode dev_diagnostic requires --enable-diagnostic.")
        selection_meta["selected_by"] = "dev_metrics_diagnostic_only"
    elif args.selection_mode == "train_holdout":
        split = _train_holdout_indices(y_train, args)
        if split is None:
            selection_meta["selected_by"] = "fallback_first_config_insufficient_train_holdout"
        else:
            train_idx, val_idx = split
            selection_candidates: list[dict[str, Any]] = []
            for config in configs:
                result, _, _ = _train_and_evaluate(
                    model_id=model_id,
                    train_texts=_subset(train_texts, train_idx),
                    y_train=_subset(y_train, train_idx),
                    dev_texts=_subset(train_texts, val_idx),
                    y_dev=_subset(y_train, val_idx),
                    tfidf_config={
                        "max_features": config["tfidf_max_features"],
                        "ngram_range": tuple(config["tfidf_ngram_range"]),
                    },
                    model_config={"C": config["C"]},
                    args=args,
                )
                result["selection_config"] = config
                selection_candidates.append(result)
            selection_candidates.sort(key=_selection_score, reverse=True)
            selected_config = selection_candidates[0]["selection_config"]
            selection_meta["selected_by"] = "train_holdout_macro_f1"
            selection_meta["selection_results"] = selection_candidates

    results: list[dict[str, Any]] = []
    predicts: list[list[str]] = []
    probs: list[np.ndarray] = []

    for config in configs:
        tfidf_config = {
            "max_features": config["tfidf_max_features"],
            "ngram_range": tuple(config["tfidf_ngram_range"]),
        }
        model_config = {"C": config["C"]}
        result, y_pred, y_prob = _train_and_evaluate(
            model_id=model_id,
            train_texts=train_texts,
            y_train=y_train,
            dev_texts=dev_texts,
            y_dev=y_dev,
            tfidf_config=tfidf_config,
            model_config=model_config,
            args=args,
        )
        result["tfidf_max_features"] = config["tfidf_max_features"]
        result["tfidf_ngram_range"] = config["tfidf_ngram_range"]
        result["model_C"] = config["C"]
        result["collapse_gate"] = _collapse_gate(
            result["prediction_histogram"],
            threshold=args.collapse_threshold,
        )
        result["acceptance_gate"] = _acceptance_gate(result, args)
        result["selection_meta"] = selection_meta
        results.append(result)
        predicts.append(y_pred)
        probs.append(y_prob)

    if args.selection_mode == "dev_diagnostic":
        best_idx, best_result, best_pred, best_prob = _select_winner(results, predicts, probs)
    else:
        selected_key = _config_key(selected_config)
        best_idx = next(
            idx for idx, result in enumerate(results)
            if _config_key(result["config"]) == selected_key
        )
        best_result = results[best_idx]
        best_pred = predicts[best_idx]
        best_prob = probs[best_idx]
    best_result["selection_meta"] = selection_meta
    return best_result, results, best_pred, best_prob, best_result["confusion_matrix"]


def _write_grid_payload(
    outdir: Path,
    run_id: str,
    model_id: str,
    results: list[dict[str, Any]],
    best: dict[str, Any],
) -> Path:
    path = outdir / f"{run_id}_{model_id}_grid_search.json"
    write_json(path, {"model_id": model_id, "results": results, "best": best})
    return path


def _write_model_artifacts(
    outdir: Path,
    run_id: str,
    variant_key: str,
    predictions: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    matrix: np.ndarray,
) -> dict[str, str]:
    pred_path = outdir / f"{run_id}_{variant_key}_dev_predictions.json"
    metric_path = outdir / f"{run_id}_{variant_key}_metrics.json"
    report_path = outdir / f"{run_id}_{variant_key}_classification_report.txt"
    conf_path = outdir / f"{run_id}_{variant_key}_confusion_matrix.csv"
    write_json(pred_path, predictions)
    write_json(metric_path, metrics)
    report_path.write_text(metrics["classification_report"], encoding="utf-8")
    _write_confusion_csv(conf_path, np.array(matrix), LABEL_ORDER)
    return {
        "predictions": str(pred_path),
        "metrics": str(metric_path),
        "report": str(report_path),
        "confusion_matrix": str(conf_path),
    }


def main() -> None:
    start = time.perf_counter()
    args = _parse_args()
    args.command = " ".join(sys.argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.evidence),
            str(args.train_pool),
            str(args.dev_pool),
            str(args.output_dir),
            str(args.manifest),
            str(args.record),
        ]
    )
    if forbidden_hits:
        raise SystemExit(
            "STRICT GUARD FAILED: forbidden token(s) detected\n"
            + json.dumps(forbidden_hits, sort_keys=True)
        )

    train_family, dev_family, run_mode = _ensure_family_contract(
        args.train_pool,
        args.dev_pool,
        args.enable_diagnostic,
    )
    if args.selection_mode == "dev_diagnostic":
        if not args.enable_diagnostic:
            raise SystemExit("--selection-mode dev_diagnostic requires --enable-diagnostic.")
        run_mode = "diagnostic-only"

    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    if not train_claims or not dev_claims:
        raise SystemExit("Input claim files must be non-empty.")

    train_contexts, train_context_diag = _build_contexts(
        train_claims,
        evidence=evidence,
        pool=train_pool,
        top_k=args.train_context_k,
        source_family=train_family,
    )
    dev_contexts, dev_context_diag = _build_contexts(
        dev_claims,
        evidence=evidence,
        pool=dev_pool,
        top_k=args.dev_context_k,
        source_family=dev_family,
    )

    train_ids, train_texts, y_train = _rows_from_contexts(
        train_claims,
        train_contexts,
        evidence_token_budget=args.evidence_token_budget,
    )
    dev_ids, dev_texts, y_dev = _rows_from_contexts(
        dev_claims,
        dev_contexts,
        evidence_token_budget=args.evidence_token_budget,
    )

    if not args.run_logreg and not args.run_svm:
        args.run_logreg = True
        args.run_svm = True

    train_context_path = args.output_dir / f"{args.run_id}_train_context_top{args.train_context_k}.jsonl"
    dev_context_path = args.output_dir / f"{args.run_id}_dev_context_top{args.dev_context_k}.jsonl"
    _write_jsonl(train_context_path, [train_contexts[cid] for cid in sorted(train_contexts.keys())])
    _write_jsonl(dev_context_path, [dev_contexts[cid] for cid in sorted(dev_contexts.keys())])

    model_outputs: dict[str, Any] = {}
    model_files: list[str] = [str(train_context_path), str(dev_context_path)]
    outputs_written: list[str] = [str(args.manifest), str(args.record)]

    if args.run_logreg:
        best_result, all_results, best_pred, best_prob, best_matrix = _grid_search(
            "tfidf_logreg",
            train_texts=train_texts,
            y_train=y_train,
            dev_texts=dev_texts,
            y_dev=y_dev,
            args=args,
        )
        best_result.update(
            _evidence_assignment_metrics(
                dev_claims,
                dev_contexts,
                claim_accuracy=float(best_result["accuracy"]),
            )
        )
        variant = _variant_key("tfidf_logreg", best_result["config"])
        pred_payload = _build_predictions_payload(
            dev_ids,
            claim_lookup=dev_claims,
            contexts=dev_contexts,
            y_pred=best_pred,
            y_prob=best_prob,
        )
        artifact_paths = _write_model_artifacts(
            outdir=args.output_dir,
            run_id=args.run_id,
            variant_key=variant,
            predictions=pred_payload,
            metrics=best_result,
            matrix=np.array(best_matrix),
        )
        grid_path = _write_grid_payload(
            outdir=args.output_dir,
            run_id=args.run_id,
            model_id="tfidf_logreg",
            results=all_results,
            best=best_result,
        )
        artifact_paths["grid_search"] = str(grid_path)
        model_outputs["tfidf_logreg"] = {
            **artifact_paths,
            "selected_variant": variant,
            "status": best_result["collapse_gate"]["status"],
            "family": train_family,
        }
        model_files.extend(artifact_paths.values())

    if args.run_svm:
        best_result, all_results, best_pred, best_prob, best_matrix = _grid_search(
            "linear_svm",
            train_texts=train_texts,
            y_train=y_train,
            dev_texts=dev_texts,
            y_dev=y_dev,
            args=args,
        )
        best_result.update(
            _evidence_assignment_metrics(
                dev_claims,
                dev_contexts,
                claim_accuracy=float(best_result["accuracy"]),
            )
        )
        variant = _variant_key("linear_svm", best_result["config"])
        pred_payload = _build_predictions_payload(
            dev_ids,
            claim_lookup=dev_claims,
            contexts=dev_contexts,
            y_pred=best_pred,
            y_prob=best_prob,
        )
        artifact_paths = _write_model_artifacts(
            outdir=args.output_dir,
            run_id=args.run_id,
            variant_key=variant,
            predictions=pred_payload,
            metrics=best_result,
            matrix=np.array(best_matrix),
        )
        grid_path = _write_grid_payload(
            outdir=args.output_dir,
            run_id=args.run_id,
            model_id="linear_svm",
            results=all_results,
            best=best_result,
        )
        artifact_paths["grid_search"] = str(grid_path)
        model_outputs["linear_svm"] = {
            **artifact_paths,
            "selected_variant": variant,
            "status": best_result["collapse_gate"]["status"],
            "family": train_family,
        }
        model_files.extend(artifact_paths.values())

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate" if run_mode == "strict-candidate" else "diagnostic-only",
        mode="STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
        stage="o_c3_strict_context",
        command=args.command,
        working_directory=Path.cwd(),
        random_seed=args.random_seed,
    )
    manifest["input_files"] = [
        {"path": str(args.train_claims), "sha256": sha256_file(args.train_claims), "split": "train", "labels_used": True},
        {"path": str(args.dev_claims), "sha256": sha256_file(args.dev_claims), "split": "dev", "labels_used": True},
        {"path": str(args.evidence), "sha256": sha256_file(args.evidence), "split": "evidence", "labels_used": False},
        {"path": str(args.train_pool), "sha256": sha256_file(args.train_pool), "split": "current_run_train_artifact", "labels_used": False},
        {"path": str(args.dev_pool), "sha256": sha256_file(args.dev_pool), "split": "current_run_dev_artifact", "labels_used": False},
    ]
    manifest["forbidden_input_scan"] = {
        "passed": not bool(forbidden_hits),
        "notes": "Input path scan passed." if not forbidden_hits else str(forbidden_hits),
    }
    manifest["output_files"] = sorted(set(model_files))
    manifest["metrics"] = {
        "train_context_family": train_family,
        "dev_context_family": dev_family,
        "strict_family_match": run_mode == "strict-candidate",
        "train_context_source": str(args.train_pool),
        "dev_context_source": str(args.dev_pool),
        "train_context_k": args.train_context_k,
        "dev_context_k": args.dev_context_k,
        "evidence_token_budget": args.evidence_token_budget,
        "train_context_diag": train_context_diag,
        "dev_context_diag": dev_context_diag,
        "models": model_outputs,
        "train_context_path": str(train_context_path),
        "dev_context_path": str(dev_context_path),
        "selection_mode": args.selection_mode,
        "strict_selection_rule": (
            "train_holdout_or_fixed_first; dev_diagnostic forces diagnostic-only"
        ),
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Built explicit train/dev contexts from provided pools with no hidden fallback. "
        f"Train family={train_family}, dev family={dev_family}. "
        "Trained grid variants for TF-IDF+LogReg and TF-IDF+calibrated linear SVM."
    )
    manifest["split_isolation_summary"] = (
        "Dev split is used only for fixed post-hoc scoring; no train/dev context "
        "crossover, no fallback context retrieval."
    )
    write_json(args.manifest, manifest)

    record = {
        "run_id": args.run_id,
        "stage": "o_c3_strict_context",
        "mode": "STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
        "status": run_mode,
        "split": "train/dev",
        "command": args.command,
        "command_args": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "claims_train": len(train_claims),
        "claims_dev": len(dev_claims),
        "evidence_count": len(evidence),
        "train_context_source": train_family,
        "dev_context_source": dev_family,
        "train_pool": str(args.train_pool),
        "dev_pool": str(args.dev_pool),
        "files_written": [str(args.manifest), str(args.record), *sorted(set(model_files))],
        "model_results": model_outputs,
        "selection_mode": args.selection_mode,
        "wall_seconds": manifest["runtime"]["wall_seconds"],
    }
    write_json(args.record, record)

    print(f"Wrote train context file: {train_context_path}")
    print(f"Wrote dev context file: {dev_context_path}")
    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote record: {args.record}")


if __name__ == "__main__":
    main()
