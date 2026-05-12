#!/usr/bin/env python
"""Round18 O-C2 baseline pipeline.

Builds classifier context for:
  - dev: from current O-A2 aggregate pool
  - train: lexical TF-IDF fallback over raw evidence corpus (no train-side aggregate pool)

Trains:
  - TF-IDF + Logistic Regression
  - TF-IDF + linear SVM (CalibratedClassifierCV)
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

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

LABEL_ORDER = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_POOL = Path(
    "round18/outputs/o_aggregate/o_a2_calibrated_rank/dev_full_dev_o_a2_calibrated_rank_strict_ce64_sparse_backfill_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_classifier/o_c2_baselines")
DEFAULT_MANIFEST = Path(
    "round18/outputs/o_classifier/o_c2_baselines/run_manifest.json"
)
DEFAULT_RECORD = Path("round18/outputs/o_classifier/o_c2_baselines/run_record.json")
DEFAULT_RUN_ID = "o_c2_baselines"


@dataclass
class ContextItem:
    evidence_id: str
    rank: int
    score: float
    source: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run O-C2 TF-IDF + logistic/SVM baselines."
    )
    parser.add_argument(
        "--train-claims",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS,
        help="Train claims json path.",
    )
    parser.add_argument(
        "--dev-claims",
        type=Path,
        default=DEFAULT_DEV_CLAIMS,
        help="Dev claims json path.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence corpus json.",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=DEFAULT_POOL,
        help="Current-run dev aggregate pool used for dev context.",
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
    parser.add_argument(
        "--train-context-k",
        type=int,
        default=20,
        help="Number of per-claim evidence context items for train.",
    )
    parser.add_argument(
        "--dev-context-k",
        type=int,
        default=20,
        help="Number of per-claim evidence context items for dev.",
    )
    parser.add_argument(
        "--ret-max-features",
        type=int,
        default=30_000,
        help="Max TF-IDF features used for train lexical retrieval.",
    )
    parser.add_argument(
        "--ret-min-df",
        type=int,
        default=2,
        help="Min document frequency used for train lexical retrieval.",
    )
    parser.add_argument(
        "--clf-max-features",
        type=int,
        default=50_000,
        help="Max TF-IDF features used for the classifier text model.",
    )
    parser.add_argument("--svm-cv", type=int, default=3, help="CV folds for SVM calibration.")
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Bins for top-1 expected calibration error.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Jobs for sklearn models.")
    parser.add_argument(
        "--run-svm",
        action="store_true",
        help="Train linear SVM model.",
    )
    parser.add_argument(
        "--run-logreg",
        action="store_true",
        help="Train TF-IDF + Logistic Regression model.",
    )
    return parser.parse_args()


def _infer_split(path: Path) -> str:
    if str(path) == str(DEFAULT_TRAIN_CLAIMS):
        return "train"
    if str(path) == str(DEFAULT_DEV_CLAIMS):
        return "dev"
    if str(path) == str(DEFAULT_EVIDENCE):
        return "evidence"
    if "o_a2_calibrated_rank" in str(path):
        return "current_run_artifact"
    return "code"


def _claim_context_text(claim_text: str, evidence_items: list[ContextItem]) -> str:
    parts = [f"CLAIM: {claim_text.strip()}"]
    for item in evidence_items:
        if item.text:
            parts.append(f"EVIDENCE_{item.rank}: {item.text}")
    return "\n".join(parts)


def _to_context_item_map(
    raw_items: list[dict[str, Any]],
    evidence: dict[str, str],
    top_k: int,
    source: str,
) -> list[ContextItem]:
    items: list[ContextItem] = []
    seen = set()
    if top_k <= 0:
        return items
    for pos, entry in enumerate(raw_items or []):
        if len(items) >= top_k:
            break
        if not isinstance(entry, dict):
            continue
        evidence_id = str(entry.get("evidence_id", "")).strip()
        if not evidence_id or evidence_id in seen:
            continue
        text = evidence.get(evidence_id, "")
        if text is None:
            continue
        score = float(entry.get("score", 0.0))
        items.append(
            ContextItem(
                evidence_id=evidence_id,
                rank=pos + 1,
                score=score,
                source=source,
                text=str(text),
            )
        )
        seen.add(evidence_id)
    return items


def _build_pool_contexts(
    claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    pool: dict[str, list[dict[str, Any]]],
    top_k: int,
    fallback_retriever: "LexicalRetriever | None" = None,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    missing_claims = []

    for claim_id, claim in claims.items():
        raw_pool_items = pool.get(claim_id, [])
        parsed = _to_context_item_map(raw_pool_items, evidence, top_k, source="o_a2_pool")
        if len(parsed) < top_k and fallback_retriever is not None:
            need = top_k - len(parsed)
            fallback_items = fallback_retriever.topk(
                [claim.get("claim_text", "")], top_k=need, start_rank=len(parsed)
            )[0]
            for item in fallback_items:
                if item.evidence_id not in {x.evidence_id for x in parsed}:
                    item.source = "train_lexical_fallback"
                    parsed.append(item)
            parsed = parsed[:top_k]

        if len(parsed) < top_k:
            missing_claims.append(claim_id)

        contexts[claim_id] = {
            "claim_id": claim_id,
            "claim_text": claim.get("claim_text", ""),
            "claim_label": claim.get("claim_label"),
            "source": "o_a2_strict_pool",
            "classifier_context_top_k": top_k,
            "final_evidence_candidates": [item.evidence_id for item in parsed],
            "classifier_evidence_context": [item.__dict__ for item in parsed],
        }

    return contexts, missing_claims


class LexicalRetriever:
    """Simple lexical fallback retriever from raw evidence for claims without pool context."""

    def __init__(
        self,
        evidence: dict[str, str],
        max_features: int,
        min_df: int,
        random_seed: int,
    ):
        evidence_items = list(evidence.items())
        self.evidence_ids = [k for k, _ in evidence_items]
        evidence_texts = [v for _, v in evidence_items]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=min_df,
            dtype=np.float32,
        )
        self.evidence_matrix = self.vectorizer.fit_transform(evidence_texts)
        self.neighbor = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self.neighbor.fit(self.evidence_matrix)

    def topk(self, queries: list[str], top_k: int, start_rank: int = 0) -> list[list[ContextItem]]:
        if top_k <= 0:
            return [[] for _ in queries]
        if len(self.evidence_ids) == 0:
            return [[] for _ in queries]
        k = min(top_k, len(self.evidence_ids))
        query_matrix = self.vectorizer.transform(queries)
        distances, indexes = self.neighbor.kneighbors(query_matrix, n_neighbors=k)
        outputs: list[list[ContextItem]] = []
        for query_row, (dist_row, idx_row) in enumerate(zip(distances, indexes)):
            row_items: list[ContextItem] = []
            for local_rank, (dist, idx) in enumerate(zip(dist_row, idx_row), start=1):
                evidence_id = self.evidence_ids[int(idx)]
                score = 1.0 - float(dist)
                row_items.append(
                    ContextItem(
                        evidence_id=evidence_id,
                        rank=start_rank + local_rank,
                        score=score,
                        source="train_lexical_tfidf",
                        text="",
                    )
                )
            outputs.append(row_items)
        return outputs


def _build_train_contexts(
    train_claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    top_k: int,
    retriever: LexicalRetriever,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    claim_ids = list(train_claims.keys())
    claim_texts = [train_claims[cid].get("claim_text", "") for cid in claim_ids]
    # Retrieve candidate evidence ids/scores
    raw_rows = retriever.topk(claim_texts, top_k=top_k, start_rank=0)

    for claim_id, raw_items in zip(claim_ids, raw_rows):
        # attach texts from raw evidence dict
        items = []
        for item in raw_items:
            item.text = evidence.get(item.evidence_id, "")
            items.append(item)
        claim = train_claims[claim_id]
        contexts[claim_id] = {
            "claim_id": claim_id,
            "claim_text": claim.get("claim_text", ""),
            "claim_label": claim.get("claim_label"),
            "source": "raw_lexical_tfidf",
            "classifier_context_top_k": top_k,
            "final_evidence_candidates": [item.evidence_id for item in items],
            "classifier_evidence_context": [item.__dict__ for item in items],
        }
    return contexts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _expected_calibration_error(
    y_true: list[str],
    proba: np.ndarray,
    label_order: list[str],
    bins: int,
) -> dict[str, Any]:
    if proba.size == 0:
        return {"calibration_ece": 0.0, "ece_bins": []}

    y_true_idx = np.array([label_order.index(y) for y in y_true], dtype=int)
    pred_idx = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    if conf.size == 0:
        return {"calibration_ece": 0.0, "ece_bins": []}

    bin_size = 1.0 / bins
    ece = 0.0
    bin_rows: list[dict[str, float | int]] = []
    n_samples = len(conf)
    for b in range(bins):
        lower = b * bin_size
        upper = (b + 1) * bin_size
        if b == bins - 1:
            mask = (conf >= lower) & (conf <= upper + 1e-12)
        else:
            mask = (conf >= lower) & (conf < upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(pred_idx[mask] == y_true_idx[mask]))
        confidence = float(np.mean(conf[mask]))
        weight = int(mask.sum())
        gap = abs(accuracy - confidence)
        ece += (weight / n_samples) * gap
        bin_rows.append(
            {
                "bin": b,
                "count": weight,
                "confidence_mean": confidence,
                "accuracy": accuracy,
                "gap": gap,
                "lower": lower,
                "upper": upper,
            }
        )
    return {"calibration_ece": float(ece), "ece_bins": bin_rows}


def _evidence_metrics(
    claims: dict[str, dict[str, Any]],
    context_lookup: dict[str, dict[str, Any]],
) -> dict[str, float]:
    assignment_f = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        predicted_ids = set(context_lookup.get(claim_id, {}).get("final_evidence_candidates", []))
        overlap = gold & predicted_ids
        if not gold:
            assignment_f.append(0.0)
            continue
        precision = len(overlap) / len(predicted_ids) if predicted_ids else 0.0
        recall = len(overlap) / len(gold)
        if precision == 0.0 and recall == 0.0:
            assignment_f.append(0.0)
        else:
            assignment_f.append(2 * precision * recall / (precision + recall))
    return {"assignment_f": float(np.mean(assignment_f) if assignment_f else 0.0)}


def _evaluate(
    claims: dict[str, dict[str, Any]],
    pred_labels: list[str],
    y_true: list[str],
    y_pred_proba: np.ndarray,
    context_lookup: dict[str, dict[str, Any]],
    label_order: list[str],
    ece_bins: int,
) -> dict[str, Any]:
    assignment_f = _evidence_metrics(claims, context_lookup)["assignment_f"]
    claim_accuracy = accuracy_score(y_true, pred_labels)
    claim_macro_f1 = f1_score(
        y_true,
        pred_labels,
        labels=label_order,
        average="macro",
        zero_division=0,
    )
    ece = _expected_calibration_error(y_true, y_pred_proba, label_order, ece_bins)
    return {
        "claim_accuracy": float(claim_accuracy),
        "claim_macro_f1": float(claim_macro_f1),
        "assignment_f": assignment_f,
        "harmonic_mean_F_A": float(
            (2 * claim_accuracy * assignment_f / (claim_accuracy + assignment_f))
            if (claim_accuracy + assignment_f) > 0
            else 0.0
        ),
        "calibration_ece": ece["calibration_ece"],
        "ece_bins": ece["ece_bins"],
    }


def _safe_labels_and_context(
    claims: dict[str, dict[str, Any]],
    context_lookup: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[list[ContextItem]]]:
    claim_ids = list(claims.keys())
    texts = []
    gold = []
    candidate_lists = []
    for claim_id in claim_ids:
        claim = claims[claim_id]
        ctx = context_lookup.get(claim_id)
        context_items = [ContextItem(**item) for item in ctx.get("classifier_evidence_context", [])] if ctx else []
        texts.append(_claim_context_text(claim.get("claim_text", ""), context_items))
        gold.append(claim.get("claim_label"))
        candidate_lists.append(ctx.get("final_evidence_candidates", []) if ctx else [])
    return claim_ids, texts, gold, candidate_lists


def _train_and_predict(
    model_id: str,
    label_order: list[str],
    train_claims: dict[str, dict[str, Any]],
    dev_claims: dict[str, dict[str, Any]],
    train_contexts: dict[str, dict[str, Any]],
    dev_contexts: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    train_claim_ids, train_texts, y_train, _ = _safe_labels_and_context(
        train_claims, train_contexts
    )
    dev_claim_ids, dev_texts, y_dev, dev_candidates = _safe_labels_and_context(
        dev_claims, dev_contexts
    )
    if len(train_claim_ids) == 0 or len(dev_claim_ids) == 0:
        raise SystemExit("No train or dev claim context available for training/evaluation.")

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=args.clf_max_features,
        dtype=np.float32,
    )
    x_train = tfidf.fit_transform(train_texts)
    x_dev = tfidf.transform(dev_texts)

    if model_id == "tfidf_logreg":
        estimator = LogisticRegression(
            solver="saga",
            max_iter=2_000,
            class_weight="balanced",
            random_state=args.random_seed,
        )
    elif model_id == "linear_svm":
        base = LinearSVC(
            class_weight="balanced",
            random_state=args.random_seed,
            dual=True,
        )
        estimator = CalibratedClassifierCV(
            estimator=base,
            cv=args.svm_cv,
            method="sigmoid",
        )
    else:
        raise ValueError(f"Unknown model id: {model_id}")

    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    y_pred = estimator.predict(x_dev).tolist()
    proba = estimator.predict_proba(x_dev).astype(float)

    metrics = _evaluate(
        dev_claims,
        y_pred,
        y_dev,
        proba,
        {cid: {"final_evidence_candidates": cands} for cid, cands in zip(dev_claim_ids, dev_candidates)},
        label_order,
        args.ece_bins,
    )
    metrics["fit_seconds"] = float(fit_seconds)
    metrics["classes_"] = estimator.classes_.tolist()

    predictions = {
        cid: {
            "claim_text": dev_claims[cid].get("claim_text", ""),
            "claim_label": pred,
            "evidences": dev_contexts[cid]["final_evidence_candidates"],
        }
        for cid, pred in zip(dev_claim_ids, y_pred)
    }

    report = classification_report(
        y_dev,
        y_pred,
        labels=label_order,
        zero_division=0,
    )
    matrix = confusion_matrix(y_dev, y_pred, labels=label_order)

    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
    }, report


def _write_confusion_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(v) for v in row]])


def _write_records(
    outdir: Path,
    run_id: str,
    train_claims: dict[str, dict[str, Any]],
    dev_claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    train_contexts: dict[str, dict[str, Any]],
    dev_contexts: dict[str, dict[str, Any]],
    model_results: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    train_context_path = outdir / f"{run_id}_train_context_{timestamp}_top{args.train_context_k}.jsonl"
    dev_context_path = outdir / f"{run_id}_dev_context_{timestamp}_top{args.dev_context_k}.jsonl"
    summary_path = outdir / f"{run_id}_summary_{timestamp}.json"
    blocker_path = outdir / f"{run_id}_blockers_{timestamp}.json"

    _write_jsonl(
        train_context_path,
        [train_contexts[cid] for cid in sorted(train_contexts.keys())],
    )
    _write_jsonl(
        dev_context_path,
        [dev_contexts[cid] for cid in sorted(dev_contexts.keys())],
    )

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "train_claims": len(train_claims),
        "dev_claims": len(dev_claims),
        "evidence_count": len(evidence),
        "train_context_k": args.train_context_k,
        "dev_context_k": args.dev_context_k,
        "train_context_source": "raw_lexical_tfidf",
        "dev_context_source": "o_a2_aggregate_pool",
        "models": list(model_results.keys()),
        "model_metrics": model_results,
    }
    write_json(summary_path, summary)
    blocker = {
        "status": "blocked_training_only_in_name_not_data",
        "train_context_strategy": (
            "No train-side aggregate pool path is present in current inputs. "
            "Train classifier contexts are built from raw evidence via lexical fallback."
        ),
        "risks": [
            "Train and dev contexts use different construction sources (aggregate pool for dev, lexical corpus fallback for train).",
            "No dev-label-based tuning; model hyperparameters are fixed defaults.",
            "ECE uses top-1 reliability binning, not full per-class calibration decomposition.",
        ],
    }
    write_json(blocker_path, blocker)
    return train_context_path, dev_context_path, summary_path, blocker_path


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    args.command = " ".join(sys.argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outdir = args.output_dir

    input_files = [
        args.train_claims,
        args.dev_claims,
        args.evidence,
        args.pool,
    ]
    forbidden_hits = find_forbidden_tokens([str(p) for p in input_files])
    if forbidden_hits:
        raise SystemExit(
            "STRICT GUARD FAILED: input path contains forbidden token(s): "
            + json.dumps(forbidden_hits, sort_keys=True)
        )

    print(f"Loading inputs...")
    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    pool = load_json(args.pool)

    if not isinstance(train_claims, dict) or not isinstance(dev_claims, dict):
        raise SystemExit("Claim files must be dictionaries keyed by claim_id.")
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence file must be a dictionary keyed by evidence_id.")
    if not isinstance(pool, dict):
        raise SystemExit("Aggregate pool must be a dictionary keyed by claim_id.")

    # Train-side lexical fallback indexer (raw evidence)
    print("Building train-side lexical retrieval index...")
    retriever = LexicalRetriever(
        evidence,
        max_features=args.ret_max_features,
        min_df=args.ret_min_df,
        random_seed=args.random_seed,
    )

    # Build contexts (no dev labels in training process)
    print("Building train contexts from raw lexical retrieval...")
    train_contexts = _build_train_contexts(
        train_claims, evidence, args.train_context_k, retriever
    )

    print("Building dev contexts from current O-A2 aggregate pool...")
    dev_contexts, missing_dev_claims = _build_pool_contexts(
        dev_claims,
        evidence,
        pool,
        args.dev_context_k,
        fallback_retriever=None,
    )
    if missing_dev_claims:
        print(
            f"Warning: {len(missing_dev_claims)} dev claims had incomplete aggregate context from strict pool."
        )

    # Write shared context + blocker files and get their paths
    (
        train_context_path,
        dev_context_path,
        summary_path,
        blocker_path,
    ) = _write_records(
        outdir=outdir,
        run_id=args.run_id,
        train_claims=train_claims,
        dev_claims=dev_claims,
        evidence=evidence,
        train_contexts=train_contexts,
        dev_contexts=dev_contexts,
        model_results={},
        args=args,
    )

    model_results: dict[str, Any] = {}
    model_outputs: list[str] = [
        str(train_context_path),
        str(dev_context_path),
        str(summary_path),
        str(blocker_path),
    ]

    if not (args.run_logreg or args.run_svm):
        args.run_logreg = True
        args.run_svm = True

    if args.run_logreg:
        print("Training TF-IDF + Logistic Regression...")
        result, report = _train_and_predict(
            "tfidf_logreg",
            LABEL_ORDER,
            train_claims,
            dev_claims,
            train_contexts,
            dev_contexts,
            args,
        )
        pred_path = outdir / f"{args.run_id}_tfidf_logreg_dev_predictions.json"
        metric_path = outdir / f"{args.run_id}_tfidf_logreg_metrics.json"
        report_path = outdir / f"{args.run_id}_tfidf_logreg_classification_report.txt"
        conf_path = outdir / f"{args.run_id}_tfidf_logreg_confusion_matrix.csv"
        write_json(pred_path, result["predictions"])
        write_json(metric_path, result["metrics"])
        report_path.write_text(result["report"], encoding="utf-8")
        _write_confusion_csv(conf_path, result["confusion_matrix"], LABEL_ORDER)
        model_results["tfidf_logreg"] = {
            "predictions": str(pred_path),
            "metrics": str(metric_path),
            "report": str(report_path),
            "confusion_matrix": str(conf_path),
        }
        model_outputs += [
            str(pred_path),
            str(metric_path),
            str(report_path),
            str(conf_path),
        ]

    if args.run_svm:
        print("Training TF-IDF + linear SVM (calibrated)...")
        result, report = _train_and_predict(
            "linear_svm",
            LABEL_ORDER,
            train_claims,
            dev_claims,
            train_contexts,
            dev_contexts,
            args,
        )
        pred_path = outdir / f"{args.run_id}_linear_svm_dev_predictions.json"
        metric_path = outdir / f"{args.run_id}_linear_svm_metrics.json"
        report_path = outdir / f"{args.run_id}_linear_svm_classification_report.txt"
        conf_path = outdir / f"{args.run_id}_linear_svm_confusion_matrix.csv"
        write_json(pred_path, result["predictions"])
        write_json(metric_path, result["metrics"])
        report_path.write_text(result["report"], encoding="utf-8")
        _write_confusion_csv(conf_path, result["confusion_matrix"], LABEL_ORDER)
        model_results["linear_svm"] = {
            "predictions": str(pred_path),
            "metrics": str(metric_path),
            "report": str(report_path),
            "confusion_matrix": str(conf_path),
        }
        model_outputs += [
            str(pred_path),
            str(metric_path),
            str(report_path),
            str(conf_path),
        ]

    # now that model results are available, add them to summary
    summary = load_json(summary_path)
    summary["model_metrics"] = model_results
    summary["models"] = list(model_results.keys())
    write_json(summary_path, summary)

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate",
        mode="STRICT",
        stage="o_c2_baselines",
        command=" ".join(sys.argv),
        working_directory=Path.cwd(),
        random_seed=args.random_seed,
    )
    manifest["input_files"] = [
        {
            "path": str(args.train_claims),
            "sha256": sha256_file(args.train_claims),
            "split": _infer_split(args.train_claims),
            "labels_used": True,
        },
        {
            "path": str(args.dev_claims),
            "sha256": sha256_file(args.dev_claims),
            "split": _infer_split(args.dev_claims),
            "labels_used": True,
        },
        {
            "path": str(args.evidence),
            "sha256": sha256_file(args.evidence),
            "split": _infer_split(args.evidence),
            "labels_used": False,
        },
        {
            "path": str(args.pool),
            "sha256": sha256_file(args.pool),
            "split": _infer_split(args.pool),
            "labels_used": False,
        },
    ]
    manifest["forbidden_input_scan"] = {
        "passed": not bool(forbidden_hits),
        "notes": "Input path scan for forbidden tokens passed.",
    }
    manifest["metrics"] = {
        "model_results": {
            model: load_json(Path(paths["metrics"]))
            for model, paths in model_results.items()
        },
        "data_flow_summary": {
            "train_context_source": "raw_lexical_tfidf",
            "dev_context_source": "o_a2_pool",
            "train_context_k": args.train_context_k,
            "dev_context_k": args.dev_context_k,
            "train_context_path": str(train_context_path),
            "dev_context_path": str(dev_context_path),
            "blocker_file": str(blocker_path),
        },
        "summary_path": str(summary_path),
    }
    manifest["output_files"] = sorted(set(model_outputs + [str(summary_path), str(args.manifest), str(args.record), str(blocker_path)]))
    manifest["data_flow_summary"] = (
        "Built train contexts from raw evidence with a TF-IDF cosine lexical retriever and "
        "built dev contexts from the current-run O-A2 strict aggregate pool. "
        "Trained TF-IDF logistic and calibrated linear-SVM classifiers on train contexts and "
        "evaluated only on dev claims."
    )
    manifest["split_isolation_summary"] = (
        "Train labels from data/train-claims.json are used only for fitting. "
        "Dev labels are used only for post-hoc fixed-model evaluation; no dev-based tuning or "
        "hyperparameter search is performed."
    )
    manifest["leakage_risk"] = "medium"
    manifest["reproducibility_risk"] = "low"
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["runtime"]["device"] = "cpu"
    write_json(args.manifest, manifest)
    model_outputs.append(str(args.manifest))

    record = {
        "run_id": args.run_id,
        "command": " ".join(sys.argv),
        "command_args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "status": "strict-candidate",
        "mode": "STRICT",
        "split": "train/dev",
        "stage": "o_c2_baselines",
        "claims_train": len(train_claims),
        "claims_dev": len(dev_claims),
        "evidence_count": len(evidence),
        "blocked": [
            "Train-side aggregate pool not available",
        ],
        "blocker_file": str(blocker_path),
        "data_flow": {
            "train_context_source": "raw_lexical_tfidf",
            "dev_context_source": "o_a2_pool",
            "train_context_k": args.train_context_k,
            "dev_context_k": args.dev_context_k,
            "missing_dev_pool_claims": missing_dev_claims,
        },
        "files_written": model_outputs,
        "model_metrics": model_results,
        "wall_seconds": manifest["runtime"]["wall_seconds"],
    }
    write_json(args.record, record)

    print("Wrote:")
    for path in model_outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
