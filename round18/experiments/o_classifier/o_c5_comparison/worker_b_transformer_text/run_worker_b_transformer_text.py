#!/usr/bin/env python
"""Round18 O-C5 Worker B: transformer-based strict-context classifier.

This runner compares transformer tokenized sequence classifiers against the existing
TF-IDF setup under strict train/dev context family constraints.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

SCRIPT_PATH = Path(__file__).resolve()
ROUND18_ROOT = next((path for path in SCRIPT_PATH.parents if path.name == "round18"), None)
if ROUND18_ROOT is None:
    raise SystemExit("Unable to locate repository root containing round18 package directory.")
REPO_ROOT = ROUND18_ROOT.parent
for _path in (REPO_ROOT, ROUND18_ROOT / "src"):
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
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_TRAIN_POOL = Path(
    "round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json"
)
DEFAULT_DEV_POOL = Path(
    "round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "run_manifest.json"
DEFAULT_RECORD = DEFAULT_OUTPUT_DIR / "run_record.json"
DEFAULT_RUN_ID = "o_c5_worker_b_transformer_text"


@dataclass(frozen=True)
class ContextItem:
    evidence_id: str
    rank: int
    score: float
    source: str
    text: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate strict-context transformer claim classifier."
    )
    parser.add_argument(
        "--train-claims",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS,
    )
    parser.add_argument(
        "--dev-claims",
        type=Path,
        default=DEFAULT_DEV_CLAIMS,
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )
    parser.add_argument(
        "--train-pool",
        type=Path,
        required=True,
        help="Strict-context candidate pool for train claims.",
    )
    parser.add_argument(
        "--dev-pool",
        type=Path,
        required=True,
        help="Strict-context candidate pool for dev claims.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--train-context-k",
        type=int,
        default=20,
        help="Number of ranked evidence items per claim in train context.",
    )
    parser.add_argument(
        "--dev-context-k",
        type=int,
        default=20,
        help="Number of ranked evidence items per claim in dev context.",
    )
    parser.add_argument(
        "--evidence-token-budget",
        type=int,
        default=0,
        help="Optional token budget per evidence item in context text; 0 means full text.",
    )
    parser.add_argument(
        "--model-name",
        default="distilroberta-base",
        help="Transformer tokenizer/classifier model.",
    )
    parser.add_argument(
        "--lr-grid",
        default="2e-5,3e-5",
        help="Comma-separated learning-rate candidates for train-holdout selection.",
    )
    parser.add_argument(
        "--epoch-grid",
        default="2,3",
        help="Comma-separated epoch candidates for train-holdout selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for train and dev loops.",
    )
    parser.add_argument(
        "--max-length-grid",
        default="256",
        help="Comma-separated candidate max sequence lengths.",
    )
    parser.add_argument(
        "--train-val-fraction",
        type=float,
        default=0.2,
        help="Holdout fraction from train split for hyperparameter selection.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help="If >0, only train first N and dev first N rows for smoke run.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the smoke-sized train/holdout selection and stop.",
    )
    parser.add_argument(
        "--smoke-lr-grid",
        default="2e-5",
        help="Learning-rate grid for smoke.",
    )
    parser.add_argument(
        "--smoke-epoch-grid",
        default="1",
        help="Epoch grid for smoke.",
    )
    parser.add_argument(
        "--smoke-max-length-grid",
        default="128",
        help="Max-length grid for smoke.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
    )
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    parser.add_argument("--min-macro-f1", type=float, default=0.4719276094276095)
    parser.add_argument(
        "--enable-diagnostic",
        action="store_true",
        help="Allow train/dev context family mismatch (diagnostic mode).",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def _parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("float grid must not be empty.")
    return values


def _parse_int_grid(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        ivalue = int(item)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("all grid values must be > 0")
        values.append(ivalue)
    if not values:
        raise argparse.ArgumentTypeError("int grid must not be empty.")
    return values


def _extract_pool_family(pool_path: Path) -> str:
    stem = pool_path.name.replace("-", "_")
    stem = stem.removesuffix(".json")
    stem = re.sub(r"_top\d+_candidates$", "", stem)
    stem = re.sub(r"_candidates$", "", stem)
    match = re.search(r"(?:^|_)(o_[a-z0-9_]+)(?:_|$)", stem)
    if not match:
        raise SystemExit(f"Cannot infer pool family from pool filename: {pool_path}")
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
            "Strict context source mismatch (train/dev pool family). "
            f"{train_family!r} vs {dev_family!r}. "
            "Pass --enable-diagnostic only for diagnostic-only mode."
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
    diagnostics = {"missing_evidence_text": 0}
    seen: set[str] = set()

    for pos, row in enumerate(rows):
        if len(items) >= top_k:
            break
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("evidence_id", "")).strip()
        if not evidence_id or evidence_id in seen:
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


def _build_contexts(
    claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    pool: dict[str, list[dict[str, Any]]],
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
                f"Strict context failure: missing claim_id in pool -> {claim_id}"
            )
        raw_rows = pool[claim_id]
        if not isinstance(raw_rows, list):
            raise SystemExit(f"Expected list per claim in pool: {claim_id}")
        items, item_diag = _build_context_items(
            raw_rows,
            evidence=evidence,
            top_k=top_k,
            source_family=source_family,
        )
        diagnostics["missing_evidence_text"] += item_diag["missing_evidence_text"]
        diagnostics["total_candidates"] += len(items)
        if len(items) < top_k:
            diagnostics["insufficient_context_claims"].append(claim_id)
            raise SystemExit(
                f"Strict context failure: insufficient evidence for {claim_id} "
                f"(need {top_k}, got {len(items)})."
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


def _build_claim_context_text(
    claim_text: str,
    evidence_items: list[ContextItem],
    evidence_token_budget: int = 0,
) -> str:
    blocks = [f"CLAIM: {claim_text.strip()}"]
    for item in evidence_items:
        text = item.text or ""
        if evidence_token_budget > 0:
            text = " ".join(text.split()[:evidence_token_budget])
        blocks.append(f"EVIDENCE_{item.rank}: {text}")
    return "\n".join(blocks)


def _rows_from_contexts(
    claims: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    evidence_token_budget: int,
) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    claim_ids = list(claims.keys())
    texts: list[str] = []
    labels: list[str] = []
    candidates: list[list[str]] = []
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
        candidates.append(context["final_evidence_candidates"])
    return claim_ids, texts, labels, candidates


def _validate_labels(claims: dict[str, dict[str, Any]], split_name: str) -> None:
    unknown = sorted(
        {claim.get("claim_label") for claim in claims.values() if claim.get("claim_label") not in LABEL_ORDER}
    )
    if unknown:
        raise SystemExit(f"{split_name} contains unknown label(s): {unknown}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _parse_inputs(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    for path in (
        args.train_claims,
        args.dev_claims,
        args.evidence,
        args.train_pool,
        args.dev_pool,
    ):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pool = load_json(args.train_pool)
    dev_pool = load_json(args.dev_pool)

    if not isinstance(train_claims, dict):
        raise SystemExit("train-claims must be a dict keyed by claim_id.")
    if not isinstance(dev_claims, dict):
        raise SystemExit("dev-claims must be a dict keyed by claim_id.")
    if not isinstance(evidence, dict):
        raise SystemExit("evidence must be a dict keyed by evidence_id.")
    if not isinstance(train_pool, dict):
        raise SystemExit("train-pool must be a dict keyed by claim_id.")
    if not isinstance(dev_pool, dict):
        raise SystemExit("dev-pool must be a dict keyed by claim_id.")

    _validate_labels(train_claims, "train")
    _validate_labels(dev_claims, "dev")
    return train_claims, dev_claims, evidence, train_pool, dev_pool


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_confusion_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(v) for v in row]])


def _collapse_gate(prediction_hist: dict[str, int], threshold: float) -> dict[str, Any]:
    total = sum(prediction_hist.values())
    if total == 0:
        return {
            "status": "failed",
            "reason": "no predictions",
            "threshold": float(threshold),
            "max_class_share": 0.0,
            "collapsed_classes": [],
        }
    max_label, max_count = max(prediction_hist.items(), key=lambda kv: (kv[1], kv[0]))
    share = max_count / total
    collapsed = [lbl for lbl, cnt in prediction_hist.items() if cnt / total > threshold]
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


def _evaluate_predictions(
    y_true: list[str],
    y_pred_idx: list[int],
    y_prob: np.ndarray,
) -> dict[str, Any]:
    y_pred = [ID_TO_LABEL[idx] for idx in y_pred_idx]
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="micro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    per_class_recall: dict[str, float] = {}
    row_totals = matrix.sum(axis=1).astype(float)
    for idx, label in enumerate(LABEL_ORDER):
        if row_totals[idx] <= 0:
            per_class_recall[label] = 0.0
        else:
            per_class_recall[label] = float(matrix[idx, idx] / row_totals[idx])
    hist = {label: 0 for label in LABEL_ORDER}
    for pred in y_pred:
        hist[pred] = hist.get(pred, 0) + 1
    max_share = max(hist.values()) / len(y_pred) if y_pred else 0.0
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_recall": float(macro_recall),
        "classification_report": report,
        "per_class_recall": per_class_recall,
        "prediction_histogram": hist,
        "top_class_share": float(max_share),
        "top_class": max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0],
        "confusion_matrix": matrix.tolist(),
        "proba": y_prob.tolist(),
    }


def _selection_score(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result["macro_f1"]),
        float(result["accuracy"]),
        float(result["macro_recall"]),
        -float(result["top_class_share"]),
    )


def _acceptance_gate(metrics: dict[str, Any], threshold: float, min_macro_f1: float) -> dict[str, Any]:
    failures: list[str] = []
    hist = metrics.get("prediction_histogram", {})
    recalls = metrics.get("per_class_recall", {})
    if any(int(hist.get(lbl, 0)) <= 0 for lbl in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_predictions")
    if any(float(recalls.get(lbl, 0.0)) <= 0.0 for lbl in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_recall")
    if float(metrics.get("top_class_share", 1.0)) > float(threshold):
        failures.append("max_class_share_above_threshold")
    if float(metrics.get("macro_f1", 0.0)) < float(min_macro_f1):
        failures.append("macro_f1_below_baseline")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "thresholds": {
            "max_class_share": float(threshold),
            "min_macro_f1": float(min_macro_f1),
            "require_nonzero_predictions_for_all_classes": True,
            "require_nonzero_recall_for_all_classes": True,
        },
    }


class ClaimConcatDataset(Dataset):
    def __init__(self, claim_ids: list[str], texts: list[str], labels: list[str]):
        self.claim_ids = claim_ids
        self.texts = texts
        self.labels = [LABEL_TO_ID[label] for label in labels]

    def __len__(self) -> int:
        return len(self.claim_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "id": self.claim_ids[index],
            "text": self.texts[index],
            "label": self.labels[index],
        }


def _collate_batch(
    batch: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict[str, Any]:
    ids = [row["id"] for row in batch]
    texts = [row["text"] for row in batch]
    labels = [row["label"] for row in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    encoded["claim_ids"] = ids
    return encoded


def _softmax_numpy(logits: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(float)
    return probs


def _train_candidate(
    texts_train: list[str],
    labels_train: list[str],
    texts_val: list[str],
    labels_val: list[str],
    candidate: dict[str, Any],
    tokenizer: AutoTokenizer,
    model_name: str,
    batch_size: int,
    random_seed: int,
    max_length: int,
) -> dict[str, Any]:
    _seed_everything(random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_ORDER),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["lr"])

    train_ids = [f"train-{i}" for i in range(len(texts_train))]
    val_ids = [f"val-{i}" for i in range(len(texts_val))]

    train_loader = DataLoader(
        ClaimConcatDataset(train_ids, texts_train, labels_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_batch(batch, tokenizer, max_length),
    )
    val_loader = DataLoader(
        ClaimConcatDataset(val_ids, texts_val, labels_val),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_batch(batch, tokenizer, max_length),
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_epoch_metrics: dict[str, Any] = {}
    epoch_trace: list[dict[str, Any]] = []
    best_prob: np.ndarray = np.empty((0, len(LABEL_ORDER)))
    best_preds_idx: list[int] = []
    best_claim_ids: list[str] = []

    for epoch in range(1, candidate["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train e{epoch}", leave=False):
            batch_labels = batch.pop("labels").to(device)
            batch_inputs = {key: value.to(device) for key, value in batch.items() if key != "claim_ids"}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch_inputs, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_pred_idx: list[int] = []
        val_true_idx: list[int] = []
        val_prob: list[np.ndarray] = []
        val_claim_ids: list[str] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val e{epoch}", leave=False):
                batch_labels = batch.pop("labels").to(device)
                batch_claim_ids = batch.pop("claim_ids")
                batch_inputs = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch_inputs, labels=batch_labels)
                val_loss += float(outputs.loss.detach().cpu())
                probs = _softmax_numpy(outputs.logits)
                preds = np.argmax(probs, axis=1).astype(int).tolist()
                val_pred_idx.extend(preds)
                val_true_idx.extend(batch_labels.detach().cpu().tolist())
                val_prob.append(probs)
                val_claim_ids.extend(batch_claim_ids)

        val_prob_arr = np.concatenate(val_prob, axis=0) if val_prob else np.empty((0, len(LABEL_ORDER)))
        val_true = [ID_TO_LABEL[idx] for idx in val_true_idx]
        val_metrics = _evaluate_predictions(val_true, val_pred_idx, val_prob_arr)
        val_metrics["train_loss"] = train_loss
        val_metrics["val_loss"] = val_loss / max(1, len(val_loader))
        val_metrics["epoch"] = epoch
        epoch_trace.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss / max(1, len(val_loader))),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_top_class_share": float(val_metrics["top_class_share"]),
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            best_epoch_metrics = {
                **val_metrics,
                "epoch": epoch,
            }
            best_claim_ids = list(val_claim_ids)
            best_preds_idx = list(val_pred_idx)
            best_prob = val_prob_arr.copy()

    if best_state is None:
        raise RuntimeError("No best model state from candidate training.")

    selected = {
        "candidate": candidate,
        "best_epoch": int(best_epoch),
        "best_holdout_macro_f1": float(best_val_f1),
        "best_holdout_micro_f1": float(best_epoch_metrics.get("micro_f1", 0.0)),
        "best_holdout_accuracy": float(best_epoch_metrics.get("accuracy", 0.0)),
        "best_holdout_top_class_share": float(best_epoch_metrics.get("top_class_share", 0.0)),
        "epoch_trace": epoch_trace,
        "holdout_metrics": best_epoch_metrics,
        "holdout_predictions": {
            cid: {
                "pred_label": ID_TO_LABEL[int(pred)],
                "pred_proba": {LABEL_ORDER[j]: float(p) for j, p in enumerate(best_prob[i])}
                if best_prob.size else {},
            }
            for i, (cid, pred) in enumerate(zip(best_claim_ids, best_preds_idx))
        },
    }
    selected["holdout_prob_matrix"] = best_prob.tolist() if best_prob.size else []
    model.load_state_dict(best_state)
    return selected, model


def _train_full(
    texts_train: list[str],
    labels_train: list[str],
    texts_test: list[str],
    labels_test: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    max_length: int,
    random_seed: int,
    claim_ids_test: list[str],
) -> tuple[dict[str, Any], np.ndarray, float]:
    _seed_everything(random_seed + 31)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    train_ids = [f"train-{i}" for i in range(len(texts_train))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))

    train_loader = DataLoader(
        ClaimConcatDataset(train_ids, texts_train, labels_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_batch(batch, tokenizer, max_length),
    )
    test_loader = DataLoader(
        ClaimConcatDataset(claim_ids_test, texts_test, labels_test),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_batch(batch, tokenizer, max_length),
    )

    train_start = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"full-train e{epoch}", leave=False):
            batch_labels = batch.pop("labels").to(device)
            batch_inputs = {key: value.to(device) for key, value in batch.items() if key != "claim_ids"}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch_inputs, labels=batch_labels).loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        _ = epoch_loss / max(1, len(train_loader))
    full_train_seconds = time.perf_counter() - train_start

    model.eval()
    pred_idx: list[int] = []
    true_idx: list[int] = []
    prob_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test-eval", leave=False):
            batch_labels = batch.pop("labels").to(device)
            batch_inputs = {key: value.to(device) for key, value in batch.items() if key != "claim_ids"}
            outputs = model(**batch_inputs)
            probs = _softmax_numpy(outputs.logits)
            pred_idx.extend(np.argmax(probs, axis=1).astype(int).tolist())
            true_idx.extend(batch_labels.detach().cpu().tolist())
            prob_chunks.append(probs)
    prob_array = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.empty((0, len(LABEL_ORDER)))
    y_true = [ID_TO_LABEL[idx] for idx in true_idx]
    metrics = _evaluate_predictions(y_true, pred_idx, prob_array)
    return metrics, prob_array, full_train_seconds


def _format_lr_for_name(lr: float) -> str:
    if lr >= 1:
        return f"{lr:.0f}"
    scaled = str(lr).replace(".", "p").replace("-", "m")
    return scaled


def _format_config_prefix(args: argparse.Namespace, max_length: int, lr: float, epochs: int) -> str:
    return (
        f"{args.run_id}_k{args.train_context_k}_ml{max_length}_"
        f"lr{_format_lr_for_name(lr)}_ep{epochs}"
    )


def _subset_rows(
    claim_ids: list[str],
    texts: list[str],
    labels: list[str],
    candidates: list[list[str]],
    n: int,
) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    if n <= 0 or n >= len(claim_ids):
        return claim_ids, texts, labels, candidates
    keep = claim_ids[:n]
    indexes = {cid: idx for idx, cid in enumerate(claim_ids)}
    subset_indexes = [indexes[cid] for cid in keep]
    return (
        [claim_ids[idx] for idx in subset_indexes],
        [texts[idx] for idx in subset_indexes],
        [labels[idx] for idx in subset_indexes],
        [candidates[idx] for idx in subset_indexes],
    )


def _train_holdout_indices(
    labels: list[str],
    train_val_fraction: float,
    random_seed: int,
) -> tuple[list[int], list[int]]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=train_val_fraction,
        random_state=random_seed,
    )
    indices = np.arange(len(labels))
    split = next(splitter.split(indices, labels))
    return split[0].tolist(), split[1].tolist()


def main() -> None:
    args = _parse_args()
    args.command = " ".join(sys.argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    forbidden_hits = find_forbidden_tokens(
        [str(args.train_claims), str(args.dev_claims), str(args.evidence), str(args.train_pool), str(args.dev_pool)]
    )
    if forbidden_hits:
        raise SystemExit(
            "STRICT GUARD FAILED: forbidden token(s) in input path(s): "
            + json.dumps(forbidden_hits, sort_keys=True)
        )

    train_family, dev_family, run_mode = _ensure_family_contract(
        args.train_pool,
        args.dev_pool,
        args.enable_diagnostic,
    )

    train_claims, dev_claims, evidence, train_pool, dev_pool = _parse_inputs(args)
    _seed_everything(args.random_seed)
    start = time.perf_counter()

    train_contexts, train_context_diag = _build_contexts(
        train_claims,
        evidence,
        train_pool,
        top_k=args.train_context_k,
        source_family=train_family,
    )
    dev_contexts, dev_context_diag = _build_contexts(
        dev_claims,
        evidence,
        dev_pool,
        top_k=args.dev_context_k,
        source_family=dev_family,
    )

    train_ids_full, train_texts_full, y_train_full, _ = _rows_from_contexts(
        train_claims,
        train_contexts,
        evidence_token_budget=args.evidence_token_budget,
    )
    dev_ids_full, dev_texts_full, y_dev_full, dev_candidates = _rows_from_contexts(
        dev_claims,
        dev_contexts,
        evidence_token_budget=args.evidence_token_budget,
    )

    if args.smoke_rows > 0:
        train_ids_full, train_texts_full, y_train_full, _ = _subset_rows(
            train_ids_full,
            train_texts_full,
            y_train_full,
            [[] for _ in train_ids_full],
            args.smoke_rows,
        )
        dev_ids_full, dev_texts_full, y_dev_full, dev_candidates = _subset_rows(
            dev_ids_full,
            dev_texts_full,
            y_dev_full,
            dev_candidates,
            min(args.smoke_rows, len(dev_ids_full)),
        )

    train_context_path = args.output_dir / f"{args.run_id}_train_context_top{args.train_context_k}.jsonl"
    dev_context_path = args.output_dir / f"{args.run_id}_dev_context_top{args.dev_context_k}.jsonl"
    _write_jsonl(train_context_path, [train_contexts[cid] for cid in train_ids_full])
    _write_jsonl(dev_context_path, [dev_contexts[cid] for cid in dev_ids_full])

    lr_grid = _parse_float_grid(args.smoke_lr_grid if args.smoke_only else args.lr_grid)
    epoch_grid = _parse_int_grid(args.smoke_epoch_grid if args.smoke_only else args.epoch_grid)
    max_length_grid = _parse_int_grid(args.smoke_max_length_grid if args.smoke_only else args.max_length_grid)

    holdout_idx, val_idx = _train_holdout_indices(
        y_train_full,
        args.train_val_fraction if not args.smoke_only else 0.2,
        args.random_seed,
    )
    if len(holdout_idx) == 0 or len(val_idx) == 0:
        raise SystemExit("Unable to build train/holdout split from train data.")
    train_ids = [train_ids_full[i] for i in holdout_idx]
    train_texts = [train_texts_full[i] for i in holdout_idx]
    train_labels = [y_train_full[i] for i in holdout_idx]
    val_ids = [train_ids_full[i] for i in val_idx]
    val_texts = [train_texts_full[i] for i in val_idx]
    val_labels = [y_train_full[i] for i in val_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    candidate_results: list[dict[str, Any]] = []
    selected_result: dict[str, Any] = {}
    selected_candidate: dict[str, Any] | None = None
    selected_config: dict[str, float | int] = {}
    selected_state_model = None
    selected_score: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, 1.0)

    for max_length in max_length_grid:
        for lr in lr_grid:
            for epochs in epoch_grid:
                candidate = {"model_name": args.model_name, "max_length": int(max_length), "lr": float(lr), "epochs": int(epochs)}
                holdout_result, trained_model = _train_candidate(
                    train_texts,
                    train_labels,
                    val_texts,
                    val_labels,
                    candidate,
                    tokenizer=tokenizer,
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                    random_seed=args.random_seed,
                    max_length=int(max_length),
                )
                candidate_trace = {
                    "candidate": candidate,
                    "best_holdout_epoch": holdout_result["best_epoch"],
                    "best_holdout_macro_f1": holdout_result["best_holdout_macro_f1"],
                    "best_holdout_accuracy": holdout_result["best_holdout_accuracy"],
                    "best_holdout_top_class_share": holdout_result["best_holdout_top_class_share"],
                    "epoch_trace": holdout_result["epoch_trace"],
                    "holdout_predictions": holdout_result["holdout_predictions"],
                    "holdout_prob_matrix": holdout_result.get("holdout_prob_matrix", []),
                    "holdout_metrics": holdout_result["holdout_metrics"],
                }
                candidate_results.append(candidate_trace)
                score_tuple = _selection_score(holdout_result["holdout_metrics"])
                if selected_result == {} or score_tuple > selected_score:
                    selected_result = candidate_trace
                    selected_config = {
                        "max_length": int(max_length),
                        "lr": float(lr),
                        "epochs": int(epochs),
                    }
                    selected_state_model = trained_model
                    selected_candidate = candidate_trace
                    selected_score = score_tuple

    selection = {
        "selection_mode": "train_holdout",
        "selection_fraction": float(args.train_val_fraction if not args.smoke_only else 0.2),
        "selection_results": candidate_results,
        "selection_results_sorted": sorted(
            candidate_results,
            key=lambda row: (
                row["best_holdout_macro_f1"],
                row["best_holdout_accuracy"],
                row["holdout_metrics"]["macro_recall"],
                -row["best_holdout_top_class_share"],
            ),
            reverse=True,
        ),
        "selected_by": "best_holdout_macro_f1_then_accuracy_then_macro_recall_then_lowest_top_class_share",
        "selected_config": {
            **selected_config,
            "model_name": args.model_name,
        },
        "selected_result": selected_result,
    }

    selection_trace_path = args.output_dir / f"{args.run_id}_selection_trace.json"
    write_json(selection_trace_path, selection)

    if args.smoke_only:
        selected_smoke_candidate = selected_candidate or candidate_results[0]
        holdout_predictions_payload = {
            claim_id: payload
            for claim_id, payload in selected_smoke_candidate["holdout_predictions"].items()
        }
        holdout_pred_path = args.output_dir / f"{args.run_id}_smoke_train_holdout_predictions.json"
        holdout_proba_path = args.output_dir / f"{args.run_id}_smoke_train_holdout_proba.json"
        write_json(holdout_pred_path, holdout_predictions_payload)
        write_json(
            holdout_proba_path,
            {
                "claim_ids": list(holdout_predictions_payload.keys()),
                "holdout_probabilities": selected_smoke_candidate.get("holdout_prob_matrix", []),
                "label_order": LABEL_ORDER,
            },
        )
        manifest = manifest_base(
            run_id=args.run_id,
            status="strict-candidate" if run_mode == "strict-candidate" else "diagnostic-only",
            mode="STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
            stage="o_c5_comparison",
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
        manifest["output_files"] = sorted(
            [str(args.manifest), str(args.record), str(train_context_path), str(dev_context_path)]
        )
        manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
        manifest["runtime"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        manifest["metrics"] = {"selection_trace": str(selection_trace_path)}
        manifest["data_flow_summary"] = (
            "Smoke run only. Strict context pools used for both train/dev. "
            "Transformer holdout selection on train split."
        )
        write_json(args.manifest, manifest)
        record = {
            "run_id": args.run_id,
            "stage": "o_c5_comparison",
            "mode": "STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
            "status": "smoke-only",
            "claims_train": len(train_ids_full),
            "claims_dev": len(dev_ids_full),
            "selection_trace_path": str(selection_trace_path),
            "train_holdout_predictions": str(holdout_pred_path),
            "train_holdout_proba": str(holdout_proba_path),
            "train_context_family": train_family,
            "dev_context_family": dev_family,
            "selected_config": selection["selected_config"],
            "files_written": [
                str(args.manifest),
                str(args.record),
                str(selection_trace_path),
                str(holdout_pred_path),
                str(holdout_proba_path),
                str(train_context_path),
                str(dev_context_path),
            ],
            "wall_seconds": manifest["runtime"]["wall_seconds"],
        }
        write_json(args.record, record)
        print(f"Wrote smoke selection trace: {selection_trace_path}")
        print(f"Wrote manifest: {args.manifest}")
        return

    # full training selected model
    final_prefix = _format_config_prefix(
        args,
        max_length=int(selected_config["max_length"]),
        lr=float(selected_config["lr"]),
        epochs=int(selected_config["epochs"]),
    )
    final_model = selected_state_model
    if final_model is None:
        raise SystemExit("No candidate model selected.")

    final_metrics, dev_probs, full_train_seconds = _train_full(
        train_texts_full,
        y_train_full,
        dev_texts_full,
        y_dev_full,
        model=final_model,
        tokenizer=tokenizer,
        learning_rate=float(selected_config["lr"]),
        epochs=int(selected_config["epochs"]),
        batch_size=args.batch_size,
        max_length=int(selected_config["max_length"]),
        random_seed=args.random_seed,
        claim_ids_test=dev_ids_full,
    )
    final_metrics["selection_meta"] = selection
    final_metrics["model"] = {
        "name": args.model_name,
        "max_length": int(selected_config["max_length"]),
        "epochs": int(selected_config["epochs"]),
        "learning_rate": float(selected_config["lr"]),
        "label_order": LABEL_ORDER,
    }
    final_metrics["selection_gate"] = _acceptance_gate(
        final_metrics,
        threshold=args.collapse_threshold,
        min_macro_f1=args.min_macro_f1,
    )
    final_metrics["runtime"] = {
        "full_train_wall_seconds": float(full_train_seconds),
        "total_wall_seconds": float(time.perf_counter() - start),
    }
    final_predictions_payload: dict[str, Any] = {}
    pred_idx = np.argmax(dev_probs, axis=1).astype(int).tolist() if dev_probs.size else [0] * len(dev_ids_full)
    for idx, cid in enumerate(dev_ids_full):
        dist = [
            {"label": label, "prob": float(p)}
            for label, p in zip(LABEL_ORDER, dev_probs[idx].tolist() if dev_probs.size else [])
        ]
        final_predictions_payload[cid] = {
            "claim_text": dev_claims[cid].get("claim_text", ""),
            "claim_label": LABEL_ORDER[int(pred_idx[idx])],
            "evidences": dev_contexts[cid]["final_evidence_candidates"],
            "label_distribution": dist,
            "pred_class_probability": float(max(dev_probs[idx])) if dev_probs.size else 0.0,
            "pred_class": LABEL_ORDER[int(pred_idx[idx])] if pred_idx else "",
        }

    dev_pred_path = args.output_dir / f"{final_prefix}_dev_predictions.json"
    dev_metric_path = args.output_dir / f"{final_prefix}_metrics.json"
    dev_report_path = args.output_dir / f"{final_prefix}_classification_report.txt"
    conf_matrix_path = args.output_dir / f"{final_prefix}_confusion_matrix.csv"
    holdout_pred_path = args.output_dir / f"{final_prefix}_train_holdout_predictions.json"
    holdout_proba_path = args.output_dir / f"{final_prefix}_train_holdout_proba.json"
    dev_proba_path = args.output_dir / f"{final_prefix}_dev_proba.json"

    write_json(dev_pred_path, final_predictions_payload)
    write_json(dev_metric_path, final_metrics)
    dev_report_path.write_text(final_metrics["classification_report"], encoding="utf-8")
    _write_confusion_csv(conf_matrix_path, np.array(final_metrics["confusion_matrix"]), LABEL_ORDER)
    write_json(dev_proba_path, {"label_order": LABEL_ORDER, "probabilities": dev_probs.tolist()})

    selected_candidate = selected_candidate or next(
        item
        for item in selection["selection_results"]
        if item["candidate"]["max_length"] == selected_config["max_length"]
        and item["candidate"]["lr"] == selected_config["lr"]
        and item["candidate"]["epochs"] == selected_config["epochs"]
    )
    write_json(
        holdout_pred_path,
        {
            "claim_ids": list(selected_candidate["holdout_predictions"].keys()),
            "predictions": selected_candidate["holdout_predictions"],
        },
    )
    write_json(
        holdout_proba_path,
        {
            "label_order": LABEL_ORDER,
            "probabilities": selected_candidate.get("holdout_prob_matrix", []),
            "claim_ids": list(selected_candidate["holdout_predictions"].keys()),
        },
    )

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate" if run_mode == "strict-candidate" else "diagnostic-only",
        mode="STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
        stage="o_c5_comparison",
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
    manifest["metrics"] = {
        "dev": {
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "macro_recall": final_metrics["macro_recall"],
            "top_class_share": final_metrics["top_class_share"],
            "per_class_recall": final_metrics["per_class_recall"],
            "selection_trace": str(selection_trace_path),
            "selection_config": selection["selected_config"],
        },
        "train_context_family": train_family,
        "dev_context_family": dev_family,
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["runtime"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    manifest["runtime"]["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3) if torch.cuda.is_available() else None
    manifest["data_flow_summary"] = (
        f"Strict strict context pools used. Candidate: {args.model_name}. "
        f"Context k train/dev: {args.train_context_k}/{args.dev_context_k}. "
        f"Train-holdout selection and full retrain on all train."
    )
    manifest["output_files"] = sorted(
        [
            str(args.manifest),
            str(args.record),
            str(train_context_path),
            str(dev_context_path),
            str(selection_trace_path),
            str(dev_pred_path),
            str(dev_metric_path),
            str(dev_report_path),
            str(conf_matrix_path),
            str(holdout_pred_path),
            str(holdout_proba_path),
            str(dev_proba_path),
        ]
    )
    write_json(args.manifest, manifest)

    record = {
        "run_id": args.run_id,
        "stage": "o_c5_comparison",
        "mode": "STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
        "status": run_mode,
        "selection_mode": "train_holdout",
        "smoke_only": False,
        "split": "train/dev",
        "train_claims": len(train_claims),
        "dev_claims": len(dev_claims),
        "selected_config": selection["selected_config"],
        "selection_trace_path": str(selection_trace_path),
        "train_holdout_predictions": str(holdout_pred_path),
        "train_holdout_prob_path": str(holdout_proba_path),
        "dev_metrics_path": str(dev_metric_path),
        "dev_predictions_path": str(dev_pred_path),
        "dev_confusion_matrix": str(conf_matrix_path),
        "train_context_path": str(train_context_path),
        "dev_context_path": str(dev_context_path),
        "files_written": [
            str(args.manifest),
            str(args.record),
            str(train_context_path),
            str(dev_context_path),
            str(selection_trace_path),
            str(dev_pred_path),
            str(dev_metric_path),
            str(dev_report_path),
            str(conf_matrix_path),
            str(holdout_pred_path),
            str(holdout_proba_path),
            str(dev_proba_path),
        ],
        "selection": selection,
        "dev": final_metrics,
        "wall_seconds": manifest["runtime"]["wall_seconds"],
    }
    write_json(args.record, record)

    print(f"Selection trace: {selection_trace_path}")
    print(f"Dev metrics: {dev_metric_path}")
    print(f"Dev predictions: {dev_pred_path}")
    print(f"Confusion matrix: {conf_matrix_path}")
    print(f"Holdout probs for fusion: {holdout_proba_path}")


if __name__ == "__main__":
    main()
