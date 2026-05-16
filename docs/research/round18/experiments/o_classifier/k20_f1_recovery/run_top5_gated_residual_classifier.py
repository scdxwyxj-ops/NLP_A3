#!/usr/bin/env python
"""Top5-gated residual classifier for top20 context.

The classifier has two fixed branches:
  - head branch: claim + top5 evidence
  - tail branch: rank 6-20 evidence

A learned scalar gate uses only top5-derived features:
  final_logits = head_logits + sigmoid(top5_features @ w) * (tail_logits - head_logits)

This lets the model degenerate to top5 when the gate is closed, while allowing
extra top20 context only when top5 signals suggest it is useful.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler


def _resolve_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "round18" / "tools").exists():
            return candidate
    return Path(__file__).resolve().parents[4]


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
    _collapse_gate,
    _ensure_family_contract,
    _load_inputs,
    _train_holdout_indices,
)
from round18.tools.common import find_forbidden_tokens, manifest_base, write_json  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/k20_top5_gated_residual")
DEFAULT_REPORT_PATH = Path("round18/reports/k20_top5_gated_residual_report.md")


@dataclass(frozen=True)
class GateSpec:
    name: str
    max_features: int
    c_value: float
    gate_l2: float
    gate_feature_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top5-gated residual classifier audit.")
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--dev-pool", type=Path, default=DEFAULT_DEV_POOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--context-k", type=int, default=20)
    parser.add_argument("--head-k", type=int, default=5)
    parser.add_argument("--evidence-token-budget", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    return parser.parse_args()


def build_specs() -> list[GateSpec]:
    specs: list[GateSpec] = []
    for max_features in [30000, 60000]:
        for c_value in [1.0, 2.0, 4.0]:
            for gate_l2 in [0.0, 0.01, 0.1, 1.0, 10.0]:
                for mode in ["uncertainty", "uncertainty_plus_shallow"]:
                    specs.append(
                        GateSpec(
                            name=(
                                f"gated_head5_mf{max_features}"
                                f"_c{str(c_value).replace('.', 'p')}"
                                f"_l2{str(gate_l2).replace('.', 'p')}"
                                f"_{mode}"
                            ),
                            max_features=max_features,
                            c_value=c_value,
                            gate_l2=gate_l2,
                            gate_feature_mode=mode,
                        )
                    )
    return specs


def context_text(claim_text: str, items: list[ContextItem], evidence_token_budget: int, include_claim: bool) -> str:
    blocks: list[str] = []
    if include_claim:
        blocks.append(f"CLAIM: {claim_text.strip()}")
    for item in items:
        text = item.text
        if evidence_token_budget > 0:
            text = " ".join(text.split()[:evidence_token_budget])
        blocks.append(f"EVIDENCE_{item.rank}: {text}")
    return "\n".join(blocks)


def build_rows(
    claims: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    head_k: int,
    evidence_token_budget: int,
) -> tuple[list[str], list[str], list[str], list[str], dict[str, list[ContextItem]], dict[str, list[ContextItem]]]:
    claim_ids = list(claims.keys())
    head_texts: list[str] = []
    tail_texts: list[str] = []
    labels: list[str] = []
    head_items: dict[str, list[ContextItem]] = {}
    tail_items: dict[str, list[ContextItem]] = {}
    for claim_id in claim_ids:
        items = [ContextItem(**row) for row in contexts[claim_id]["classifier_evidence_context"]]
        head = items[:head_k]
        tail = items[head_k:]
        head_items[claim_id] = head
        tail_items[claim_id] = tail
        claim_text = claims[claim_id].get("claim_text", "")
        head_texts.append(context_text(claim_text, head, evidence_token_budget, include_claim=True))
        tail_texts.append(context_text("", tail, evidence_token_budget, include_claim=False))
        labels.append(str(claims[claim_id].get("claim_label")))
    return claim_ids, head_texts, tail_texts, labels, head_items, tail_items


def build_branch_matrix(
    train_texts: list[str],
    dev_texts: list[str],
    train_side: csr_matrix,
    dev_side: csr_matrix,
    max_features: int,
) -> tuple[csr_matrix, csr_matrix, dict[str, int]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        dtype=np.float32,
    )
    train_text_x = vectorizer.fit_transform(train_texts)
    dev_text_x = vectorizer.transform(dev_texts)
    scaler = MaxAbsScaler()
    train_side_x = scaler.fit_transform(train_side)
    dev_side_x = scaler.transform(dev_side)
    return (
        hstack([train_text_x, train_side_x], format="csr"),
        hstack([dev_text_x, dev_side_x], format="csr"),
        {"word_dim": int(train_text_x.shape[1]), "side_dim": int(train_side.shape[1])},
    )


def train_branch(x_train: csr_matrix, y_train: list[str], c_value: float) -> OneVsRestClassifier:
    model = OneVsRestClassifier(
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=1337,
        )
    )
    model.fit(x_train, y_train)
    return model


def aligned_logits(model: OneVsRestClassifier, x_dev: csr_matrix) -> np.ndarray:
    raw = model.decision_function(x_dev).astype(np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    out = np.zeros((raw.shape[0], len(LABEL_ORDER)), dtype=np.float64)
    class_to_idx = {str(label): i for i, label in enumerate(model.classes_)}
    for j, label in enumerate(LABEL_ORDER):
        if label in class_to_idx:
            out[:, j] = raw[:, class_to_idx[label]]
        else:
            out[:, j] = -20.0
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def gate_features(head_logits: np.ndarray, head_side: csr_matrix, mode: str) -> np.ndarray:
    probs = softmax(head_logits)
    sorted_probs = np.sort(probs, axis=1)
    max_prob = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / math.log(probs.shape[1])
    base = [max_prob, margin, entropy]
    if mode == "uncertainty_plus_shallow":
        dense_side = head_side.toarray().astype(np.float64)
        side_mean = dense_side.mean(axis=0, keepdims=True)
        side_std = dense_side.std(axis=0, keepdims=True) + 1e-6
        dense_side = (dense_side - side_mean) / side_std
        base.append(dense_side)
    z = np.column_stack(base) if len(base) == 3 else np.hstack([np.column_stack(base[:3]), base[3]])
    return np.hstack([np.ones((z.shape[0], 1), dtype=np.float64), z])


def label_indices(labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(LABEL_ORDER)}
    return np.array([index[label] for label in labels], dtype=np.int64)


def fit_gate(
    head_logits: np.ndarray,
    tail_logits: np.ndarray,
    gate_x: np.ndarray,
    labels: list[str],
    gate_l2: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    y = label_indices(labels)
    n, k = head_logits.shape
    delta = tail_logits - head_logits
    class_counts = np.bincount(y, minlength=k).astype(np.float64)
    class_weights = n / np.maximum(k * class_counts, 1.0)
    sample_weights = class_weights[y]

    def loss_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        gate = 1.0 / (1.0 + np.exp(-np.clip(gate_x @ w, -40, 40)))
        logits = head_logits + gate[:, None] * delta
        probs = softmax(logits)
        losses = -np.log(np.maximum(probs[np.arange(n), y], 1e-12))
        weighted_losses = sample_weights * losses
        loss = float(weighted_losses.mean() + 0.5 * gate_l2 * np.dot(w[1:], w[1:]))
        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits *= (sample_weights / n)[:, None]
        dgate = np.sum(dlogits * delta, axis=1)
        local = dgate * gate * (1.0 - gate)
        grad = gate_x.T @ local
        grad[1:] += gate_l2 * w[1:]
        return loss, grad

    init = np.zeros(gate_x.shape[1], dtype=np.float64)
    result = minimize(lambda w: loss_grad(w), init, jac=True, method="L-BFGS-B", options={"maxiter": 250})
    return result.x.astype(np.float64), {"success": bool(result.success), "loss": float(result.fun), "nit": int(result.nit)}


def predict_with_gate(head_logits: np.ndarray, tail_logits: np.ndarray, gate_x: np.ndarray, gate_w: np.ndarray) -> tuple[list[str], np.ndarray]:
    gate = 1.0 / (1.0 + np.exp(-np.clip(gate_x @ gate_w, -40, 40)))
    logits = head_logits + gate[:, None] * (tail_logits - head_logits)
    pred_idx = np.argmax(logits, axis=1)
    return [LABEL_ORDER[i] for i in pred_idx], gate


def evaluate_predictions(y_true: list[str], y_pred: list[str], gate_values: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    hist = {label: int(sum(1 for item in y_pred if item == label)) for label in LABEL_ORDER}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "top_class_share": float(max(hist.values()) / max(1, len(y_pred))),
        "prediction_histogram": hist,
        "collapse_gate": _collapse_gate(hist, args.collapse_threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).astype(int).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0),
        "gate_mean": float(np.mean(gate_values)),
        "gate_std": float(np.std(gate_values)),
        "gate_min": float(np.min(gate_values)),
        "gate_max": float(np.max(gate_values)),
    }


def run_spec(
    spec: GateSpec,
    train_head: list[str],
    train_tail: list[str],
    y_train: list[str],
    dev_head: list[str],
    dev_tail: list[str],
    y_dev: list[str],
    train_head_side: csr_matrix,
    train_tail_side: csr_matrix,
    dev_head_side: csr_matrix,
    dev_tail_side: csr_matrix,
    args: argparse.Namespace,
) -> dict[str, Any]:
    head_train_x, head_dev_x, head_dims = build_branch_matrix(train_head, dev_head, train_head_side, dev_head_side, spec.max_features)
    tail_train_x, tail_dev_x, tail_dims = build_branch_matrix(train_tail, dev_tail, train_tail_side, dev_tail_side, spec.max_features)
    head_model = train_branch(head_train_x, y_train, spec.c_value)
    tail_model = train_branch(tail_train_x, y_train, spec.c_value)
    head_train_logits = aligned_logits(head_model, head_train_x)
    tail_train_logits = aligned_logits(tail_model, tail_train_x)
    head_dev_logits = aligned_logits(head_model, head_dev_x)
    tail_dev_logits = aligned_logits(tail_model, tail_dev_x)
    gate_train_x = gate_features(head_train_logits, train_head_side, spec.gate_feature_mode)
    gate_dev_x = gate_features(head_dev_logits, dev_head_side, spec.gate_feature_mode)
    gate_w, gate_fit = fit_gate(head_train_logits, tail_train_logits, gate_train_x, y_train, spec.gate_l2)
    y_pred, gate_values = predict_with_gate(head_dev_logits, tail_dev_logits, gate_dev_x, gate_w)
    metrics = evaluate_predictions(y_dev, y_pred, gate_values, args)
    metrics.update(
        {
            "name": spec.name,
            "max_features": spec.max_features,
            "C": spec.c_value,
            "gate_l2": spec.gate_l2,
            "gate_feature_mode": spec.gate_feature_mode,
            "gate_fit": gate_fit,
            "head_dims": head_dims,
            "tail_dims": tail_dims,
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    args.command = " ".join(sys.argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden_hits = find_forbidden_tokens(
        [str(args.train_claims), str(args.dev_claims), str(args.evidence), str(args.train_pool), str(args.dev_pool), str(args.output_root)]
    )
    if forbidden_hits:
        raise SystemExit("STRICT GUARD FAILED: forbidden token(s) detected\n" + json.dumps(forbidden_hits, sort_keys=True))

    train_family, dev_family = _ensure_family_contract(args.train_pool, args.dev_pool)
    if train_family != dev_family:
        raise SystemExit("Train/dev context family mismatch.")
    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    train_contexts, train_diag = _build_contexts(train_claims, evidence, train_pool, top_k=args.context_k, source_family=train_family)
    dev_contexts, dev_diag = _build_contexts(dev_claims, evidence, dev_pool, top_k=args.context_k, source_family=dev_family)
    train_ids, train_head, train_tail, y_train, train_head_items, train_tail_items = build_rows(
        train_claims, train_contexts, args.head_k, args.evidence_token_budget
    )
    dev_ids, dev_head, dev_tail, y_dev, dev_head_items, dev_tail_items = build_rows(
        dev_claims, dev_contexts, args.head_k, args.evidence_token_budget
    )
    train_profiles = _build_profile_cache(train_claims)
    dev_profiles = _build_profile_cache(dev_claims)
    train_head_side, _ = _build_shallow_matrix(train_ids, train_profiles, train_head_items, evidence)
    train_tail_side, _ = _build_shallow_matrix(train_ids, train_profiles, train_tail_items, evidence)
    dev_head_side, _ = _build_shallow_matrix(dev_ids, dev_profiles, dev_head_items, evidence)
    dev_tail_side, _ = _build_shallow_matrix(dev_ids, dev_profiles, dev_tail_items, evidence)

    split = _train_holdout_indices(y_train, args)
    if split is None:
        raise SystemExit("Could not build train holdout split.")
    tr_idx, val_idx = split

    selection_results: list[dict[str, Any]] = []
    dev_results: list[dict[str, Any]] = []
    specs = build_specs()
    start = time.perf_counter()
    for idx, spec in enumerate(specs, start=1):
        selection = run_spec(
            spec,
            [train_head[i] for i in tr_idx],
            [train_tail[i] for i in tr_idx],
            [y_train[i] for i in tr_idx],
            [train_head[i] for i in val_idx],
            [train_tail[i] for i in val_idx],
            [y_train[i] for i in val_idx],
            train_head_side[tr_idx],
            train_tail_side[tr_idx],
            train_head_side[val_idx],
            train_tail_side[val_idx],
            args,
        )
        selection_results.append(selection)
        dev_results.append(
            run_spec(
                spec,
                train_head,
                train_tail,
                y_train,
                dev_head,
                dev_tail,
                y_dev,
                train_head_side,
                train_tail_side,
                dev_head_side,
                dev_tail_side,
                args,
            )
        )
        if idx % 10 == 0:
            print(f"finished {idx}/{len(specs)} specs", flush=True)

    selection_results.sort(key=lambda r: (r["macro_f1"], r["accuracy"], -r["top_class_share"]), reverse=True)
    dev_results.sort(key=lambda r: (r["macro_f1"], r["accuracy"], -r["top_class_share"]), reverse=True)
    selected_name = selection_results[0]["name"]
    selected_dev = next(row for row in dev_results if row["name"] == selected_name)

    k5_metrics = json.loads((REPO_ROOT / "round18/outputs/o_classifier/o_c5_comparison/worker_g_final_candidate/worker_g_final_candidate_candidate_dev_metrics.json").read_text())
    residual_metrics = json.loads((REPO_ROOT / "round18/outputs/o_classifier/k20_rank_residual/k20_rank_residual_results.json").read_text())

    payload = {
        "run_id": "k20_top5_gated_residual",
        "selection_rule": "train-holdout macro-F1, accuracy tie-break, lower top-class-share tie-break",
        "strict_selected_train_holdout": selection_results[0],
        "strict_selected_dev": selected_dev,
        "diagnostic_dev_best": dev_results[0],
        "top_train_holdout": selection_results[:10],
        "top_dev": dev_results[:10],
        "reference_k5_shallow_dev_best": {
            "macro_f1": k5_metrics["macro_f1"],
            "accuracy": k5_metrics["accuracy"],
            "top_class_share": k5_metrics["top_class_share"],
        },
        "reference_global_residual_dev_best": {
            "macro_f1": residual_metrics["diagnostic_dev_best"]["macro_f1"],
            "accuracy": residual_metrics["diagnostic_dev_best"]["accuracy"],
            "top_class_share": residual_metrics["diagnostic_dev_best"]["top_class_share"],
        },
        "manifest": manifest_base(
            run_id="k20_top5_gated_residual",
            status="diagnostic_with_train_selected_result",
            mode="STRICT_SELECTION_PLUS_DIAGNOSTIC_DEV_BEST",
            stage="k20_top5_gated_residual",
            command=args.command,
            working_directory=Path.cwd(),
            random_seed=args.random_seed,
        ),
        "runtime_seconds": float(time.perf_counter() - start),
        "context_diagnostics": {"train": train_diag, "dev": dev_diag},
    }
    out_path = args.output_root / "k20_top5_gated_residual_results.json"
    write_json(out_path, payload)

    lines = [
        "# k20 Top5-Gated Residual Classifier Audit",
        "",
        "Architecture: `final_logits = head_logits + sigmoid(top5_features @ w) * (tail_logits - head_logits)`.",
        "Gate features are derived from top5 only, so the tail branch cannot decide by looking at its own text.",
        "",
        f"- Reference top5 shallow dev-best macro-F1: `{k5_metrics['macro_f1']:.6f}`.",
        f"- Reference global residual dev-best macro-F1: `{residual_metrics['diagnostic_dev_best']['macro_f1']:.6f}`.",
        "",
        "## Train-Holdout Selected Result",
        "",
        f"- selected config: `{selected_name}`",
        f"- train-holdout macro-F1: `{selection_results[0]['macro_f1']:.6f}`",
        f"- dev macro-F1: `{selected_dev['macro_f1']:.6f}`",
        f"- dev accuracy: `{selected_dev['accuracy']:.6f}`",
        f"- gate mean/std: `{selected_dev['gate_mean']:.4f}` / `{selected_dev['gate_std']:.4f}`",
        "",
        "## Dev-Best Diagnostic Result",
        "",
        f"- best dev config: `{dev_results[0]['name']}`",
        f"- dev macro-F1: `{dev_results[0]['macro_f1']:.6f}`",
        f"- dev accuracy: `{dev_results[0]['accuracy']:.6f}`",
        f"- gate mean/std: `{dev_results[0]['gate_mean']:.4f}` / `{dev_results[0]['gate_std']:.4f}`",
        "",
        "## Top Dev Configs",
        "",
        "| rank | config | macro-F1 | accuracy | top class share | gate mean | gate std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(dev_results[:10], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['top_class_share']:.6f} | {row['gate_mean']:.4f} | {row['gate_std']:.4f} |"
        )
    lines.extend(["", f"Artifacts: `{out_path}`"])
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "results": str(out_path),
                "report": str(args.report_path),
                "selected_dev_macro_f1": selected_dev["macro_f1"],
                "dev_best_macro_f1": dev_results[0]["macro_f1"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
