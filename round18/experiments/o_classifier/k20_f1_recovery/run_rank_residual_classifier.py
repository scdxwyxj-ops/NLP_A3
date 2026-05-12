#!/usr/bin/env python
"""Rank-block residual classifier for top20 context.

Main branch: claim + top5 evidence.
Residual branch: rank 6-20 evidence, scaled by tail_alpha.

When tail_alpha=0, the model can degenerate to a top5 classifier. This tests
whether top20 context can add useful information without forcing noisy tail text
into the main representation.
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


DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/k20_rank_residual")
DEFAULT_REPORT_PATH = Path("round18/reports/k20_rank_residual_report.md")


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    max_features: int
    c_value: float
    tail_alpha: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank-block residual classifier audit.")
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
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def build_specs() -> list[ResidualSpec]:
    specs: list[ResidualSpec] = []
    for max_features in [30000, 60000]:
        for c_value in [0.5, 1.0, 2.0, 4.0]:
            for tail_alpha in [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]:
                specs.append(
                    ResidualSpec(
                        name=f"head5_mf{max_features}_c{str(c_value).replace('.', 'p')}_tail{str(tail_alpha).replace('.', 'p')}",
                        max_features=max_features,
                        c_value=c_value,
                        tail_alpha=tail_alpha,
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


def vectorize(
    train_head: list[str],
    train_tail: list[str],
    dev_head: list[str],
    dev_tail: list[str],
    train_head_side: csr_matrix,
    train_tail_side: csr_matrix,
    dev_head_side: csr_matrix,
    dev_tail_side: csr_matrix,
    spec: ResidualSpec,
) -> tuple[csr_matrix, csr_matrix, dict[str, int]]:
    head_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=spec.max_features,
        dtype=np.float32,
    )
    tail_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=spec.max_features,
        dtype=np.float32,
    )
    train_head_x = head_vec.fit_transform(train_head)
    dev_head_x = head_vec.transform(dev_head)
    train_tail_x = tail_vec.fit_transform(train_tail) * spec.tail_alpha
    dev_tail_x = tail_vec.transform(dev_tail) * spec.tail_alpha

    head_scaler = MaxAbsScaler()
    tail_scaler = MaxAbsScaler()
    train_head_side_x = head_scaler.fit_transform(train_head_side)
    dev_head_side_x = head_scaler.transform(dev_head_side)
    train_tail_side_x = tail_scaler.fit_transform(train_tail_side) * spec.tail_alpha
    dev_tail_side_x = tail_scaler.transform(dev_tail_side) * spec.tail_alpha

    x_train = hstack([train_head_x, train_tail_x, train_head_side_x, train_tail_side_x], format="csr")
    x_dev = hstack([dev_head_x, dev_tail_x, dev_head_side_x, dev_tail_side_x], format="csr")
    dims = {
        "head_word_dim": int(train_head_x.shape[1]),
        "tail_word_dim": int(train_tail_x.shape[1]),
        "head_side_dim": int(train_head_side.shape[1]),
        "tail_side_dim": int(train_tail_side.shape[1]),
    }
    return x_train, x_dev, dims


def evaluate(
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
    spec: ResidualSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_train, x_dev, dims = vectorize(
        train_head,
        train_tail,
        dev_head,
        dev_tail,
        train_head_side,
        train_tail_side,
        dev_head_side,
        dev_tail_side,
        spec,
    )
    model = OneVsRestClassifier(
        LogisticRegression(
            C=spec.c_value,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=args.random_seed,
        )
    )
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    pred = model.predict(x_dev).astype(str).tolist()
    hist = {label: int(sum(1 for x in pred if x == label)) for label in LABEL_ORDER}
    return {
        "name": spec.name,
        "max_features": spec.max_features,
        "C": spec.c_value,
        "tail_alpha": spec.tail_alpha,
        "accuracy": float(accuracy_score(y_dev, pred)),
        "macro_f1": float(f1_score(y_dev, pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "top_class_share": float(max(hist.values()) / max(1, len(pred))),
        "prediction_histogram": hist,
        "collapse_gate": _collapse_gate(hist, args.collapse_threshold),
        "confusion_matrix": confusion_matrix(y_dev, pred, labels=LABEL_ORDER).astype(int).tolist(),
        "classification_report": classification_report(y_dev, pred, labels=LABEL_ORDER, zero_division=0),
        "fit_seconds": float(fit_seconds),
        **dims,
    }


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
    for idx, spec in enumerate(specs, start=1):
        selection = evaluate(
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
            spec,
            args,
        )
        selection_results.append(selection)
        dev_results.append(
            evaluate(
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
                spec,
                args,
            )
        )
        if idx % 12 == 0:
            print(f"finished {idx}/{len(specs)} specs", flush=True)

    selection_results.sort(key=lambda r: (r["macro_f1"], r["accuracy"], -r["top_class_share"]), reverse=True)
    dev_results.sort(key=lambda r: (r["macro_f1"], r["accuracy"], -r["top_class_share"]), reverse=True)
    selected_name = selection_results[0]["name"]
    selected_dev = next(row for row in dev_results if row["name"] == selected_name)
    k5_metrics = json.loads((REPO_ROOT / "round18/outputs/o_classifier/o_c5_comparison/worker_g_final_candidate/worker_g_final_candidate_candidate_dev_metrics.json").read_text())

    payload = {
        "run_id": "k20_rank_residual",
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
        "manifest": manifest_base(
            run_id="k20_rank_residual",
            status="diagnostic_with_train_selected_result",
            mode="STRICT_SELECTION_PLUS_DIAGNOSTIC_DEV_BEST",
            stage="k20_rank_residual",
            command=args.command,
            working_directory=Path.cwd(),
            random_seed=args.random_seed,
        ),
        "context_diagnostics": {"train": train_diag, "dev": dev_diag},
    }
    out_path = args.output_root / "k20_rank_residual_results.json"
    write_json(out_path, payload)

    lines = [
        "# k20 Rank-Residual Classifier Audit",
        "",
        "Architecture: main branch is claim + top5 evidence; residual branch is ranks 6-20 scaled by tail_alpha.",
        "When tail_alpha=0, the architecture degenerates to a top5-style classifier.",
        "",
        f"- Reference top5 shallow dev-best macro-F1: `{k5_metrics['macro_f1']:.6f}`.",
        "",
        "## Train-Holdout Selected Result",
        "",
        f"- selected config: `{selected_name}`",
        f"- train-holdout macro-F1: `{selection_results[0]['macro_f1']:.6f}`",
        f"- dev macro-F1: `{selected_dev['macro_f1']:.6f}`",
        f"- dev accuracy: `{selected_dev['accuracy']:.6f}`",
        f"- tail_alpha: `{selected_dev['tail_alpha']}`",
        "",
        "## Dev-Best Diagnostic Result",
        "",
        f"- best dev config: `{dev_results[0]['name']}`",
        f"- dev macro-F1: `{dev_results[0]['macro_f1']:.6f}`",
        f"- dev accuracy: `{dev_results[0]['accuracy']:.6f}`",
        f"- tail_alpha: `{dev_results[0]['tail_alpha']}`",
        "",
        "## Top Dev Configs",
        "",
        "| rank | config | tail_alpha | macro-F1 | accuracy | top class share |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(dev_results[:10], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | {row['tail_alpha']:.2f} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['top_class_share']:.6f} |"
        )
    lines.extend(["", f"Artifacts: `{out_path}`"])
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(out_path), "report": str(args.report_path), "selected_dev_macro_f1": selected_dev["macro_f1"], "dev_best_macro_f1": dev_results[0]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
