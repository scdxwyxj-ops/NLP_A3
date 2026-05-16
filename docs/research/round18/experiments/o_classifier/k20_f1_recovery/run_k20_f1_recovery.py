#!/usr/bin/env python
"""Focused k20 classifier tuning audit.

This script answers a narrow question: can a top20-context classifier be tuned
to beat the previously observed top5 shallow dev macro-F1, without selecting
hyperparameters on dev?

Outputs are diagnostic unless the selected config comes from train holdout.
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
    _rows_from_contexts,
    _train_holdout_indices,
)
from round18.tools.common import find_forbidden_tokens, manifest_base, write_json  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/k20_f1_recovery")
DEFAULT_REPORT_PATH = Path("round18/reports/k20_f1_recovery_report.md")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    word_ngram: tuple[int, int]
    word_max_features: int
    char_ngram: tuple[int, int] | None
    char_max_features: int
    use_shallow: bool
    c_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused top20 classifier tuning audit.")
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--dev-pool", type=Path, default=DEFAULT_DEV_POOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--context-k", type=int, default=20)
    parser.add_argument("--evidence-token-budget", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def build_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    targeted_grid = [
        ((1, 2), 30000, None, 0, [0.5, 1.0, 2.0, 4.0, 8.0]),
        ((1, 2), 60000, None, 0, [0.5, 1.0, 2.0, 4.0, 8.0]),
        ((1, 3), 20000, None, 0, [2.0, 4.0, 8.0]),
        ((1, 3), 30000, None, 0, [2.0, 4.0, 8.0]),
        ((1, 2), 60000, (3, 5), 20000, [1.0, 2.0, 4.0]),
        ((1, 2), 60000, (3, 5), 30000, [1.0, 2.0, 4.0]),
    ]
    for word_ngram, word_mf, char_ngram, char_mf, c_grid in targeted_grid:
        for use_shallow in [False, True]:
            for c_value in c_grid:
                parts = [f"w{word_ngram[0]}x{word_ngram[1]}_mf{word_mf}"]
                if char_ngram is not None:
                    parts.append(f"char{char_ngram[0]}x{char_ngram[1]}_mf{char_mf}")
                if use_shallow:
                    parts.append("shallow")
                parts.append(f"c{str(c_value).replace('.', 'p')}")
                specs.append(
                    FeatureSpec(
                        name="_".join(parts),
                        word_ngram=word_ngram,
                        word_max_features=word_mf,
                        char_ngram=char_ngram,
                        char_max_features=char_mf,
                        use_shallow=use_shallow,
                        c_value=c_value,
                    )
                )
    return specs


def vectorize(
    train_texts: list[str],
    dev_texts: list[str],
    spec: FeatureSpec,
    train_side: csr_matrix | None,
    dev_side: csr_matrix | None,
) -> tuple[csr_matrix, csr_matrix, dict[str, int]]:
    matrices_train = []
    matrices_dev = []
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=spec.word_ngram,
        max_features=spec.word_max_features,
        dtype=np.float32,
    )
    matrices_train.append(word.fit_transform(train_texts))
    matrices_dev.append(word.transform(dev_texts))
    dims = {"word_dim": int(matrices_train[-1].shape[1]), "char_dim": 0, "side_dim": 0}

    if spec.char_ngram is not None:
        char = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=spec.char_ngram,
            max_features=spec.char_max_features,
            dtype=np.float32,
        )
        matrices_train.append(char.fit_transform(train_texts))
        matrices_dev.append(char.transform(dev_texts))
        dims["char_dim"] = int(matrices_train[-1].shape[1])

    if spec.use_shallow:
        if train_side is None or dev_side is None:
            raise ValueError("shallow features requested without side matrices")
        scaler = MaxAbsScaler()
        matrices_train.append(scaler.fit_transform(train_side))
        matrices_dev.append(scaler.transform(dev_side))
        dims["side_dim"] = int(train_side.shape[1])

    return hstack(matrices_train, format="csr"), hstack(matrices_dev, format="csr"), dims


def evaluate(
    train_texts: list[str],
    y_train: list[str],
    dev_texts: list[str],
    y_dev: list[str],
    spec: FeatureSpec,
    args: argparse.Namespace,
    train_side: csr_matrix | None,
    dev_side: csr_matrix | None,
) -> dict[str, Any]:
    x_train, x_dev, dims = vectorize(train_texts, dev_texts, spec, train_side, dev_side)
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
    cm = confusion_matrix(y_dev, pred, labels=LABEL_ORDER)
    hist = {label: int(sum(1 for x in pred if x == label)) for label in LABEL_ORDER}
    top_class_share = max(hist.values()) / max(1, len(pred))
    return {
        "name": spec.name,
        "word_ngram": list(spec.word_ngram),
        "word_max_features": spec.word_max_features,
        "char_ngram": list(spec.char_ngram) if spec.char_ngram else None,
        "char_max_features": spec.char_max_features,
        "use_shallow": spec.use_shallow,
        "C": spec.c_value,
        "accuracy": float(accuracy_score(y_dev, pred)),
        "macro_f1": float(f1_score(y_dev, pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "top_class_share": float(top_class_share),
        "prediction_histogram": hist,
        "collapse_gate": _collapse_gate(hist, args.collapse_threshold),
        "confusion_matrix": cm.astype(int).tolist(),
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
        raise SystemExit("STRICT GUARD FAILED: forbidden token(s) detected\\n" + json.dumps(forbidden_hits, sort_keys=True))

    train_family, dev_family = _ensure_family_contract(args.train_pool, args.dev_pool)
    if train_family != dev_family:
        raise SystemExit("Train/dev context family mismatch.")

    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    train_contexts, train_diag = _build_contexts(train_claims, evidence, train_pool, top_k=args.context_k, source_family=train_family)
    dev_contexts, dev_diag = _build_contexts(dev_claims, evidence, dev_pool, top_k=args.context_k, source_family=dev_family)
    train_ids, train_texts, y_train, train_context_items = _rows_from_contexts(train_claims, train_contexts, evidence_token_budget=args.evidence_token_budget)
    dev_ids, dev_texts, y_dev, dev_context_items = _rows_from_contexts(dev_claims, dev_contexts, evidence_token_budget=args.evidence_token_budget)

    train_profiles = _build_profile_cache(train_claims)
    dev_profiles = _build_profile_cache(dev_claims)
    train_side, _ = _build_shallow_matrix(train_ids, train_profiles, train_context_items, evidence)
    dev_side, _ = _build_shallow_matrix(dev_ids, dev_profiles, dev_context_items, evidence)

    split = _train_holdout_indices(y_train, args)
    if split is None:
        raise SystemExit("Could not build train holdout split.")
    tr_idx, val_idx = split
    train_texts_sub = [train_texts[i] for i in tr_idx]
    val_texts_sub = [train_texts[i] for i in val_idx]
    y_train_sub = [y_train[i] for i in tr_idx]
    y_val = [y_train[i] for i in val_idx]

    specs = build_specs()
    selection_results: list[dict[str, Any]] = []
    dev_results: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        use_train_side = train_side if spec.use_shallow else None
        use_dev_side = dev_side if spec.use_shallow else None
        val_train_side = train_side[tr_idx] if spec.use_shallow else None
        val_side = train_side[val_idx] if spec.use_shallow else None
        selection = evaluate(train_texts_sub, y_train_sub, val_texts_sub, y_val, spec, args, val_train_side, val_side)
        selection_results.append(selection)
        dev_result = evaluate(train_texts, y_train, dev_texts, y_dev, spec, args, use_train_side, use_dev_side)
        dev_results.append(dev_result)
        if idx % 20 == 0:
            print(f"finished {idx}/{len(specs)} specs", flush=True)

    selection_results.sort(key=lambda r: (r["macro_f1"], r["accuracy"], -r["top_class_share"]), reverse=True)
    dev_results.sort(key=lambda r: (r["macro_f1"], r["accuracy"], -r["top_class_share"]), reverse=True)
    selected_name = selection_results[0]["name"]
    selected_dev = next(row for row in dev_results if row["name"] == selected_name)

    promoted_metrics = json.loads((REPO_ROOT / "round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_metrics.json").read_text())
    k5_metrics = json.loads((REPO_ROOT / "round18/outputs/o_classifier/o_c5_comparison/worker_g_final_candidate/worker_g_final_candidate_candidate_dev_metrics.json").read_text())

    payload = {
        "run_id": "k20_f1_recovery",
        "context_k": args.context_k,
        "selection_rule": "train-holdout macro-F1, accuracy tie-break, lower top-class-share tie-break",
        "strict_selected": selected_dev,
        "strict_selected_train_holdout": selection_results[0],
        "diagnostic_dev_best": dev_results[0],
        "top_train_holdout": selection_results[:10],
        "top_dev": dev_results[:10],
        "reference_promoted_k20": {
            "macro_f1": promoted_metrics["macro_f1"],
            "accuracy": promoted_metrics["accuracy"],
            "top_class_share": promoted_metrics["top_class_share"],
        },
        "reference_k5_shallow_dev_best": {
            "macro_f1": k5_metrics["macro_f1"],
            "accuracy": k5_metrics["accuracy"],
            "top_class_share": k5_metrics["top_class_share"],
        },
        "manifest": manifest_base(
            run_id="k20_f1_recovery",
            status="diagnostic_with_train_selected_result",
            mode="STRICT_SELECTION_PLUS_DIAGNOSTIC_DEV_BEST",
            stage="k20_f1_recovery",
            command=args.command,
            working_directory=Path.cwd(),
            random_seed=args.random_seed,
        ),
        "context_diagnostics": {"train": train_diag, "dev": dev_diag},
    }
    out_path = args.output_root / "k20_f1_recovery_results.json"
    write_json(out_path, payload)

    report_lines = [
        "# k20 F1 Recovery Audit",
        "",
        "Goal: test whether a top20-context classifier can be tuned to exceed the top5 shallow dev macro-F1 without dev-selected hyperparameters.",
        "",
        "## References",
        "",
        f"- Promoted k20 logistic: macro-F1 `{promoted_metrics['macro_f1']:.6f}`, accuracy `{promoted_metrics['accuracy']:.6f}`.",
        f"- Top5 shallow dev-best: macro-F1 `{k5_metrics['macro_f1']:.6f}`, accuracy `{k5_metrics['accuracy']:.6f}`.",
        "",
        "## Train-Holdout Selected k20 Result",
        "",
        f"- selected config: `{selected_name}`",
        f"- train-holdout macro-F1: `{selection_results[0]['macro_f1']:.6f}`",
        f"- dev macro-F1: `{selected_dev['macro_f1']:.6f}`",
        f"- dev accuracy: `{selected_dev['accuracy']:.6f}`",
        f"- top class share: `{selected_dev['top_class_share']:.6f}`",
        f"- collapse: `{selected_dev['collapse_gate']['status']}`",
        "",
        "## Dev-Best Diagnostic k20 Result",
        "",
        f"- best dev config: `{dev_results[0]['name']}`",
        f"- dev macro-F1: `{dev_results[0]['macro_f1']:.6f}`",
        f"- dev accuracy: `{dev_results[0]['accuracy']:.6f}`",
        f"- top class share: `{dev_results[0]['top_class_share']:.6f}`",
        "",
        "## Top Dev Configs",
        "",
        "| rank | config | macro-F1 | accuracy | top class share |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(dev_results[:10], start=1):
        report_lines.append(
            f"| {rank} | `{row['name']}` | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['top_class_share']:.6f} |"
        )
    report_lines.extend(["", f"Artifacts: `{out_path}`"])
    args.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(out_path), "report": str(args.report_path), "selected_dev_macro_f1": selected_dev["macro_f1"], "dev_best_macro_f1": dev_results[0]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
