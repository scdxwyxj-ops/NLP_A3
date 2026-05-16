#!/usr/bin/env python
"""Bootstrap score distribution audit for classifier variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier


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
    _ensure_family_contract,
    _load_inputs,
)
from round18.experiments.o_classifier.k20_f1_recovery.run_rank_residual_classifier import (  # noqa: E402
    ResidualSpec,
    build_rows as build_residual_rows,
    vectorize as residual_vectorize,
)
from round18.experiments.o_classifier.k20_f1_recovery.run_top5_gated_residual_classifier import (  # noqa: E402
    GateSpec,
    aligned_logits,
    build_branch_matrix,
    build_rows as build_gate_rows,
    fit_gate,
    gate_features,
    predict_with_gate,
    train_branch,
)
from round18.tools.common import find_forbidden_tokens, manifest_base, write_json  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/score_distribution_audit")
DEFAULT_REPORT_PATH = Path("round18/reports/score_distribution_audit_report.md")
DEFAULT_FIG_PATH = Path("round18/reports/figures/score_distribution_audit.png")

METHOD_COLORS = {
    "Evidence-Aware Logistic": "#4C78A8",
    "Top5 Shallow Candidate": "#D37295",
    "Global Residual Diagnostic": "#72B7B2",
    "Top5-Gated Residual Diagnostic": "#54A24B",
    "Top5-Gated Train-Selected": "#8C9AA9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap score distributions for classifier variants.")
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--dev-pool", type=Path, default=DEFAULT_DEV_POOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--fig-path", type=Path, default=DEFAULT_FIG_PATH)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260510)
    parser.add_argument("--context-k", type=int, default=20)
    parser.add_argument("--head-k", type=int, default=5)
    parser.add_argument("--evidence-token-budget", type=int, default=0)
    return parser.parse_args()


def load_prediction_file(path: Path, gold_lookup: dict[str, str]) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        claim_id: {
            "gold": str(gold_lookup[claim_id]),
            "pred": str(row["pred_class"]),
        }
        for claim_id, row in payload.items()
    }


def train_residual_predictions(args: argparse.Namespace, spec: ResidualSpec) -> dict[str, dict[str, str]]:
    train_family, dev_family = _ensure_family_contract(args.train_pool, args.dev_pool)
    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    train_contexts, _ = _build_contexts(train_claims, evidence, train_pool, top_k=args.context_k, source_family=train_family)
    dev_contexts, _ = _build_contexts(dev_claims, evidence, dev_pool, top_k=args.context_k, source_family=dev_family)
    train_ids, train_head, train_tail, y_train, train_head_items, train_tail_items = build_residual_rows(
        train_claims, train_contexts, args.head_k, args.evidence_token_budget
    )
    dev_ids, dev_head, dev_tail, y_dev, dev_head_items, dev_tail_items = build_residual_rows(
        dev_claims, dev_contexts, args.head_k, args.evidence_token_budget
    )
    train_profiles = _build_profile_cache(train_claims)
    dev_profiles = _build_profile_cache(dev_claims)
    train_head_side, _ = _build_shallow_matrix(train_ids, train_profiles, train_head_items, evidence)
    train_tail_side, _ = _build_shallow_matrix(train_ids, train_profiles, train_tail_items, evidence)
    dev_head_side, _ = _build_shallow_matrix(dev_ids, dev_profiles, dev_head_items, evidence)
    dev_tail_side, _ = _build_shallow_matrix(dev_ids, dev_profiles, dev_tail_items, evidence)
    x_train, x_dev, _ = residual_vectorize(
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
        LogisticRegression(C=spec.c_value, solver="liblinear", max_iter=1000, class_weight="balanced", random_state=1337)
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_dev).astype(str).tolist()
    return {claim_id: {"gold": gold, "pred": yhat} for claim_id, gold, yhat in zip(dev_ids, y_dev, pred)}


def train_gated_predictions(args: argparse.Namespace, spec: GateSpec) -> dict[str, dict[str, str]]:
    train_family, dev_family = _ensure_family_contract(args.train_pool, args.dev_pool)
    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    train_contexts, _ = _build_contexts(train_claims, evidence, train_pool, top_k=args.context_k, source_family=train_family)
    dev_contexts, _ = _build_contexts(dev_claims, evidence, dev_pool, top_k=args.context_k, source_family=dev_family)
    train_ids, train_head, train_tail, y_train, train_head_items, train_tail_items = build_gate_rows(
        train_claims, train_contexts, args.head_k, args.evidence_token_budget
    )
    dev_ids, dev_head, dev_tail, y_dev, dev_head_items, dev_tail_items = build_gate_rows(
        dev_claims, dev_contexts, args.head_k, args.evidence_token_budget
    )
    train_profiles = _build_profile_cache(train_claims)
    dev_profiles = _build_profile_cache(dev_claims)
    train_head_side, _ = _build_shallow_matrix(train_ids, train_profiles, train_head_items, evidence)
    train_tail_side, _ = _build_shallow_matrix(train_ids, train_profiles, train_tail_items, evidence)
    dev_head_side, _ = _build_shallow_matrix(dev_ids, dev_profiles, dev_head_items, evidence)
    dev_tail_side, _ = _build_shallow_matrix(dev_ids, dev_profiles, dev_tail_items, evidence)

    head_train_x, head_dev_x, _ = build_branch_matrix(train_head, dev_head, train_head_side, dev_head_side, spec.max_features)
    tail_train_x, tail_dev_x, _ = build_branch_matrix(train_tail, dev_tail, train_tail_side, dev_tail_side, spec.max_features)
    head_model = train_branch(head_train_x, y_train, spec.c_value)
    tail_model = train_branch(tail_train_x, y_train, spec.c_value)
    head_train_logits = aligned_logits(head_model, head_train_x)
    tail_train_logits = aligned_logits(tail_model, tail_train_x)
    head_dev_logits = aligned_logits(head_model, head_dev_x)
    tail_dev_logits = aligned_logits(tail_model, tail_dev_x)
    gate_train_x = gate_features(head_train_logits, train_head_side, spec.gate_feature_mode)
    gate_dev_x = gate_features(head_dev_logits, dev_head_side, spec.gate_feature_mode)
    gate_w, _ = fit_gate(head_train_logits, tail_train_logits, gate_train_x, y_train, spec.gate_l2)
    pred, gate_values = predict_with_gate(head_dev_logits, tail_dev_logits, gate_dev_x, gate_w)
    return {
        claim_id: {"gold": gold, "pred": yhat, "gate": float(gate)}
        for claim_id, gold, yhat, gate in zip(dev_ids, y_dev, pred, gate_values)
    }


def vectors(predictions: dict[str, dict[str, dict[str, str]]]) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    claim_ids = sorted(next(iter(predictions.values())).keys())
    gold = np.array([next(iter(predictions.values()))[claim_id]["gold"] for claim_id in claim_ids], dtype=object)
    pred_vectors = {}
    for name, rows in predictions.items():
        if sorted(rows.keys()) != claim_ids:
            raise ValueError(f"claim ID mismatch for {name}")
        pred_vectors[name] = np.array([rows[claim_id]["pred"] for claim_id in claim_ids], dtype=object)
    return claim_ids, pred_vectors, gold


def score(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(gold.tolist(), pred.tolist(), labels=LABEL_ORDER, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold.tolist(), pred.tolist())),
    }


def bootstrap_scores(
    gold: np.ndarray,
    pred_vectors: dict[str, np.ndarray],
    rng: np.random.Generator,
    samples: int,
) -> dict[str, dict[str, np.ndarray]]:
    n = len(gold)
    out = {name: {"macro_f1": np.empty(samples), "accuracy": np.empty(samples)} for name in pred_vectors}
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        for name, pred in pred_vectors.items():
            s = score(gold[idx], pred[idx])
            out[name]["macro_f1"][i] = s["macro_f1"]
            out[name]["accuracy"][i] = s["accuracy"]
    return out


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p2_5": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.50)),
        "p97_5": float(np.quantile(values, 0.975)),
    }


def plot_distributions(
    fig_path: Path,
    boot: dict[str, dict[str, np.ndarray]],
    point_scores: dict[str, dict[str, float]],
    deltas: dict[str, np.ndarray],
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax = axes[0]
    for name, row in boot.items():
        values = row["macro_f1"]
        ax.hist(values, bins=55, alpha=0.42, density=True, label=f"{name} ({point_scores[name]['macro_f1']:.1%})", color=METHOD_COLORS.get(name))
        ax.axvline(point_scores[name]["macro_f1"], color=METHOD_COLORS.get(name), lw=1.6)
    ax.set_title("Bootstrap macro-F1 distributions on dev")
    ax.set_xlabel("macro-F1")
    ax.set_ylabel("density")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for name, values in deltas.items():
        ax.hist(values, bins=55, alpha=0.48, density=True, label=name, color=METHOD_COLORS.get(name.replace(" minus Top5 Shallow Candidate", "")))
        ax.axvline(np.mean(values), color=METHOD_COLORS.get(name.replace(" minus Top5 Shallow Candidate", "")), lw=1.6)
    ax.axvline(0.0, color="#2D3748", lw=1.1, ls="--")
    ax.set_title("Paired bootstrap delta vs Top5 Shallow Candidate")
    ax.set_xlabel("macro-F1 delta")
    ax.set_ylabel("density")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


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

    dev_claims_for_gold = json.loads(args.dev_claims.read_text(encoding="utf-8"))
    gold_lookup = {claim_id: str(row["claim_label"]) for claim_id, row in dev_claims_for_gold.items()}

    predictions: dict[str, dict[str, dict[str, str]]] = {
        "Evidence-Aware Logistic": load_prediction_file(
            REPO_ROOT / "round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_dev_predictions.json",
            gold_lookup,
        ),
        "Top5 Shallow Candidate": load_prediction_file(
            REPO_ROOT / "round18/outputs/o_classifier/o_c5_comparison/worker_g_final_candidate/worker_g_final_candidate_candidate_dev_predictions.json",
            gold_lookup,
        ),
    }
    predictions["Global Residual Diagnostic"] = train_residual_predictions(
        args,
        ResidualSpec(
            name="head5_mf60000_c2p0_tail0p05",
            max_features=60000,
            c_value=2.0,
            tail_alpha=0.05,
        ),
    )
    predictions["Top5-Gated Residual Diagnostic"] = train_gated_predictions(
        args,
        GateSpec(
            name="gated_head5_mf30000_c2p0_l20p0_uncertainty",
            max_features=30000,
            c_value=2.0,
            gate_l2=0.0,
            gate_feature_mode="uncertainty",
        ),
    )
    predictions["Top5-Gated Train-Selected"] = train_gated_predictions(
        args,
        GateSpec(
            name="gated_head5_mf60000_c2p0_l20p0_uncertainty_plus_shallow",
            max_features=60000,
            c_value=2.0,
            gate_l2=0.0,
            gate_feature_mode="uncertainty_plus_shallow",
        ),
    )

    claim_ids, pred_vectors, gold = vectors(predictions)
    point_scores = {name: score(gold, pred) for name, pred in pred_vectors.items()}
    rng = np.random.default_rng(args.random_seed)
    boot = bootstrap_scores(gold, pred_vectors, rng, args.bootstrap_samples)

    baseline = "Top5 Shallow Candidate"
    deltas = {}
    delta_summaries = {}
    for name in pred_vectors:
        if name == baseline:
            continue
        values = boot[name]["macro_f1"] - boot[baseline]["macro_f1"]
        label = f"{name} minus {baseline}"
        deltas[label] = values
        delta_summaries[label] = {
            **summarize_distribution(values),
            "p_delta_gt_0": float(np.mean(values > 0.0)),
        }

    summaries = {
        name: {
            "point": point_scores[name],
            "bootstrap_macro_f1": summarize_distribution(boot[name]["macro_f1"]),
            "bootstrap_accuracy": summarize_distribution(boot[name]["accuracy"]),
        }
        for name in pred_vectors
    }

    plot_distributions(args.fig_path, boot, point_scores, deltas)
    payload = {
        "run_id": "score_distribution_audit",
        "claim_count": int(len(claim_ids)),
        "bootstrap_samples": int(args.bootstrap_samples),
        "point_scores": point_scores,
        "summaries": summaries,
        "paired_delta_vs_top5_shallow": delta_summaries,
        "figure": str(args.fig_path),
        "manifest": manifest_base(
            run_id="score_distribution_audit",
            status="diagnostic_distribution_audit",
            mode="DIAGNOSTIC",
            stage="score_distribution_audit",
            command=args.command,
            working_directory=Path.cwd(),
            random_seed=args.random_seed,
        ),
    }
    result_path = args.output_root / "score_distribution_audit_results.json"
    write_json(result_path, payload)

    lines = [
        "# Score Distribution Audit",
        "",
        "Method: paired bootstrap on the fixed dev set. This estimates sampling noise across dev claims, not a full train-randomness distribution.",
        "",
        "## Point Scores And Bootstrap Intervals",
        "",
        "| model | point macro-F1 | bootstrap mean | 95% interval | point accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summaries.items():
        ci = row["bootstrap_macro_f1"]
        lines.append(
            f"| {name} | {row['point']['macro_f1']:.6f} | {ci['mean']:.6f} | [{ci['p2_5']:.6f}, {ci['p97_5']:.6f}] | {row['point']['accuracy']:.6f} |"
        )
    lines.extend(["", "## Paired Delta Versus Top5 Shallow Candidate", ""])
    lines.extend(["| model delta | mean delta | 95% interval | P(delta > 0) |", "| --- | ---: | ---: | ---: |"])
    for name, row in delta_summaries.items():
        lines.append(
            f"| {name} | {row['mean']:.6f} | [{row['p2_5']:.6f}, {row['p97_5']:.6f}] | {row['p_delta_gt_0']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If a paired-delta interval crosses zero, the apparent dev improvement is weak evidence and may be sampling noise.",
            "- This audit does not rescue dev-selected methods; train-only selection stability remains required for promotion.",
            "",
            f"Figure: `{args.fig_path}`",
            f"Results: `{result_path}`",
        ]
    )
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(result_path), "report": str(args.report_path), "figure": str(args.fig_path)}, indent=2))


if __name__ == "__main__":
    main()
