#!/usr/bin/env python3
"""Generate ACL-ready figures for the report from current notebook/meeting artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = ROOT / "submissions" / "report" / "latex" / "figures"
SCRIPT_DIR = ROOT / "submissions" / "report" / "latex" / "scripts"
THIRD_MEETING_DIR = ROOT / "docs" / "group_meetings" / "third_meeting"
REPO = ROOT / "docs" / "research" / "round18" / "reports" / "tutorial_rn_curves"
TUTORIAL_CURVE_CSV = REPO / "tutorial_rn_curves_summary.csv"

PALETTE = [
    "#4e79a7",
    "#f28e2c",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#9c755f",
    "#76b7b2",
    "#edc948",
    "#bab0ac",
]

FINAL_COLAB_CLASSIFIER = {
    "accuracy": 0.5000,
    "macro_f1": 0.4659220505617978,
    "confusion_matrix": [
        [38, 10, 18, 2],
        [4, 13, 9, 1],
        [10, 8, 20, 3],
        [5, 6, 1, 6],
    ],
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.5,
            "lines.markersize": 6.0,
            "figure.dpi": 300,
        }
    )


def _figsize(width_px: int, height_px: int) -> Tuple[float, float]:
    return width_px / 300.0, height_px / 300.0


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _crop_to_content(img: Image.Image, threshold: int = 245, margin: int = 2) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    dark = (arr[:, :, 0] < threshold) | (arr[:, :, 1] < threshold) | (arr[:, :, 2] < threshold)
    rows = np.where(dark.mean(axis=1) > 0.002)[0]
    cols = np.where(dark.mean(axis=0) > 0.002)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(arr.shape[1] - 1, x1 + margin)
    y1 = min(arr.shape[0] - 1, y1 + margin)
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def _fit_ratio_by_crop(img: Image.Image, min_ratio: float, max_ratio: float) -> Image.Image:
    width, height = img.size
    ratio = width / height
    if ratio > max_ratio:
        target_w = int(round(height * max_ratio))
        excess = width - target_w
        if excess > 0:
            left = excess // 2
            right = excess - left
            img = img.crop((left, 0, width - right, height))
        return img
    if ratio < min_ratio:
        target_h = int(round(width / min_ratio))
        excess = height - target_h
        if excess > 0:
            top = excess // 2
            bottom = excess - top
            img = img.crop((0, top, width, height - bottom))
        return img
    return img


def _extract_claim_counts(path: Path) -> Dict[str, int]:
    claims = _load_json(path)
    return dict(Counter(p["claim_label"] for p in claims.values()))


def _load_retrieval_curves() -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    p = THIRD_MEETING_DIR / "retrieval_curves_from_artifacts.json"
    payload = _load_json(p)
    return payload["curves"]


def _load_top500_summary() -> Dict[str, float]:
    by_name: Dict[str, float] = {}
    with open(TUTORIAL_CURVE_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(float(row["top_n"])) == 500:
                by_name[row["method_id"]] = float(row["macro_recall"])
    return by_name


def _format_pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def create_dataset_cost_overview(output: Path) -> None:
    third = _load_json(THIRD_MEETING_DIR / "third_tutorial_metrics.json")
    train_counts = _extract_claim_counts(ROOT / "data" / "train-claims.json")
    dev_counts = _extract_claim_counts(ROOT / "data" / "dev-claims.json")

    class_order = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    train_vals = [train_counts.get(x, 0) for x in class_order]
    dev_vals = [dev_counts.get(x, 0) for x in class_order]
    totals = [sum(train_vals), sum(dev_vals)]

    curves = _load_retrieval_curves()
    cands = curves["candidate_final_sparse_fusion"]["points"]
    s64 = curves["top64_embedding_plus_shallow"]["points"]
    top3 = curves["top3_ce_prefilter_fusion"]["points"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=_figsize(2760, 1060), dpi=300, sharey=False)
    set_style()

    x = np.arange(len(class_order))
    barw = 0.35
    ax_left.bar(x - barw / 2, train_vals, barw, color=PALETTE[0], label="Train", edgecolor="white", linewidth=0.5)
    ax_left.bar(x + barw / 2, dev_vals, barw, color=PALETTE[2], label="Dev", edgecolor="white", linewidth=0.5)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(["SUPPORTS", "REFUTES", "NEI", "DISPUTED"], rotation=12, ha="right")
    ax_left.set_title(f"Claim-label distribution (train={totals[0]}, dev={totals[1]})")
    ax_left.set_ylabel("Claim count")
    ax_left.grid(axis="y", alpha=0.2)
    ax_left.legend(frameon=False, loc="upper right", ncols=2)

    for idx, (tv, dv) in enumerate(zip(train_vals, dev_vals)):
        ax_left.text(idx - barw / 2, tv + max(train_vals) * 0.01, f"{tv}", ha="center", va="bottom", fontsize=8)
        ax_left.text(idx + barw / 2, dv + max(dev_vals) * 0.01, f"{dv}", ha="center", va="bottom", fontsize=8)

    ax_right.set_title("Retrieval-cost motivation")
    for pts, lab, col in [
        (cands, "Sparse fusion", PALETTE[0]),
        (s64, "Top64 + shallow", PALETTE[2]),
        (top3, "Top3 fusion", PALETTE[3]),
    ]:
        ks = [p["k"] for p in pts]
        recalls = [p["macro_recall"] for p in pts]
        ax_right.plot(ks, recalls, marker="o", color=col, label=lab, markerfacecolor="white")
    ax_right.set_xscale("log")
    ax_right.set_xticks([1, 3, 5, 10, 20, 32, 64, 100, 500])
    ax_right.set_xticklabels(["1", "3", "5", "10", "20", "32", "64", "100", "500"])
    ax_right.set_xlabel("Candidate budget k")
    ax_right.set_ylabel("Macro recall")
    ax_right.set_ylim(0, 0.8)
    ax_right.grid(True, alpha=0.25)
    ax_right.legend(loc="lower right", frameon=False)

    # Highlight operating points used by the system.
    ax_right.scatter([500], [third["summary"]["candidate"]["macro_recall@500"]], s=45, marker="D", color=PALETTE[4])
    ax_right.scatter([64], [third["summary"]["top64"]["macro_recall@64"]], s=45, marker="D", color=PALETTE[5])
    ax_right.scatter([3], [third["summary"]["top3"]["macro_recall@3"]], s=45, marker="D", color=PALETTE[6])
    ax_right.text(
        0.03,
        0.94,
        "Broad sparse recall first;\nneural scoring only after pruning",
        ha="left",
        va="top",
        transform=ax_right.transAxes,
        fontsize=8.2,
    )

    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_sparse_multigate(output: Path) -> None:
    third = _load_json(THIRD_MEETING_DIR / "third_tutorial_metrics.json")
    ablation = _load_json(ROOT / "docs" / "research" / "round18" / "reports" / "colab_sklearn_candidate_experiment" / "summary.json")
    sparse_curves = third["sparse_curves"]

    curve_specs = [
        ("BM25 word gate", "Word", PALETTE[0]),
        ("Character n-gram gate", "Char", PALETTE[1]),
        ("Structured cue gate", "Structured", PALETTE[2]),
        ("Query expansion gate", "PRF", PALETTE[4]),
        ("Fixed sparse score-fusion gate", "Fusion", PALETTE[3]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=_figsize(3000, 1120), dpi=300)
    set_style()

    axw, axc, axb = axes

    ablation_metrics = ablation["metrics"]
    ablation_names = [
        "Structured\nalone",
        "No\nstructured",
        "+Structured\nsmall weight",
        "Final\nweighted fusion",
    ]
    ablation_vals = np.array(
        [
            ablation_metrics["structured"]["macro_recall@500"],
            ablation_metrics["fused_no_structured"]["macro_recall@500"],
            ablation_metrics["fused_struct_025"]["macro_recall@500"],
            ablation_metrics["fused_char_200_struct_025_prf_050"]["macro_recall@500"],
        ],
        dtype=float,
    )
    y = np.arange(len(ablation_names))
    axw.barh(y, ablation_vals, color=[PALETTE[2], PALETTE[0], PALETTE[4], PALETTE[3]], edgecolor="white", linewidth=0.5)
    axw.set_title("Low-recall structured still helps fusion")
    axw.set_xlabel("Top-500 macro recall")
    axw.set_yticks(y)
    axw.set_yticklabels(ablation_names)
    axw.set_xlim(0, 0.72)
    axw.invert_yaxis()
    axw.grid(axis="x", alpha=0.22)
    for yi, v in zip(y, ablation_vals):
        axw.text(v + 0.012, yi, _format_pct(v), ha="left", va="center", fontsize=8.1)

    for key, label, col in curve_specs:
        pts = sparse_curves[key]
        ks = [p["k"] for p in pts]
        recalls = [p["macro_recall"] for p in pts]
        axc.plot(ks, recalls, marker="o", color=col, label=label, markerfacecolor="white")
    axc.set_title("Sparse recall curves")
    axc.set_xlabel("Top-k")
    axc.set_ylabel("Macro recall")
    axc.set_xscale("log")
    axc.set_xticks([1, 3, 5, 10, 20, 32, 64, 100, 500])
    axc.set_ylim(0, 0.8)
    axc.legend(frameon=False, loc="lower right", fontsize=7.6)

    tnames = [label for _, label, _ in curve_specs]
    tvals = []
    tcols = []
    for key, _, col in curve_specs:
        top500 = next(p["macro_recall"] for p in sparse_curves[key] if p["k"] == 500)
        tvals.append(top500)
        tcols.append(col)
    x = np.arange(len(tvals))
    axb.bar(x, tvals, color=tcols, edgecolor="white", linewidth=0.5)
    axb.set_title("Top-500 recall by sparse source")
    axb.set_xticks(x)
    axb.set_xticklabels(tnames, rotation=20, ha="right")
    axb.set_ylim(0, max(tvals) * 1.12 if len(tvals) else 1.0)
    axb.set_ylabel("Macro recall")
    for xi, v in zip(x, tvals):
        axb.text(xi, v + max(tvals) * 0.02, _format_pct(v), ha="center", va="bottom", fontsize=8)
    # Emphasize selected variants in this run.
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _copy_from_source_with_crop(src: Path, out: Path, min_ratio: float, max_ratio: float) -> None:
    img = Image.open(src).convert("RGB")
    img = _crop_to_content(img)
    img = _fit_ratio_by_crop(img, min_ratio, max_ratio)
    img.save(out, dpi=(300, 300))


def create_staging_cost_benefit(output: Path) -> None:
    third = _load_json(THIRD_MEETING_DIR / "third_tutorial_metrics.json")["summary"]
    n_claims = len(_load_json(ROOT / "data" / "dev-claims.json"))
    n_evidence = len(_load_json(ROOT / "data" / "evidence.json"))

    stages = [
        ("Candidate\npool", n_claims * 500, third["candidate"]["macro_recall@500"], PALETTE[0]),
        ("Reranked\ncontext", n_claims * 64, third["top64"]["macro_recall@64"], PALETTE[2]),
        ("Submitted\nevidence", n_claims * 3, third["top3"]["macro_recall@3"], PALETTE[3]),
        ("Full evidence\nscoring", n_claims * n_evidence, 0.0, PALETTE[8]),
    ]

    fig, ax = plt.subplots(1, 1, figsize=_figsize(1660, 920), dpi=300)
    set_style()
    for label, cost, recall, color in stages:
        ax.scatter(cost, recall, s=70, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        dy = 0.035 if recall > 0.1 else 0.055
        va = "bottom" if recall > 0.1 else "bottom"
        ax.text(cost, recall + dy, label, ha="center", va=va, fontsize=8.0)
    ax.plot([s[1] for s in stages[:3]], [s[2] for s in stages[:3]], color=PALETTE[0], alpha=0.45, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Claim-evidence pairs retained or scored (log scale)")
    ax.set_ylabel("Macro recall")
    ax.set_title("Cost-benefit view of staged retrieval")
    ax.set_ylim(0, 0.76)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.05,
        "The pipeline spends expensive scoring only on progressively smaller pools.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_feature_fusion(output: Path) -> None:
    hand_metrics = _load_json(
        ROOT
        / "docs"
        / "research"
        / "round18"
        / "outputs"
        / "o_sparse"
        / "o_s10_wide_hand_feature_bm25_char_rrf_top1000"
        / "dev_full_dev_o_s10_wide_hand_feature_bm25_char_rrf_top1000_metrics.json"
    )
    family_map: Dict[str, float] = {}
    for item in hand_metrics.get("feature_importance", []):
        feature = item["feature"]
        if feature.startswith("claim_key"):
            family = "Claim-key coverage"
        elif feature.startswith("char") or feature.startswith("content") or "gram" in feature:
            family = "Lexical overlap"
        elif "rank" in feature or "score" in feature:
            family = "Source rank / score"
        elif "entity" in feature:
            family = "Entity match"
        elif "number" in feature or "year" in feature:
            family = "Number / year"
        elif "length" in feature or feature.endswith("_len"):
            family = "Length / shape"
        else:
            family = "Logic cues"
        family_map[family] = family_map.get(family, 0.0) + float(item.get("importance", 0.0))
    ordered = sorted(family_map.items(), key=lambda kv: kv[1], reverse=True)

    emb_summary = _load_json(
        ROOT / "docs" / "research" / "round18" / "reports" / "embedding_shallow_feature_kfold" / "embedding_shallow_feature_kfold_summary.json"
    )
    emb_rows = {row["variant"]: row for row in emb_summary["rows"]}
    curves = _load_retrieval_curves()
    direct_curve = [p for p in curves["candidate_final_sparse_fusion"]["points"] if p["k"] in [1, 3, 5, 10, 64]]

    def variant_curve(row, ks=(1, 3, 5, 10, 64)):
        return [{"k": k, "macro_recall": row[f"dev_macro_recall@{k}"]} for k in ks]

    shallow_curve = variant_curve(emb_rows["shallow_only"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=_figsize(2520, 900), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.1]})
    set_style()
    names = [name for name, _ in ordered][::-1]
    vals = [val for _, val in ordered][::-1]
    ax0.barh(names, vals, color=PALETTE[4], edgecolor="white", linewidth=0.5)
    ax0.set_xlabel("Total feature importance")
    ax0.set_title("Shallow feature families")
    ax0.tick_params(axis="y", labelsize=8.2)

    for curve, label, color, marker in [
        (direct_curve, "Sparse candidate order", PALETTE[6], "o"),
        (shallow_curve, "Shallow-feature selector", PALETTE[4], "s"),
    ]:
        ax1.plot(
            [x["k"] for x in curve],
            [x["macro_recall"] for x in curve],
            marker=marker,
            color=color,
            label=label,
            markerfacecolor="white",
        )
    ax1.set_xscale("log")
    ax1.set_xticks([1, 3, 5, 10, 64])
    ax1.set_xticklabels(["1", "3", "5", "10", "64"])
    ax1.set_ylim(0.05, 0.52)
    ax1.set_xlabel("Evidence kept")
    ax1.set_ylabel("Macro recall")
    ax1.set_title("Shallow features improve ordering")
    ax1.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.6, w_pad=2.0)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_top64_diagnostic(output: Path) -> None:
    third = _load_json(THIRD_MEETING_DIR / "third_tutorial_metrics.json")
    selector = third["top64_selector_comparison"]
    labels = ["Sparse cutoff top64", "Cross-encoder + factual cues", "Embedding + factual cues"]
    values = [selector[l] for l in labels]

    fig, ax = plt.subplots(1, 1, figsize=_figsize(1880, 1080), dpi=300)
    set_style()
    xpos = np.arange(len(labels))
    colors = [PALETTE[0], PALETTE[2], PALETTE[4]]
    bars = ax.bar(xpos, values, color=colors, width=0.62, edgecolor="white", linewidth=0.5)
    ax.set_title("Top-64 selector diagnostic")
    ax.set_ylabel("Macro recall @64")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Sparse cutoff", "CE + cues", "Emb. + cues"], rotation=12)
    ax.set_ylim(0, max(values) * 1.2)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + max(values) * 0.02, _format_pct(h), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_shallow_complement(output: Path) -> None:
    _copy_from_source_with_crop(
        ROOT / "docs" / "group_meetings" / "second_meeting" / "figures" / "round18_revised_shallow_complement_two_panel_rn.png",
        output,
        2.6,
        3.0,
    )


def create_top3_evidence_fusion(output: Path) -> None:
    third = _load_json(THIRD_MEETING_DIR / "retrieval_figure_metrics.json")
    metrics = _load_json(THIRD_MEETING_DIR / "third_tutorial_metrics.json")["summary"]

    weight_params = third["top3"]["method_parameters"]
    weight_params = weight_params.get("fusion_weights", weight_params)
    gate_weights = {
        "CE score": weight_params.get("ce_score", 0.0),
        "CE rank": weight_params.get("ce_rank", 0.0),
        "Embedding score": weight_params.get("embedding_score", 0.0),
        "Embedding rank": weight_params.get("embedding_rank", 0.0),
        "Source rank": weight_params.get("source_rank", 0.0),
    }
    # Drop zero-weight fields for compactness.
    gate_labels = [k for k, v in gate_weights.items() if v > 0]
    gate_values = [v for v in gate_weights.values() if v > 0]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=_figsize(3000, 1100), dpi=300)
    set_style()

    xpos = np.arange(len(gate_labels))
    ax_left.bar(xpos, gate_values, color=PALETTE[3], edgecolor="white", linewidth=0.5)
    ax_left.set_title("Top-3 fusion gate weights")
    ax_left.set_xticks(xpos)
    ax_left.set_xticklabels(gate_labels, rotation=20, ha="right")
    ax_left.set_ylabel("Weight")
    ax_left.set_ylim(0, 0.3)
    for x, v in zip(xpos, gate_values):
        ax_left.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.2)

    stage_names = ["Candidate", "Top-64", "Top-3"]
    recall_vals = [
        metrics["candidate"]["macro_recall@500"],
        metrics["top64"]["macro_recall@64"],
        metrics["top3"]["macro_recall@3"],
    ]
    hit_vals = [
        metrics["candidate"]["hit_any@500"],
        metrics["top64"]["hit_any@64"],
        metrics["top3"]["hit_any@3"],
    ]
    x = np.arange(len(stage_names))
    ax_right.plot(x, recall_vals, marker="o", color=PALETTE[0], label="Macro recall", linewidth=1.6, markersize=6.5)
    ax_right.plot(x, hit_vals, marker="s", color=PALETTE[2], label="Hit-any", linewidth=1.4, markersize=6)
    ax_right.set_title("Final submission comparison")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(stage_names)
    ax_right.set_ylim(0, 1.0)
    ax_right.set_ylabel("Rate")
    ax_right.legend(frameon=False, loc="best", ncols=1)
    for i, (r, h) in enumerate(zip(recall_vals, hit_vals)):
        ax_right.text(i, r + 0.02, _format_pct(r), ha="center", va="bottom", fontsize=8.0, color=PALETTE[0])
        ax_right.text(i, h - 0.08, _format_pct(h), ha="center", va="top", fontsize=8.0, color=PALETTE[2])

    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _class_report_from_matrix(matrix: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cm = np.asarray(matrix, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1


def create_classifier_summary(output: Path) -> None:
    cls = FINAL_COLAB_CLASSIFIER
    labels = ["SUPPORTS", "REFUTES", "NEI", "DISPUTED"]
    conf = np.asarray(cls["confusion_matrix"], dtype=float)
    precision, recall, f1 = _class_report_from_matrix(conf)
    per_class = {
        l: {"precision": float(p), "recall": float(r), "f1": float(fi)}
        for l, p, r, fi in zip(labels, precision, recall, f1)
    }

    fig, (ax_cm, ax_bars) = plt.subplots(1, 2, figsize=_figsize(3200, 1100), dpi=300)
    set_style()

    im = ax_cm.imshow(conf, cmap="Blues", aspect="auto")
    ax_cm.set_title("Confusion matrix")
    ax_cm.set_xticks(np.arange(len(labels)))
    ax_cm.set_yticks(np.arange(len(labels)))
    ax_cm.set_xticklabels(labels, rotation=20, ha="right")
    ax_cm.set_yticklabels(labels)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            color = "white" if conf[i, j] > conf.max() * 0.65 else "black"
            ax_cm.text(j, i, f"{int(conf[i, j])}", ha="center", va="center", color=color, fontsize=8.4)
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Count")

    metrics = [("Precision", [per_class[l]["precision"] for l in labels], PALETTE[0]),
               ("Recall", [per_class[l]["recall"] for l in labels], PALETTE[2]),
               ("F1", [per_class[l]["f1"] for l in labels], PALETTE[3])]
    x = np.arange(len(labels))
    width = 0.22
    for idx, (name, vals, color) in enumerate(metrics):
        ax_bars.bar(x + idx * width, vals, width, label=name, color=color, edgecolor="white", linewidth=0.5)
    ax_bars.set_xticks(x + width)
    ax_bars.set_xticklabels(labels, rotation=12, ha="right")
    ax_bars.set_title("Per-class metrics")
    ax_bars.set_ylim(0, 1)
    ax_bars.set_ylabel("Score")
    ax_bars.legend(frameon=False, loc="upper right")

    ax_bars.text(
        0.03,
        0.93,
        f"Accuracy = {cls['accuracy']:.3f}\nMacro F1 = {cls['macro_f1']:.3f}",
        transform=ax_bars.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_classifier_sensitivity(output: Path) -> None:
    rows = _load_json(THIRD_MEETING_DIR / "third_tutorial_metrics.json")["classifier_sensitivity"]
    by_exp: Dict[str, List[Dict[str, float]]] = {"C_sensitivity": [], "context_depth_sensitivity": []}
    for row in rows:
        by_exp[row["experiment"]].append(row)
    for key in by_exp:
        by_exp[key] = sorted(by_exp[key], key=lambda x: x["param_value"])

    fig, ax = plt.subplots(1, 1, figsize=_figsize(2520, 960), dpi=300)
    set_style()
    if by_exp["C_sensitivity"]:
        x = [r["param_value"] for r in by_exp["C_sensitivity"]]
        y = [r["macro_f1"] for r in by_exp["C_sensitivity"]]
        ax.plot(x, y, marker="o", color=PALETTE[0], label="Regularization C", linewidth=1.6, markersize=6)
    if by_exp["context_depth_sensitivity"]:
        x = [r["param_value"] for r in by_exp["context_depth_sensitivity"]]
        y = [r["macro_f1"] for r in by_exp["context_depth_sensitivity"]]
        ax.plot(x, y, marker="s", color=PALETTE[2], label="Evidence budget", linewidth=1.6, markersize=6)

    ax.set_title("Classifier sensitivity")
    ax.set_xlabel("Hyper-parameter")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.36, 0.52)
    ax.set_xticks([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 5, 10, 20, 32, 64])
    ax.set_xticklabels(["0.03125", "0.0625", "0.125", "0.25", "0.5", "1", "", "10", "", "", "64"])
    ax.set_xlim(0.02, 70)
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    create_dataset_cost_overview(FIG_DIR / "acl_dataset_cost_overview.png")
    create_sparse_multigate(FIG_DIR / "acl_sparse_multigate.png")
    create_staging_cost_benefit(FIG_DIR / "acl_staging_cost_benefit.png")
    create_feature_fusion(FIG_DIR / "acl_feature_fusion.png")
    create_top64_diagnostic(FIG_DIR / "acl_top64_diagnostic.png")
    create_shallow_complement(FIG_DIR / "acl_shallow_complement.png")
    create_top3_evidence_fusion(FIG_DIR / "acl_top3_evidence_fusion.png")
    create_classifier_summary(FIG_DIR / "acl_classifier_summary.png")
    create_classifier_sensitivity(FIG_DIR / "acl_classifier_sensitivity.png")


if __name__ == "__main__":
    main()
