#!/usr/bin/env python
"""Visual audit for the n/N residual grid.

This script reads the grid search output produced by worker A, renders a
summary figure, and writes a compact markdown audit. The figure separates
train-selected results from diagnostic dev-best results so the promotion
boundary stays visible.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _resolve_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "round18" / "tools").exists():
            return candidate
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _resolve_repo_root()
DEFAULT_INPUT_JSON = REPO_ROOT / "round18/outputs/o_classifier/nN_residual_grid/nN_residual_grid_results.json"
DEFAULT_FIG_PATH = REPO_ROOT / "round18/reports/figures/nN_residual_grid_summary.png"
DEFAULT_MD_PATH = REPO_ROOT / "round18/reports/nN_residual_grid_visual_summary.md"


@dataclass(frozen=True)
class GridRow:
    n: int
    N: int
    train_selected_dev_macro_f1: float
    selected_tail_alpha: float | None
    diagnostic_dev_best_macro_f1: float | None
    diagnostic_status: str | None
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the n/N residual grid results.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--fig-path", type=Path, default=DEFAULT_FIG_PATH)
    parser.add_argument("--md-path", type=Path, default=DEFAULT_MD_PATH)
    return parser.parse_args()


def _fail_missing_input(path: Path) -> None:
    raise SystemExit(
        "Missing input JSON: "
        f"{path}\nExpected worker A output at "
        "round18/outputs/o_classifier/nN_residual_grid/nN_residual_grid_results.json."
    )


def _load_json(path: Path) -> Any:
    if not path.exists():
        _fail_missing_input(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc


def _lookup_any(node: Any, *names: str) -> Any:
    """Breadth-first lookup for a field name anywhere in a nested record."""

    queue: list[Any] = [node]
    seen: set[int] = set()
    while queue:
        cur = queue.pop(0)
        marker = id(cur)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(cur, dict):
            for name in names:
                if name in cur:
                    value = cur[name]
                    if value is not None and value != "":
                        return value
            for value in cur.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(cur, list):
            for value in cur:
                if isinstance(value, (dict, list)):
                    queue.append(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?\d+\.0+", text):
            return int(float(text))
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100.0
            except ValueError:
                return None
        try:
            out = float(text)
        except ValueError:
            return None
        if math.isfinite(out):
            return out
    return None


def _format_alpha(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) < 1e-12:
        return "0"
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _format_metric(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _row_score(row: dict[str, Any]) -> int:
    score = 0
    if _coerce_int(_lookup_any(row, "n", "n_value", "grid_n", "lower_n", "head_n")) is not None:
        score += 2
    if _coerce_int(_lookup_any(row, "N", "N_value", "grid_N", "upper_N", "context_N")) is not None:
        score += 2
    if _coerce_float(
        _lookup_any(
            row,
            "train_selected_dev_macro_f1",
            "selected_train_dev_macro_f1",
            "strict_train_selected_dev_macro_f1",
            "strict_selected_dev_macro_f1",
            "selected_dev_macro_f1",
            "dev_macro_f1",
            "macro_f1",
        )
    ) is not None:
        score += 2
    if _coerce_float(_lookup_any(row, "selected_tail_alpha", "tail_alpha", "alpha")) is not None:
        score += 1
    if _coerce_float(
        _lookup_any(
            row,
            "diagnostic_dev_best_macro_f1",
            "diagnostic_macro_f1",
            "dev_best_macro_f1",
            "diagnostic_dev_macro_f1",
        )
    ) is not None:
        score += 1
    return score


def _collect_row_candidates(obj: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if _row_score(node) >= 3:
                candidates.append(node)
            for child in node.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(node, list):
            if node and all(isinstance(item, dict) for item in node):
                if sum(_row_score(item) for item in node) >= 3:
                    candidates.extend(item for item in node if _row_score(item) >= 3)
            for child in node:
                if isinstance(child, (dict, list)):
                    walk(child)

    walk(obj)
    return candidates


def _preferred_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = [item for item in payload if isinstance(item, dict) and _row_score(item) >= 3]
        if rows:
            return rows

    if isinstance(payload, dict):
        for key in (
            "grid_results",
            "results",
            "rows",
            "records",
            "items",
            "evaluations",
            "grid",
            "data",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                rows = [item for item in value if isinstance(item, dict) and _row_score(item) >= 3]
                if rows:
                    return rows
            elif isinstance(value, dict):
                rows = [item for item in value.values() if isinstance(item, dict) and _row_score(item) >= 3]
                if rows:
                    return rows

    candidates = _collect_row_candidates(payload)
    if candidates:
        # Deduplicate by identity while preserving order.
        unique: list[dict[str, Any]] = []
        seen: set[int] = set()
        for row in sorted(candidates, key=_row_score, reverse=True):
            marker = id(row)
            if marker not in seen:
                unique.append(row)
                seen.add(marker)
        return unique
    return []


def _normalize_rows(payload: Any) -> list[GridRow]:
    rows: list[GridRow] = []
    for raw in _preferred_rows(payload):
        n = _coerce_int(_lookup_any(raw, "n", "n_value", "grid_n", "lower_n", "head_n"))
        N = _coerce_int(_lookup_any(raw, "N", "N_value", "grid_N", "upper_N", "context_N"))
        train_selected = _coerce_float(
            _lookup_any(
                raw,
                "train_selected_dev_macro_f1",
                "selected_train_dev_macro_f1",
                "strict_train_selected_dev_macro_f1",
                "strict_selected_dev_macro_f1",
                "selected_dev_macro_f1",
                "dev_macro_f1",
                "macro_f1",
            )
        )
        selected_tail_alpha = _coerce_float(_lookup_any(raw, "selected_tail_alpha", "tail_alpha", "alpha"))
        diagnostic = _coerce_float(
            _lookup_any(
                raw,
                "diagnostic_dev_best_macro_f1",
                "diagnostic_macro_f1",
                "dev_best_macro_f1",
                "diagnostic_dev_macro_f1",
            )
        )
        status = _lookup_any(raw, "status", "diagnostic_status", "selection_status", "mode")
        if isinstance(status, dict):
            status = None
        if isinstance(status, str):
            status = status.strip()
        else:
            status = None
        if n is None or N is None or train_selected is None:
            continue
        rows.append(
            GridRow(
                n=n,
                N=N,
                train_selected_dev_macro_f1=train_selected,
                selected_tail_alpha=selected_tail_alpha,
                diagnostic_dev_best_macro_f1=diagnostic,
                diagnostic_status=status,
                raw=raw,
            )
        )

    if not rows:
        raise SystemExit(
            "No usable grid rows were found in the input JSON. "
            "Expected entries with n, N, and train-selected dev macro-F1 fields."
        )
    rows.sort(key=lambda row: (row.n, row.N))
    return rows


def _unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def _grid_matrix(rows: list[GridRow], getter: str) -> tuple[np.ndarray, list[int], list[int]]:
    ns = _unique_sorted(row.n for row in rows)
    Ns = _unique_sorted(row.N for row in rows)
    matrix = np.full((len(ns), len(Ns)), np.nan, dtype=float)
    n_to_idx = {n: i for i, n in enumerate(ns)}
    N_to_idx = {N: i for i, N in enumerate(Ns)}
    for row in rows:
        value = getattr(row, getter)
        if value is None:
            continue
        matrix[n_to_idx[row.n], N_to_idx[row.N]] = float(value)
    return matrix, ns, Ns


def _matrix_has_any_values(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(matrix).any())


def _annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    values_for_text: np.ndarray | None = None,
    fmt: str = "{:.3f}",
    text_color_threshold: float | None = None,
    missing_text: str = "—",
    zero_highlight: bool = False,
) -> None:
    if values_for_text is None:
        values_for_text = matrix
    finite = matrix[np.isfinite(matrix)]
    threshold = text_color_threshold
    if threshold is None and finite.size:
        threshold = float(np.nanmin(finite) + 0.65 * (np.nanmax(finite) - np.nanmin(finite)))
    if threshold is None:
        threshold = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.text(j, i, missing_text, ha="center", va="center", fontsize=9, color="#6B7280")
                continue
            display_value = values_for_text[i, j]
            if zero_highlight and abs(display_value) < 1e-12:
                text = "0"
            else:
                text = fmt.format(display_value)
            color = "white" if value >= threshold else "#111827"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color, fontweight="semibold")


def _style_heatmap_axes(ax: plt.Axes, ns: list[int], Ns: list[int], y_label: str = "n") -> None:
    ax.set_xticks(np.arange(len(Ns)))
    ax.set_yticks(np.arange(len(ns)))
    ax.set_xticklabels([str(value) for value in Ns], fontsize=9)
    ax.set_yticklabels([str(value) for value in ns], fontsize=9)
    ax.set_xlabel("N")
    ax.set_ylabel(y_label)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(Ns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ns), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.set_xlim(-0.5, len(Ns) - 0.5)
    ax.set_ylim(-0.5, len(ns) - 0.5)


def _plot_metric_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    ns: list[int],
    Ns: list[int],
    title: str,
    cbar_label: str,
    cmap_name: str,
    annotate_fmt: str = "{:.3f}",
    note: str | None = None,
) -> None:
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#E5E7EB")
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    _style_heatmap_axes(ax, ns, Ns)
    ax.set_title(title, fontsize=12, pad=10)
    _annotate_heatmap(ax, matrix, fmt=annotate_fmt)
    if note:
        ax.text(
            0.5,
            -0.18,
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#4B5563",
        )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def _plot_tail_alpha_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    ns: list[int],
    Ns: list[int],
) -> None:
    positive = np.ma.masked_where(~(np.isfinite(matrix) & (matrix > 0.0)), matrix)
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_bad("#E5E7EB")
    finite_positive = matrix[np.isfinite(matrix) & (matrix > 0.0)]
    if finite_positive.size:
        vmax = float(np.nanmax(finite_positive))
        if vmax <= 0.0:
            vmax = 1.0
    else:
        vmax = 1.0
    im = ax.imshow(positive, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    _style_heatmap_axes(ax, ns, Ns)
    ax.set_title("Selected tail_alpha by (n, N)", fontsize=12, pad=10)
    _annotate_heatmap(ax, matrix, fmt="{:.2f}", zero_highlight=True)
    ax.text(
        0.5,
        -0.18,
        "gray = tail_alpha = 0; orange scale = tail_alpha > 0",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4B5563",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("tail_alpha (positive values)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def _plot_summary_figure(rows: list[GridRow], fig_path: Path) -> bool:
    strict_matrix, ns, Ns = _grid_matrix(rows, "train_selected_dev_macro_f1")
    tail_matrix, _, _ = _grid_matrix(rows, "selected_tail_alpha")
    diag_matrix, _, _ = _grid_matrix(rows, "diagnostic_dev_best_macro_f1")
    has_diag = _matrix_has_any_values(diag_matrix)

    if has_diag:
        fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.08, 1.0, 1.08]})
        axes_list = list(axes)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1.0]})
        axes_list = list(axes)

    _plot_metric_heatmap(
        axes_list[0],
        strict_matrix,
        ns,
        Ns,
        "Dev macro-F1 after train-only alpha selection",
        "macro-F1",
        "viridis",
        annotate_fmt="{:.3f}",
        note="inside each cell, alpha is chosen on train holdout",
    )
    _plot_tail_alpha_heatmap(axes_list[1], tail_matrix, ns, Ns)
    if has_diag:
        _plot_metric_heatmap(
            axes_list[2],
            diag_matrix,
            ns,
            Ns,
            "Diagnostic dev-best macro-F1 by (n, N)",
            "macro-F1",
            "magma",
            annotate_fmt="{:.3f}",
            note="diagnostic = dev-best comparison only; non-promoted",
        )

    fig.suptitle("n/N residual grid audit", fontsize=14, y=1.02)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return has_diag


def _best_row(rows: list[GridRow], attr: str) -> GridRow | None:
    values = [row for row in rows if getattr(row, attr) is not None]
    if not values:
        return None
    return max(values, key=lambda row: float(getattr(row, attr)))


def _build_markdown(rows: list[GridRow], fig_path: Path, md_path: Path, has_diag: bool) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    n_values = _unique_sorted(row.n for row in rows)
    N_values = _unique_sorted(row.N for row in rows)
    total_cells = len(rows)
    zero_cells = sum(1 for row in rows if row.selected_tail_alpha is not None and abs(row.selected_tail_alpha) < 1e-12)
    positive_cells = sum(1 for row in rows if row.selected_tail_alpha is not None and row.selected_tail_alpha > 0.0)
    strict_best = _best_row(rows, "train_selected_dev_macro_f1")
    diag_best = _best_row(rows, "diagnostic_dev_best_macro_f1") if has_diag else None
    lines: list[str] = [
        "# n/N Residual Grid Visual Summary",
        "",
        f"- Input JSON: `{DEFAULT_INPUT_JSON}`",
        f"- Evaluated grid cells: `{total_cells}`",
        f"- Unique n values: `{', '.join(str(v) for v in n_values)}`",
        f"- Unique N values: `{', '.join(str(v) for v in N_values)}`",
        f"- Figure: `{fig_path}`",
        "",
        "The left panel shows dev confirmation after choosing tail_alpha on the train holdout inside each (n, N) cell. "
        "Comparing cells by dev is still diagnostic unless the cell was pre-registered. "
        "The diagnostic panel, when present, shows dev-best comparison values and is explicitly non-promoted.",
        "",
        "## Grid Table",
        "",
        "| n | N | train-selected dev macro-F1 | selected tail_alpha | diagnostic dev-best macro-F1 | diagnostic status |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        status = row.diagnostic_status or ("non-promoted" if row.diagnostic_dev_best_macro_f1 is not None else "n/a")
        lines.append(
            "| "
            f"{row.n} | {row.N} | {_format_metric(row.train_selected_dev_macro_f1)} | "
            f"{_format_alpha(row.selected_tail_alpha)} | {_format_metric(row.diagnostic_dev_best_macro_f1)} | "
            f"{status} |"
        )

    lines.extend(
        [
            "",
            "## Key Readout",
            "",
            f"- Best dev-confirmed cell after train-only alpha selection: `n={strict_best.n if strict_best else 'n/a'}`, `N={strict_best.N if strict_best else 'n/a'}`, "
            f"macro-F1 `{_format_metric(strict_best.train_selected_dev_macro_f1, 6) if strict_best else 'n/a'}`.",
        ]
    )
    if strict_best is not None:
        lines.append(f"- Best train-selected tail_alpha: `{_format_alpha(strict_best.selected_tail_alpha)}`.")
        lines.append("- This cell-level comparison uses dev labels; treat it as an audit result unless the cell was pre-registered.")
    lines.append(f"- tail_alpha = 0 cells: `{zero_cells}`; tail_alpha > 0 cells: `{positive_cells}`.")
    if diag_best is not None:
        lines.extend(
            [
                f"- Best diagnostic cell: `n={diag_best.n}`, `N={diag_best.N}`, "
                f"macro-F1 `{_format_metric(diag_best.diagnostic_dev_best_macro_f1, 6)}`.",
                "- Diagnostic dev-best is shown for audit context only; it is not the promoted choice.",
            ]
        )
    else:
        lines.append("- Diagnostic dev-best values were not present in the input JSON.")

    lines.extend(["", f"Artifacts: `{fig_path}`", f"Summary: `{md_path}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = _load_json(args.input_json)
    rows = _normalize_rows(payload)
    has_diag = _plot_summary_figure(rows, args.fig_path)
    _build_markdown(rows, args.fig_path, args.md_path, has_diag)
    print(
        json.dumps(
            {
                "input": str(args.input_json),
                "figure": str(args.fig_path),
                "summary": str(args.md_path),
                "cells": len(rows),
                "diagnostic_panel": has_diag,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
