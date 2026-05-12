#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / "round18").exists():
        break
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from round18.tools.common import load_json


TOP_N_DEFAULT = [1, 3, 5, 10, 20, 32, 50, 64, 100, 200, 300, 500, 1000, 2000]


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    category: str
    candidate_path: Path
    source: str
    description: str = ""


@dataclass(frozen=True)
class RecallRow:
    top_n: int
    macro_recall: float | None
    micro_recall: float | None
    claims_with_evidence: int
    claim_recall_hits: list[float]
    tp: int
    total_gold: int
    hit_any: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round18 tutorial ablation summary for R-n curves."
    )
    parser.add_argument(
        "--claims-path",
        default=Path("data/dev-claims.json"),
        type=Path,
        help="Claims file for evaluation.",
    )
    parser.add_argument(
        "--top-n",
        default=",".join(str(v) for v in TOP_N_DEFAULT),
        help="Comma-separated recall-k values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/reports/tutorial_rn_curves"),
        help="Directory for JSON/CSV outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("group_meetings/second_meeting /figures"),
        help="Directory for PNG figures.",
    )
    parser.add_argument(
        "--figure-prefix",
        default="round18_tutorial",
        help="File name prefix for figure artifacts.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    return parser.parse_args()


def parse_top_n(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Top-n value must be int: {item}") from exc
    if not values:
        raise argparse.ArgumentTypeError("--top-n must include at least one integer")
    return sorted(set(values))


def normalize_label(raw: str) -> str:
    return re.sub(r"_+", " ", raw).strip()


def default_methods() -> list[MethodSpec]:
    methods: list[MethodSpec] = [
        MethodSpec(
            method_id="single_bm25",
            label="BM25",
            category="single_sparse",
            candidate_path=Path(
                "round18/outputs/o_sparse/o_s1_lexical_index_experiments/"
                "dev_full_bm25_dev_bm25_top500_candidates.json"
            ),
            source="BM25 leaf pool",
        ),
        MethodSpec(
            method_id="single_char_tfidf",
            label="char Tfidf",
            category="single_sparse",
            candidate_path=Path(
                "round18/outputs/o_sparse/o_s6_char_tfidf/"
                "dev_full_dev_o_s6_char_tfidf_tfidf_char_top500_candidates.json"
            ),
            source="char Tfidf leaf pool",
        ),
        MethodSpec(
            method_id="single_structured",
            label="structured",
            category="single_sparse",
            candidate_path=Path("round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposed_candidates.json"),
            source="structured leaf pool",
        ),
        MethodSpec(
            method_id="single_prf",
            label="PRF",
            category="single_sparse",
            candidate_path=Path("round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json"),
            source="PRF leaf pool",
        ),
    ]

    for policy in (
        "strict_rrf_equal_bm25_char",
        "strict_rrf_equal_bm25_char_structured_prf",
        "strict_rrf_char_heavy",
        "strict_rrf_bm25_heavy",
    ):
        methods.append(
            MethodSpec(
                method_id=f"s7_{policy}",
                label=normalize_label(policy),
                category="os7",
                candidate_path=Path(
                    "round18/outputs/o_sparse/o_s7_plain_leaf_fusion/"
                    f"dev_full_dev_o_s7_plain_leaf_fusion_{policy}_top500_candidates.json"
                ),
                source="o_s7 strict fusion",
            )
        )

    for policy in (
        "rrf",
        "round_robin",
        "priority",
        "union_upper",
    ):
        methods.extend(
            [
                MethodSpec(
                    method_id=f"os9_bm25_char_{policy}_1000",
                    label=f"{policy}",
                    category="os9_bm25_char",
                    candidate_path=Path(
                        "round18/outputs/o_sparse/"
                        "o_s9_union_gate_bm25_char_dev/"
                        f"dev_full_dev_o_s9_union_gate_bm25_char_{policy}_top1000_candidates.json"
                    ),
                    source="O-S9 BM25+char (top1000)",
                ),
                MethodSpec(
                    method_id=f"os9_four_source_{policy}_2000",
                    label=f"{policy}",
                    category="os9_four_source",
                    candidate_path=Path(
                        "round18/outputs/o_sparse/"
                        "o_s9_union_gate_four_source_wide_dev/"
                        f"dev_full_dev_o_s9_union_gate_four_source_wide_{policy}_top2000_candidates.json"
                    ),
                    source="O-S9 four-source wide (top2000)",
                ),
            ]
        )

    methods.extend(
        [
            MethodSpec(
                method_id="os8_hand_feature",
                label="O-S8 hand-feature",
                category="os8",
                candidate_path=Path(
                    "round18/outputs/o_sparse/"
                    "o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json"
                ),
                source="O-S8 hand-feature",
            )
        ]
    )

    o_s10_dirs = sorted(
        Path("round18/outputs/o_sparse").glob("o_s10_wide_hand_feature*")
    )
    for run_dir in o_s10_dirs:
        for candidate_path in sorted(run_dir.glob("*_top500_candidates.json")):
            match = re.search(r"dev_full_dev_(.+)_top500_candidates\.json$", candidate_path.name)
            if not match:
                continue
            method_id = f"os10_{match.group(1)}"
            methods.append(
                MethodSpec(
                    method_id=method_id,
                    label=match.group(1).replace("_", " "),
                    category="os10",
                    candidate_path=candidate_path,
                    source=f"O-S10 variant in {run_dir.name}",
                )
            )

    for method_id in (
        "strict_sparse_only",
        "strict_ce64_sparse_backfill",
        "strict_rrf_sparse_ce",
    ):
        methods.append(
            MethodSpec(
                method_id=f"oa2c_{method_id}",
                label=method_id.replace("_", " "),
                category="oa2c",
                candidate_path=Path(
                    "round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/"
                    f"dev_full_dev_o_a2c_plain_leaf_rank_s8_{method_id}_top500_candidates.json"
                ),
                source="O-A2c strict aggregate",
            )
        )
    return methods


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_candidate_pool(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"candidate pool must be a mapping: {path}")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for claim_id, rows in raw.items():
        if not isinstance(rows, list):
            continue
        normalized[str(claim_id)] = [row for row in rows if isinstance(row, dict)]
    return normalized


def extract_evidence_ids(rows: list[dict[str, Any]]) -> list[str]:
    ranked_rows = []
    for idx, row in enumerate(rows):
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str):
            continue
        ranked_rows.append((safe_int(row.get("rank"), idx), idx, evidence_id))
    ranked_rows.sort(key=lambda item: (item[0] if item[0] is not None else 10**12, item[1]))
    deduped: list[str] = []
    seen: set[str] = set()
    for _, _, evidence_id in ranked_rows:
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        deduped.append(evidence_id)
    return deduped


def load_gold_supports(claims_path: Path) -> dict[str, set[str]]:
    claims = load_json(claims_path)
    if not isinstance(claims, dict):
        raise ValueError(f"claims path must map claim id to claim payload: {claims_path}")
    gold: dict[str, set[str]] = {}
    for claim_id, payload in claims.items():
        evidences = payload.get("evidences") if isinstance(payload, dict) else None
        if not isinstance(evidences, list):
            gold[str(claim_id)] = set()
            continue
        evidence_set = {str(e) for e in evidences if isinstance(e, str)}
        gold[str(claim_id)] = evidence_set
    return gold


def evaluate_method(
    method: MethodSpec,
    gold: dict[str, set[str]],
    top_n: list[int],
) -> dict[str, Any]:
    try:
        pool = load_candidate_pool(method.candidate_path)
    except Exception as exc:
        return {
            "method_id": method.method_id,
            "label": method.label,
            "category": method.category,
            "source": str(method.candidate_path),
            "description": method.description,
            "status": "missing_or_invalid_candidate_pool",
            "error": str(exc),
            "rows": [],
        }

    claim_candidate_ids: dict[str, list[str]] = {}
    per_claim_sizes: list[int] = []
    missing_pool_claims: list[str] = []
    for claim_id in gold:
        rows = pool.get(claim_id, [])
        if not rows:
            missing_pool_claims.append(claim_id)
        evidence_ids = extract_evidence_ids(rows)
        claim_candidate_ids[claim_id] = evidence_ids
        per_claim_sizes.append(len(evidence_ids))

    if not top_n:
        return {
            "method_id": method.method_id,
            "label": method.label,
            "category": method.category,
            "source": str(method.candidate_path),
            "description": method.description,
            "status": "no_top_n",
            "rows": [],
        }

    method_max_candidate = max(per_claim_sizes) if per_claim_sizes else 0
    rows_out: list[RecallRow] = []

    claims_with_evidence = sum(1 for ev in gold.values() if ev)
    for k in top_n:
        if k > method_max_candidate:
            rows_out.append(
                RecallRow(
                    top_n=k,
                    macro_recall=None,
                    micro_recall=None,
                    claims_with_evidence=claims_with_evidence,
                    claim_recall_hits=[],
                    tp=0,
                    total_gold=0,
                    hit_any=0,
                )
            )
            continue

        total_gold = 0
        total_tp = 0
        per_claim_recalls: list[float] = []
        hit_any = 0
        for claim_id, gold_ids in gold.items():
            if not gold_ids:
                continue
            candidates = claim_candidate_ids.get(claim_id, [])
            top_candidates = set(candidates[:k])
            tp = len(top_candidates & gold_ids)
            total_tp += tp
            total_gold += len(gold_ids)
            if tp > 0:
                hit_any += 1
            per_claim_recalls.append(tp / len(gold_ids))

        macro_recall = (
            sum(per_claim_recalls) / len(per_claim_recalls)
            if per_claim_recalls
            else 0.0
        )
        micro_recall = total_tp / total_gold if total_gold else 0.0
        rows_out.append(
            RecallRow(
                top_n=k,
                macro_recall=macro_recall,
                micro_recall=micro_recall,
                claims_with_evidence=claims_with_evidence,
                claim_recall_hits=per_claim_recalls,
                tp=total_tp,
                total_gold=total_gold,
                hit_any=hit_any,
            )
        )

    return {
        "method_id": method.method_id,
        "label": method.label,
        "category": method.category,
        "source": str(method.candidate_path),
        "description": method.description,
        "status": "ok",
        "rows": [
            row.__dict__ | {"claim_recall_hits": []} for row in rows_out
        ],
        "max_candidate_count": method_max_candidate,
        "avg_candidate_count": sum(per_claim_sizes) / len(per_claim_sizes) if per_claim_sizes else 0.0,
        "min_candidate_count": min(per_claim_sizes) if per_claim_sizes else 0,
        "missing_pool_claims": sorted(missing_pool_claims),
        "num_claims": len(gold),
        "num_claims_with_evidence": claims_with_evidence,
    }


def to_long_row(result: dict[str, Any], top_n: int) -> dict[str, Any]:
    row_map: dict[int, dict[str, Any]] = {row["top_n"]: row for row in result.get("rows", [])}
    base = {
        "method_id": result["method_id"],
        "label": result["label"],
        "category": result["category"],
        "source": result["source"],
        "status": result.get("status", ""),
        "max_candidate_count": result.get("max_candidate_count", 0),
        "avg_candidate_count": result.get("avg_candidate_count", 0.0),
        "min_candidate_count": result.get("min_candidate_count", 0),
        "num_claims": result.get("num_claims", 0),
        "num_claims_with_evidence": result.get("num_claims_with_evidence", 0),
        "missing_pool_claims": len(result.get("missing_pool_claims", [])),
    }
    row = dict(base)
    row.update(
        {
            "top_n": top_n,
            "macro_recall": row_map.get(top_n, {}).get("macro_recall"),
            "micro_recall": row_map.get(top_n, {}).get("micro_recall"),
            "hit_any": row_map.get(top_n, {}).get("hit_any"),
        }
    )
    return row


def write_summary_csv(results: list[dict[str, Any]], top_n: list[int], out_path: Path) -> None:
    columns = [
        "method_id",
        "label",
        "category",
        "source",
        "status",
        "max_candidate_count",
        "avg_candidate_count",
        "min_candidate_count",
        "num_claims",
        "num_claims_with_evidence",
        "missing_pool_claims",
        "top_n",
        "macro_recall",
        "micro_recall",
        "hit_any",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for k in top_n:
                writer.writerow(to_long_row(result, k))


def write_summary_json(results: list[dict[str, Any]], top_n: list[int], out_path: Path) -> None:
    payload = {
        "top_n": top_n,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def collect_oc3_runs() -> list[dict[str, Any]]:
    o_c3_base = Path("round18/outputs/o_classifier")
    runs = []
    for run_dir in sorted(o_c3_base.glob("o_c3*")):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        metrics = manifest.get("metrics")
        if not isinstance(metrics, dict):
            continue
        train_k = manifest.get("train_context_k", metrics.get("train_context_k"))
        dev_k = manifest.get("dev_context_k", metrics.get("dev_context_k"))
        budget = manifest.get("evidence_token_budget")
        if budget is None:
            budget = metrics.get("evidence_token_budget")
        if budget is None:
            budget_match = re.search(r"budget(\\d+)", run_dir.name)
            if budget_match:
                budget = int(budget_match.group(1))

        model_infos: list[dict[str, Any]] = []
        model_map = metrics.get("models", {})
        if isinstance(model_map, dict):
            for model_name, model_payload in model_map.items():
                if not isinstance(model_payload, dict):
                    continue
                metrics_path = model_payload.get("metrics")
                model_metric_path = Path(metrics_path) if metrics_path else None
                if not model_metric_path or not model_metric_path.exists():
                    continue
                model_metrics = load_json(model_metric_path)
                model_infos.append(
                    {
                        "model_name": model_name,
                        "status": model_payload.get("status"),
                        "selected_variant": model_payload.get("selected_variant"),
                        "model_metrics_path": str(model_metric_path),
                        "accuracy": model_metrics.get("accuracy"),
                        "macro_f1": model_metrics.get("macro_f1"),
                        "macro_recall": model_metrics.get("macro_recall"),
                        "micro_f1": model_metrics.get("micro_f1"),
                        "assignment_f": model_metrics.get("assignment_f"),
                        "harmonic_mean_F_A": model_metrics.get("harmonic_mean_F_A"),
                        "per_class_recall": model_metrics.get("per_class_recall", {}),
                    }
                )
        runs.append(
            {
                "run_id": manifest.get("run_id"),
                "run_dir": str(run_dir),
                "stage": manifest.get("stage"),
                "status": manifest.get("status"),
                "train_context_k": train_k,
                "dev_context_k": dev_k,
                "evidence_token_budget": budget,
                "train_context_source": metrics.get("train_context_source"),
                "dev_context_source": metrics.get("dev_context_source"),
                "models": model_infos,
            }
        )
    return runs


def write_classifier_summary_csv(classifier_runs: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "run_dir",
        "stage",
        "status",
        "train_context_k",
        "dev_context_k",
        "evidence_token_budget",
        "model_name",
        "model_status",
        "selected_variant",
        "accuracy",
        "macro_f1",
        "macro_recall",
        "micro_f1",
        "assignment_f",
        "harmonic_mean_F_A",
        "model_metrics_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for run in classifier_runs:
            if run["models"]:
                for model in run["models"]:
                    row = {c: run.get(c) for c in columns}
                    row.update(
                        {
                            "model_name": model.get("model_name"),
                            "model_status": model.get("status"),
                            "selected_variant": model.get("selected_variant"),
                            "accuracy": model.get("accuracy"),
                            "macro_f1": model.get("macro_f1"),
                            "macro_recall": model.get("macro_recall"),
                            "micro_f1": model.get("micro_f1"),
                            "assignment_f": model.get("assignment_f"),
                            "harmonic_mean_F_A": model.get("harmonic_mean_F_A"),
                            "model_metrics_path": model.get("model_metrics_path"),
                        }
                    )
                    writer.writerow(row)
            else:
                writer.writerow({
                    "run_id": run.get("run_id"),
                    "run_dir": run.get("run_dir"),
                    "stage": run.get("stage"),
                    "status": run.get("status"),
                    "train_context_k": run.get("train_context_k"),
                    "dev_context_k": run.get("dev_context_k"),
                    "evidence_token_budget": run.get("evidence_token_budget"),
                })


def write_classifier_summary_json(classifier_runs: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": classifier_runs}, f, indent=2, sort_keys=True)
        f.write("\n")


def _to_plot_points(result: dict[str, Any], metric_key: str) -> tuple[list[int], list[float]]:
    points_x: list[int] = []
    points_y: list[float] = []
    max_k = result.get("max_candidate_count", 0)
    for row in result.get("rows", []):
        k = row["top_n"]
        if k > max_k:
            continue
        value = row.get(metric_key)
        if value is None:
            continue
        points_x.append(k)
        points_y.append(float(value))
    return points_x, points_y


def plot_category_curves(
    results: list[dict[str, Any]],
    top_n: list[int],
    metric_key: str,
    figure_path: Path,
    title: str,
) -> None:
    if not results:
        return

    categories = [
        "single_sparse",
        "os7",
        "os9_bm25_char",
        "os9_four_source",
        "os8",
        "os10",
        "oa2c",
    ]
    category_rows = {cat: [r for r in results if r["category"] == cat] for cat in categories}

    rows = [r for r in category_rows.values() if r]
    if not rows:
        return

    fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
    axes = [ax for row in axes for ax in row]
    fig.suptitle(title, fontsize=16)

    plotted = False
    for ax, category in zip(axes, categories):
        methods = category_rows.get(category, [])
        if not methods:
            ax.set_axis_off()
            continue
        plotted = True
        for result in methods:
            xs, ys = _to_plot_points(result, metric_key)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=result["label"])
        ax.set_title(category.replace("_", " ").title())
        ax.set_xlabel("top-n")
        ax.set_ylabel("recall")
        ax.set_xticks(top_n)
        ax.set_xticklabels(top_n, rotation=45)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=False,
        )

    for ax in axes[len(categories):]:
        ax.set_axis_off()

    if not plotted:
        plt.close(fig)
        return

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def deduplicate_methods(method_specs: list[MethodSpec]) -> list[MethodSpec]:
    seen: set[str] = set()
    out = []
    for spec in method_specs:
        method_id = spec.method_id
        if method_id in seen:
            continue
        seen.add(method_id)
        out.append(spec)
    return out


def main() -> None:
    args = parse_args()
    top_n = parse_top_n(args.top_n)
    claims_gold = load_gold_supports(args.claims_path)

    method_specs = deduplicate_methods(default_methods())
    results = [evaluate_method(spec, claims_gold, top_n) for spec in method_specs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = args.output_dir / "tutorial_rn_curves_summary.json"
    summary_csv_path = args.output_dir / "tutorial_rn_curves_summary.csv"
    write_summary_json(results, top_n, summary_json_path)
    write_summary_csv(results, top_n, summary_csv_path)

    classifier_runs = collect_oc3_runs()
    classifier_json_path = args.output_dir / "tutorial_o_c3_classifier_summary.json"
    classifier_csv_path = args.output_dir / "tutorial_o_c3_classifier_summary.csv"
    write_classifier_summary_json(classifier_runs, classifier_json_path)
    write_classifier_summary_csv(classifier_runs, classifier_csv_path)

    if not args.skip_plots:
        plot_category_curves(
            results=results,
            top_n=top_n,
            metric_key="macro_recall",
            figure_path=args.figure_dir / f"{args.figure_prefix}_macro_rn_curves.png",
            title="Round18 tutorial R-n macro recall",
        )
        plot_category_curves(
            results=results,
            top_n=top_n,
            metric_key="micro_recall",
            figure_path=args.figure_prefix
            and args.figure_dir / f"{args.figure_prefix}_micro_rn_curves.png",
            title="Round18 tutorial R-n micro recall",
        )


if __name__ == "__main__":
    main()
