from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CLAIMS = Path("data/dev-claims.json")
DEFAULT_CANDIDATES = Path(
    "round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/"
    "dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_rerank/top3_fusion_grid")
DEFAULT_REPORT = Path("round18/reports/top3_fusion_grid_report.md")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(var)
    if std <= 1e-12:
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def parse_pool(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate file must be a JSON object: {path}")
    out: dict[str, list[dict[str, Any]]] = {}
    for claim_id, rows in payload.items():
        if not isinstance(rows, list):
            continue
        normalized = []
        for pos, row in enumerate(rows):
            if not isinstance(row, dict) or "evidence_id" not in row:
                continue
            item = dict(row)
            item["rank"] = int(item.get("rank", pos + 1) or pos + 1)
            item["reranker_score"] = float(item.get("reranker_score", item.get("score", 0.0)) or 0.0)
            item["source_score"] = float(item.get("source_score", 0.0) or 0.0)
            item["source_rank"] = int(item.get("source_rank", pos + 1) or pos + 1)
            normalized.append(item)
        normalized.sort(key=lambda x: (x["rank"], x["evidence_id"]))
        out[claim_id] = normalized
    return out


def rerank(pool: dict[str, list[dict[str, Any]]], ce_w: float, hf_w: float, rank_w: float) -> dict[str, list[dict[str, Any]]]:
    ranked: dict[str, list[dict[str, Any]]] = {}
    for claim_id, rows in pool.items():
        ce_z = zscore([float(r["reranker_score"]) for r in rows])
        hf_z = zscore([float(r["source_score"]) for r in rows])
        ce_mm = minmax([float(r["reranker_score"]) for r in rows])
        hf_mm = minmax([float(r["source_score"]) for r in rows])
        scored = []
        for idx, row in enumerate(rows):
            source_rr = 1.0 / (60.0 + float(row["source_rank"]))
            ce_rank_rr = 1.0 / (60.0 + float(row["rank"]))
            score = (
                ce_w * (0.65 * ce_z[idx] + 0.35 * ce_mm[idx])
                + hf_w * (0.65 * hf_z[idx] + 0.35 * hf_mm[idx])
                + rank_w * (0.65 * source_rr + 0.35 * ce_rank_rr)
            )
            item = dict(row)
            item["rank"] = 0
            item["score"] = float(score)
            item["fusion_score"] = float(score)
            item["fusion_weights"] = {"ce": ce_w, "hand_feature": hf_w, "rank_prior": rank_w}
            scored.append(item)
        scored.sort(key=lambda x: (-x["fusion_score"], x["source_rank"], x["rank"], x["evidence_id"]))
        for rank, item in enumerate(scored, start=1):
            item["rank"] = rank
        ranked[claim_id] = scored
    return ranked


def evaluate(claims: dict[str, Any], ranked: dict[str, list[dict[str, Any]]], ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        recalls = []
        tp = 0
        gold_total = 0
        hit = 0
        for claim_id, claim in claims.items():
            gold = set(claim.get("evidences", []))
            if not gold:
                continue
            pred = {row["evidence_id"] for row in ranked.get(claim_id, [])[:k]}
            inter = gold & pred
            recalls.append(len(inter) / len(gold))
            tp += len(inter)
            gold_total += len(gold)
            hit += bool(inter)
        out[f"macro_recall_at_{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        out[f"micro_recall_at_{k}"] = tp / gold_total if gold_total else 0.0
        out[f"hit_any_at_{k}"] = hit / len(recalls) if recalls else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--eval-k", default="1,3,5,10,64,100,500")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.eval_k.split(",") if x.strip()]
    claims = load_json(args.claims)
    pool = parse_pool(args.candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ce_weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    hf_weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rank_weights = [0.0, 0.25, 0.5, 1.0]
    rows = []
    best = None
    for ce_w in ce_weights:
        for hf_w in hf_weights:
            for rank_w in rank_weights:
                if ce_w == 0.0 and hf_w == 0.0 and rank_w == 0.0:
                    continue
                ranked = rerank(pool, ce_w=ce_w, hf_w=hf_w, rank_w=rank_w)
                metrics = evaluate(claims, ranked, ks)
                row = {"ce_w": ce_w, "hf_w": hf_w, "rank_w": rank_w, **metrics}
                rows.append(row)
                key = (row["macro_recall_at_3"], row["macro_recall_at_10"], row["macro_recall_at_64"])
                if best is None or key > best[0]:
                    best = (key, row, ranked)

    rows.sort(key=lambda r: (r["macro_recall_at_3"], r["macro_recall_at_10"], r["macro_recall_at_64"]), reverse=True)
    csv_path = args.output_dir / "top3_fusion_grid.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_row = rows[0]
    best_ranked = rerank(pool, best_row["ce_w"], best_row["hf_w"], best_row["rank_w"])
    best_candidates_path = args.output_dir / "dev_top3_fusion_grid_diagnostic_best_candidates.json"
    write_json(best_candidates_path, best_ranked)
    write_json(args.output_dir / "top3_fusion_grid_summary.json", {"status": "diagnostic-dev-selected", "best": best_row, "top10": rows[:10]})

    report = [
        "# Top3 Fusion Grid Diagnostic",
        "",
        "Status: diagnostic only. Weights are selected on dev metrics and must not be promoted without train-only selection.",
        "",
        f"Input candidates: `{args.candidates}`",
        f"Best candidate file: `{best_candidates_path}`",
        "",
        "## Best by macro recall@3, then @10, then @64",
        "",
        "| ce_w | hf_w | rank_w | mR@1 | mR@3 | mR@5 | mR@10 | mR@64 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:10]:
        report.append(
            f"| {row['ce_w']:.2f} | {row['hf_w']:.2f} | {row['rank_w']:.2f} | "
            f"{row['macro_recall_at_1']:.4f} | {row['macro_recall_at_3']:.4f} | "
            f"{row['macro_recall_at_5']:.4f} | {row['macro_recall_at_10']:.4f} | "
            f"{row['macro_recall_at_64']:.4f} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {best_candidates_path}")
    print(f"Wrote {args.report}")
    print(json.dumps(best_row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
