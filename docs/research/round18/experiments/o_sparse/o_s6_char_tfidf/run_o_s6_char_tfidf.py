#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from round18.tools.common import (
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-S6 char TF-IDF candidate generator/evaluator from raw claim/evidence JSON."
        )
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/dev-claims.json"),
        help="Claims split used for generation and recall evaluation.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("data/evidence.json"),
        help="Raw evidence corpus JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s6_char_tfidf"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-id",
        default="o_s6_char_tfidf",
        help="Run identifier used in output filenames.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s6_char_tfidf/run_manifest.json"),
        help="Manifest output path.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Top-k candidate pool to write.",
    )
    parser.add_argument(
        "--eval-k",
        default="100,500",
        help="Comma-separated recall-k values, e.g. 100,500.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=300_000,
        help="Max char n-gram vocabulary size for the vectorizer.",
    )
    parser.add_argument(
        "--char-ngram-range",
        default="3,5",
        help="Character n-gram range for TF-IDF, e.g. 3,5.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use bounded claim/evidence slices instead of the full corpus.",
    )
    parser.add_argument("--max-claims", type=int, default=0, help="Optional claim cap (0 means no cap).")
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=0,
        help="Optional evidence cap (0 means no cap).",
    )
    parser.add_argument(
        "--smoke-claims",
        type=int,
        default=64,
        help="Smoke cap for claims when --smoke is enabled.",
    )
    parser.add_argument(
        "--smoke-evidence",
        type=int,
        default=120_000,
        help="Smoke cap for evidence docs when --smoke is enabled.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Reproducibility marker.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s6_char_tfidf/run_record.json"),
        help="JSON execution record output path.",
    )
    parser.add_argument(
        "--stage",
        default="o_s6_char_tfidf",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("eval-k must contain at least one integer.")
    return sorted(set(values))


def parse_char_ngram_range(raw: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 2 or parts[0] <= 0 or parts[1] < parts[0]:
        raise argparse.ArgumentTypeError(
            "char-ngram-range must be two integers in ascending order, e.g. 3,5"
        )
    return parts[0], parts[1]


def _build_inferred_split_name(path: Path) -> str:
    if str(path) == "data/train-claims.json":
        return "train"
    if str(path) == "data/dev-claims.json":
        return "dev"
    return "claims"


def _is_allowed_claim_path(claims_path: Path) -> None:
    if claims_path not in {Path("data/train-claims.json"), Path("data/dev-claims.json")}:
        raise SystemExit(
            "O-S6 requires claims from data/train-claims.json or data/dev-claims.json."
        )
    if claims_path == Path("data/test-claims-unlabelled.json"):
        raise SystemExit("O-S6 does not support test run without explicit prediction mode.")


def _input_records(path: Path, evidence_path: Path, split_name: str) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "split": split_name,
            "labels_used": split_name in {"train", "dev"},
        },
        {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "split": "evidence",
            "labels_used": False,
        },
    ]


def limit_dict(data: dict[str, Any], max_items: int) -> dict[str, Any]:
    if max_items <= 0 or len(data) <= max_items:
        return dict(data)
    head = list(data.items())[:max_items]
    return {key: value for key, value in head}


def top_indices_with_fallback(row, top_k: int, fallback_count: int) -> list[int]:
    if top_k <= 0 or fallback_count <= 0:
        return []

    if row.nnz == 0:
        return list(range(min(top_k, fallback_count)))

    indices = np.asarray(row.indices, dtype=np.int64)
    scores = np.asarray(row.data, dtype=np.float32)
    order = np.lexsort((indices, -scores))
    ranked = indices[order]
    selected = ranked[: min(top_k, ranked.size)].tolist()

    if len(selected) < top_k:
        selected_set = set(selected)
        needed = top_k - len(selected)
        for idx in range(fallback_count):
            if idx in selected_set:
                continue
            selected.append(idx)
            needed -= 1
            if needed <= 0:
                break

    return selected[:top_k]


def build_candidates(
    claims: dict[str, Any],
    evidence: dict[str, str],
    candidate_k: int,
    max_features: int,
    ngram_range: tuple[int, int],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    evidence_ids = list(evidence.keys())
    evidence_texts = [evidence[evidence_id] for evidence_id in evidence_ids]
    claim_texts = [claim["claim_text"] for claim in claims.values()]

    if not evidence_ids or not claim_texts:
        empty_pool = {claim_id: [] for claim_id in claims}
        return empty_pool, evidence_ids

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
    )
    evidence_matrix = vectorizer.fit_transform(evidence_texts)
    claim_matrix = vectorizer.transform(claim_texts)
    similarities = claim_matrix @ evidence_matrix.T

    pool: dict[str, list[dict[str, Any]]] = {}
    for row_idx, claim_id in enumerate(claims.keys()):
        row = similarities.getrow(row_idx)
        selected_indexes = top_indices_with_fallback(
            row=row,
            top_k=candidate_k,
            fallback_count=len(evidence_ids),
        )
        row_scores = {idx: score for idx, score in zip(row.indices.tolist(), row.data.tolist())}
        candidates = [
            {
                "claim_id": claim_id,
                "evidence_id": evidence_ids[idx],
                "rank": rank,
                "score": float(row_scores.get(idx, 0.0)),
            }
            for rank, idx in enumerate(selected_indexes, start=1)
        ]
        pool[claim_id] = candidates

    return pool, evidence_ids


def evaluate_recall_at_k(claims: dict[str, Any], pool: dict[str, list[dict[str, Any]]], ks: list[int]) -> dict[str, Any]:
    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    per_k = {k: [] for k in ks}
    totals = {k: {"tp": 0, "gold": 0, "claims_hit": 0} for k in ks}
    label_totals: dict[str, int] = {}
    label_hits: dict[str, dict[int, int]] = {k: {} for k in ks}

    claims_with_evidence = 0
    for claim_id, claim in claims.items():
        gold = claim.get("evidences", [])
        if not gold:
            continue
        claims_with_evidence += 1
        gold_set = set(gold)
        ranked = [candidate["evidence_id"] for candidate in pool.get(claim_id, [])]
        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0) + 1

        for k in ks:
            predicted = set(ranked[:k])
            tp = len(gold_set.intersection(predicted))
            totals[k]["tp"] += tp
            totals[k]["gold"] += len(gold_set)
            if tp > 0:
                totals[k]["claims_hit"] += 1
                label_hits[k][label] = label_hits[k].get(label, 0) + 1
            recall = tp / len(gold_set)
            per_k[k].append(recall)

    rows: dict[str, Any] = {
        "claims_with_evidence": claims_with_evidence,
        "evaluated_ks": ks,
    }
    for k in ks:
        total_tp = float(totals[k]["tp"])
        total_gold = float(totals[k]["gold"])
        recalls = per_k[k]
        rows[f"macro_recall_at_{k}"] = float(np.mean(recalls)) if recalls else 0.0
        rows[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        rows[f"hit_any_at_{k}"] = totals[k]["claims_hit"] / claims_with_evidence if claims_with_evidence else 0.0
        for label in sorted(label_totals):
            hits = label_hits[k].get(label, 0)
            rows[f"{label.lower()}_hit_any_at_{k}"] = hits / label_totals[label] if label_totals[label] else 0.0
    return rows


def build_data_flow_report(
    args: argparse.Namespace,
    split_name: str,
    claims: dict[str, Any],
    evidence_count: int,
    candidate_path: Path,
    metrics_path: Path,
    metrics: dict[str, Any],
    eval_ks: list[int],
    input_records: list[dict[str, Any]],
    data_flow_path: Path,
) -> dict[str, Any]:
    return {
        "pipeline": "O-S6 char TF-IDF candidate lane (leaf flat output).",
        "split": split_name,
        "inputs": {
            "claims": str(args.claims),
            "evidence": str(args.evidence),
        },
        "input_file_hashes": {record["path"]: record["sha256"] for record in input_records},
        "processing": {
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "char_analyzer": "char_wb",
            "char_ngram_range": args.char_ngram_range,
            "max_features": args.max_features,
            "smoke": args.smoke,
            "max_claims": args.max_claims,
            "max_evidence": args.max_evidence,
            "random_seed": args.random_seed,
        },
        "counts": {
            "claims_in_file": len(claims),
            "evidence_in_file": evidence_count,
        },
        "outputs": {
            "candidate_pool": str(candidate_path),
            "metrics": str(metrics_path),
            "data_flow_report": str(data_flow_path),
            "run_record": str(args.record_path),
            "manifest": str(args.manifest),
        },
        "reported_metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_command = "python round18/experiments/o_sparse/o_s6_char_tfidf/run_o_s6_char_tfidf.py"

    _is_allowed_claim_path(args.claims)
    if args.evidence != Path("data/evidence.json"):
        raise SystemExit("Evidence path must be data/evidence.json for strict mode.")

    if args.smoke:
        if args.max_claims <= 0:
            args.max_claims = args.smoke_claims
        if args.max_evidence <= 0:
            args.max_evidence = args.smoke_evidence

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    if not isinstance(claims, dict):
        raise SystemExit("Claims JSON must be a dict keyed by claim id.")
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence JSON must be a dict keyed by evidence id.")

    claims = limit_dict(claims, args.max_claims)
    evidence = limit_dict(evidence, args.max_evidence)

    split_name = _build_inferred_split_name(args.claims)
    eval_ks = parse_k_list(args.eval_k)
    max_eval_k = max(eval_ks)
    if max_eval_k > args.candidate_k:
        args.candidate_k = max_eval_k

    input_scan_targets = [str(args.claims), str(args.evidence), str(args.output_dir)]
    forbidden_hits = find_forbidden_tokens(input_scan_targets)

    ngram_range = parse_char_ngram_range(args.char_ngram_range)
    pool, evidence_ids = build_candidates(
        claims=claims,
        evidence=evidence,
        candidate_k=args.candidate_k,
        max_features=args.max_features,
        ngram_range=ngram_range,
    )
    metrics = evaluate_recall_at_k(claims=claims, pool=pool, ks=eval_ks)
    metrics.update(
        {
            "method": "char_tfidf",
            "char_analyzer": "char_wb",
            "char_ngram_range": list(ngram_range),
            "candidate_k": args.candidate_k,
            "candidate_pool_size": len(pool),
            "claims_input_count": len(claims),
            "evidence_items_available": len(evidence_ids),
            "max_features": args.max_features,
            "smoke": args.smoke,
            "max_claims": args.max_claims,
            "max_evidence": args.max_evidence,
            "random_seed": args.random_seed,
            "candidate_output_type": "leaf_plain",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_file_suffix = f"{split_name}_full_{split_name}"
    candidate_output = args.output_dir / f"{split_file_suffix}_{args.run_id}_tfidf_char_top{args.candidate_k}_candidates.json"
    metrics_output = args.output_dir / f"{split_file_suffix}_{args.run_id}_tfidf_char_top{args.candidate_k}_metrics.json"
    data_flow_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_data_flow_report.json"

    write_json(candidate_output, pool)
    write_json(metrics_output, metrics)
    output_files = [str(candidate_output), str(metrics_output)]

    command = (
        f"{base_command} --claims {args.claims} --evidence {args.evidence} "
        f"--candidate-k {args.candidate_k} --eval-k {','.join(map(str, eval_ks))} "
        f"--max-features {args.max_features} --char-ngram-range {args.char_ngram_range} "
        f"--run-id {args.run_id} --manifest {args.manifest} --output-dir {args.output_dir} "
        f"--stage {args.stage} --random-seed {args.random_seed} "
        f"{'--smoke' if args.smoke else ''} "
        f"{'--max-claims ' + str(args.max_claims) if args.max_claims > 0 else ''} "
        f"{'--max-evidence ' + str(args.max_evidence) if args.max_evidence > 0 else ''}"
    ).strip()

    input_records = _input_records(args.claims, args.evidence, split_name)
    for item in input_records:
        if item["path"] == str(args.claims):
            item["labels_used"] = split_name in {"train", "dev"}

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate",
        mode="STRICT",
        stage=args.stage,
        command=command,
        working_directory=Path.cwd(),
        config_path=str(args.manifest),
        config_hash="",
        random_seed=args.random_seed,
        cv_seed=None,
    )
    manifest["input_files"] = input_records
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input/evidence path scan against forbidden strict tokens passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = output_files + [str(args.record_path), str(data_flow_path), str(args.manifest)]
    manifest["metrics"] = {
        "recall": metrics,
        "candidate_file": str(candidate_output),
        "candidate_top_k": args.candidate_k,
        "candidate_output_type": "leaf_plain",
        "claims_input_count": len(claims),
        "evidence_input_count": len(evidence_ids),
        "evidence_requested_count": len(evidence),
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Loaded raw claims/evidence JSON, built raw char-level TF-IDF sparse similarities, "
        "materialized top-k leaf candidates per claim, and wrote plain per-claim candidate list with recall metrics."
    )
    manifest["split_isolation_summary"] = (
        "Train labels are used only for train-split smoke diagnostics when explicitly selected. "
        "This run defaults to dev and does not inspect test labels."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low"
    manifest["notes"] = (
        "Candidate output is a plain leaf list per claim (claim_id/evidence_id/rank/score only), "
        "matching Round18 sparse input schema."
    )
    write_json(args.manifest, manifest)
    output_files.append(str(args.manifest))

    record = {
        "run_id": args.run_id,
        "command": command,
        "command_args": {k: str(v) for k, v in vars(args).items()},
        "mode": "STRICT",
        "status": "strict-candidate",
        "stage": args.stage,
        "split": split_name,
        "forbidden_hits": forbidden_hits,
        "files_written": output_files + [str(data_flow_path)],
        "claims_count": len(claims),
        "evidence_count": len(evidence),
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "smoke": args.smoke,
        "smoke_claims": args.max_claims if args.smoke and args.max_claims > 0 else 0,
        "smoke_evidence": args.max_evidence if args.smoke and args.max_evidence > 0 else 0,
        "macro_recall_at_500": metrics.get("macro_recall_at_500"),
    }
    write_json(args.record_path, record)
    output_files.append(str(args.record_path))

    data_flow = build_data_flow_report(
        args=args,
        split_name=split_name,
        claims=claims,
        evidence_count=len(evidence_ids),
        candidate_path=candidate_output,
        metrics_path=metrics_output,
        metrics=metrics,
        eval_ks=eval_ks,
        input_records=input_records,
        data_flow_path=data_flow_path,
    )
    write_json(data_flow_path, data_flow)
    output_files.append(str(data_flow_path))

    print(f"Wrote candidate pool: {candidate_output}")
    print(f"Wrote metrics: {metrics_output}")
    print(f"Wrote data-flow report: {data_flow_path}")
    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote record: {args.record_path}")
    for key in sorted(metrics.keys()):
        if key.endswith("@500") or key.startswith("macro_recall_at_") or key.startswith("claims"):
            print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
