from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from a3_factcheck.retrieval.bm25 import build_bm25_similarities
from a3_factcheck.retrieval.tfidf import build_tfidf_similarities

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
            "Round18 lexical sparse candidate generator/evaluator using only raw course "
            "JSON inputs."
        )
    )
    parser.add_argument(
        "--claims",
        default="data/dev-claims.json",
        type=Path,
        help="Claim split used for generation and recall evaluation.",
    )
    parser.add_argument(
        "--evidence",
        default="data/evidence.json",
        type=Path,
        help="Evidence corpus JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("round18/outputs/o_sparse/o_s1_lexical_index_experiments"),
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--run-id",
        default="o_s1_lexical_index_experiments",
        help="Run identifier used in output file names.",
    )
    parser.add_argument(
        "--manifest",
        default=Path("round18/outputs/o_sparse/o_s1_lexical_index_experiments/run_manifest.json"),
        type=Path,
        help="Where to write run_manifest.json.",
    )
    parser.add_argument(
        "--method",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Sparse retrieval method.",
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
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use bounded data slices instead of the full raw corpus for a quick, "
            "reproducible smoke run."
        ),
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=0,
        help="Optional claim cap (0 means no cap).",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=0,
        help="Optional evidence cap (0 means no cap).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Seed recorded for reproducibility if any subset sampling is added in future.",
    )
    parser.add_argument(
        "--record-path",
        default=Path("round18/outputs/o_sparse/o_s1_lexical_index_experiments/run_record.json"),
        type=Path,
        help="Optional JSON record of files read/written and command context.",
    )
    parser.add_argument(
        "--stage",
        default="o_s1_lexical_index_experiments",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    ks = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not ks:
        raise argparse.ArgumentTypeError("eval-k must contain at least one integer.")
    return ks


def limit_dict(data: dict[str, Any], max_items: int) -> dict[str, Any]:
    if max_items <= 0 or len(data) <= max_items:
        return dict(data)
    head = list(data.items())[:max_items]
    return {k: v for k, v in head}


def top_indices_with_fallback(row, top_k: int, fallback_count: int) -> list[int]:
    if top_k <= 0 or fallback_count <= 0:
        return []

    if row.nnz == 0:
        return list(range(min(top_k, fallback_count)))

    indices = np.asarray(row.indices, dtype=np.int64)
    scores = np.asarray(row.data, dtype=np.float32)
    order = np.lexsort((indices, -scores))  # score desc, index asc for deterministic ties.
    ranked = indices[order]
    selected = ranked[:min(top_k, ranked.size)].tolist()

    if len(selected) < top_k:
        in_selected = set(selected)
        needed = top_k - len(selected)
        for idx in range(fallback_count):
            if idx not in in_selected:
                selected.append(idx)
                needed -= 1
                if needed <= 0:
                    break

    return selected[:top_k]


def build_candidates(claims: dict[str, Any], evidence: dict[str, str], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    if args.method == "bm25":
        evidence_ids, similarities = build_bm25_similarities(
            claims=claims,
            evidence=evidence,
            max_features=args.max_features,
            k1=args.k1,
            b=args.b,
        )
    else:
        evidence_ids, similarities = build_tfidf_similarities(
            claims=claims,
            evidence=evidence,
            max_features=args.max_features,
        )

    pool: dict[str, list[dict[str, Any]]] = {}
    for row_idx, claim_id in enumerate(claims.keys()):
        row = similarities.getrow(row_idx)
        selected_indexes = top_indices_with_fallback(
            row=row,
            top_k=args.candidate_k,
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


def evaluate_recall_at_k(claims: dict[str, Any], pool: dict[str, list[dict[str, Any]]], ks: Iterable[int]) -> dict[str, Any]:
    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    k_lookup = {k: {"tp": 0, "gold": 0, "label_gold_counts": {}} for k in ks}
    hit_counts = {k: 0 for k in ks}
    label_hit_counts = {k: {} for k in ks}

    claim_recall_by_k = {k: [] for k in ks}
    claim_count = 0
    for claim_id, claim in claims.items():
        gold = claim.get("evidences", [])
        if not gold:
            continue
        claim_count += 1
        gold_set = set(gold)
        label = claim.get("claim_label", "UNLABELED")
        ranked = pool.get(claim_id, [])
        ranked_ids = [candidate["evidence_id"] for candidate in ranked]

        for k in ks:
            predicted = set(ranked_ids[:k])
            tp = len(gold_set & predicted)
            k_lookup[k]["tp"] += tp
            k_lookup[k]["gold"] += len(gold_set)
            if tp > 0:
                hit_counts[k] += 1
                label_hit_counts[k][label] = label_hit_counts[k].get(label, 0) + 1

            k_lookup[k]["label_gold_counts"][label] = k_lookup[k]["label_gold_counts"].get(label, 0) + 1
            recall = tp / len(gold_set)
            claim_recall_by_k[k].append(recall)

    results: dict[str, Any] = {
        "claims_with_evidence": claim_count,
        "evaluated_ks": ks,
    }
    for k in ks:
        total_tp = float(k_lookup[k]["tp"])
        total_gold = float(k_lookup[k]["gold"])
        recalls = claim_recall_by_k[k]
        results[f"macro_recall_at_{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        results[f"hit_any_at_{k}"] = hit_counts[k] / claim_count if claim_count else 0.0
        for label, label_gold_total in k_lookup[k]["label_gold_counts"].items():
            recall_hits = label_hit_counts[k].get(label, 0)
            if label_gold_total:
                results[f"{label.lower()}_hit_any_at_{k}"] = recall_hits / label_gold_total
    return results


def _build_inferred_split_name(path: Path) -> str:
    if str(path) == "data/train-claims.json":
        return "train"
    if str(path) == "data/dev-claims.json":
        return "dev"
    if str(path) == "data/test-claims-unlabelled.json":
        return "test"
    return "claims"


def _is_allowed_claim_path(claims_path: Path, evidence_path: Path) -> None:
    if claims_path == Path("data/test-claims-unlabelled.json"):
        raise SystemExit(
            "Do not inspect test labels in strict candidate generation. "
            "Use a separate prediction-only mode only."
        )
    if claims_path not in {Path("data/train-claims.json"), Path("data/dev-claims.json")}:
        raise SystemExit(
            "Claims file must be data/train-claims.json or data/dev-claims.json for this sparse baseline run."
        )
    if evidence_path != Path("data/evidence.json"):
        raise SystemExit(
            "Evidence path must be data/evidence.json for this strict candidate run."
        )


def _input_records(input_paths: list[Path], split_name: str) -> list[dict[str, Any]]:
    records = []
    for path in input_paths:
        if path.name == "train-claims.json":
            labels_used = True
            split = "train"
        elif path.name == "dev-claims.json":
            labels_used = True
            split = "dev"
        elif path.name == "test-claims-unlabelled.json":
            labels_used = False
            split = "test"
        elif path.name == "evidence.json":
            labels_used = False
            split = "evidence"
        else:
            labels_used = split_name not in {"train", "dev", "test"}
            split = split_name
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "split": split,
                "labels_used": labels_used,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_command = "python round18/experiments/o_sparse/o_s1_lexical_index_experiments/run_lexical_index_experiment.py"

    _is_allowed_claim_path(args.claims, args.evidence)

    if args.smoke:
        if args.max_claims <= 0:
            args.max_claims = 64
        if args.max_evidence <= 0:
            args.max_evidence = 120_000

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)

    if not isinstance(claims, dict) or not isinstance(evidence, dict):
        raise SystemExit("Claims and evidence inputs must be dictionaries keyed by id.")

    claims = limit_dict(claims, args.max_claims)
    evidence = limit_dict(evidence, args.max_evidence)

    split_name = _build_inferred_split_name(args.claims)
    eval_ks = parse_k_list(args.eval_k)
    max_eval_k = max(eval_ks)
    if max_eval_k > args.candidate_k:
        # Keep candidate pool large enough to support requested recall positions.
        args.candidate_k = max_eval_k

    # Keep manifest and reproducibility trail honest:
    input_scan_targets = [
        "data/train-claims.json",
        "data/dev-claims.json",
        "data/evidence.json",
        str(args.claims),
        str(args.evidence),
    ]
    forbidden_hits = find_forbidden_tokens(input_scan_targets)

    input_files = _input_records([args.claims, args.evidence], split_name)
    # If claim file is train, make sure split is marked train.
    for item in input_files:
        if item["path"].endswith("train-claims.json"):
            item["split"] = "train"
            item["labels_used"] = True

    pool, evidence_ids = build_candidates(claims=claims, evidence=evidence, args=args)
    metrics = evaluate_recall_at_k(claims=claims, pool=pool, ks=eval_ks)
    metrics["method"] = args.method
    metrics["candidate_k"] = args.candidate_k
    metrics["candidate_pool_size"] = len(pool)
    metrics["evidence_items_available"] = len(evidence_ids)
    metrics["max_features"] = args.max_features
    metrics["k1"] = args.k1
    metrics["b"] = args.b
    metrics["smoke"] = args.smoke
    metrics["max_claims"] = args.max_claims
    metrics["max_evidence"] = args.max_evidence
    metrics["random_seed"] = args.random_seed

    output_files: list[str] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_output = args.output_dir / f"{split_name}_{args.run_id}_{args.method}_top{args.candidate_k}_candidates.json"
    write_json(candidate_output, pool)
    output_files.append(str(candidate_output))

    metrics_output = args.output_dir / f"{split_name}_{args.run_id}_metrics.json"
    write_json(metrics_output, metrics)
    output_files.append(str(metrics_output))

    command = (
        f"{base_command} --claims {args.claims} --evidence {args.evidence} "
        f"--method {args.method} --candidate-k {args.candidate_k} --eval-k {','.join(map(str, eval_ks))} "
        f"--max-features {args.max_features} --k1 {args.k1} --b {args.b} "
        f"--stage {args.stage} --run-id {args.run_id} "
        f"{'--smoke' if args.smoke else ''} "
        f"{'--max-claims ' + str(args.max_claims) if args.max_claims > 0 else ''} "
        f"{'--max-evidence ' + str(args.max_evidence) if args.max_evidence > 0 else ''}"
    ).strip()

    record = {
        "run_id": args.run_id,
        "command": command,
        "command_args": {k: str(v) for k, v in vars(args).items()},
        "mode": "STRICT",
        "stage": args.stage,
        "split": split_name,
        "smoke": args.smoke,
        "forbidden_hits": forbidden_hits,
        "files_read": [str(args.claims), str(args.evidence)],
        "files_written": output_files + [str(args.manifest), str(args.record_path)],
    }
    if args.max_claims > 0:
        record["subset_claim_count"] = len(claims)
    if args.max_evidence > 0:
        record["subset_evidence_count"] = len(evidence)
    write_json(args.record_path, record)
    output_files.append(str(args.record_path))

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate",
        mode="STRICT",
        stage=args.stage,
        command=command,
        working_directory=Path.cwd(),
        config_path="",
        config_hash="",
        random_seed=args.random_seed,
        cv_seed=None,
    )
    manifest["input_files"] = input_files
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan against forbidden strict-mode tokens passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = output_files + [str(args.manifest)]
    manifest["metrics"] = {
        "recall": metrics,
        "command": command,
        "candidate_file": str(candidate_output),
        "candidate_top_k": args.candidate_k,
        "claims_input_count": len(claims),
        "evidence_input_count": len(evidence_ids),
        "max_claims": args.max_claims,
        "max_evidence": args.max_evidence,
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Loaded allowed raw JSON inputs, built a deterministic sparse retrieval index, "
        "wrote raw candidate pool and recall metrics."
    )
    manifest["split_isolation_summary"] = (
        "No test labels were loaded. Train labels are not required for this run. "
        "Only claim/evidence labels for the selected split are used for raw recall reporting."
    )
    if args.smoke:
        manifest["notes"] = (
            "This run used bounded smoke caps: max-claims/evidence were constrained. "
            "Metrics are diagnostic-scope with respect to the capped slice."
        )
        manifest["leakage_risk"] = "low"
        manifest["reproducibility_risk"] = "low"
    else:
        manifest["notes"] = "Full corpus run using fixed algorithmic configuration; no stochastic components."
        manifest["leakage_risk"] = "low"
        manifest["reproducibility_risk"] = "low"

    write_json(args.manifest, manifest)
    output_files.append(str(args.manifest))

    print(f"Wrote candidate pool: {candidate_output}")
    print(f"Wrote metrics: {metrics_output}")
    print(f"Wrote record: {args.record_path}")
    print(f"Wrote manifest: {args.manifest}")
    for key, value in sorted(metrics.items()):
        if key.endswith(f"@100") or key.endswith(f"@500") or key.endswith("evaluated_ks") or key.startswith("claims_"):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
