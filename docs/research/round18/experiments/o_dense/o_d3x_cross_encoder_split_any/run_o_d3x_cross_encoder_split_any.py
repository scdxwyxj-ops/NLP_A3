from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.experiments.o_dense.o_d3b_cross_encoder_s7 import (  # noqa: E402
    run_o_d3b_cross_encoder_s7 as base,
)
from round18.tools.common import (  # noqa: E402
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


DEFAULT_CLAIMS = Path("data/train-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_SPARSE_POOL = Path(
    "round18/outputs/o_sparse/o_s9_union_gate_bm25_char_train/"
    "train_full_train_o_s9_union_gate_bm25_char_rrf_top1000_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_dense/o_d3x_cross_encoder_split_any")
DEFAULT_RUN_ID = "o_d3x_cross_encoder_split_any"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 split-aware CE/MiniLM scorer for train/dev candidate pools. "
            "This is for train-only tuning experiments; it does not select hyperparameters on dev."
        )
    )
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--sparse-pool", type=Path, default=DEFAULT_SPARSE_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "run_manifest.json")
    parser.add_argument("--record-path", type=Path, default=DEFAULT_OUTPUT_DIR / "run_record.json")
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--eval-k", default="1,3,5,10,64,100,500")
    parser.add_argument("--model-name", default=base.DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-claims", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1818)
    parser.add_argument("--stage", default="o_d3x_cross_encoder_split_any")
    return parser.parse_args()


def enforce_inputs(args: argparse.Namespace) -> None:
    if args.claims not in {Path("data/train-claims.json"), Path("data/dev-claims.json")}:
        raise SystemExit("claims must be data/train-claims.json or data/dev-claims.json.")
    if args.evidence != DEFAULT_EVIDENCE:
        raise SystemExit("evidence must be data/evidence.json.")
    if not args.sparse_pool.exists():
        raise SystemExit(f"Missing sparse pool: {args.sparse_pool}")
    if not str(args.sparse_pool).startswith("round18/outputs/"):
        raise SystemExit("sparse pool must be a Round18 output artifact.")
    if "round18/outputs" not in str(args.output_dir):
        raise SystemExit("output-dir must be under round18/outputs.")
    if args.candidate_k <= 0:
        raise SystemExit("--candidate-k must be positive.")


def input_records(args: argparse.Namespace, split_name: str) -> list[dict[str, Any]]:
    return [
        {
            "path": str(args.claims),
            "sha256": sha256_file(args.claims),
            "split": split_name,
            "labels_used": False,
        },
        {
            "path": str(args.evidence),
            "sha256": sha256_file(args.evidence),
            "split": "evidence",
            "labels_used": False,
        },
        {
            "path": str(args.sparse_pool),
            "sha256": sha256_file(args.sparse_pool),
            "split": f"{split_name}_candidate_artifact",
            "labels_used": False,
        },
    ]


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    enforce_inputs(args)

    eval_ks = base.parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        args.candidate_k = max(eval_ks)

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    if not isinstance(claims, dict):
        raise SystemExit("claims must be a JSON object.")
    if not isinstance(evidence, dict):
        raise SystemExit("evidence must be a JSON object.")
    claims = base.subset_claims(claims, args.max_claims if args.max_claims > 0 else 0)

    sparse_pool = base.parse_sparse_pool(args.sparse_pool, args.candidate_k)
    sparse_pool, missing_claims = base.build_sparse_to_evidence_inputs(claims, sparse_pool)
    if not sparse_pool:
        raise SystemExit("No sparse candidates loaded for selected claims.")

    needed_evidence_ids = {
        evidence_id for rows in sparse_pool.values() for evidence_id, _ in rows
    }
    evidence_text = {eid: evidence[eid] for eid in needed_evidence_ids if eid in evidence}
    missing_evidence = len(needed_evidence_ids) - len(evidence_text)

    ranked_pool, reranker_type, reranker_info = base.rerank_with_cross_encoder(
        candidates=sparse_pool,
        claims=claims,
        evidence_text=evidence_text,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    for rows in ranked_pool.values():
        for idx, item in enumerate(rows, start=1):
            item["rank"] = idx

    metrics = base.evaluate_recall_at_k(claims=claims, pool=ranked_pool, eval_ks=eval_ks)
    metrics.update(
        {
            "candidate_top_k": args.candidate_k,
            "claims_input_count": len(claims),
            "claims_with_sparse_candidates": len(ranked_pool),
            "claims_with_source_evidence_text": len(evidence_text),
            "sparse_missing_claims": missing_claims,
            "evidence_missing_count": missing_evidence,
            "reranker_type": reranker_type,
            "reranker_info": reranker_info,
            "evidence_available_total": len(evidence),
            "sparse_candidate_count": sum(len(v) for v in sparse_pool.values()),
            "reranked_candidate_count": sum(len(v) for v in ranked_pool.values()),
        }
    )

    split_name = base.infer_split_name(args.claims)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{split_name}_full_{split_name}_{args.run_id}_strict_top{args.candidate_k}"
    candidate_path = args.output_dir / f"{prefix}_candidates.json"
    metrics_path = args.output_dir / f"{prefix}_metrics.json"
    data_flow_path = args.output_dir / f"{prefix}_data_flow_report.json"

    write_json(candidate_path, ranked_pool)
    write_json(metrics_path, metrics)

    command = (
        "python round18/experiments/o_dense/o_d3x_cross_encoder_split_any/"
        "run_o_d3x_cross_encoder_split_any.py "
        f"--claims {args.claims} --evidence {args.evidence} --sparse-pool {args.sparse_pool} "
        f"--output-dir {args.output_dir} --run-id {args.run_id} --candidate-k {args.candidate_k} "
        f"--eval-k {','.join(map(str, eval_ks))} --model-name {args.model_name} "
        f"--batch-size {args.batch_size} --max-length {args.max_length} --device {args.device}"
    )
    if args.max_claims:
        command += f" --max-claims {args.max_claims}"

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.claims),
            str(args.evidence),
            str(args.sparse_pool),
            str(args.output_dir),
            command,
        ]
    )

    data_flow = {
        "pipeline": "split-aware CE/MiniLM scoring over fixed candidate pool",
        "inputs": {
            "claims": str(args.claims),
            "evidence": str(args.evidence),
            "sparse_pool": str(args.sparse_pool),
        },
        "processing": {
            "labels_used_for_scoring": False,
            "candidate_k": args.candidate_k,
            "model_name": args.model_name,
            "reranker_type": reranker_type,
        },
        "outputs": {
            "candidate_pool": str(candidate_path),
            "metrics": str(metrics_path),
        },
        "notes": [
            "Claims labels are used only by the metrics block for labeled splits.",
            "No hyperparameters are selected here.",
        ],
    }
    write_json(data_flow_path, data_flow)

    record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT" if split_name == "train" else "DEV_CONFIRMATION",
        "status": "score-artifact",
        "command": command,
        "forbidden_hits": forbidden_hits,
        "files_written": [
            str(candidate_path),
            str(metrics_path),
            str(data_flow_path),
            str(args.record_path),
            str(args.manifest),
        ],
    }
    write_json(args.record_path, record)

    manifest = manifest_base(
        run_id=args.run_id,
        status="score-artifact",
        mode="STRICT" if split_name == "train" else "DEV_CONFIRMATION",
        stage=args.stage,
        command=command,
        working_directory=Path.cwd(),
        config_path=str(args.manifest),
        config_hash="",
        random_seed=args.random_seed,
        cv_seed=None,
    )
    manifest["input_files"] = input_records(args, split_name)
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan passed." if not forbidden_hits else str(forbidden_hits),
    }
    manifest["output_files"] = record["files_written"][:3]
    manifest["metrics"] = {
        "metrics_file": str(metrics_path),
        "macro_recall_at_64": metrics.get("macro_recall_at_64"),
        "macro_recall_at_500": metrics.get("macro_recall_at_500"),
        "hit_any_at_64": metrics.get("hit_any_at_64"),
        "reranker_type": reranker_type,
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = reranker_info.get("device", args.device)
    manifest["data_flow_summary"] = (
        f"Scored a fixed {split_name} candidate pool with CE/MiniLM. "
        "No labels or dev-selected hyperparameters are used for scoring."
    )
    manifest["split_isolation_summary"] = (
        "This script only scores candidates. Train labels are reserved for downstream k-fold tuning; "
        "dev labels are reserved for final confirmation metrics."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "medium"
    write_json(args.manifest, manifest)

    print(f"Wrote candidate pool: {candidate_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"reranker_type: {reranker_type}")
    print(f"macro_recall_at_64: {metrics.get('macro_recall_at_64')}")


if __name__ == "__main__":
    main()
