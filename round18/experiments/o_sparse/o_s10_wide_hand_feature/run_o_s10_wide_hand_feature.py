#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rerank.round16_branch_a_requirement_selector import (  # noqa: E402
    FEATURES,
    grouped_rows,
    matrix_from_rows,
    stream_ranked_predictions,
)
from round18.tools.common import (  # noqa: E402
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round18 O-S10 train-only hand-feature compressor over explicit wide pools."
    )
    parser.add_argument("--train-claims", type=Path, default=Path("data/train-claims.json"))
    parser.add_argument("--dev-claims", type=Path, default=Path("data/dev-claims.json"))
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence.json"))
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--dev-pool", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s10_wide_hand_feature"),
    )
    parser.add_argument("--run-id", default="o_s10_wide_hand_feature")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s10_wide_hand_feature/run_manifest.json"),
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s10_wide_hand_feature/run_record.json"),
    )
    parser.add_argument("--train-pool-limit", type=int, default=1000)
    parser.add_argument("--dev-pool-limit", type=int, default=1000)
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--eval-k", default="3,10,64,100,500")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--random-seed", type=int, default=1810)
    parser.add_argument("--stage", default="o_s10_wide_hand_feature")
    parser.add_argument(
        "--mode",
        choices=("STRICT", "DIAGNOSTIC"),
        default="STRICT",
        help="STRICT requires allowed train/dev raw claims and no forbidden input tokens.",
    )
    parser.add_argument(
        "--status",
        choices=("strict-candidate", "diagnostic-only", "rejected"),
        default="strict-candidate",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--eval-k must contain at least one integer.")
    return sorted(set(values))


def infer_split(path: Path) -> str:
    if path == Path("data/train-claims.json"):
        return "train"
    if path == Path("data/dev-claims.json"):
        return "dev"
    if path == Path("data/evidence.json"):
        return "evidence"
    return "current_run_artifact" if "round18/outputs" in str(path) else "input"


def validate_contract(args: argparse.Namespace) -> None:
    if args.train_claims != Path("data/train-claims.json"):
        raise SystemExit("O-S10 requires --train-claims data/train-claims.json")
    if args.dev_claims != Path("data/dev-claims.json"):
        raise SystemExit("O-S10 requires --dev-claims data/dev-claims.json")
    if args.evidence != Path("data/evidence.json"):
        raise SystemExit("O-S10 requires --evidence data/evidence.json")
    for path in (args.train_pool, args.dev_pool):
        if not path.exists():
            raise SystemExit(f"Missing pool input: {path}")
        if "round18/outputs" not in str(path):
            raise SystemExit(f"O-S10 pool inputs must be Round18 outputs: {path}")
    if "round18/outputs" not in str(args.output_dir):
        raise SystemExit("O-S10 must write under round18/outputs.")
    if args.mode == "STRICT" and args.status != "strict-candidate":
        raise SystemExit("STRICT mode requires status=strict-candidate.")


def evaluate_recall_at_k(
    claims: dict[str, Any],
    ranked: dict[str, list[dict[str, Any]]],
    ks: list[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "claims_with_evidence": 0,
        "evaluated_ks": ks,
        "avg_candidate_count": 0.0,
        "union_candidates": 0,
    }
    per_k_claim_recalls = {k: [] for k in ks}
    per_k_tp = {k: 0.0 for k in ks}
    per_k_gold = {k: 0.0 for k in ks}
    per_k_hit = {k: 0 for k in ks}
    label_totals: dict[str, int] = {}
    label_hit_counts: dict[int, dict[str, int]] = {k: {} for k in ks}

    total_candidates = 0
    union_ids: set[str] = set()
    claim_count = 0
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claim_count += 1
        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0) + 1
        rows = ranked.get(claim_id, [])
        ids = [row["evidence_id"] for row in rows]
        total_candidates += len(ids)
        union_ids.update(ids)
        for k in ks:
            predicted = set(ids[:k])
            tp = float(len(gold & predicted))
            per_k_tp[k] += tp
            per_k_gold[k] += float(len(gold))
            per_k_claim_recalls[k].append(tp / float(len(gold)))
            if tp > 0:
                per_k_hit[k] += 1
                label_hit_counts[k][label] = label_hit_counts[k].get(label, 0) + 1

    result["claims_with_evidence"] = claim_count
    result["avg_candidate_count"] = float(total_candidates / claim_count) if claim_count else 0.0
    result["union_candidates"] = len(union_ids)
    for k in ks:
        recalls = per_k_claim_recalls[k]
        result[f"macro_recall_at_{k}"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
        result[f"micro_recall_at_{k}"] = per_k_tp[k] / per_k_gold[k] if per_k_gold[k] else 0.0
        result[f"hit_any_at_{k}"] = per_k_hit[k] / claim_count if claim_count else 0.0
        for label, total in label_totals.items():
            if total:
                result[f"{label.lower()}_hit_any_at_{k}"] = label_hit_counts[k].get(label, 0) / total
    return result


def input_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for path in (
        args.train_claims,
        args.dev_claims,
        args.evidence,
        args.train_pool,
        args.dev_pool,
    ):
        split = infer_split(path)
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "split": split,
                "labels_used": split in {"train", "dev"},
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    validate_contract(args)

    eval_ks = parse_k_list(args.eval_k)
    args.candidate_k = max(args.candidate_k, max(eval_ks))

    command = " ".join(sys.argv)
    forbidden_hits = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.evidence),
            str(args.train_pool),
            str(args.dev_pool),
            str(args.output_dir),
            command,
        ]
    )
    if args.mode == "STRICT" and forbidden_hits:
        raise SystemExit(f"STRICT GUARD FAILED: {forbidden_hits}")

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pool = load_json(args.train_pool)
    dev_pool = load_json(args.dev_pool)
    for name, payload in {
        "train_claims": train_claims,
        "dev_claims": dev_claims,
        "evidence": evidence,
        "train_pool": train_pool,
        "dev_pool": dev_pool,
    }.items():
        if not isinstance(payload, dict):
            raise SystemExit(f"{name} must be a JSON object.")

    train_rows, train_y, train_groups, _, _ = grouped_rows(
        claims=train_claims,
        evidence=evidence,
        pool=train_pool,
        pool_limit=args.train_pool_limit,
        source_indexes={},
        include_gold=True,
    )
    if not train_rows:
        raise SystemExit("No train rows available for O-S10.")
    train_x = matrix_from_rows(train_rows)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=12,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_seed,
        n_jobs=-1,
        verbose=-1,
    )
    weights = np.ones(len(train_y), dtype=np.float32)
    weights[train_y == 1] = args.positive_weight
    ranker.fit(train_x, train_y, group=train_groups, sample_weight=weights)

    ranked, dev_positive_rows = stream_ranked_predictions(
        claims=dev_claims,
        evidence=evidence,
        pool=dev_pool,
        pool_limit=args.dev_pool_limit,
        source_indexes={},
        ranker=ranker,
    )
    for claim_id, rows in ranked.items():
        ranked[claim_id] = rows[: args.candidate_k]
        for rank, item in enumerate(ranked[claim_id], start=1):
            item["rank"] = rank

    metrics = evaluate_recall_at_k(dev_claims, ranked, eval_ks)
    metrics.update(
        {
            "train_rows": len(train_rows),
            "train_positive_rows": int(train_y.sum()),
            "train_groups": len(train_groups),
            "dev_positive_rows_in_pool": int(dev_positive_rows),
            "candidate_top_k": args.candidate_k,
            "train_pool_limit": args.train_pool_limit,
            "dev_pool_limit": args.dev_pool_limit,
            "model": "LGBMRanker lambdarank",
            "feature_names": FEATURES,
            "feature_importance": [
                {"feature": name, "importance": float(value)}
                for name, value in sorted(
                    zip(FEATURES, ranker.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / f"dev_full_dev_{args.run_id}_top{args.candidate_k}_candidates.json"
    metrics_path = args.output_dir / f"dev_full_dev_{args.run_id}_metrics.json"
    data_flow_path = args.output_dir / f"dev_full_dev_{args.run_id}_data_flow_report.json"

    write_json(candidate_path, ranked)
    write_json(metrics_path, metrics)
    write_json(
        data_flow_path,
        {
            "pipeline": "O-S10 train-only hand-feature compressor over explicit wide pools.",
            "inputs": {
                "train_pool": str(args.train_pool),
                "dev_pool": str(args.dev_pool),
            },
            "processing": {
                "train_labels": "train evidences only",
                "dev_labels_used_for_training": False,
                "train_pool_limit": args.train_pool_limit,
                "dev_pool_limit": args.dev_pool_limit,
                "candidate_k": args.candidate_k,
            },
            "outputs": {
                "candidate_pool": str(candidate_path),
                "metrics": str(metrics_path),
            },
        },
    )

    manifest = manifest_base(
        run_id=args.run_id,
        status=args.status,
        mode=args.mode,
        stage=args.stage,
        command=command,
        working_directory=Path.cwd(),
        random_seed=args.random_seed,
    )
    manifest["input_files"] = input_records(args)
    manifest["forbidden_input_scan"] = {
        "passed": not bool(forbidden_hits),
        "notes": "Input path scan passed." if not forbidden_hits else str(forbidden_hits),
    }
    manifest["output_files"] = [
        str(candidate_path),
        str(metrics_path),
        str(data_flow_path),
        str(args.record_path),
        str(args.manifest),
    ]
    manifest["metrics"] = {
        "metrics_file": str(metrics_path),
        "macro_recall_at_64": metrics.get("macro_recall_at_64"),
        "macro_recall_at_100": metrics.get("macro_recall_at_100"),
        "macro_recall_at_500": metrics.get("macro_recall_at_500"),
        "hit_any_at_500": metrics.get("hit_any_at_500"),
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Trained a hand-feature LambdaRank compressor on train claims and explicit train pool, "
        "then reranked the explicit dev pool."
    )
    manifest["split_isolation_summary"] = (
        "Train labels are used for fitting only; dev labels are used only for post-hoc metrics."
    )
    manifest["leakage_risk"] = "low" if args.mode == "STRICT" else "medium"
    manifest["reproducibility_risk"] = "medium"
    manifest["notes"] = "Generalized O-S8 from a fixed top500 pool to explicit wide pools."
    write_json(args.manifest, manifest)

    record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": args.mode,
        "status": args.status,
        "command": command,
        "forbidden_hits": forbidden_hits,
        "files_written": manifest["output_files"],
        "metrics": manifest["metrics"],
    }
    write_json(args.record_path, record)

    print(f"Wrote candidate pool: {candidate_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"macro_recall_at_64: {metrics.get('macro_recall_at_64')}")
    print(f"macro_recall_at_500: {metrics.get('macro_recall_at_500')}")


if __name__ == "__main__":
    main()

