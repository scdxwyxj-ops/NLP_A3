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

from experiments.rerank.round16_branch_a_requirement_selector import (
    FEATURES,
    grouped_rows,
    matrix_from_rows,
    rrf_fuse_ranked,
    stream_ranked_predictions,
)
from round18.tools.common import (
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_TRAIN_BM25 = Path(
    "round18/outputs/o_sparse/o_s1_lexical_index_experiments/"
    "train_train_full_bm25_bm25_top500_candidates.json"
)
DEFAULT_TRAIN_CHAR = Path(
    "round18/outputs/o_sparse/o_s6_char_tfidf/"
    "train_full_train_train_full_o_s6_char_tfidf_tfidf_char_top500_candidates.json"
)
DEFAULT_DEV_POOL = Path(
    "round18/outputs/o_sparse/o_s7_plain_leaf_fusion/"
    "dev_full_dev_o_s7_plain_leaf_fusion_strict_rrf_char_heavy_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_sparse/o_s8_hand_feature_ranker")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "run_manifest.json"
DEFAULT_RECORD = DEFAULT_OUTPUT_DIR / "run_record.json"
DEFAULT_RUN_ID = "o_s8_hand_feature_ranker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-S8 train-only LightGBM hand-feature reranker over the O-S7 strict top500 leaf."
        )
    )
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-bm25-pool", type=Path, default=DEFAULT_TRAIN_BM25)
    parser.add_argument("--train-char-pool", type=Path, default=DEFAULT_TRAIN_CHAR)
    parser.add_argument("--dev-pool", type=Path, default=DEFAULT_DEV_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record-path", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--train-pool-limit", type=int, default=500)
    parser.add_argument("--eval-k", default="1,3,5,10,64,100,500")
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--char-weight", type=float, default=1.5)
    parser.add_argument("--random-seed", type=int, default=1808)
    parser.add_argument("--stage", default="o_s8_hand_feature_ranker")
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--eval-k must contain at least one integer.")
    return sorted(set(values))


def enforce_contract(args: argparse.Namespace) -> None:
    required = {
        "train_claims": (args.train_claims, DEFAULT_TRAIN_CLAIMS),
        "dev_claims": (args.dev_claims, DEFAULT_DEV_CLAIMS),
        "evidence": (args.evidence, DEFAULT_EVIDENCE),
        "train_bm25_pool": (args.train_bm25_pool, DEFAULT_TRAIN_BM25),
        "train_char_pool": (args.train_char_pool, DEFAULT_TRAIN_CHAR),
        "dev_pool": (args.dev_pool, DEFAULT_DEV_POOL),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise SystemExit(f"O-S8 strict run requires {name}={expected}, got {actual}.")
        if not actual.exists():
            raise SystemExit(f"Missing O-S8 strict input {name}: {actual}")
    if "round18/outputs" not in str(args.output_dir):
        raise SystemExit("O-S8 must write under round18/outputs.")


def evaluate_recall_at_k(claims: dict[str, Any], ranked: dict[str, list[dict[str, Any]]], ks: list[int]) -> dict[str, Any]:
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

    total_candidates = 0
    union_ids: set[str] = set()
    claim_count = 0
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claim_count += 1
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

    result["claims_with_evidence"] = claim_count
    result["avg_candidate_count"] = float(total_candidates / claim_count) if claim_count else 0.0
    result["union_candidates"] = len(union_ids)
    for k in ks:
        recalls = per_k_claim_recalls[k]
        result[f"macro_recall_at_{k}"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
        result[f"micro_recall_at_{k}"] = per_k_tp[k] / per_k_gold[k] if per_k_gold[k] else 0.0
        result[f"hit_any_at_{k}"] = per_k_hit[k] / claim_count if claim_count else 0.0
    return result


def input_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "path": str(args.train_claims),
            "sha256": sha256_file(args.train_claims),
            "split": "train",
            "labels_used": True,
        },
        {
            "path": str(args.dev_claims),
            "sha256": sha256_file(args.dev_claims),
            "split": "dev",
            "labels_used": True,
        },
        {
            "path": str(args.evidence),
            "sha256": sha256_file(args.evidence),
            "split": "evidence",
            "labels_used": False,
        },
        {
            "path": str(args.train_bm25_pool),
            "sha256": sha256_file(args.train_bm25_pool),
            "split": "current_run_train_artifact",
            "labels_used": False,
        },
        {
            "path": str(args.train_char_pool),
            "sha256": sha256_file(args.train_char_pool),
            "split": "current_run_train_artifact",
            "labels_used": False,
        },
        {
            "path": str(args.dev_pool),
            "sha256": sha256_file(args.dev_pool),
            "split": "current_run_dev_artifact",
            "labels_used": False,
        },
    ]


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    enforce_contract(args)
    eval_ks = parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        args.candidate_k = max(eval_ks)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_bm25 = load_json(args.train_bm25_pool)
    train_char = load_json(args.train_char_pool)
    dev_pool = load_json(args.dev_pool)
    for name, payload in {
        "train_claims": train_claims,
        "dev_claims": dev_claims,
        "evidence": evidence,
        "train_bm25": train_bm25,
        "train_char": train_char,
        "dev_pool": dev_pool,
    }.items():
        if not isinstance(payload, dict):
            raise SystemExit(f"{name} must be a JSON object.")

    train_pool = rrf_fuse_ranked(
        {"bm25": train_bm25, "char": train_char},
        {"bm25": args.bm25_weight, "char": args.char_weight},
        rrf_k=args.rrf_k,
    )
    train_rows, train_y, train_groups, _, _ = grouped_rows(
        claims=train_claims,
        evidence=evidence,
        pool=train_pool,
        pool_limit=args.train_pool_limit,
        source_indexes={},
        include_gold=True,
    )
    train_x = matrix_from_rows(train_rows)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=260,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=12,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_seed,
        n_jobs=-1,
        verbose=-1,
    )
    weights = np.ones(len(train_y), dtype=np.float32)
    weights[train_y == 1] = 8.0
    ranker.fit(train_x, train_y, group=train_groups, sample_weight=weights)

    ranked, dev_positive_rows = stream_ranked_predictions(
        claims=dev_claims,
        evidence=evidence,
        pool=dev_pool,
        pool_limit=args.candidate_k,
        source_indexes={},
        ranker=ranker,
    )
    metrics = evaluate_recall_at_k(dev_claims, ranked, eval_ks)
    metrics.update(
        {
            "train_rows": len(train_rows),
            "train_positive_rows": int(train_y.sum()),
            "train_groups": len(train_groups),
            "dev_positive_rows_in_pool": int(dev_positive_rows),
            "candidate_top_k": args.candidate_k,
            "train_pool_limit": args.train_pool_limit,
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
    train_pool_path = args.output_dir / f"train_full_{args.run_id}_rrf_bm25_char_top{args.train_pool_limit}_pool.json"
    data_flow_path = args.output_dir / f"dev_full_dev_{args.run_id}_data_flow_report.json"

    write_json(candidate_path, ranked)
    write_json(metrics_path, metrics)
    write_json(train_pool_path, {claim_id: rows[: args.train_pool_limit] for claim_id, rows in train_pool.items()})

    command = (
        "python round18/experiments/o_sparse/o_s8_hand_feature_ranker/run_o_s8_hand_feature_ranker.py "
        f"--train-claims {args.train_claims} --dev-claims {args.dev_claims} --evidence {args.evidence} "
        f"--train-bm25-pool {args.train_bm25_pool} --train-char-pool {args.train_char_pool} "
        f"--dev-pool {args.dev_pool} --candidate-k {args.candidate_k} --train-pool-limit {args.train_pool_limit} "
        f"--eval-k {','.join(map(str, eval_ks))} --rrf-k {args.rrf_k} "
        f"--bm25-weight {args.bm25_weight} --char-weight {args.char_weight} --stage {args.stage}"
    )
    forbidden_hits = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.evidence),
            str(args.train_bm25_pool),
            str(args.train_char_pool),
            str(args.dev_pool),
            str(args.output_dir),
            command,
        ]
    )

    data_flow = {
        "pipeline": "train-only hand-feature LambdaRank leaf over strict O-S7 dev pool",
        "inputs": {
            "train_claims": str(args.train_claims),
            "dev_claims": str(args.dev_claims),
            "evidence": str(args.evidence),
            "train_bm25_pool": str(args.train_bm25_pool),
            "train_char_pool": str(args.train_char_pool),
            "dev_pool": str(args.dev_pool),
        },
        "processing": {
            "train_pool": "fixed RRF over train BM25 and train char leaves",
            "train_labels": "train evidences only",
            "dev_labels_used_for_training": False,
            "candidate_k": args.candidate_k,
            "features": "rank, lexical overlap, claim-key coverage, entity/number/year/logic cues, length",
        },
        "outputs": {
            "candidate_pool": str(candidate_path),
            "metrics": str(metrics_path),
            "train_pool": str(train_pool_path),
        },
        "notes": [
            "This is a plain leaf reranker; it does not consume old Round14/Round16 artifacts.",
            "Dev labels are used only for evaluation metrics, not for model fitting.",
        ],
    }
    write_json(data_flow_path, data_flow)

    record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT",
        "status": "strict-candidate",
        "command": command,
        "forbidden_hits": forbidden_hits,
        "files_written": [
            str(candidate_path),
            str(metrics_path),
            str(train_pool_path),
            str(data_flow_path),
            str(args.record_path),
            str(args.manifest),
        ],
    }
    write_json(args.record_path, record)

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
    manifest["input_files"] = input_records(args)
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan against forbidden strict tokens passed."
        if not forbidden_hits
        else str(forbidden_hits),
    }
    manifest["output_files"] = [
        str(candidate_path),
        str(metrics_path),
        str(train_pool_path),
        str(data_flow_path),
        str(args.record_path),
        str(args.manifest),
    ]
    manifest["metrics"] = {
        "metrics_file": str(metrics_path),
        "macro_recall_at_100": metrics.get("macro_recall_at_100"),
        "macro_recall_at_500": metrics.get("macro_recall_at_500"),
        "micro_recall_at_100": metrics.get("micro_recall_at_100"),
        "micro_recall_at_500": metrics.get("micro_recall_at_500"),
        "hit_any_at_100": metrics.get("hit_any_at_100"),
        "hit_any_at_500": metrics.get("hit_any_at_500"),
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Built a fixed train BM25+char RRF pool from current Round18 train leaves, "
        "trained hand-feature LambdaRank on train gold labels, and reranked the O-S7 strict dev top500 pool."
    )
    manifest["split_isolation_summary"] = (
        "Training uses train claims/evidence labels only. Dev labels are used only in metrics."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "medium"
    manifest["notes"] = (
        "This reproduces the old Branch-A hand-feature idea as a Round18 leaf without old artifact inputs."
    )
    write_json(args.manifest, manifest)

    print(f"Wrote candidate pool: {candidate_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"macro_recall_at_100: {metrics.get('macro_recall_at_100')}")
    print(f"macro_recall_at_500: {metrics.get('macro_recall_at_500')}")


if __name__ == "__main__":
    main()
