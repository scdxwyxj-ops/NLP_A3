#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.tools.common import (
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)

SourceRows = dict[str, list[dict[str, Any]]]


POLICIES = ("rrf", "round_robin", "priority", "union_upper")
SOURCE_ORDER = ("bm25", "char", "structured", "prf")


@dataclass(frozen=True)
class PolicyConfig:
    policy: str
    variant_id: str
    label: str
    status: str = "strict-candidate"


POLICY_ORDER = (
    PolicyConfig("rrf", "strict_rrf", "RRF"),
    PolicyConfig("round_robin", "strict_round_robin", "Round-robin"),
    PolicyConfig("priority", "strict_priority", "Priority"),
    PolicyConfig("union_upper", "strict_union_upper", "Union-upper"),
)


DEFAULT_TOP_K = 500
RRF_K_DEFAULT = 60

TRAIN_DEFAULT_PATHS = {
    "bm25": Path("round18/outputs/o_sparse/o_s1_lexical_index_experiments/train_train_full_bm25_bm25_top500_candidates.json"),
    "char": Path(
        "round18/outputs/o_sparse/o_s6_char_tfidf/"
        "train_full_train_train_full_o_s6_char_tfidf_tfidf_char_top500_candidates.json"
    ),
    "structured": Path("__absent_train_structured_pool__"),
    "prf": Path("__absent_train_prf_pool__"),
}
DEV_DEFAULT_PATHS = {
    "bm25": Path(
        "round18/outputs/o_sparse/o_s1_lexical_index_experiments/"
        "dev_full_bm25_dev_bm25_top500_candidates.json"
    ),
    "char": Path(
        "round18/outputs/o_sparse/o_s6_char_tfidf/"
        "dev_full_dev_o_s6_char_tfidf_tfidf_char_top500_candidates.json"
    ),
    "structured": Path("round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposed_candidates.json"),
    "prf": Path("round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round18 O-S9 strict/diagnostic union gate over leaf sparse pools."
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/dev-claims.json"),
        help="Claims split path (train or dev).",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("data/evidence.json"),
        help="Evidence corpus path.",
    )
    parser.add_argument(
        "--bm25-pool",
        type=Path,
        default=None,
        help="BM25 leaf pool path; defaults per split if omitted.",
    )
    parser.add_argument(
        "--char-pool",
        type=Path,
        default=None,
        help="Character TF-IDF leaf pool path; defaults per split if omitted.",
    )
    parser.add_argument(
        "--structured-pool",
        type=Path,
        default=None,
        help="Structured route leaf pool path; optional.",
    )
    parser.add_argument(
        "--prf-pool",
        type=Path,
        default=None,
        help="PRF leaf pool path; optional.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s9_union_gate"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-id",
        default="o_s9_union_gate",
        help="Run identifier used in filenames.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s9_union_gate/run_manifest.json"),
        help="Run manifest output path.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s9_union_gate/run_record.json"),
        help="Run record output path.",
    )
    parser.add_argument(
        "--eval-k",
        default="100,500",
        help="Comma-separated recall-k values.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Final per-claim candidate count to write.",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Per-source BM25 cap before fusion.",
    )
    parser.add_argument(
        "--char-top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Per-source char TF-IDF cap before fusion.",
    )
    parser.add_argument(
        "--structured-top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Per-source structured cap before fusion.",
    )
    parser.add_argument(
        "--prf-top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Per-source PRF cap before fusion.",
    )
    parser.add_argument(
        "--policies",
        default="rrf,round_robin,priority,union_upper",
        help="Comma-separated policies to materialize as strict-candidate outputs.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K_DEFAULT,
        help="RRF denominator term used by policy 'rrf'.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="BM25 source weight.",
    )
    parser.add_argument(
        "--char-weight",
        type=float,
        default=1.0,
        help="Char source weight.",
    )
    parser.add_argument(
        "--structured-weight",
        type=float,
        default=1.0,
        help="Structured source weight.",
    )
    parser.add_argument(
        "--prf-weight",
        type=float,
        default=1.0,
        help="PRF source weight.",
    )
    parser.add_argument(
        "--enable-diagnostic",
        action="store_true",
        default=True,
        help="Enable dev-only diagnostic best-policy selection.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Reproducibility marker.",
    )
    parser.add_argument(
        "--stage",
        default="o_s9_union_gate",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--eval-k must include at least one integer.")
    return sorted(set(values))


def _infer_split_name(claims_path: Path) -> str:
    filename = claims_path.name
    if filename == "train-claims.json":
        return "train"
    if filename == "dev-claims.json":
        return "dev"
    if filename == "test-claims-unlabelled.json":
        return "test"
    return "claims"


def _split_file_prefix(split_name: str) -> str:
    if split_name == "train":
        return "train_full_train"
    if split_name == "dev":
        return "dev_full_dev"
    if split_name == "test":
        return "test_full_test"
    return "claims_full"


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_policies(raw: str) -> list[str]:
    if not raw:
        return list(POLICIES)
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [value for value in values if value not in set(POLICIES)]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown policy values: {', '.join(unknown)}")
    return values


def _default_source_paths(split_name: str) -> dict[str, Path]:
    if split_name == "train":
        return dict(TRAIN_DEFAULT_PATHS)
    if split_name == "dev":
        return dict(DEV_DEFAULT_PATHS)
    return dict(DEV_DEFAULT_PATHS)


def _resolve_source_paths(
    args: argparse.Namespace, split_name: str
) -> dict[str, Path]:
    defaults = _default_source_paths(split_name)
    source_paths = {
        "bm25": args.bm25_pool or defaults["bm25"],
        "char": args.char_pool or defaults["char"],
        "structured": args.structured_pool or defaults["structured"],
        "prf": args.prf_pool or defaults["prf"],
    }
    return source_paths


def _build_source_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "bm25": float(args.bm25_weight),
        "char": float(args.char_weight),
        "structured": float(args.structured_weight),
        "prf": float(args.prf_weight),
    }


def stable_deduplicate_evidence_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pure helper: remove duplicate evidence rows by first-seen evidence_id."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        evidence_id = str(row.get("evidence_id", ""))
        if evidence_id in seen:
            continue
        if evidence_id:
            seen.add(evidence_id)
            deduped.append(row)
    return deduped


def enforce_source_limit(
    rows: list[dict[str, Any]], source_limit: int
) -> list[dict[str, Any]]:
    if source_limit <= 0:
        return []
    return rows[:source_limit]


def limit_evidence_count(rows: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    return rows[:max_items]


def parse_candidate_pool(
    path: Path,
    source_name: str,
    source_top_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    if not path or not path.exists() or path.is_dir():
        return {}, {"claims_seen": 0, "claims_loaded": 0, "items_seen": 0}

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Candidate pool is not a dict: {path}")

    parsed: dict[str, list[dict[str, Any]]] = {}
    claims_seen = len(payload)
    items_seen = 0

    for claim_id, entries in payload.items():
        if not isinstance(entries, list):
            continue
        normalized_rows: list[tuple[int, str, float, int]] = []
        for idx, raw in enumerate(entries):
            if not isinstance(raw, dict):
                continue
            evidence_id = str(raw.get("evidence_id", "")).strip()
            if not evidence_id:
                continue
            rank = _coerce_int(raw.get("rank"))
            if rank <= 0:
                rank = idx + 1
            score = _coerce_float(raw.get("score"))
            normalized_rows.append((rank, evidence_id, score, idx))

        if not normalized_rows:
            continue
        normalized_rows.sort(key=lambda item: (item[0], item[3], item[1]))
        ordered_rows: list[dict[str, Any]] = []
        for rank, evidence_id, score, _ in normalized_rows:
            ordered_rows.append(
                {
                    "evidence_id": evidence_id,
                    "rank": int(rank),
                    "score": float(score),
                    "source": source_name,
                }
            )

        ordered_rows = stable_deduplicate_evidence_rows(ordered_rows)
        ordered_rows = enforce_source_limit(ordered_rows, source_top_k)
        if ordered_rows:
            parsed[claim_id] = ordered_rows
            items_seen += len(ordered_rows)

    claims_loaded = len(parsed)
    return parsed, {
        "claims_seen": claims_seen,
        "claims_loaded": claims_loaded,
        "items_seen": items_seen,
    }


def _build_trace_entry(
    source: str,
    candidate: dict[str, Any],
    weight: float,
    contribution: float,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_rank": int(candidate["rank"]),
        "source_score": float(candidate["score"]),
        "weight": float(weight),
        "contribution": float(contribution),
    }


def _build_output_rows(
    ordered_evidence: list[str],
    score_by_evidence: dict[str, float],
    trace_by_evidence: dict[str, list[dict[str, Any]]],
    candidate_k: int,
    source_order: tuple[str, ...],
    policy: str,
) -> list[dict[str, Any]]:
    source_position = {source: idx for idx, source in enumerate(source_order)}
    rows: list[dict[str, Any]] = []
    for rank_idx, evidence_id in enumerate(ordered_evidence[:candidate_k], start=1):
        traces = trace_by_evidence.get(evidence_id, [])
        # Preserve deterministic source ordering in provenance rows.
        traces = sorted(
            traces,
            key=lambda row: (
                source_position.get(str(row.get("source")), len(source_position)),
                int(row.get("source_rank", 0)),
                str(row.get("source")),
            ),
        )
        rows.append(
            {
                "claim_id": None,
                "evidence_id": evidence_id,
                "rank": rank_idx,
                "score": float(score_by_evidence[evidence_id]),
                "fusion_policy": policy,
                "sources": sorted({str(entry.get("source")) for entry in traces}),
                "source_breakdown": traces,
                "source_count": len(traces),
            }
        )
    return rows


def _rrf_candidates(
    claim_source_rows: dict[str, SourceRows],
    source_weights: dict[str, float],
    candidate_k: int,
    rrf_k: int,
    source_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    score_by_evidence: dict[str, float] = {}
    trace_by_evidence: dict[str, list[dict[str, Any]]] = {}
    for source in source_order:
        weight = float(source_weights.get(source, 1.0))
        if weight <= 0:
            continue
        for candidate in claim_source_rows.get(source, []):
            rank = int(candidate["rank"])
            contribution = weight / (float(rrf_k) + max(1, float(rank)))
            score_by_evidence[candidate["evidence_id"]] = (
                score_by_evidence.get(candidate["evidence_id"], 0.0) + contribution
            )
            trace_by_evidence.setdefault(candidate["evidence_id"], []).append(
                _build_trace_entry(source, candidate, weight, contribution)
            )
    ordered = sorted(
        score_by_evidence.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return _build_output_rows(
        [eid for eid, _ in ordered],
        score_by_evidence,
        trace_by_evidence,
        candidate_k,
        source_order,
        "rrf",
    )


def _round_robin_candidates(
    claim_source_rows: dict[str, SourceRows],
    source_weights: dict[str, float],
    candidate_k: int,
    source_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    proposals: list[tuple[int, str, dict[str, Any], str, float]] = []
    max_len = max((len(claim_source_rows.get(source, [])) for source in source_order), default=0)
    sequence = 0
    for offset in range(max_len):
        for source in source_order:
            rows = claim_source_rows.get(source, [])
            if offset >= len(rows):
                continue
            candidate = rows[offset]
            if float(source_weights.get(source, 1.0)) <= 0:
                continue
            contribution = float(source_weights[source]) / (1.0 + float(candidate["rank"]))
            proposals.append(
                (
                    sequence,
                    candidate["evidence_id"],
                    {"source": source, "candidate": candidate},
                    source,
                    contribution,
                )
            )
            sequence += 1

    first_position: dict[str, int] = {}
    trace_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence_idx, evidence_id, payload, source, contribution in proposals:
        weight = float(source_weights.get(source, 1.0))
        first_position.setdefault(evidence_id, sequence_idx)
        trace_by_evidence[evidence_id].append(
            _build_trace_entry(source, payload["candidate"], weight, contribution)
        )

    order = sorted(first_position.items(), key=lambda item: (item[1], item[0]))
    score_by_evidence = {eid: 1.0 / (1.0 + pos) for eid, pos in first_position.items()}
    return _build_output_rows(
        [eid for eid, _ in order],
        score_by_evidence,
        trace_by_evidence,
        candidate_k,
        source_order,
        "round_robin",
    )


def _priority_candidates(
    claim_source_rows: dict[str, SourceRows],
    source_weights: dict[str, float],
    candidate_k: int,
    source_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    order: list[tuple[int, str]] = []
    score_by_evidence: dict[str, float] = {}
    trace_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source in source_order:
        rows = claim_source_rows.get(source, [])
        weight = float(source_weights.get(source, 1.0))
        if weight <= 0:
            continue
        for candidate in rows:
            evidence_id = candidate["evidence_id"]
            contribution = weight / (1.0 + float(candidate["rank"]))
            score_by_evidence.setdefault(
                evidence_id,
                contribution,
            )
            trace_by_evidence[evidence_id].append(
                _build_trace_entry(source, candidate, weight, contribution)
            )
        for candidate in rows:
            evidence_id = candidate["evidence_id"]
            if any(item == evidence_id for _, item in order):
                continue
            order.append((score_by_evidence[evidence_id], evidence_id))

    order = sorted(order, key=lambda item: (-item[0], item[1]))
    return _build_output_rows(
        [eid for _, eid in order],
        score_by_evidence,
        trace_by_evidence,
        candidate_k,
        source_order,
        "priority",
    )


def _union_upper_candidates(
    claim_source_rows: dict[str, SourceRows],
    source_weights: dict[str, float],
    candidate_k: int,
    source_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    score_by_evidence: dict[str, float] = {}
    trace_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in source_order:
        weight = float(source_weights.get(source, 1.0))
        if weight <= 0:
            continue
        for candidate in claim_source_rows.get(source, []):
            evidence_id = candidate["evidence_id"]
            contribution = weight / (1.0 + float(candidate["rank"]))
            score_by_evidence[evidence_id] = max(
                score_by_evidence.get(evidence_id, -1.0),
                contribution,
            )
            trace_by_evidence[evidence_id].append(
                _build_trace_entry(source, candidate, weight, contribution)
            )

    ordered = sorted(score_by_evidence.items(), key=lambda item: (-item[1], item[0]))
    return _build_output_rows(
        [eid for eid, _ in ordered],
        score_by_evidence,
        trace_by_evidence,
        candidate_k,
        source_order,
        "union_upper",
    )


def build_policy_candidates(
    claim_source_rows: dict[str, list[dict[str, Any]]],
    policy: str,
    candidate_k: int,
    source_weights: dict[str, float],
    rrf_k: int,
) -> list[dict[str, Any]]:
    if policy == "rrf":
        return _rrf_candidates(
            claim_source_rows,
            source_weights=source_weights,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            source_order=SOURCE_ORDER,
        )
    if policy == "round_robin":
        return _round_robin_candidates(
            claim_source_rows,
            source_weights=source_weights,
            candidate_k=candidate_k,
            source_order=SOURCE_ORDER,
        )
    if policy == "priority":
        return _priority_candidates(
            claim_source_rows,
            source_weights=source_weights,
            candidate_k=candidate_k,
            source_order=SOURCE_ORDER,
        )
    if policy == "union_upper":
        return _union_upper_candidates(
            claim_source_rows,
            source_weights=source_weights,
            candidate_k=candidate_k,
            source_order=SOURCE_ORDER,
        )
    raise ValueError(f"Unknown policy: {policy}")


def _candidate_pool_for_policy(
    claims: dict[str, Any],
    source_pools: dict[str, SourceRows],
    policy: str,
    candidate_k: int,
    source_weights: dict[str, float],
    rrf_k: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        rows = build_policy_candidates(
            {
                source: source_pools.get(source, {}).get(claim_id, [])
                for source in SOURCE_ORDER
            },
            policy=policy,
            candidate_k=candidate_k,
            source_weights=source_weights,
            rrf_k=rrf_k,
        )
        # Ensure deterministic output always has explicit per-claim rows.
        out[claim_id] = rows
    return out


def evaluate_recall_at_k(
    claims: dict[str, Any],
    ranked_pool: dict[str, list[dict[str, Any]]],
    eval_ks: list[int],
) -> dict[str, Any]:
    ks = sorted(set(int(k) for k in eval_ks if int(k) > 0))
    rows = {
        "claims_with_evidence": 0,
        "evaluated_ks": ks,
        "avg_candidate_count": 0.0,
        "union_candidates": 0,
    }

    if not claims:
        return rows

    per_k = {k: {"tp": 0, "gold": 0, "claims_hit": 0, "recalls": []} for k in ks}
    label_totals: dict[str, int] = {}
    label_hits: dict[int, dict[str, int]] = {k: {} for k in ks}

    claims_with_evidence = 0
    total_count = 0
    union: set[str] = set()

    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claims_with_evidence += 1
        ranked = [row["evidence_id"] for row in ranked_pool.get(claim_id, [])]
        total_count += len(ranked)
        union.update(ranked)

        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0) + 1

        for k in ks:
            predicted = set(ranked[:k])
            tp = len(gold.intersection(predicted))
            per_k[k]["tp"] += tp
            per_k[k]["gold"] += len(gold)
            recall = tp / len(gold) if gold else 0.0
            per_k[k]["recalls"].append(recall)
            if tp > 0:
                per_k[k]["claims_hit"] += 1
                label_hits[k][label] = label_hits[k].get(label, 0) + 1

    rows["claims_with_evidence"] = claims_with_evidence
    rows["avg_candidate_count"] = float(total_count / claims_with_evidence) if claims_with_evidence else 0.0
    rows["union_candidates"] = len(union)

    for k in ks:
        recall_rows = per_k[k]["recalls"]
        rows[f"macro_recall_at_{k}"] = float(sum(recall_rows) / len(recall_rows)) if recall_rows else 0.0
        total_tp = float(per_k[k]["tp"])
        total_gold = float(per_k[k]["gold"])
        rows[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        rows[f"hit_any_at_{k}"] = per_k[k]["claims_hit"] / claims_with_evidence if claims_with_evidence else 0.0
        for label, total in label_totals.items():
            if total <= 0:
                continue
            rows[f"{label.lower()}_hit_any_at_{k}"] = label_hits[k].get(label, 0) / total

    return rows


def _build_input_records(
    claims_path: Path,
    evidence_path: Path,
    source_paths: dict[str, Path],
    source_meta: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    split_name = _infer_split_name(claims_path)
    records = [
        {
            "path": str(claims_path),
            "sha256": sha256_file(claims_path),
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
    for source_name in SOURCE_ORDER:
        if source_meta[source_name]["claims_loaded"] > 0:
            records.append(
                {
                    "path": str(source_paths[source_name]),
                    "sha256": sha256_file(source_paths[source_name]),
                    "split": "current_run_artifact",
                    "labels_used": False,
                }
            )
    return records


def _score_row_for_diagnostic_selection(metrics: dict[str, Any]) -> tuple:
    return (
        metrics.get("micro_recall_at_500", 0.0),
        metrics.get("macro_recall_at_500", 0.0),
        metrics.get("hit_any_at_500", 0.0),
        metrics.get("micro_recall_at_100", 0.0),
        metrics.get("macro_recall_at_100", 0.0),
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_command = "python round18/experiments/o_sparse/o_s9_union_gate/run_o_s9_union_gate.py"

    split_name = _infer_split_name(args.claims)
    if split_name not in {"train", "dev"}:
        raise SystemExit("O-S9 currently supports data/train-claims.json or data/dev-claims.json.")

    source_paths = _resolve_source_paths(args, split_name)
    active_policies = parse_policies(args.policies)
    eval_ks = parse_k_list(args.eval_k)
    max_eval_k = max(eval_ks)
    if args.candidate_k < max_eval_k:
        args.candidate_k = max_eval_k

    if args.candidate_k <= 0:
        raise SystemExit("candidate-k must be a positive integer.")

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")

    claims = load_json(args.claims)
    if not isinstance(claims, dict):
        raise SystemExit("Claims file must be a dict keyed by claim id.")
    evidence = load_json(args.evidence)
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence file must be a dict keyed by evidence id.")

    source_top_k = {
        "bm25": args.bm25_top_k,
        "char": args.char_top_k,
        "structured": args.structured_top_k,
        "prf": args.prf_top_k,
    }

    source_pools: dict[str, SourceRows] = {}
    source_meta: dict[str, dict[str, int]] = {}
    for source in SOURCE_ORDER:
        pool, meta = parse_candidate_pool(
            path=source_paths[source],
            source_name=source,
            source_top_k=source_top_k[source],
        )
        source_pools[source] = pool
        source_meta[source] = {"claims_seen": int(meta["claims_seen"]), "claims_loaded": int(meta["claims_loaded"]), "items_seen": int(meta["items_seen"])}

    active_sources = [
        source for source in SOURCE_ORDER if source_meta[source]["claims_loaded"] > 0
    ]
    if not active_sources:
        raise SystemExit("No source pools available for fusion.")

    source_weights = _build_source_weights(args)
    source_weights = {
        source: max(0.0, float(source_weights.get(source, 1.0)))
        for source in SOURCE_ORDER
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_prefix = _split_file_prefix(split_name)
    files_written: list[str] = []

    strict_results: list[Any] = []
    candidate_rows: list[dict[str, Any]] = []

    for policy in active_policies:
        if policy not in set(POLICIES):
            raise SystemExit(f"Unknown policy: {policy}")

        strict_config = next((cfg for cfg in POLICY_ORDER if cfg.policy == policy), None)
        if strict_config is None:
            strict_config = PolicyConfig(policy, f"strict_{policy}", policy.upper())

        ranked = _candidate_pool_for_policy(
            claims=claims,
            source_pools=source_pools,
            policy=policy,
            candidate_k=args.candidate_k,
            source_weights=source_weights,
            rrf_k=args.rrf_k,
        )
        metrics = evaluate_recall_at_k(claims=claims, ranked_pool=ranked, eval_ks=eval_ks)
        metrics["policy"] = policy
        metrics["policy_label"] = strict_config.label
        metrics["fusion_status"] = strict_config.status
        metrics["candidate_top_k"] = args.candidate_k
        metrics["eval_k"] = eval_ks
        metrics["split"] = split_name
        metrics["smoke"] = False
        metrics["source_items_available"] = {
            source: source_meta[source]["items_seen"] for source in SOURCE_ORDER
        }
        metrics["source_claims_loaded"] = {
            source: source_meta[source]["claims_loaded"] for source in SOURCE_ORDER
        }
        metrics["evidence_items_available"] = len(evidence)
        metrics["source_weights"] = source_weights
        metrics["rrf_k"] = args.rrf_k

        candidate_path = output_dir / f"{split_prefix}_{args.run_id}_{policy}_top{args.candidate_k}_candidates.json"
        metrics_path = output_dir / f"{split_prefix}_{args.run_id}_{policy}_metrics.json"

        for claim_id in claims:
            for idx, row in enumerate(ranked[claim_id], start=1):
                row["claim_id"] = claim_id
                row["rank"] = idx
        write_json(candidate_path, ranked)
        write_json(metrics_path, metrics)
        candidate_rows.append(
            {
                "policy": policy,
                "candidate_file": str(candidate_path),
                "metrics_file": str(metrics_path),
            }
        )

        strict_results.append((strict_config, metrics, ranked))
        files_written.extend([str(candidate_path), str(metrics_path)])

    comparison_rows = []
    for config, metrics, _ in strict_results:
        comparison_rows.append(
            {
                "policy": config.policy,
                "label": config.label,
                "status": config.status,
                "policy_id": config.variant_id,
                "macro_recall@100": metrics.get("macro_recall_at_100", 0.0),
                "macro_recall@500": metrics.get("macro_recall_at_500", 0.0),
                "micro_recall@100": metrics.get("micro_recall_at_100", 0.0),
                "micro_recall@500": metrics.get("micro_recall_at_500", 0.0),
                "hit_any@100": metrics.get("hit_any_at_100", 0.0),
                "hit_any@500": metrics.get("hit_any_at_500", 0.0),
            }
        )
    comparison_rows.sort(
        key=lambda row: (
            row["micro_recall@500"],
            row["macro_recall@500"],
            row["hit_any@500"],
            row["micro_recall@100"],
            row["macro_recall@100"],
            row["policy"] == "rrf",
        ),
        reverse=True,
    )
    comparison_csv = output_dir / f"{split_prefix}_{args.run_id}_comparison_table.csv"
    comparison_json = output_dir / f"{split_prefix}_{args.run_id}_comparison_table.json"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(comparison_rows[0].keys())
            if comparison_rows
            else [
                "policy",
                "label",
                "status",
                "policy_id",
                "macro_recall@100",
                "macro_recall@500",
                "micro_recall@100",
                "micro_recall@500",
                "hit_any@100",
                "hit_any@500",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    write_json(comparison_json, comparison_rows)
    files_written.extend([str(comparison_csv), str(comparison_json)])

    all_metrics = {
        "split": split_name,
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "policies": active_policies,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_claims_loaded": {name: meta["claims_loaded"] for name, meta in source_meta.items()},
        "source_items_available": {name: meta["items_seen"] for name, meta in source_meta.items()},
        "comparison_rows": comparison_rows,
        "strict_variant_count": len(strict_results),
        "diagnostic_enabled": bool(args.enable_diagnostic),
        "rrf_k": args.rrf_k,
        "source_weights": source_weights,
    }

    all_metrics_path = output_dir / f"{split_prefix}_{args.run_id}_all_metrics.json"
    write_json(all_metrics_path, all_metrics)
    files_written.append(str(all_metrics_path))

    diagnostic_selected: dict[str, Any] | None = None
    if split_name == "dev" and args.enable_diagnostic and strict_results:
        scored = []
        for config, metrics, pool in strict_results:
            scored.append((_score_row_for_diagnostic_selection(metrics), config.policy, metrics, pool))
        scored.sort(key=lambda item: item[0], reverse=True)
        diag_policy = scored[0][1]
        diag_metrics = dict(scored[0][2])
        diag_pool = scored[0][3]
        diag_metrics["policy"] = diag_policy
        diag_metrics["policy_label"] = f"{diag_policy} (diagnostic)"
        diag_metrics["fusion_status"] = "diagnostic-only"
        diag_metrics["diagnostic_mode"] = "dev-only"

        diagnostic_candidate_path = output_dir / f"{split_prefix}_{args.run_id}_diagnostic_best_top{args.candidate_k}_candidates.json"
        diagnostic_metrics_path = output_dir / f"{split_prefix}_{args.run_id}_diagnostic_best_metrics.json"
        selection_path = output_dir / f"{split_prefix}_{args.run_id}_diagnostic_selection.json"

        write_json(diagnostic_candidate_path, diag_pool)
        write_json(diagnostic_metrics_path, diag_metrics)
        write_json(
            selection_path,
            {
                "selection_mode": "dev-only",
                "best_policy": diag_policy,
                "selection_rule": "argmax over [micro@500, macro@500, hit_any@500, micro@100, macro@100]",
                "candidates_evaluated": len(scored),
            },
        )
        files_written.extend(
            [
                str(diagnostic_candidate_path),
                str(diagnostic_metrics_path),
                str(selection_path),
            ]
        )
        diagnostic_selected = {"policy": diag_policy}
    all_metrics["diagnostic_policy"] = diagnostic_selected["policy"] if diagnostic_selected else None
    all_metrics["diagnostic_candidate_file"] = str(
        output_dir / f"{split_prefix}_{args.run_id}_diagnostic_best_top{args.candidate_k}_candidates.json"
    ) if diagnostic_selected else None

    write_json(all_metrics_path, all_metrics)
    files_written.append(str(all_metrics_path))

    all_metrics["candidate_rows"] = candidate_rows
    write_json(all_metrics_path, all_metrics)

    data_flow_path = output_dir / f"{split_prefix}_{args.run_id}_data_flow_report.json"
    write_json(
        data_flow_path,
        {
            "pipeline": "O-S9 leaf strict gate with union/round policies",
            "split": split_name,
            "policy_set": active_policies,
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "source_metadata": source_meta,
            "per_source_top_k": source_top_k,
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "rrf_k": args.rrf_k,
            "diagnostic_mode": bool(args.enable_diagnostic and split_name == "dev"),
            "diagnostic_selected": diagnostic_selected,
            "outputs": {
                "candidate_summary_file": str(comparison_csv),
                "all_metrics_file": str(all_metrics_path),
            },
        },
    )
    files_written.append(str(data_flow_path))

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.claims),
            str(args.evidence),
            str(args.bm25_pool),
            str(args.char_pool),
            str(args.structured_pool),
            str(args.prf_pool),
            str(args.output_dir),
            str(args.manifest),
            str(args.record_path),
        ]
    )
    if forbidden_hits:
        raise SystemExit(f"Forbidden token(s) in inputs: {forbidden_hits}")

    command = (
        f"{base_command} --claims {args.claims} --evidence {args.evidence} "
        f"--bm25-pool {source_paths['bm25']} --char-pool {source_paths['char']} "
        f"--structured-pool {source_paths['structured']} --prf-pool {source_paths['prf']} "
        f"--candidate-k {args.candidate_k} --eval-k {','.join(map(str, eval_ks))} "
        f"--bm25-top-k {args.bm25_top_k} --char-top-k {args.char_top_k} "
        f"--structured-top-k {args.structured_top_k} --prf-top-k {args.prf_top_k} "
        f"--policies {','.join(active_policies)} --rrf-k {args.rrf_k} "
        f"--run-id {args.run_id} --output-dir {args.output_dir} --stage {args.stage}"
    )

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
    manifest["input_files"] = _build_input_records(
        claims_path=args.claims,
        evidence_path=args.evidence,
        source_paths=source_paths,
        source_meta=source_meta,
    )
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = files_written + [str(args.manifest), str(args.record_path)]
    manifest["metrics"] = {
        "comparison_rows": comparison_rows,
        "all_metrics_file": str(all_metrics_path),
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
        "data_flow_report": str(data_flow_path),
        "diagnostic_policy": diagnostic_selected["policy"] if diagnostic_selected else None,
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Loaded available leaf pools (BM25/char/structured/PRF), applied strict gate policies "
        "including RRF, round-robin, priority, and union-upper, and wrote per-policy candidate + metric files."
    )
    manifest["split_isolation_summary"] = (
        f"Run uses {split_name} split only. Diagnostic best-policy selection is enabled only for dev."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low"
    manifest["notes"] = (
        "Diagnostic winner is marked diagnostic-only and is not promoted to strict status."
    )

    write_json(args.manifest, manifest)
    files_written.append(str(args.manifest))

    record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT",
        "status": "strict-candidate",
        "split": split_name,
        "command": command,
        "command_args": {key: str(value) for key, value in vars(args).items()},
        "claims_count": len(claims),
        "evidence_count": len(evidence),
        "strict_variant_count": len(strict_results),
        "diagnostic_enabled": bool(args.enable_diagnostic),
        "diagnostic_selected": diagnostic_selected,
        "policy_set": active_policies,
        "files_written": files_written + [str(args.record_path)],
        "forbidden_hits": forbidden_hits,
    }
    write_json(args.record_path, record)
    files_written.append(str(args.record_path))

    all_metrics["record_file"] = str(args.record_path)
    all_metrics["manifest_file"] = str(args.manifest)
    write_json(all_metrics_path, all_metrics)

    print(f"Wrote run manifest: {args.manifest}")
    print(f"Wrote run record: {args.record_path}")
    print(f"Strict variant count: {len(strict_results)}")
    if diagnostic_selected:
        print(f"Diagnostic winner (dev): {diagnostic_selected['policy']}")


if __name__ == "__main__":
    main()
