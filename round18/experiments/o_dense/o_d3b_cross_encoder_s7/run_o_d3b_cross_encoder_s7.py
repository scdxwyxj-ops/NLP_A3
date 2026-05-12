from __future__ import annotations

import argparse
import sys
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

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


DEFAULT_SPLIT_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_SPARSE_POOL = Path(
    "round18/outputs/o_sparse/o_s7_plain_leaf_fusion/"
    "dev_full_dev_o_s7_plain_leaf_fusion_strict_rrf_char_heavy_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_dense/o_d3b_cross_encoder_s7")
DEFAULT_RUN_ID = "o_d3b_cross_encoder_s7"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "run_manifest.json"
DEFAULT_RECORD_PATH = DEFAULT_OUTPUT_DIR / "run_record.json"
DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "hers",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "would",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round18 O-D3b strict bounded cross-encoder reranker over the O-S7 plain sparse leaf pool."
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_SPLIT_CLAIMS,
        help="Claims split path (dev-only by contract).",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence corpus path.",
    )
    parser.add_argument(
        "--sparse-pool",
        type=Path,
        default=DEFAULT_SPARSE_POOL,
        help="Current-run top-500 sparse candidate pool path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory under round18/outputs.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run identifier for output file names.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest path.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=DEFAULT_RECORD_PATH,
        help="Per-run record path.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=64,
        help="Top-k candidates to rerank per claim (default 64, smoke-safe).",
    )
    parser.add_argument(
        "--eval-k",
        default="1,3,5,10,64",
        help="Comma-separated evaluation k values.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Cross-encoder model name. Falls back if unavailable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Cross-encoder inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length for cross-encoder inputs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model inference ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke subset mode (max-claims default is 64).",
    )
    parser.add_argument(
        "--smoke-claims",
        type=int,
        default=64,
        help="Max claims when --smoke is set.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=0,
        help="Optional hard cap on claim count for any run (0 = disabled).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Reproducibility marker only.",
    )
    parser.add_argument(
        "--stage",
        default="o_d3b_cross_encoder_s7",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--eval-k must include at least one integer.")
    return sorted(set(values))


def parse_sparse_pool(path: Path, candidate_k: int) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise SystemExit(f"Missing sparse pool file: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Sparse pool must be a dict: {path}")

    parsed: dict[str, list[dict[str, Any]]] = {}
    for claim_id, entries in payload.items():
        if not isinstance(entries, list):
            continue
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        for row in entries:
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            rank = int(row.get("rank", 0) or 0)
            score = row.get("score")
            selected.append(
                {
                    "evidence_id": evidence_id,
                    "source_rank": rank if rank > 0 else len(selected) + 1,
                    "source_score": float(score) if isinstance(score, (int, float)) else 0.0,
                    "source_mode": str(row.get("fusion_mode", "sparse")),
                    "source_count": int(row.get("source_count", 1) or 1),
                }
            )
            if len(selected) >= candidate_k:
                break
        parsed[claim_id] = selected
    return parsed


def subset_claims(claims: dict[str, dict[str, Any]], max_claims: int) -> dict[str, dict[str, Any]]:
    if max_claims <= 0:
        return claims
    return {claim_id: claims[claim_id] for claim_id in list(claims.keys())[:max_claims]}


def infer_split_name(claims_path: Path) -> str:
    name = str(claims_path)
    if name == "data/dev-claims.json":
        return "dev"
    if name == "data/train-claims.json":
        return "train"
    if name == "data/test-claims-unlabelled.json":
        return "test"
    return "claims"


def _enforce_contract(claims: Path, evidence: Path, sparse_pool: Path) -> None:
    if claims != DEFAULT_SPLIT_CLAIMS:
        raise SystemExit(
            "O-D3b in this scope is defined for dev claims only: claims must be data/dev-claims.json."
        )
    if evidence != DEFAULT_EVIDENCE:
        raise SystemExit("O-D3b in this scope requires data/evidence.json as evidence input.")
    if sparse_pool != DEFAULT_SPARSE_POOL:
        raise SystemExit(
            "O-D3b must use the current-run O-S7 plain sparse top-500 pool: "
            f"{DEFAULT_SPARSE_POOL}."
        )


def evaluate_recall_at_k(
    claims: dict[str, dict[str, Any]],
    pool: dict[str, list[dict[str, Any]]],
    eval_ks: list[int],
) -> dict[str, Any]:
    ks = sorted(set(int(k) for k in eval_ks if int(k) > 0))
    rows: dict[str, Any] = {
        "claims_with_evidence": 0,
        "evaluated_ks": ks,
        "avg_candidate_count": 0.0,
        "union_candidates": 0,
    }
    if not claims:
        return rows

    per_k_hits = {
        k: {"tp": 0, "gold": 0, "claims_hit": 0, "recalls": []} for k in ks
    }
    label_totals: dict[str, int] = {}
    label_hits: dict[int, dict[str, int]] = {k: {} for k in ks}
    claim_recall_count = 0

    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claim_recall_count += 1
        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0) + 1
        ranked = [candidate["evidence_id"] for candidate in pool.get(claim_id, [])]
        for k in ks:
            predicted = set(ranked[:k])
            tp = len(gold.intersection(predicted))
            per_k_hits[k]["tp"] += tp
            per_k_hits[k]["gold"] += len(gold)
            recall = tp / len(gold) if gold else 0.0
            per_k_hits[k]["recalls"].append(recall)
            if tp > 0:
                per_k_hits[k]["claims_hit"] += 1
                label_hits[k][label] = label_hits[k].get(label, 0) + 1

    total_candidates = sum(len(v) for v in pool.values())
    rows["claims_with_evidence"] = claim_recall_count
    rows["avg_candidate_count"] = float(total_candidates / claim_recall_count) if claim_recall_count else 0.0
    rows["union_candidates"] = len({cand["evidence_id"] for values in pool.values() for cand in values})

    for k in ks:
        recalls = per_k_hits[k]["recalls"]
        rows[f"macro_recall_at_{k}"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
        total_tp = float(per_k_hits[k]["tp"])
        total_gold = float(per_k_hits[k]["gold"])
        rows[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        rows[f"hit_any_at_{k}"] = per_k_hits[k]["claims_hit"] / claim_recall_count if claim_recall_count else 0.0
        for label in sorted(label_totals):
            if label_totals[label] == 0:
                continue
            hits = label_hits[k].get(label, 0)
            rows[f"{label.lower()}_hit_any_at_{k}"] = hits / label_totals[label]
    return rows


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def lexical_overlap_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = tokenize(claim_text)
    evidence_tokens = tokenize(evidence_text)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    claim_counter = Counter(claim_tokens)
    evidence_counter = Counter(evidence_tokens)
    token_inter = sum(
        min(claim_counter[token], evidence_counter[token]) for token in claim_counter
    )
    claim_bigrams = {" ".join(claim_tokens[i : i + 2]) for i in range(max(0, len(claim_tokens) - 1))}
    evidence_bigrams = {
        " ".join(evidence_tokens[i : i + 2]) for i in range(max(0, len(evidence_tokens) - 1))
    }
    bigram_inter = len(claim_bigrams & evidence_bigrams)
    denom_terms = len(set(claim_tokens) | set(evidence_tokens))
    denom_bigrams = len(claim_bigrams | evidence_bigrams)
    token_j = token_inter / denom_terms if denom_terms else 0.0
    bigram_j = (2 * bigram_inter / denom_bigrams) if denom_bigrams else 0.0
    return min(1.0, 0.7 * token_j + 0.3 * bigram_j)


def build_sparse_to_evidence_inputs(
    claims: dict[str, dict[str, Any]],
    sparse_pool: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[tuple[str, dict[str, Any]]]], int]:
    claim_to_items: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    missing_claims = 0
    for claim_id in claims:
        rows = sparse_pool.get(claim_id, [])
        if not rows:
            missing_claims += 1
            continue
        claim_to_items[claim_id] = [(row["evidence_id"], row) for row in rows]
    return claim_to_items, missing_claims


def rerank_with_cross_encoder(
    candidates: dict[str, list[tuple[str, dict[str, Any]]]],
    claims: dict[str, dict[str, Any]],
    evidence_text: dict[str, str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, Any]]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except Exception as exc:  # pragma: no cover - dependency optional
        return rerank_with_lexical_fallback(
            candidates=candidates,
            claims=claims,
            evidence_text=evidence_text,
        ), "lexical_fallback", {"fallback_reason": f"transformers_import_failed:{exc.__class__.__name__}"}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        selected_device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(selected_device)
        model.eval()
        reranked: dict[str, list[dict[str, Any]]] = {}
        for claim_id, items in candidates.items():
            claim_text = claims[claim_id].get("claim_text", "")
            valid_items = [
                (evidence_id, item)
                for evidence_id, item in items
                if evidence_id in evidence_text
            ]
            if not valid_items:
                reranked[claim_id] = []
                continue
            evidence_ids = [item[0] for item in valid_items]
            evidence_texts = [evidence_text[eid] for eid in evidence_ids]
            scores: list[float] = []
            with torch.no_grad():
                for start in range(0, len(evidence_ids), batch_size):
                    batch_claims = [claim_text] * min(batch_size, len(evidence_ids) - start)
                    batch_evidence = evidence_texts[start : start + batch_size]
                    encoded = tokenizer(
                        batch_claims,
                        batch_evidence,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(selected_device) for key, value in encoded.items()}
                    logits = model(**encoded).logits
                    if logits.shape[-1] == 1:
                        batch_scores = torch.sigmoid(logits[:, 0]).detach().cpu().numpy().astype(float)
                    else:
                        batch_scores = torch.softmax(logits, dim=-1)[:, -1].detach().cpu().numpy().astype(float)
                    scores.extend([float(value) for value in batch_scores])
                    del encoded, logits
            scored = [
                {
                    "claim_id": claim_id,
                    "evidence_id": evidence_ids[idx],
                    "reranker_score": float(scores[idx]),
                    "score": float(scores[idx]),
                    "source_rank": valid_items[idx][1]["source_rank"],
                    "source_score": valid_items[idx][1]["source_score"],
                    "source_mode": valid_items[idx][1]["source_mode"],
                    "source_count": valid_items[idx][1]["source_count"],
                }
                for idx in range(len(evidence_ids))
            ]
            scored.sort(
                key=lambda row: (-row["reranker_score"], row["source_rank"], row["evidence_id"])
            )
            ranked: list[dict[str, Any]] = []
            for rank, row in enumerate(scored, start=1):
                row = dict(row)
                row["rank"] = rank
                row["reranker_mode"] = "cross_encoder"
                row["model_name"] = model_name
                ranked.append(row)
            reranked[claim_id] = ranked
        return reranked, "cross_encoder", {
            "model_name": model_name,
            "batch_size": batch_size,
            "max_length": max_length,
            "device": selected_device,
            "source": "cross_encoder/ms-marco-MiniLM-L6-v2",
        }
    except Exception as exc:
        return rerank_with_lexical_fallback(
            candidates=candidates,
            claims=claims,
            evidence_text=evidence_text,
        ), "lexical_fallback", {"fallback_reason": f"cross_encoder_inference_failed:{exc.__class__.__name__}"}


def rerank_with_lexical_fallback(
    candidates: dict[str, list[tuple[str, dict[str, Any]]]],
    claims: dict[str, dict[str, Any]],
    evidence_text: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    reranked: dict[str, list[dict[str, Any]]] = {}
    for claim_id, items in candidates.items():
        claim_text = claims[claim_id].get("claim_text", "")
        valid_items = [
            (evidence_id, meta)
            for evidence_id, meta in items
            if evidence_id in evidence_text
        ]
        scored: list[dict[str, Any]] = []
        for evidence_id, meta in valid_items:
            score = lexical_overlap_score(claim_text, evidence_text[evidence_id])
            row = {
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "reranker_score": float(score),
                "score": float(score),
                "source_rank": meta["source_rank"],
                "source_score": meta["source_score"],
                "source_mode": meta["source_mode"],
                "source_count": meta["source_count"],
                "reranker_mode": "lexical_fallback",
            }
            scored.append(row)
        scored.sort(key=lambda row: (-row["reranker_score"], row["source_rank"], row["evidence_id"]))
        for rank, row in enumerate(scored, start=1):
            row["rank"] = rank
        reranked[claim_id] = scored
    return reranked


def format_input_records(
    claims_path: Path,
    evidence_path: Path,
    sparse_pool: Path,
    split_name: str,
) -> list[dict[str, Any]]:
    return [
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
        {
            "path": str(sparse_pool),
            "sha256": sha256_file(sparse_pool),
            "split": "current_run_artifact",
            "labels_used": False,
        },
    ]


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    _enforce_contract(args.claims, args.evidence, args.sparse_pool)
    if args.candidate_k <= 0:
        raise SystemExit("--candidate-k must be > 0")
    if args.smoke:
        if args.max_claims <= 0:
            args.max_claims = args.smoke_claims
        args.candidate_k = min(args.candidate_k, 64)

    eval_ks = parse_k_list(args.eval_k)
    if max(eval_ks) > args.candidate_k:
        args.candidate_k = max(eval_ks)

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")
    if not args.sparse_pool.exists():
        raise SystemExit(f"Missing sparse pool file: {args.sparse_pool}")

    claims = load_json(args.claims)
    if not isinstance(claims, dict):
        raise SystemExit("Claims must be a JSON object.")
    claims = subset_claims(claims, args.max_claims if args.max_claims > 0 else 0)
    evidence = load_json(args.evidence)
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence must be a JSON object keyed by evidence id.")

    sparse_pool = parse_sparse_pool(args.sparse_pool, args.candidate_k)
    sparse_pool, missing_claims = build_sparse_to_evidence_inputs(claims, sparse_pool)
    if not sparse_pool:
        raise SystemExit("No sparse candidates were loaded for selected claims.")

    needed_evidence_ids = {evidence_id for rows in sparse_pool.values() for evidence_id, _ in rows}
    evidence_text = {eid: evidence[eid] for eid in needed_evidence_ids if eid in evidence}
    missing_evidence = len(needed_evidence_ids) - len(evidence_text)

    reranker_type, reranker_info = None, {}
    if args.smoke:
        split_file_prefix = "dev_full_dev_smoke"
    else:
        split_file_prefix = "dev_full_dev"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = (
        f"{split_file_prefix}_{args.run_id}_strict_top{args.candidate_k}"
    )
    candidate_path = args.output_dir / f"{file_prefix}_candidates.json"
    metrics_path = args.output_dir / f"{file_prefix}_metrics.json"
    data_flow_path = args.output_dir / f"{file_prefix}_data_flow_report.json"

    ranked_pool, mode, reranker_info = rerank_with_cross_encoder(
        candidates=sparse_pool,
        claims=claims,
        evidence_text=evidence_text,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    reranker_type = mode
    for candidates in ranked_pool.values():
        for idx, item in enumerate(candidates, start=1):
            item["rank"] = idx

    metrics = evaluate_recall_at_k(claims=claims, pool=ranked_pool, eval_ks=eval_ks)
    metrics["candidate_top_k"] = args.candidate_k
    metrics["claims_input_count"] = len(claims)
    metrics["claims_with_sparse_candidates"] = len(ranked_pool)
    metrics["claims_with_source_evidence_text"] = len(evidence_text)
    metrics["sparse_missing_claims"] = missing_claims
    metrics["evidence_missing_count"] = missing_evidence
    metrics["smoke"] = args.smoke
    metrics["reranker_type"] = reranker_type
    metrics["reranker_info"] = reranker_info
    metrics["evidence_available_total"] = len(evidence)
    metrics["sparse_candidate_count"] = sum(len(v) for v in sparse_pool.values())
    metrics["reranked_candidate_count"] = sum(len(v) for v in ranked_pool.values())

    write_json(candidate_path, ranked_pool)
    write_json(metrics_path, metrics)

    split_name = infer_split_name(args.claims)
    command = (
        f"python round18/experiments/o_dense/o_d3b_cross_encoder_s7/run_o_d3b_cross_encoder_s7.py "
        f"--claims {args.claims} --evidence {args.evidence} --sparse-pool {args.sparse_pool} "
        f"--output-dir {args.output_dir} --run-id {args.run_id} --candidate-k {args.candidate_k} "
        f"--eval-k {','.join(map(str, eval_ks))} --model-name {args.model_name} "
        f"--batch-size {args.batch_size} --max-length {args.max_length} --device {args.device}"
    )
    if args.smoke:
        command += " --smoke"
    if args.max_claims:
        command += f" --max-claims {args.max_claims}"

    split_file_suffix = "dev_full_dev"
    forbidden_hits = find_forbidden_tokens(
        [
            str(args.claims),
            str(args.evidence),
            str(args.sparse_pool),
            str(args.output_dir),
            command,
            str(args.model_name),
        ]
    )

    data_flow = {
        "pipeline": "sparse top-k gate -> bounded rerank per claim",
        "split": split_name,
        "smoke": args.smoke,
        "inputs": {
            "claims": str(args.claims),
            "evidence": str(args.evidence),
            "sparse_pool": str(args.sparse_pool),
        },
        "input_hashes": {
            "claims": sha256_file(args.claims),
            "evidence": sha256_file(args.evidence),
            "sparse_pool": sha256_file(args.sparse_pool),
        },
        "processing": {
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "reranker_type": reranker_type,
            "reranker_model": args.model_name,
            "reranker_mode": reranker_type,
            "fallback_reason": reranker_info.get("fallback_reason"),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": args.device,
        },
        "counts": {
            "claims_selected": len(claims),
            "claims_with_sparse_input": len(sparse_pool),
            "sparse_candidates_loaded": sum(len(v) for v in sparse_pool.values()),
            "reranked_candidates": sum(len(v) for v in ranked_pool.values()),
            "missing_sparse_claims": missing_claims,
            "missing_sparse_evidence_text": missing_evidence,
        },
        "outputs": {
            "candidate_pool": str(candidate_path),
            "metrics": str(metrics_path),
            "status": "strict-candidate",
        },
        "notes": [
            "No corpus-wide reranking: only the top-k sparse candidates per claim were scored.",
            "If model loading or inference fails, a deterministic lexical overlap fallback is used."
            + ("" if reranker_type == "cross_encoder" else " The fallback was used."),
        ],
    }
    write_json(data_flow_path, data_flow)

    input_records = format_input_records(args.claims, args.evidence, args.sparse_pool, split_name)
    base_command = command
    command_args = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            command_args[key] = str(value)
        else:
            command_args[key] = value

    record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT",
        "status": "strict-candidate",
        "split": split_name,
        "command": base_command,
        "command_args": command_args,
        "claims_count": len(claims),
        "evidence_count": len(evidence),
        "forbidden_hits": forbidden_hits,
        "reranker_type": reranker_type,
        "reranker_model": args.model_name,
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
        status="strict-candidate",
        mode="STRICT",
        stage=args.stage,
        command=base_command,
        working_directory=Path.cwd(),
        config_path=str(args.manifest),
        random_seed=args.random_seed,
        cv_seed=None,
    )
    manifest["input_files"] = input_records
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input and runtime arguments were scanned and passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = [
        str(candidate_path),
        str(metrics_path),
        str(data_flow_path),
        str(args.record_path),
        str(args.manifest),
    ]
    manifest["metrics"] = {
        "candidate_file": str(candidate_path),
        "metrics_file": str(metrics_path),
        "data_flow_report": str(data_flow_path),
        "reranker_type": reranker_type,
        "reranker_model": args.model_name,
        "smoke": args.smoke,
        "recalls": {k: v for k, v in metrics.items() if k.startswith("macro_recall_at_") or k.startswith("micro_recall_at_")},
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = args.device
    manifest["model_names"] = [args.model_name] if reranker_type == "cross_encoder" else []
    manifest["data_flow_summary"] = (
        "Loaded dev claims/evidence and a current-run top-500 sparse pool, "
        "scored only the per-claim top-k sparse candidates with a bounded reranker, "
        "and wrote strict-candidate reranked candidates + recall metrics."
    )
    manifest["split_isolation_summary"] = (
        "Run is restricted to dev claims. Evidence/evidence ids are consumed only through "
        "the sparse candidate gate; no full-corpus scoring is performed."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low" if reranker_type == "cross_encoder" else "medium"
    fallback_note = reranker_info.get("fallback_reason")
    manifest["notes"] = (
        "Model-based reranking was attempted first. "
        + (f"Fallback engaged: {fallback_note}" if fallback_note else "Cross-encoder was used.")
    )
    write_json(args.manifest, manifest)

    print(f"Wrote reranked candidates: {candidate_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote data flow report: {data_flow_path}")


if __name__ == "__main__":
    main()
