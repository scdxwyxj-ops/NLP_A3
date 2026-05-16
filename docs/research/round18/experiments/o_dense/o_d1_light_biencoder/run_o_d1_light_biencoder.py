from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

REPO_ROOT = Path(__file__).resolve().parents[4]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.tools.common import (
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


DEFAULT_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_POOL = Path(
    "round18/outputs/o_sparse/o_s4_fusion/"
    "dev_full_dev_o_s4_fusion_strict_rrf_prf_heavy_top500_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("round18/outputs/o_dense/o_d1_light_biencoder")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "run_manifest.json"
DEFAULT_RECORD_PATH = DEFAULT_OUTPUT_DIR / "run_record.json"
DEFAULT_RUN_ID = "o_d1_light_biencoder"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PREFERRED_FALLBACK_RANDOM_STATE = 1337


@dataclass(frozen=True)
class CandidateItem:
    evidence_id: str
    sparse_rank: int
    sparse_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-D1 light bi-encoder rerank constrained to sparse candidate IDs."
        )
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_CLAIMS,
        help="Claim split JSON (dev only in this strict scope).",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence corpus JSON.",
    )
    parser.add_argument(
        "--sparse-pool",
        type=Path,
        default=DEFAULT_POOL,
        help=(
            "Current-run sparse candidate pool. Candidates are used as the only scoring "
            "universe (no full-corpus dense retrieval)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for rerank candidates, metrics, and reports.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run identifier used in output filenames.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Dense model name. Preference is sentence-transformers/all-MiniLM-L6-v2.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Optional device hint (currently not required by fallback mode).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Number of dense-reranked candidates to keep per claim.",
    )
    parser.add_argument(
        "--eval-k",
        default="100,500",
        help="Comma-separated recall-k values.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=500,
        help="Number of sparse candidates to score per claim.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run bounded smoke mode with limited claims and evidence text for quick verification.",
    )
    parser.add_argument(
        "--smoke-claims",
        type=int,
        default=16,
        help="Smoke claim cap when --smoke is set.",
    )
    parser.add_argument(
        "--smoke-max-candidates",
        type=int,
        default=80,
        help="Optional cap for candidate docs per claim in smoke mode.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest output path.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=DEFAULT_RECORD_PATH,
        help="Run record output path.",
    )
    parser.add_argument(
        "--run-stage",
        default="o_d1_light_biencoder",
        help="Manifest stage label.",
    )
    parser.add_argument("--random-seed", type=int, default=1337, help="Determinism marker.")
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    ks = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not ks:
        raise argparse.ArgumentTypeError("eval-k must include at least one integer.")
    return sorted(set(int(k) for k in ks if int(k) > 0))


def _infer_split_name(path: Path) -> str:
    if str(path) == "data/train-claims.json":
        return "train"
    if str(path) == "data/dev-claims.json":
        return "dev"
    if str(path) == "data/test-claims-unlabelled.json":
        return "test"
    return "claims"


def _enforce_strict_inputs(claims: Path, evidence: Path, sparse_pool: Path) -> None:
    if claims != DEFAULT_CLAIMS:
        raise SystemExit("Dense-in-sparse strict scope is fixed to data/dev-claims.json.")
    if evidence != DEFAULT_EVIDENCE:
        raise SystemExit("Dense-in-sparse strict scope is fixed to data/evidence.json.")
    if sparse_pool != DEFAULT_POOL:
        raise SystemExit(
            "Dense-in-sparse strict scope requires the current-run sparse pool at: "
            f"{DEFAULT_POOL}"
        )


def _limit_dict(payload: dict[str, Any], max_items: int) -> dict[str, Any]:
    if max_items <= 0 or len(payload) <= max_items:
        return dict(payload)
    selected = list(payload.items())[:max_items]
    return {k: v for k, v in selected}


def parse_sparse_pool(
    pool_payload: dict[str, Any],
    allowed_claims: dict[str, Any],
    max_candidates: int,
) -> tuple[dict[str, list[CandidateItem]], dict[str, Any]]:
    parsed: dict[str, list[CandidateItem]] = {}
    diagnostics = {"claim_count": 0, "entries_with_missing_id": 0, "top_rank_entries": 0}
    for claim_id, entries in pool_payload.items():
        if claim_id not in allowed_claims:
            continue
        if not isinstance(entries, list):
            continue
        diagnostics["claim_count"] += 1
        ranked: list[CandidateItem] = []
        seen: set[str] = set()
        for position, item in enumerate(entries, start=1):
            if len(ranked) >= max_candidates:
                break
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id:
                diagnostics["entries_with_missing_id"] += 1
                continue
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            sparse_score = item.get("score")
            if not isinstance(sparse_score, (int, float)):
                sparse_score = 0.0
            rank = item.get("rank")
            sparse_rank = int(rank) if isinstance(rank, int) and rank > 0 else position
            ranked.append(CandidateItem(evidence_id=evidence_id, sparse_rank=sparse_rank, sparse_score=float(sparse_score)))
            diagnostics["top_rank_entries"] += 1
        parsed[claim_id] = ranked
    return parsed, diagnostics


def try_load_sentence_transformer(model_name: str, device: str) -> tuple[bool, str, Any]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as error:  # pragma: no cover - environment dependent
        return False, f"import failed: {error}", None
    try:
        model = SentenceTransformer(model_name)
    except Exception as error:  # pragma: no cover - environment dependent
        return False, f"load failed: {error}", None
    return True, "", model


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _svd_reduce(
    matrix: np.ndarray,
    *,
    max_components: int = 64,
    random_state: int = PREFERRED_FALLBACK_RANDOM_STATE,
) -> np.ndarray:
    n_samples, n_features = matrix.shape
    if n_features <= 1 or n_samples <= 1:
        return matrix
    max_supported = min(max_components, n_samples - 1, n_features - 1)
    if max_supported <= 1:
        return matrix
    projector = TruncatedSVD(
        n_components=max_supported,
        n_iter=10,
        random_state=random_state,
    )
    return projector.fit_transform(matrix)


def fallback_tfidf_svd_scores(claim_text: str, candidate_texts: list[str]) -> list[float]:
    docs = [claim_text, *candidate_texts]
    if len(candidate_texts) == 0:
        return []
    if all(not doc.strip() for doc in docs):
        return [0.0 for _ in candidate_texts]

    vectorizers = [
        {"name": "word", "args": dict(analyzer="word", ngram_range=(1, 2), max_features=20000)},
        {"name": "char", "args": dict(analyzer="char_wb", ngram_range=(3, 5), max_features=12000)},
    ]
    parts: list[np.ndarray] = []
    for conf in vectorizers:
        vectorizer = TfidfVectorizer(**conf["args"])
        sparse_matrix = vectorizer.fit_transform(docs)
        matrix = sparse_matrix.toarray().astype(np.float32, copy=False)
        matrix = _svd_reduce(matrix, max_components=64)
        matrix = _normalise_rows(matrix)
        parts.append(matrix)

    stacked = np.hstack(parts) if len(parts) > 1 else parts[0]
    if stacked.shape[1] == 0:
        return [0.0 for _ in candidate_texts]
    stacked = _normalise_rows(stacked)
    query = stacked[0]
    candidates = stacked[1:]
    scores = candidates @ query
    return [float(score) for score in scores]


def build_sentence_scores(
    model: Any,
    cache: dict[str, np.ndarray],
    claim_text: str,
    candidates: list[tuple[str, str]],
) -> list[float]:
    evidence_ids = [evidence_id for evidence_id, _ in candidates]
    candidate_texts = [text for _, text in candidates]
    missing_ids = [eid for eid in evidence_ids if eid not in cache]
    if missing_ids:
        unique_missing = []
        seen: set[str] = set()
        for eid in missing_ids:
            if eid in seen:
                continue
            seen.add(eid)
            index = evidence_ids.index(eid)
            unique_missing.append(candidate_texts[index])
        embeddings = model.encode(
            unique_missing,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        start = 0
        for eid in missing_ids:
            if eid in cache:
                continue
            cache[eid] = embeddings[start]
            start += 1
    candidate_matrix = np.vstack([cache[eid] for eid in evidence_ids])
    query = model.encode(
        [claim_text],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]
    scores = candidate_matrix @ query
    return [float(score) for score in scores]


def build_dense_scores(
    claim_text: str,
    candidates: list[tuple[str, str]],
    use_fallback: bool,
    fallback_name: str,
    model: Any | None,
    model_cache: dict[str, np.ndarray],
    diagnostics: dict[str, int],
) -> tuple[list[float], str]:
    if use_fallback or model is None:
        diagnostics["fallback_calls"] = diagnostics.get("fallback_calls", 0) + 1
        return fallback_tfidf_svd_scores(claim_text, [text for _, text in candidates]), fallback_name

    scores = build_sentence_scores(model=model, cache=model_cache, claim_text=claim_text, candidates=candidates)
    return scores, "sentence-transformers"


def build_reranked_pool(
    claims: dict[str, Any],
    sparse_pool: dict[str, list[CandidateItem]],
    evidence_text_lookup: dict[str, str],
    candidate_k: int,
    use_fallback: bool,
    model_name: str,
    model: Any,
    random_seed: int,
    fallback_name: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    model_cache: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {
        "dense_scored_claims": 0,
        "dense_missing_evidence_text": 0,
        "fallback_calls": 0,
        "model_name": model_name,
        "dense_backend": fallback_name,
    }

    reranked_pool: dict[str, list[dict[str, Any]]] = {}
    for claim_id, claim in claims.items():
        candidates = sparse_pool.get(claim_id, [])
        candidate_pairs: list[tuple[str, str]] = []
        source_rows: list[CandidateItem] = []

        for candidate in candidates:
            text = evidence_text_lookup.get(candidate.evidence_id, "")
            if not text:
                diagnostics["dense_missing_evidence_text"] += 1
                continue
            candidate_pairs.append((candidate.evidence_id, text))
            source_rows.append(candidate)

        if not candidate_pairs:
            reranked_pool[claim_id] = []
            continue

        scores, backend_used = build_dense_scores(
            claim_text=claim["claim_text"],
            candidates=candidate_pairs,
            use_fallback=use_fallback or model is None,
            fallback_name=fallback_name,
            model=model,
            model_cache=model_cache,
            diagnostics=diagnostics,
        )

        diagnostics["dense_backend"] = backend_used if backend_used else diagnostics["dense_backend"]
        scored = [
            (
                source_rows[idx].evidence_id,
                scores[idx],
                source_rows[idx].sparse_rank,
                source_rows[idx].sparse_score,
            )
            for idx in range(len(scores))
        ]

        # deterministic tie-break: score desc, sparse-rank asc, evidence id asc
        scored = sorted(scored, key=lambda row: (-row[1], row[2], row[0]))

        candidates_out = []
        for rank, (evidence_id, score, sparse_rank, sparse_score) in enumerate(scored, start=1):
            candidates_out.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": evidence_id,
                    "rank": rank,
                    "score": float(score),
                    "reranker": backend_used,
                    "sparse_rank": sparse_rank,
                    "sparse_score": float(sparse_score),
                }
            )
            if rank >= candidate_k:
                break
        diagnostics["dense_scored_claims"] += 1
        reranked_pool[claim_id] = candidates_out

    diagnostics["rng_seed"] = random_seed
    diagnostics["rng_used"] = True
    return reranked_pool, diagnostics


def evaluate_recall_at_k(
    claims: dict[str, Any],
    pool: dict[str, list[dict[str, Any]]],
    eval_ks: list[int],
) -> dict[str, Any]:
    ks = sorted(set(eval_ks))
    total_claims = 0
    totals: dict[int, dict[str, float]] = {
        k: {"tp": 0.0, "gold": 0.0, "claims_hit": 0.0}
        for k in ks
    }
    label_totals: dict[str, float] = {}
    label_hits: dict[int, dict[str, float]] = {k: {} for k in ks}
    recall_by_claim: dict[int, list[float]] = {k: [] for k in ks}
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        total_claims += 1
        gold_size = float(len(gold))
        ranked = [item["evidence_id"] for item in pool.get(claim_id, [])]
        label = str(claim.get("claim_label", "UNLABELED"))
        label_totals[label] = label_totals.get(label, 0.0) + 1.0
        for k in ks:
            predicted = set(ranked[:k])
            tp = len(gold.intersection(predicted))
            totals[k]["tp"] += float(tp)
            totals[k]["gold"] += gold_size
            recall = tp / gold_size if gold_size else 0.0
            recall_by_claim[k].append(recall)
            if tp > 0:
                totals[k]["claims_hit"] += 1
                label_hits[k][label] = label_hits[k].get(label, 0.0) + 1.0

    results: dict[str, Any] = {
        "claims_with_evidence": int(total_claims),
        "eval_ks": ks,
    }
    for k in ks:
        claim_count = float(total_claims)
        micro_recall = totals[k]["tp"] / totals[k]["gold"] if totals[k]["gold"] else 0.0
        hit_any = totals[k]["claims_hit"] / claim_count if claim_count else 0.0
        macro_recall = sum(recall_by_claim[k]) / claim_count if claim_count else 0.0
        results[f"micro_recall_at_{k}"] = float(micro_recall)
        results[f"hit_any_at_{k}"] = float(hit_any)
        results[f"macro_recall_at_{k}"] = float(macro_recall)
        for label, label_total in sorted(label_totals.items()):
            if label_total <= 0:
                continue
            label_hit = label_hits[k].get(label, 0.0)
            results[f"{label.lower()}_hit_any_at_{k}"] = float(label_hit / label_total)
    return results


def build_input_records(claims_path: Path, evidence_path: Path, sparse_pool_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(claims_path),
            "sha256": sha256_file(claims_path),
            "split": "dev",
            "labels_used": True,
        },
        {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "split": "evidence",
            "labels_used": False,
        },
        {
            "path": str(sparse_pool_path),
            "sha256": sha256_file(sparse_pool_path),
            "split": "current_run_artifact",
            "labels_used": False,
        },
    ]


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    base_command = "python round18/experiments/o_dense/o_d1_light_biencoder/run_o_d1_light_biencoder.py"
    _enforce_strict_inputs(args.claims, args.evidence, args.sparse_pool)

    eval_ks = parse_k_list(args.eval_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.record_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")
    if not args.sparse_pool.exists():
        raise SystemExit(f"Missing sparse pool: {args.sparse_pool}")
    if args.smoke:
        if args.smoke_claims <= 0:
            raise SystemExit("smoke mode requires --smoke-claims > 0")

    if max(eval_ks) > args.candidate_k:
        args.candidate_k = max(eval_ks)

    args.max_candidates = min(args.max_candidates, args.candidate_k)
    if args.smoke:
        args.max_candidates = min(args.max_candidates, args.smoke_max_candidates)

    claims = load_json(args.claims)
    if not isinstance(claims, dict):
        raise SystemExit("Claims JSON must be a dict.")
    if args.smoke:
        claims = _limit_dict(claims, args.smoke_claims)

    evidence = load_json(args.evidence)
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence JSON must be a dict.")

    sparse_pool_payload = load_json(args.sparse_pool)
    if not isinstance(sparse_pool_payload, dict):
        raise SystemExit("Sparse pool must be a dict keyed by claim id.")

    sparse_pool, pool_diag = parse_sparse_pool(
        pool_payload=sparse_pool_payload,
        allowed_claims=claims,
        max_candidates=args.max_candidates,
    )

    candidate_evidence_ids = sorted(
        {candidate.evidence_id for items in sparse_pool.values() for candidate in items}
    )
    evidence_text_lookup = {
        evidence_id: str(evidence.get(evidence_id, "")) for evidence_id in candidate_evidence_ids
    }
    missing_evidence_texts = sum(1 for text in evidence_text_lookup.values() if not text)
    if not candidate_evidence_ids:
        raise SystemExit("No sparse candidates found for the selected claims.")

    candidate_pool_name = (
        "sentence_transformers_all_miniLM_l6_v2"
        if args.model == DEFAULT_MODEL_NAME
        else args.model.replace("/", "_").replace("-", "_")
    )
    split_name = _infer_split_name(args.claims)
    prefix = "dev_full_dev" if split_name == "dev" else f"{split_name}_full"

    use_fallback = False
    model = None
    fallback_reason = ""
    model_load_time = 0.0

    if args.model:
        model_start = time.perf_counter()
        loaded, reason, model = try_load_sentence_transformer(args.model, args.device)
        model_load_time = round(time.perf_counter() - model_start, 6)
        if not loaded:
            use_fallback = True
            fallback_reason = reason
            candidate_pool_name = "fallback_tfidf_svd"
    else:
        use_fallback = True
        fallback_reason = "model argument not provided"
        candidate_pool_name = "fallback_tfidf_svd"

    fallback_name = "sentence-transformers" if not use_fallback else "sklearn_tfidf_svd"

    dense_pool, dense_diag = build_reranked_pool(
        claims=claims,
        sparse_pool=sparse_pool,
        evidence_text_lookup=evidence_text_lookup,
        candidate_k=args.candidate_k,
        use_fallback=use_fallback,
        model_name=args.model,
        model=model,
        random_seed=args.random_seed,
        fallback_name=fallback_name,
    )

    metrics = evaluate_recall_at_k(
        claims=claims,
        pool=dense_pool,
        eval_ks=eval_ks,
    )

    metrics.update(
        {
            "run_id": args.run_id,
            "split": split_name,
            "candidate_k": args.candidate_k,
            "eval_k": eval_ks,
            "claims_input_count": len(claims),
            "sparse_candidate_source": str(args.sparse_pool),
            "sparse_candidate_pool_claims": len(sparse_pool),
            "sparse_candidate_count_unique": len(candidate_evidence_ids),
            "sparse_pool_entries": pool_diag["claim_count"],
            "dense_missing_evidence_text": dense_diag["dense_missing_evidence_text"],
            "model": args.model,
            "fallback_used": use_fallback,
            "fallback_reason": fallback_reason,
            "dense_backend": dense_diag["dense_backend"],
            "scoring_scope": "candidate_ids_from_sparse_pool_only",
            "candidate_k_effective": args.candidate_k,
            "max_candidates_scored": args.max_candidates,
            "model_load_seconds": model_load_time,
            "smoke": args.smoke,
            "smoke_claims": args.smoke_claims if args.smoke else 0,
            "pool_diagnostics": pool_diag,
            "rerank_diagnostics": dense_diag,
            "random_seed": args.random_seed,
        }
    )

    candidate_path = args.output_dir / f"{prefix}_{args.run_id}_{candidate_pool_name}_top{args.candidate_k}_candidates.json"
    metrics_path = args.output_dir / f"{prefix}_{args.run_id}_{candidate_pool_name}_metrics.json"
    data_flow_path = args.output_dir / f"{prefix}_{args.run_id}_{candidate_pool_name}_data_flow_report.json"
    write_json(candidate_path, dense_pool)
    write_json(metrics_path, metrics)

    evidence_with_text = len(evidence_text_lookup) - missing_evidence_texts
    data_flow = {
        "pipeline": "O-D1 dense-in-sparse rerank",
        "split": split_name,
        "inputs": {
            "claims": str(args.claims),
            "evidence": str(args.evidence),
            "sparse_pool": str(args.sparse_pool),
        },
        "scope_guard": {
            "enforced_split": "dev",
            "enforced_pool": str(DEFAULT_POOL),
            "score_only_sparse_pool": True,
        },
        "candidate_universe": {
            "sparse_claims_with_candidates": len(sparse_pool),
            "sparse_unique_candidate_evidence": len(candidate_evidence_ids),
            "sparse_candidate_evidence_with_text": evidence_with_text,
            "sparse_candidate_evidence_missing_text": missing_evidence_texts,
        },
        "reranker": {
            "requested_model": args.model,
            "fallback_used": use_fallback,
            "fallback_details": fallback_reason,
            "backend": dense_diag["dense_backend"],
            "candidate_prefix": candidate_pool_name,
            "random_seed": args.random_seed,
            "device_hint": args.device,
            "max_candidates_per_claim": args.max_candidates,
        },
        "outputs": {
            "candidate_pool": str(candidate_path),
            "metrics": str(metrics_path),
            "metric_eval_k": eval_ks,
        },
    }
    write_json(data_flow_path, data_flow)

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.claims),
            str(args.evidence),
            str(args.sparse_pool),
            str(args.output_dir),
            str(args.manifest),
        ]
    )

    command = (
        f"{base_command} --claims {args.claims} --evidence {args.evidence} "
        f"--sparse-pool {args.sparse_pool} --output-dir {args.output_dir} "
        f"--candidate-k {args.candidate_k} --eval-k {','.join(map(str, eval_ks))} "
        f"--max-candidates {args.max_candidates} --run-id {args.run_id} --run-stage {args.run_stage} "
        f"--model {args.model} --device {args.device}"
    ).strip()
    if args.smoke:
        command += (
            f" --smoke --smoke-claims {args.smoke_claims} "
            f"--smoke-max-candidates {args.smoke_max_candidates}"
        )

    command_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    run_record = {
        "run_id": args.run_id,
        "stage": args.run_stage,
        "mode": "STRICT",
        "status": "strict-candidate",
        "split": split_name,
        "command": command,
        "command_args": command_args,
        "claims_count": len(claims),
        "candidate_count_total": len(candidate_evidence_ids),
        "candidate_pool_claims": len(sparse_pool),
        "fallback_used": use_fallback,
        "fallback_reason": fallback_reason if use_fallback else "",
        "fallback_name": fallback_name,
        "files_written": [
            str(candidate_path),
            str(metrics_path),
            str(data_flow_path),
            str(args.record_path),
            str(args.manifest),
        ],
    }
    write_json(args.record_path, run_record)

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate",
        mode="STRICT",
        stage=args.run_stage,
        command=command,
        working_directory=Path.cwd(),
        config_path=str(args.manifest),
        config_hash="",
        random_seed=args.random_seed,
        cv_seed=None,
    )
    manifest["input_files"] = build_input_records(args.claims, args.evidence, args.sparse_pool)
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": (
            "Input path scan against forbidden strict tokens passed."
            if len(forbidden_hits) == 0
            else str(forbidden_hits)
        ),
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
        "smoke": args.smoke,
        "fallback_used": use_fallback,
        "sparse_scope_enforced": True,
        "candidate_k": args.candidate_k,
        "eval_k": eval_ks,
        "macro_recall@100": metrics.get("macro_recall_at_100", 0.0),
        "macro_recall@500": metrics.get("macro_recall_at_500", 0.0),
        "micro_recall@100": metrics.get("micro_recall_at_100", 0.0),
        "micro_recall@500": metrics.get("micro_recall_at_500", 0.0),
        "hit_any@100": metrics.get("hit_any_at_100", 0.0),
        "hit_any@500": metrics.get("hit_any_at_500", 0.0),
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["runtime"]["device"] = args.device if use_fallback else (args.device or "auto")
    manifest["data_flow_summary"] = (
        "Loaded strict dev claims/evidence and enforced sparse candidate pool from round18. "
        "Reranked only candidate evidence IDs in that pool with a light bi-encoder (sentence-transformers preferred, "
        "deterministic TF-IDF+SVD fallback). Wrote candidate pool, recall metrics, data-flow report, and manifest."
    )
    manifest["split_isolation_summary"] = (
        "Dev claims were used only for fixed scoring/evaluation and metrics. "
        "No train/test artifacts or full-corpus dense retrieval were used."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low" if use_fallback else "medium"
    manifest["notes"] = (
        f"Sparse-only scope enforced: {str(args.sparse_pool)}"
        if not use_fallback
        else f"Sparse-only scope enforced; fallback model used: {fallback_name}. reason={fallback_reason}"
    )
    write_json(args.manifest, manifest)

    print(f"Wrote candidate pool: {candidate_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote data-flow report: {data_flow_path}")
    print(f"Wrote manifest: {args.manifest}")
    if 100 in eval_ks:
        print(f"mR@100={metrics.get('macro_recall_at_100', 0.0):.4f}, mR@500={metrics.get('macro_recall_at_500',0.0):.4f}")
        print(f"xR@100={metrics.get('micro_recall_at_100', 0.0):.4f}, xR@500={metrics.get('micro_recall_at_500',0.0):.4f}")
    if use_fallback:
        print("Using deterministic fallback backend:", fallback_name)


if __name__ == "__main__":
    main()
