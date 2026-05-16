#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import re

from round18.tools.common import (
    ALLOWED_DATA_FILES,
    find_forbidden_tokens,
    load_json,
    manifest_base,
    utc_now,
    sha256_file,
    sha256_text,
    write_json,
)


Token = str


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
    "will",
    "with",
    "would",
}


TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


@dataclass
class Candidate:
    evidence_id: str
    rank: int
    score: float
    source: str = "sparse_baseline"
    doc_index: int = -1


def tokenize(text: str) -> list[Token]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def ngram_tokens(tokens: list[Token], k: int = 3) -> list[Token]:
    if k < 2:
        return []
    out: list[Token] = []
    for token in tokens:
        if len(token) < k:
            continue
        for i in range(len(token) - k + 1):
            out.append(token[i : i + k])
    return out


def normalize_claims(
    raw_claims: dict[str, dict[str, Any]], max_claims: int
) -> dict[str, dict[str, Any]]:
    if max_claims <= 0:
        return raw_claims
    return {k: raw_claims[k] for k in list(raw_claims.keys())[:max_claims]}


def build_evidence_index(
    evidence: dict[str, str],
    claim_terms: dict[str, list[Token]],
    max_docs: int,
    use_char_ngrams: bool,
) -> tuple[list[str], list[Counter[Token]], list[int], dict[Token, list[tuple[int, int]]]]:
    query_vocab = set()
    for terms in claim_terms.values():
        query_vocab.update(terms)

    doc_ids: list[str] = []
    doc_term_counts: list[Counter[Token]] = []
    doc_lengths: list[int] = []
    postings: dict[Token, list[tuple[int, int]]] = defaultdict(list)

    for evidence_id, raw_text in evidence.items():
        if max_docs > 0 and len(doc_ids) >= max_docs:
            break
        words = tokenize(raw_text)
        expanded = words + ngram_tokens(words) if use_char_ngrams else words
        if not expanded:
            continue
        if query_vocab and not set(expanded) & query_vocab:
            continue
        counts = Counter(expanded)
        local_idx = len(doc_ids)
        doc_ids.append(evidence_id)
        doc_term_counts.append(counts)
        doc_lengths.append(sum(counts.values()))
        for token, tf in counts.items():
            postings[token].append((local_idx, tf))

    return doc_ids, doc_term_counts, doc_lengths, postings


def _idf(total_docs: int, df: int) -> float:
    return math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)


def rank_claim(
    claim_terms: list[Token],
    doc_ids: list[str],
    doc_term_counts: list[Counter[Token]],
    doc_lengths: list[int],
    postings: dict[Token, list[tuple[int, int]]],
    k1: float,
    b: float,
    source_label: str,
) -> list[Candidate]:
    n_docs = len(doc_ids)
    if n_docs == 0 or not claim_terms:
        return []
    avg_dl = sum(doc_lengths) / n_docs
    q_terms = Counter(claim_terms)
    score_by_doc: dict[int, float] = {}

    for token, qtf in q_terms.items():
        plist = postings.get(token, [])
        if not plist:
            continue
        idf = _idf(n_docs, len(plist))
        for doc_idx, tf in plist:
            dl = doc_lengths[doc_idx]
            norm = tf + k1 * (1.0 - b + b * (dl / max(avg_dl, 1.0)))
            score = idf * (tf * (k1 + 1.0) / norm) * (qtf / (qtf + 1.0))
            score_by_doc[doc_idx] = score_by_doc.get(doc_idx, 0.0) + score

    ranked = sorted(
        score_by_doc.items(),
        key=lambda item: (-item[1], doc_ids[item[0]]),
    )
    return [
        Candidate(
            evidence_id=doc_ids[doc_idx],
            rank=idx + 1,
            score=float(score),
            source=source_label,
            doc_index=doc_idx,
        )
        for idx, (doc_idx, score) in enumerate(ranked)
    ]


def build_pool(
    claims: dict[str, dict[str, Any]],
    doc_ids: list[str],
    doc_term_counts: list[Counter[Token]],
    doc_lengths: list[int],
    postings: dict[Token, list[tuple[int, int]]],
    k1: float,
    b: float,
    top_k: int,
) -> tuple[dict[str, list[Candidate]], dict[str, list[Token]]]:
    claim_term_map: dict[str, list[Token]] = {}
    pool: dict[str, list[Candidate]] = {}
    for claim_id, claim in claims.items():
        terms = tokenize(claim.get("claim_text", ""))
        claim_term_map[claim_id] = terms
        pool[claim_id] = rank_claim(
            claim_terms=terms,
            doc_ids=doc_ids,
            doc_term_counts=doc_term_counts,
            doc_lengths=doc_lengths,
            postings=postings,
            k1=k1,
            b=b,
            source_label="sparse_baseline",
        )[:top_k]
    return pool, claim_term_map


def parse_baseline_pool(
    path: str,
    claims: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, list[Candidate]]:
    if not path or path == ".":
        return {}
    baseline_path = Path(path)
    if not baseline_path.exists():
        return {}
    payload = load_json(baseline_path)
    if not isinstance(payload, dict):
        return {}

    output: dict[str, list[Candidate]] = {}
    for claim_id, entries in payload.items():
        if claim_id not in claims or not isinstance(entries, list):
            continue
        parsed: list[Candidate] = []
        for rank, item in enumerate(entries[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id:
                continue
            parsed.append(
                Candidate(
                    evidence_id=evidence_id,
                    rank=rank,
                    score=float(item.get("score", 0.0)),
                    source="external_sparse_baseline",
                )
            )
        if parsed:
            output[claim_id] = parsed
    return output


def rerank_with_expansion(
    claims: dict[str, dict[str, Any]],
    base_pool: dict[str, list[Candidate]],
    claim_term_map: dict[str, list[Token]],
    doc_term_counts: list[Counter[Token]],
    doc_ids: list[str],
    doc_lengths: list[int],
    postings: dict[Token, list[tuple[int, int]]],
    feedback_top_k: int,
    max_expansion_terms: int,
    feedback_decay: float,
    use_char_ngrams: bool,
    top_k: int,
) -> tuple[dict[str, list[Candidate]], list[dict[str, Any]]]:
    evidence_to_doc = {evidence_id: idx for idx, evidence_id in enumerate(doc_ids)}
    expanded_log: list[dict[str, Any]] = []
    prf_pool: dict[str, list[Candidate]] = {}

    for claim_id, claim in claims.items():
        base_terms = claim_term_map.get(claim_id, [])
        seed = base_pool.get(claim_id, [])[:feedback_top_k]
        term_scores: Counter[Token] = Counter()
        used_feedback = []

        for candidate in seed:
            used_feedback.append(candidate.evidence_id)
            rank_weight = 1.0 / (feedback_decay + candidate.rank)
            source_idx = evidence_to_doc.get(candidate.evidence_id, candidate.doc_index)
            if source_idx is None or source_idx < 0 or source_idx >= len(doc_term_counts):
                continue
            for token, tf in doc_term_counts[source_idx].items():
                if token in STOPWORDS or token in base_terms:
                    continue
                term_scores[token] += float(tf) * rank_weight

        expansion_terms = [token for token, _ in term_scores.most_common(max_expansion_terms)]
        expanded_terms = list(dict.fromkeys(base_terms + expansion_terms))
        if use_char_ngrams:
            expanded_terms.extend(ngram_tokens(expanded_terms))

        reranked = rank_claim(
            claim_terms=expanded_terms,
            doc_ids=doc_ids,
            doc_term_counts=doc_term_counts,
            doc_lengths=doc_lengths,
            postings=postings,
            k1=1.5,
            b=0.75,
            source_label="prf_rm3_lite",
        )[:top_k]

        prf_pool[claim_id] = reranked
        expanded_log.append(
            {
                "claim_id": claim_id,
                "original_query": claim.get("claim_text", ""),
                "expanded_query": " ".join(expanded_terms),
                "base_feedback_docs": used_feedback,
                "feedback_docs_used": len(used_feedback),
                "expansion_terms": expansion_terms,
                "added_term_count": len(expansion_terms),
                "base_top_evidence_ids": [item.evidence_id for item in seed],
                "prf_top_evidence_ids": [item.evidence_id for item in reranked[:min(10, top_k)]],
            }
        )

    return prf_pool, expanded_log


def recall_at_k(
    pool: dict[str, list[Candidate]], claims: dict[str, dict[str, Any]], k: int
) -> float:
    recalls: list[float] = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        top_evidence = {entry.evidence_id for entry in pool.get(claim_id, [])[:k]}
        recalls.append(len(gold.intersection(top_evidence)) / len(gold))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def precision_at_k(
    pool: dict[str, list[Candidate]], claims: dict[str, dict[str, Any]], k: int
) -> float:
    precisions: list[float] = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        top_candidates = pool.get(claim_id, [])[:k]
        if not top_candidates:
            precisions.append(0.0)
            continue
        hits = sum(1 for item in top_candidates if item.evidence_id in gold)
        precisions.append(hits / k)
    return float(sum(precisions) / len(precisions)) if precisions else 0.0


def candidate_delta_rows(
    claims: dict[str, dict[str, Any]],
    base_pool: dict[str, list[Candidate]],
    prf_pool: dict[str, list[Candidate]],
    k: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        base_top = {item.evidence_id for item in base_pool.get(claim_id, [])[:k]}
        prf_top = {item.evidence_id for item in prf_pool.get(claim_id, [])[:k]}
        base_recall = len(gold.intersection(base_top)) / len(gold) if gold else 0.0
        prf_recall = len(gold.intersection(prf_top)) / len(gold) if gold else 0.0
        delta = prf_recall - base_recall
        deltas.append(delta)
        rows.append(
            {
                "claim_id": claim_id,
                "gold_count": len(gold),
                "base_recall@{}".format(k): round(base_recall, 6),
                "prf_recall@{}".format(k): round(prf_recall, 6),
                "delta_recall@{}".format(k): round(delta, 6),
                "overlap_at_k": len(base_top.intersection(prf_top)),
                               "new_in_prf_top_k": sorted(prf_top - base_top),
                "dropped_from_baseline": sorted(base_top - prf_top),
            }
        )

    macro_delta = float(sum(deltas) / len(deltas)) if deltas else 0.0
    changed = sum(1 for row in rows if row["new_in_prf_top_k"] or row["dropped_from_baseline"])
    return {
        "k": k,
        "macro_delta_recall@{}".format(k): macro_delta,
        "claims_with_candidate_change": changed,
        "rows": rows,
    }


def candidate_json(pool: dict[str, list[Candidate]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for claim_id, candidates in pool.items():
        out[claim_id] = [
            {
                "evidence_id": item.evidence_id,
                "rank": item.rank,
                "score": item.score,
                "source": item.source,
            }
            for item in candidates
        ]
    return out


def to_rows(
    claims: dict[str, dict[str, Any]],
    base_pool: dict[str, list[Candidate]],
    prf_pool: dict[str, list[Candidate]],
    top_ks: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, pool in (("baseline", base_pool), ("prf", prf_pool)):
        for k in top_ks:
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "claims": len(claims),
                    "macro_recall": recall_at_k(pool, claims, k),
                    "precision_at_k": precision_at_k(pool, claims, k),
                }
            )
    return rows


def classify_input_split(path: Path) -> str:
    filename = path.name
    if filename in ALLOWED_DATA_FILES:
        return ALLOWED_DATA_FILES[filename]
    if "claims" in filename and "train" in filename:
        return "train"
    if "claims" in filename and "dev" in filename:
        return "dev"
    if "evidence" in filename:
        return "evidence"
    return "current_run_artifact"


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()

    read_paths: list[str] = []
    baseline_cli = str(args.baseline_pool)
    if baseline_cli == ".":
        baseline_cli = ""

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    read_paths.extend([str(args.claims), str(args.evidence)])

    claims = normalize_claims(claims, args.smoke_claims)
    claim_term_map: dict[str, list[Token]] = {
        claim_id: tokenize(payload.get("claim_text", ""))
        for claim_id, payload in claims.items()
    }

    doc_ids, doc_term_counts, doc_lengths, postings = build_evidence_index(
        evidence=evidence,
        claim_terms=claim_term_map,
        max_docs=args.max_evidence_docs,
        use_char_ngrams=args.use_char_ngrams,
    )

    baseline_pool = parse_baseline_pool(
        path=baseline_cli,
        claims=claims,
        top_k=args.pool_top_k,
    )
    baseline_source = "self_contained_sparse_baseline"
    if baseline_pool:
        baseline_source = "external_sparse_baseline"
        read_paths.append(baseline_cli)
    else:
        baseline_pool, claim_term_map = build_pool(
            claims=claims,
            doc_ids=doc_ids,
            doc_term_counts=doc_term_counts,
            doc_lengths=doc_lengths,
            postings=postings,
            k1=args.k1,
            b=args.b,
            top_k=args.pool_top_k,
        )

    prf_pool, expanded_rows = rerank_with_expansion(
        claims=claims,
        base_pool=baseline_pool,
        claim_term_map=claim_term_map,
        doc_term_counts=doc_term_counts,
        doc_ids=doc_ids,
        doc_lengths=doc_lengths,
        postings=postings,
        feedback_top_k=args.feedback_top_k,
        max_expansion_terms=args.max_expansion_terms,
        feedback_decay=args.feedback_decay,
        use_char_ngrams=args.use_char_ngrams,
        top_k=args.pool_top_k,
    )

    top_ks = sorted({args.summary_k, min(args.summary_k, args.pool_top_k), args.pool_top_k})
    rows = to_rows(claims, baseline_pool, prf_pool, top_ks)
    delta = candidate_delta_rows(claims, baseline_pool, prf_pool, args.summary_k)

    expansion_vocab = set()
    for row in expanded_rows:
        expansion_vocab.update(row["expansion_terms"])

    metrics = {
        "candidate_recall_macro@100": recall_at_k(prf_pool, claims, min(100, args.pool_top_k)),
        "baseline_recall_macro@100": recall_at_k(baseline_pool, claims, min(100, args.pool_top_k)),
        "evidence_precision_at_candidate_k": precision_at_k(
            prf_pool, claims, args.summary_k
        ),
        "evidence_precision_at_candidate_k_value": args.summary_k,
        "expansion_vocab_size": len(expansion_vocab),
        "baseline_source": baseline_source,
        "pool_top_k": args.pool_top_k,
        "claims_processed": len(claims),
        "evidence_index_size": len(doc_ids),
        "smoke_claims": args.smoke_claims,
        "smoke_evidence_limit": args.max_evidence_docs,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / f"candidate_pool_baseline_top{args.pool_top_k}.json"
    prf_path = output_dir / f"candidate_pool_prf_top{args.pool_top_k}.json"
    expanded_path = output_dir / "expanded_queries.json"
    delta_path = output_dir / "per_claim_delta.json"
    rows_path = output_dir / "candidate_recall_summary.csv"
    smoke_summary_path = output_dir / "smoke_summary.json"

    write_json(baseline_path, candidate_json(baseline_pool))
    write_json(prf_path, candidate_json(prf_pool))
    write_json(expanded_path, expanded_rows)
    write_json(delta_path, delta)
    write_json(
        smoke_summary_path,
        {
            "created_utc": utc_now(),
            "claims_file": str(args.claims),
            "evidence_file": str(args.evidence),
            "arg_payload": serializable_args(args),
            "metrics": metrics,
            "summary_rows": rows,
        },
    )
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    forbidden_hits = find_forbidden_tokens(read_paths)
    manifest = manifest_base(
        run_id="o_s3_prf_smoke",
        status=args.status,
        mode=args.mode,
        stage=args.stage,
        command=" ".join(shlex.quote(item) for item in sys.argv),
        working_directory=Path.cwd(),
        config_path=str(args.manifest),
        config_hash=(
            args.command_hash
            if args.command_hash
            else sha256_text(json.dumps(serializable_args(args), sort_keys=True))
        ),
        random_seed=args.random_seed,
    )
    manifest["input_files"] = []
    input_paths = [args.claims, args.evidence]
    if baseline_cli:
        input_paths.append(Path(baseline_cli))

    for path in input_paths:
        split = classify_input_split(path)
        manifest["input_files"].append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "split": split,
                "labels_used": split in {"train", "dev"},
            }
        )

    manifest["forbidden_input_scan"] = {
        "passed": not bool(forbidden_hits),
        "notes": "No forbidden tokens found in scanned read paths." if not forbidden_hits else str(forbidden_hits),
    }
    manifest["output_files"] = [
        str(baseline_path),
        str(prf_path),
        str(expanded_path),
        str(delta_path),
        str(rows_path),
        str(smoke_summary_path),
        str(args.manifest),
    ]
    manifest["metrics"] = metrics
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 3)
    manifest["data_flow_summary"] = (
        "Claims and evidence were loaded from raw JSON, a sparse lexical index was built from evidence "
        "matching claim term vocabulary, baseline retrieval was computed (or an explicitly supplied baseline pool was used), "
        "then RM3-lite terms from top baseline feedback docs were appended and re-ranked."
    )
    manifest["split_isolation_summary"] = (
        "No labels are used in ranking formulas. For smoke, only fixed hyperparameters were used and metrics are reported only."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low"
    manifest["notes"] = (
        "Pure sparse lexical prototype: token-level retrieval, local feedback expansion, no saved neural model files."
    )
    write_json(args.manifest, manifest)

    return {
        "metrics": metrics,
        "read_paths": read_paths,
        "produced": {
            "candidate_pool_baseline_path": str(baseline_path),
            "candidate_pool_prf_path": str(prf_path),
            "expanded_query_path": str(expanded_path),
            "delta_path": str(delta_path),
            "summary_csv_path": str(rows_path),
            "smoke_summary_path": str(smoke_summary_path),
            "run_manifest": str(args.manifest),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round18 O-S3 PRF/RM3-lite lexical expansion prototype (sparse only)."
    )
    parser.add_argument("--claims", type=Path, default=Path("data/train-claims.json"))
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence.json"))
    parser.add_argument(
        "--baseline-pool",
        type=str,
        default="",
        help="Optional previous sparse candidate pool for per-claim delta baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("round18/outputs/o_sparse/o_s3_prf"))
    parser.add_argument("--manifest", type=Path, default=Path("round18/outputs/o_sparse/o_s3_prf/run_manifest.json"))
    parser.add_argument("--smoke-claims", type=int, default=32, help="Bound claims for smoke run.")
    parser.add_argument("--max-evidence-docs", type=int, default=5000, help="Bound evidence scan in smoke run.")
    parser.add_argument("--pool-top-k", type=int, default=100)
    parser.add_argument("--summary-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--feedback-top-k", type=int, default=20)
    parser.add_argument("--max-expansion-terms", type=int, default=12)
    parser.add_argument("--feedback-decay", type=float, default=1.0)
    parser.add_argument("--use-char-ngrams", action="store_true")
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument(
        "--status",
        choices=["strict-candidate", "diagnostic-only", "rejected"],
        default="strict-candidate",
    )
    parser.add_argument("--mode", choices=["STRICT", "DIAGNOSTIC"], default="STRICT")
    parser.add_argument("--stage", default="O-S3")
    parser.add_argument("--command-hash", default="", help="Optional config hash override for manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    metrics = result["metrics"]
    produced = result["produced"]
    print("Wrote:")
    for path in produced.values():
        print(f" - {path}")
    print(
        f"candidate_recall_macro@100={metrics['candidate_recall_macro@100']:.6f} "
        f"precision@{metrics['evidence_precision_at_candidate_k_value']}={metrics['evidence_precision_at_candidate_k']:.6f} "
        f"expansion_vocab_size={metrics['expansion_vocab_size']}"
    )


if __name__ == "__main__":
    main()
