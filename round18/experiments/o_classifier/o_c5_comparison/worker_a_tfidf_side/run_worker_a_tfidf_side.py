#!/usr/bin/env python
"""Round18 O-C5 strict side-feature comparison for TF-IDF claim classification.

Runs strict train/dev candidate-context classification using the existing O-C4 CE factual
candidate pool, comparing:
  - TF-IDF only
  - TF-IDF + shallow side features

Selection is constrained to train-holdout splits only; dev is always held out for final confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MaxAbsScaler

def _resolve_repo_root() -> Path:
    candidates = list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "round18" / "tools").exists():
            return candidate
    return Path(__file__).resolve().parents[5]


REPO_ROOT = _resolve_repo_root()
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from round18.tools.common import (  # noqa: E402
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


DEFAULT_TRAIN_CLAIMS = Path("data/train-claims.json")
DEFAULT_DEV_CLAIMS = Path("data/dev-claims.json")
DEFAULT_EVIDENCE = Path("data/evidence.json")
DEFAULT_TRAIN_POOL = Path(
    "round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json"
)
DEFAULT_DEV_POOL = Path(
    "round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json"
)
DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/o_c5_comparison/worker_a_tfidf_side")
DEFAULT_RUN_ID = "worker_a_tfidf_side"
DEFAULT_NGRAM_GRID = "1-2"
DEFAULT_MAX_FEATURES_GRID = "30000,60000"
DEFAULT_LOGREG_C_GRID = "0.5,1.0,2.0"
LABEL_ORDER = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}
ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9]{1,}$")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
ENTITY_STOP = {
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "If",
    "For",
    "To",
    "Of",
    "By",
    "And",
    "Or",
    "That",
    "This",
    "These",
    "Those",
    "It",
    "Its",
}
ENTITY_STOP_LOWER = {value.lower() for value in ENTITY_STOP}

SIDE_FEATURE_NAMES = [
    "claim_token_count",
    "claim_char_count",
    "claim_entity_count",
    "claim_number_count",
    "claim_year_count",
    "claim_to_evidence_ratio_mean",
    "claim_to_evidence_ratio_top1",
    "evidence_count",
    "evidence_unique_variant_count",
    "evidence_token_mean",
    "evidence_token_max",
    "evidence_token_min",
    "evidence_token_std",
    "evidence_char_mean",
    "evidence_char_max",
    "evidence_char_min",
    "evidence_char_std",
    "source_rank_top1",
    "source_rank_min",
    "source_rank_mean",
    "source_rank_max",
    "source_rank_std",
    "source_score_top1",
    "source_score_mean",
    "source_score_std",
    "source_score_min",
    "source_score_max",
    "ce_rank_top1",
    "ce_rank_mean",
    "ce_rank_std",
    "ce_score_top1",
    "ce_score_mean",
    "ce_score_std",
    "lexical_jaccard_top1",
    "lexical_jaccard_mean",
    "lexical_recall_top1",
    "lexical_recall_mean",
    "lexical_precision_top1",
    "lexical_precision_mean",
    "entity_overlap_top1",
    "entity_overlap_mean",
    "number_overlap_top1",
    "number_overlap_mean",
    "year_overlap_top1",
    "year_overlap_mean",
    "inv_source_rank_top1",
    "inv_source_rank_mean",
]


@dataclass(frozen=True)
class ContextItem:
    evidence_id: str
    rank: int
    score: float
    source_rank: int
    source_score: float
    ce_rank: int
    ce_score: float
    reranker_variant: str
    source_count: int
    source: str
    text: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-C5 strict classifier side-feature comparison over CE-factual contexts."
        )
    )
    parser.add_argument(
        "--train-claims",
        type=Path,
        default=DEFAULT_TRAIN_CLAIMS,
        help="Train claims JSON.",
    )
    parser.add_argument(
        "--dev-claims",
        type=Path,
        default=DEFAULT_DEV_CLAIMS,
        help="Dev claims JSON.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence JSON.",
    )
    parser.add_argument(
        "--train-pool",
        type=Path,
        default=DEFAULT_TRAIN_POOL,
        required=False,
        help="Explicit strict train context pool.",
    )
    parser.add_argument(
        "--dev-pool",
        type=Path,
        default=DEFAULT_DEV_POOL,
        required=False,
        help="Explicit strict dev context pool.",
    )
    parser.add_argument(
        "--context-ks",
        default="5,10,20,32,64",
        help="Comma-separated context sizes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for O-C5 experiments.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run ID prefix.",
    )
    parser.add_argument("--evidence-token-budget", type=int, default=0)
    parser.add_argument(
        "--clf-max-features-grid",
        default=DEFAULT_MAX_FEATURES_GRID,
    )
    parser.add_argument(
        "--clf-ngram-grid",
        default=DEFAULT_NGRAM_GRID,
    )
    parser.add_argument(
        "--logreg-c-grid",
        default=DEFAULT_LOGREG_C_GRID,
    )
    parser.add_argument(
        "--selection-mode",
        choices=("train_holdout",),
        default="train_holdout",
    )
    parser.add_argument("--train-val-fraction", type=float, default=0.2)
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=0.3585558388189967,
    )
    parser.add_argument("--baseline-macro-f1", type=float, default=0.4719276094276095)
    parser.add_argument("--run-tfidf-only", action="store_true")
    parser.add_argument("--run-tfidf-shallow", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _parse_int_grid(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        v = int(item)
        if v <= 0:
            raise argparse.ArgumentTypeError("context size must be >0.")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("At least one context size is required.")
    return sorted(set(out))


def _parse_float_grid(raw: str) -> list[float]:
    out: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise argparse.ArgumentTypeError("float grid must contain at least one value.")
    return out


def _parse_int_grid2(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("int grid must contain at least one value.")
    return out


def _parse_ngram_grid(raw: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", item)
        if not m:
            raise argparse.ArgumentTypeError(
                f"Invalid ngram range '{item}' (expected lo-hi, e.g., 1-2)."
            )
        lo = int(m.group(1))
        hi = int(m.group(2))
        if lo <= 0 or hi <= 0 or lo > hi:
            raise argparse.ArgumentTypeError(f"Invalid ngram range '{item}'.")
        ranges.append((lo, hi))
    if not ranges:
        raise argparse.ArgumentTypeError("ngram grid must contain at least one value.")
    return ranges


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_pool_family(pool_path: Path) -> str:
    stem = pool_path.name.replace("-", "_")
    stem = stem.removesuffix(".json")
    stem = re.sub(r"_top\d+_candidates$", "", stem)
    stem = re.sub(r"_candidates$", "", stem)
    m = re.search(r"(?:^|_)(o_[a-z0-9_]+)(?:_|$)", stem)
    if not m:
        raise SystemExit(f"Cannot infer pool family from: {pool_path}")
    return m.group(1)


def _ensure_family_contract(train_pool: Path, dev_pool: Path) -> tuple[str, str]:
    train_family = _extract_pool_family(train_pool)
    dev_family = _extract_pool_family(dev_pool)
    if train_family != dev_family:
        raise SystemExit(
            "Strict context source mismatch: "
            f"{train_family!r} (train-pool) != {dev_family!r} (dev-pool)."
        )
    return train_family, dev_family


def _validate_label_set(claims: dict[str, dict[str, Any]], split_name: str) -> None:
    unknown = set()
    for claim in claims.values():
        label = claim.get("claim_label")
        if label not in LABEL_ORDER:
            unknown.add(str(label))
    if unknown:
        raise SystemExit(
            f"{split_name} labels contain unsupported classes: {sorted(unknown)}"
        )


def _build_context_items(
    rows: list[Any],
    evidence: dict[str, str],
    top_k: int,
    source_family: str,
) -> tuple[list[ContextItem], dict[str, int]]:
    items: list[ContextItem] = []
    diagnostics: dict[str, int] = {"missing_evidence_text": 0, "deduplicated": 0}
    seen: set[str] = set()

    for pos, row in enumerate(rows):
        if len(items) >= top_k:
            break
        if not isinstance(row, dict):
            continue

        evidence_id = _coerce_text(row.get("evidence_id")).strip()
        if not evidence_id or evidence_id in seen:
            if evidence_id in seen:
                diagnostics["deduplicated"] += 1
            continue
        seen.add(evidence_id)

        evidence_text = evidence.get(evidence_id, "")
        if not evidence_text:
            diagnostics["missing_evidence_text"] += 1

        items.append(
            ContextItem(
                evidence_id=evidence_id,
                rank=_parse_int(row.get("rank"), default=pos + 1),
                score=_parse_float(row.get("score"), default=0.0),
                source_rank=_parse_int(row.get("source_rank"), default=_parse_int(row.get("rank"), default=pos + 1)),
                source_score=_parse_float(
                    row.get("source_score"),
                    default=_parse_float(row.get("score", 0.0)),
                ),
                ce_rank=_parse_int(row.get("ce_rank"), default=_parse_int(row.get("rank"), default=pos + 1)),
                ce_score=_parse_float(row.get("ce_score"), default=0.0),
                reranker_variant=_coerce_text(row.get("reranker_variant")),
                source_count=_parse_int(
                    row.get("source_count"),
                    default=1,
                ),
                source=source_family,
                text=_coerce_text(evidence_text),
            )
        )
    return items, diagnostics


def _build_contexts(
    claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    pool: dict[str, Any],
    top_k: int,
    source_family: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "missing_claims_in_pool": [],
        "insufficient_context_claims": [],
        "missing_evidence_text": 0,
        "total_candidates": 0,
    }

    for claim_id, claim in claims.items():
        if claim_id not in pool:
            raise SystemExit(f"Strict mode requires pool entry for claim: {claim_id}")
        raw_rows = pool[claim_id]
        if not isinstance(raw_rows, list):
            raise SystemExit(f"Invalid pool row type for {claim_id}: {type(raw_rows)}")
        items, diag = _build_context_items(
            raw_rows,
            evidence=evidence,
            top_k=top_k,
            source_family=source_family,
        )
        diagnostics["missing_evidence_text"] += diag["missing_evidence_text"]
        diagnostics["total_candidates"] += len(items)
        if len(items) < top_k:
            diagnostics["insufficient_context_claims"].append(claim_id)
            raise SystemExit(
                f"Strict mode: no hidden fallback, insufficient context for claim {claim_id}: "
                f"need {top_k}, got {len(items)}."
            )
        contexts[claim_id] = {
            "claim_id": claim_id,
            "claim_text": claim.get("claim_text", ""),
            "claim_label": claim.get("claim_label"),
            "context_source_family": source_family,
            "classifier_context_top_k": top_k,
            "context_item_count": len(items),
            "final_evidence_candidates": [item.evidence_id for item in items],
            "classifier_evidence_context": [item.__dict__ for item in items],
        }
    return contexts, diagnostics


def _build_claim_context_text(
    claim_text: str,
    evidence_items: list[ContextItem],
    evidence_token_budget: int = 0,
) -> str:
    blocks = [f"CLAIM: {claim_text.strip()}"]
    for item in evidence_items:
        if not item.text:
            continue
        text = item.text
        if evidence_token_budget > 0:
            text = " ".join(text.split()[:evidence_token_budget])
        blocks.append(f"EVIDENCE_{item.rank}: {text}")
    return "\n".join(blocks)


def _rows_from_contexts(
    claims: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    evidence_token_budget: int,
) -> tuple[list[str], list[str], list[str], dict[str, list[ContextItem]]]:
    claim_ids: list[str] = list(claims.keys())
    texts: list[str] = []
    labels: list[str] = []
    claim_items: dict[str, list[ContextItem]] = {}

    for claim_id in claim_ids:
        context = contexts[claim_id]
        items = [ContextItem(**row) for row in context["classifier_evidence_context"]]
        claim_items[claim_id] = items
        texts.append(
            _build_claim_context_text(
                claims[claim_id].get("claim_text", ""),
                items,
                evidence_token_budget=evidence_token_budget,
            )
        )
        labels.append(str(claims[claim_id].get("claim_label")))
    return claim_ids, texts, labels, claim_items


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(_coerce_text(text))]


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _is_capitalized_word(word: str) -> bool:
    return bool(word) and word[0].isupper() and not word.isupper()


def _finalize_entity_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) >= 3 and token not in ENTITY_STOP:
            return [token]
        return []
    phrase = " ".join(tokens)
    if len(phrase) <= 3:
        return []
    return [phrase] + tokens


def _extract_entities(text: str) -> list[str]:
    tokens = _tokenize(text)
    entities: list[str] = []
    current: list[str] = []
    for raw_token in WORD_RE.findall(text):
        lower = raw_token.lower()
        if raw_token in ("", " "):
            if current:
                entities.extend(_finalize_entity_tokens(current))
                current = []
            continue
        if len(raw_token) < 2 or lower in ENTITY_STOP_LOWER:
            if current:
                entities.extend(_finalize_entity_tokens(current))
                current = []
            continue
        if _is_capitalized_word(raw_token) or ACRONYM_RE.fullmatch(raw_token):
            current.append(raw_token)
        else:
            if current:
                entities.extend(_finalize_entity_tokens(current))
                current = []
    if current:
        entities.extend(_finalize_entity_tokens(current))
    return _dedupe_preserve([value for value in entities if value])


def _text_profile(text: str) -> dict[str, Any]:
    tokens = _tokenize(text)
    content_tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    entities = _extract_entities(text)
    return {
        "tokens": tokens,
        "token_set": set(tokens),
        "content_set": set(content_tokens),
        "char_count": len(_coerce_text(text)),
        "entities": set(e.lower() for e in entities),
        "numbers": set(NUMBER_RE.findall(text)),
        "years": set(YEAR_RE.findall(text)),
        "claim_len": max(1, len(tokens)),
        "content_len": max(1, len(content_tokens)),
    }


def _build_profile_cache(claims: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for claim_id, claim in claims.items():
        profiles[claim_id] = _text_profile(claim.get("claim_text", ""))
    return profiles


def _build_shallow_features_for_claim(
    claim_profile: dict[str, Any],
    item_features: list[tuple[ContextItem, dict[str, Any]]],
    _claim_k: int,
) -> dict[str, float]:
    if not item_features:
        return {name: 0.0 for name in SIDE_FEATURE_NAMES}

    lexical_jaccard: list[float] = []
    lexical_recall: list[float] = []
    lexical_precision: list[float] = []
    entity_overlap: list[float] = []
    number_overlap: list[float] = []
    year_overlap: list[float] = []
    ev_token_counts: list[float] = []
    ev_char_counts: list[float] = []
    source_ranks: list[float] = []
    source_scores: list[float] = []
    source_counts: set[str] = set()
    ce_ranks: list[float] = []
    ce_scores: list[float] = []
    claim_content_set = claim_profile["content_set"]
    claim_entities = claim_profile["entities"]
    claim_numbers = claim_profile["numbers"]
    claim_years = claim_profile["years"]

    for item, eprof in item_features:
        e_tokens = eprof["content_set"]
        e_chars = eprof["char_count"]
        source_ranks.append(float(item.source_rank))
        source_scores.append(_parse_float(item.source_score, default=0.0))
        source_counts.add(item.reranker_variant or "")
        ce_ranks.append(float(item.ce_rank))
        ce_scores.append(_parse_float(item.ce_score, default=0.0))
        ev_token_counts.append(max(1.0, float(len(eprof["tokens"]))))
        ev_char_counts.append(float(e_chars))

        inter = len(claim_content_set & eprof["content_set"])
        union = len(claim_content_set | eprof["content_set"])
        lexical_jaccard.append(_safe_div(inter, union))
        lexical_recall.append(_safe_div(inter, len(claim_content_set)))
        lexical_precision.append(_safe_div(inter, len(e_tokens)))

        entity_overlap.append(float(len(claim_entities & eprof["entities"])))
        number_overlap.append(_safe_div(len(claim_numbers & eprof["numbers"]), max(1, len(claim_numbers))))
        year_overlap.append(_safe_div(len(claim_years & eprof["years"]), max(1, len(claim_years))))

    def _agg(values: list[float]) -> tuple[float, float, float, float]:
        arr = np.asarray(values, dtype=float)
        return (
            float(np.mean(arr)),
            float(np.max(arr)),
            float(np.min(arr)),
            float(np.std(arr)),
        )

    ev_mean, ev_max, ev_min, ev_std = _agg(ev_token_counts)
    ev_char_mean, ev_char_max, ev_char_min, ev_char_std = _agg(ev_char_counts)
    sr_mean, sr_max, sr_min, sr_std = _agg(source_ranks)
    ss_mean, ss_max, ss_min, ss_std = _agg(source_scores)
    ce_mean, ce_max, ce_min, ce_std = _agg(ce_ranks)
    cs_mean, cs_max, cs_min, cs_std = _agg(ce_scores)
    lj_mean, lj_max, _lj_min, _lj_std = _agg(lexical_jaccard)
    lr_mean, lr_max, _lr_min, _lr_std = _agg(lexical_recall)
    lp_mean, lp_max, _lp_min, _lp_std = _agg(lexical_precision)
    eo_mean, eo_max, _eo_min, _eo_std = _agg(entity_overlap)
    no_mean, no_max, _no_min, _no_std = _agg(number_overlap)
    yo_mean, yo_max, _yo_min, _yo_std = _agg(year_overlap)

    first = item_features[0][0]
    claim_tokens_count = float(len(claim_profile["tokens"]))
    first_text_profile = item_features[0][1]
    return {
        "claim_token_count": claim_tokens_count,
        "claim_char_count": float(max(1, claim_profile.get("char_count", 0))),
        "claim_entity_count": float(len(claim_entities)),
        "claim_number_count": float(len(claim_numbers)),
        "claim_year_count": float(len(claim_years)),
        "claim_to_evidence_ratio_mean": _safe_div(np.sum(ev_token_counts), claim_tokens_count),
        "claim_to_evidence_ratio_top1": _safe_div(float(len(first_text_profile["tokens"])), claim_tokens_count),
        "evidence_count": float(len(item_features)),
        "evidence_unique_variant_count": float(len(source_counts)),
        "evidence_token_mean": ev_mean,
        "evidence_token_max": ev_max,
        "evidence_token_min": ev_min,
        "evidence_token_std": ev_std,
        "evidence_char_mean": ev_char_mean,
        "evidence_char_max": ev_char_max,
        "evidence_char_min": ev_char_min,
        "evidence_char_std": ev_char_std,
        "source_rank_top1": float(first.source_rank),
        "source_rank_min": sr_min,
        "source_rank_mean": sr_mean,
        "source_rank_max": sr_max,
        "source_rank_std": sr_std,
        "source_score_top1": float(first.source_score if first.source_score else _parse_float(first.score)),
        "source_score_mean": ss_mean,
        "source_score_std": ss_std,
        "source_score_min": ss_min,
        "source_score_max": ss_max,
        "ce_rank_top1": float(first.ce_rank),
        "ce_rank_mean": ce_mean,
        "ce_rank_std": ce_std,
        "ce_score_top1": first.ce_score,
        "ce_score_mean": cs_mean,
        "ce_score_std": cs_std,
        "lexical_jaccard_top1": lexical_jaccard[0] if lexical_jaccard else 0.0,
        "lexical_jaccard_mean": lj_mean,
        "lexical_recall_top1": lexical_recall[0] if lexical_recall else 0.0,
        "lexical_recall_mean": lr_mean,
        "lexical_precision_top1": lexical_precision[0] if lexical_precision else 0.0,
        "lexical_precision_mean": lp_mean,
        "entity_overlap_top1": entity_overlap[0] if entity_overlap else 0.0,
        "entity_overlap_mean": eo_mean,
        "number_overlap_top1": number_overlap[0] if number_overlap else 0.0,
        "number_overlap_mean": no_mean,
        "year_overlap_top1": year_overlap[0] if year_overlap else 0.0,
        "year_overlap_mean": yo_mean,
        "inv_source_rank_top1": _safe_div(1.0, float(first.source_rank)),
        "inv_source_rank_mean": _safe_div(np.sum(1.0 / (np.asarray(source_ranks) + 1.0)), len(item_features)),
    }


def _build_shallow_matrix(
    claim_ids: list[str],
    claim_profiles: dict[str, dict[str, Any]],
    claim_context_items: dict[str, list[ContextItem]],
    evidence: dict[str, str],
) -> tuple[csr_matrix, dict[str, Any]]:
    evidence_profiles: dict[str, dict[str, Any]] = {}
    rows: list[list[float]] = []
    for claim_id in claim_ids:
        items = claim_context_items[claim_id]
        pair_items: list[tuple[ContextItem, dict[str, Any]]] = []
        for item in items:
            if item.evidence_id not in evidence_profiles:
                evidence_profiles[item.evidence_id] = _text_profile(evidence.get(item.evidence_id, ""))
            pair_items.append((item, evidence_profiles[item.evidence_id]))
        feature_map = _build_shallow_features_for_claim(claim_profiles[claim_id], pair_items, len(items))
        rows.append([float(feature_map[name]) for name in SIDE_FEATURE_NAMES])
    return csr_matrix(np.asarray(rows, dtype=np.float32)), evidence_profiles


def _load_inputs(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    for path in (args.train_claims, args.dev_claims, args.evidence, args.train_pool, args.dev_pool):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")
    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pool = load_json(args.train_pool)
    dev_pool = load_json(args.dev_pool)

    if not isinstance(train_claims, dict):
        raise SystemExit("train-claims must be an object keyed by claim_id.")
    if not isinstance(dev_claims, dict):
        raise SystemExit("dev-claims must be an object keyed by claim_id.")
    if not isinstance(evidence, dict):
        raise SystemExit("evidence must be an object keyed by evidence_id.")
    if not isinstance(train_pool, dict):
        raise SystemExit("train-pool must be an object keyed by claim_id.")
    if not isinstance(dev_pool, dict):
        raise SystemExit("dev-pool must be an object keyed by claim_id.")
    _validate_label_set(train_claims, "train")
    _validate_label_set(dev_claims, "dev")
    return train_claims, dev_claims, evidence, train_pool, dev_pool


def _evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    y_prob: np.ndarray,
) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="micro", zero_division=0)
    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    per_class_recall: dict[str, float] = {}
    row_totals = matrix.sum(axis=1).astype(float)
    for idx, label in enumerate(LABEL_ORDER):
        per_class_recall[label] = _safe_div(float(matrix[idx, idx]), row_totals[idx])

    hist = {label: 0 for label in LABEL_ORDER}
    for pred in y_pred:
        hist[pred] = hist.get(pred, 0) + 1
    total_pred = max(len(y_pred), 1)
    max_share = max(hist.values()) / total_pred if total_pred else 0.0
    max_label = max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_recall": float(macro_recall),
        "classification_report": report,
        "per_class_recall": per_class_recall,
        "prediction_histogram": hist,
        "top_class_share": float(max_share),
        "top_class": max_label,
        "confusion_matrix": matrix.tolist(),
        "proba": y_prob.tolist(),
    }


def _collapse_gate(prediction_hist: dict[str, int], threshold: float) -> dict[str, Any]:
    total = sum(prediction_hist.values())
    if total == 0:
        return {
            "status": "failed",
            "reason": "no predictions produced",
            "threshold": float(threshold),
            "max_class_share": 0.0,
            "collapsed_classes": [],
        }
    max_class = max(prediction_hist.items(), key=lambda kv: (kv[1], kv[0]))
    share = max_class[1] / total
    passed = share <= threshold
    collapsed = [label for label, count in prediction_hist.items() if _safe_div(count, total) > threshold]
    return {
        "status": "passed" if passed else "failed",
        "reason": (
            "passed"
            if passed
            else f"single-class collapse above threshold {threshold}: {max_class[0]}"
        ),
        "max_class": max_class[0],
        "max_class_share": float(share),
        "threshold": float(threshold),
        "collapsed_classes": collapsed,
    }


def _acceptance_gate(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    histogram = metrics.get("prediction_histogram", {})
    per_class_recall = metrics.get("per_class_recall", {})
    if any(int(histogram.get(label, 0)) <= 0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_predictions")
    if any(float(per_class_recall.get(label, 0.0)) <= 0.0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_recall")
    if float(metrics.get("top_class_share", 1.0)) > float(args.collapse_threshold):
        failures.append("max_class_share_above_threshold")
    if float(metrics.get("macro_f1", 0.0)) < float(args.min_macro_f1):
        failures.append("macro_f1_below_threshold")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "thresholds": {
            "max_class_share": float(args.collapse_threshold),
            "min_macro_f1": float(args.min_macro_f1),
            "require_nonzero_predictions_for_all_classes": True,
            "require_nonzero_recall_for_all_classes": True,
        },
    }


def _selection_score(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result["macro_f1"]),
        float(result["micro_f1"]),
        float(result["macro_recall"]),
        -float(result["top_class_share"]),
    )


def _train_holdout_indices(labels: list[str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args.train_val_fraction),
        random_state=args.random_seed,
    )
    idx = np.arange(len(labels))
    try:
        train_idx, val_idx = next(splitter.split(idx, labels))
    except ValueError:
        return None
    return train_idx, val_idx


def _variant_key(
    feature_mode: str,
    use_side_features: bool,
    tfidf_config: dict[str, Any],
    c_value: float,
    *,
    n_side_features: int = 0,
) -> str:
    base = "tfidf_logreg"
    suffix = "only" if feature_mode == "tfidf_only" else "plus_side"
    return (
        f"{base}_{suffix}_mf{int(tfidf_config['max_features'])}"
        f"_ngram{tfidf_config['ngram_range'][0]}x{tfidf_config['ngram_range'][1]}"
        f"_c{str(c_value).replace('.', 'p')}"
        f"_sf{int(n_side_features)}"
    )


def _train_and_evaluate(
    model_id: str,
    train_texts: list[str],
    y_train: list[str],
    dev_texts: list[str],
    y_dev: list[str],
    tfidf_config: dict[str, Any],
    c_val: float,
    args: argparse.Namespace,
    train_side: csr_matrix | None,
    dev_side: csr_matrix | None,
    selected_split: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=tuple(tfidf_config["ngram_range"]),
        max_features=int(tfidf_config["max_features"]),
        dtype=np.float32,
    )
    x_train_text = tfidf.fit_transform(train_texts)
    x_dev_text = tfidf.transform(dev_texts)

    if train_side is not None and dev_side is not None and model_id == "tfidf_with_shallow":
        if selected_split is None:
            scaler = MaxAbsScaler()
            train_side_scaled = scaler.fit_transform(train_side)
            dev_side_scaled = scaler.transform(dev_side)
        else:
            tr_idx, val_idx = selected_split
            scaler = MaxAbsScaler()
            train_side_scaled = scaler.fit_transform(train_side[tr_idx])
            dev_side_scaled = scaler.transform(train_side[val_idx])
        x_train = hstack([x_train_text, train_side_scaled], format="csr")
        x_dev = hstack([x_dev_text, dev_side_scaled], format="csr")
    else:
        x_train = x_train_text
        x_dev = x_dev_text

    model = LogisticRegression(
        C=float(c_val),
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
    )

    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    y_pred = model.predict(x_dev).astype(str).tolist()
    y_prob = model.predict_proba(x_dev).astype(float)
    stats = _evaluate_predictions(y_dev, y_pred, y_prob)
    stats["fit_seconds"] = float(fit_seconds)
    stats["classes_"] = [str(c) for c in model.classes_]
    stats["vectorizer_dim"] = int(x_train_text.shape[1])
    if train_side is not None and dev_side is not None and model_id == "tfidf_with_shallow":
        stats["side_features"] = int(train_side.shape[1])
    else:
        stats["side_features"] = 0
    stats["config"] = {
        "model_id": model_id,
        "tfidf_max_features": int(tfidf_config["max_features"]),
        "tfidf_ngram_range": list(tfidf_config["ngram_range"]),
        "C": float(c_val),
        "feature_mode": "tfidf_with_shallow" if model_id == "tfidf_with_shallow" else "tfidf_only",
    }
    return stats, y_pred, y_prob


def _grid_search(
    model_id: str,
    train_texts: list[str],
    y_train: list[str],
    dev_texts: list[str],
    y_dev: list[str],
    args: argparse.Namespace,
    use_side_features: bool,
    train_side: csr_matrix | None,
    dev_side: csr_matrix | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], np.ndarray, dict[str, Any]]:
    c_grid = _parse_float_grid(args.logreg_c_grid)
    max_features_grid = _parse_int_grid2(args.clf_max_features_grid)
    ngram_grid = _parse_ngram_grid(args.clf_ngram_grid)

    configs: list[dict[str, Any]] = [
        {
            "model_id": "tfidf_with_shallow" if use_side_features else "tfidf_only",
            "tfidf_max_features": max_features,
            "tfidf_ngram_range": list(ngram_range),
            "C": c_value,
        }
        for max_features in max_features_grid
        for ngram_range in ngram_grid
        for c_value in c_grid
    ]

    selection_meta: dict[str, Any] = {
        "selection_mode": "train_holdout",
        "selected_by": "pre_registered_first_config",
        "train_val_fraction": float(args.train_val_fraction),
        "selection_results": [],
    }
    selected_config = configs[0]

    if args.selection_mode == "train_holdout":
        split = _train_holdout_indices(y_train, args)
        if split is None:
            selection_meta["selected_by"] = "fallback_first_config_insufficient_train_holdout"
        else:
            train_idx, val_idx = split
            train_texts_sub = [train_texts[i] for i in train_idx]
            dev_texts_sub = [train_texts[i] for i in val_idx]
            y_train_sub = [y_train[i] for i in train_idx]
            y_val = [y_train[i] for i in val_idx]
            for config in configs:
                result, _, _ = _train_and_evaluate(
                    model_id=config["model_id"],
                    train_texts=train_texts_sub,
                    y_train=y_train_sub,
                    dev_texts=dev_texts_sub,
                    y_dev=y_val,
                    tfidf_config={
                        "max_features": int(config["tfidf_max_features"]),
                        "ngram_range": tuple(config["tfidf_ngram_range"]),
                    },
                    c_val=config["C"],
                    args=args,
                    train_side=train_side,
                    dev_side=dev_side,
                    selected_split=(train_idx, val_idx),
                )
                result["selection_config"] = config
                result["selection_fold"] = {
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                }
                selection_results = result
                selection_meta["selection_results"].append(result)
            selection_meta["selection_results"].sort(key=_selection_score, reverse=True)
            selected_config = selection_meta["selection_results"][0]["selection_config"]
            selection_meta["selected_by"] = "train_holdout_macro_f1"

    results: list[dict[str, Any]] = []
    predicts: list[list[str]] = []
    probs: list[np.ndarray] = []

    for config in configs:
        result, y_pred, y_prob = _train_and_evaluate(
            model_id=config["model_id"],
            train_texts=train_texts,
            y_train=y_train,
            dev_texts=dev_texts,
            y_dev=y_dev,
            tfidf_config={
                "max_features": int(config["tfidf_max_features"]),
                "ngram_range": tuple(config["tfidf_ngram_range"]),
            },
            c_val=config["C"],
            args=args,
            train_side=train_side,
            dev_side=dev_side,
        )
        result["tfidf_max_features"] = int(config["tfidf_max_features"])
        result["tfidf_ngram_range"] = list(config["tfidf_ngram_range"])
        result["model_C"] = float(config["C"])
        result["collapse_gate"] = _collapse_gate(
            result["prediction_histogram"],
            threshold=args.collapse_threshold,
        )
        result["acceptance_gate"] = _acceptance_gate(result, args)
        result["selection_meta"] = selection_meta
        results.append(result)
        predicts.append(y_pred)
        probs.append(y_prob)

    if use_side_features:
        target = selected_config
        target_id = "tfidf_with_shallow"
    else:
        target = selected_config
        target_id = "tfidf_only"

    best_idx = next(
        idx
        for idx, result in enumerate(results)
        if (
            result["config"]["model_id"] == target_id
            and int(result["tfidf_max_features"]) == int(target["tfidf_max_features"])
            and list(result["config"]["tfidf_ngram_range"]) == list(target["tfidf_ngram_range"])
            and float(result["model_C"]) == float(target["C"])
        )
    )
    best_result = results[best_idx]
    best_result["selection_trace"] = selection_meta
    return best_result, results, predicts[best_idx], probs[best_idx], np.array(best_result["confusion_matrix"])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_confusion_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(v) for v in row]])


def _write_grid_payload(
    outdir: Path,
    model_id: str,
    results: list[dict[str, Any]],
    best: dict[str, Any],
) -> Path:
    path = outdir / f"{model_id}_grid_search.json"
    write_json(path, {"model_id": model_id, "results": results, "best": best})
    return path


def _build_predictions_payload(
    claim_ids: list[str],
    claim_lookup: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    y_pred: list[str],
    y_prob: np.ndarray,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for claim_id, pred, probs in zip(claim_ids, y_pred, y_prob):
        label_dist = [{"label": label, "prob": float(prob)} for label, prob in zip(LABEL_ORDER, probs.tolist())]
        payload[claim_id] = {
            "claim_text": claim_lookup[claim_id].get("claim_text", ""),
            "claim_label": pred,
            "evidences": contexts[claim_id].get("final_evidence_candidates", []),
            "label_distribution": label_dist,
            "pred_class_probability": float(max(probs)),
            "pred_class": pred,
        }
    return payload


def _write_model_artifacts(
    outdir: Path,
    model_id: str,
    predictions: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    matrix: np.ndarray,
) -> tuple[dict[str, str], list[str]]:
    pred_path = outdir / f"{model_id}_dev_predictions.json"
    metric_path = outdir / f"{model_id}_metrics.json"
    report_path = outdir / f"{model_id}_classification_report.txt"
    conf_path = outdir / f"{model_id}_confusion_matrix.csv"
    selection_path = outdir / f"{model_id}_selection_trace.json"
    write_json(pred_path, predictions)
    write_json(metric_path, metrics)
    report_path.write_text(metrics["classification_report"], encoding="utf-8")
    _write_confusion_csv(conf_path, matrix, LABEL_ORDER)
    write_json(selection_path, {"selection_trace": metrics.get("selection_trace", metrics.get("selection_meta", {}))})
    paths = [
        str(pred_path),
        str(metric_path),
        str(report_path),
        str(conf_path),
        str(selection_path),
    ]
    return {
        "predictions": str(pred_path),
        "metrics": str(metric_path),
        "report": str(report_path),
        "confusion_matrix": str(conf_path),
        "selection_trace": str(selection_path),
    }, paths


def main() -> None:
    start = time.perf_counter()
    args = _parse_args()
    args.command = " ".join(sys.argv)
    args.context_ks = _parse_int_grid(args.context_ks)
    if not args.run_tfidf_only and not args.run_tfidf_shallow:
        args.run_tfidf_only = True
        args.run_tfidf_shallow = True

    args.output_root.mkdir(parents=True, exist_ok=True)

    forbidden_hits = find_forbidden_tokens(
        [
            str(args.train_claims),
            str(args.dev_claims),
            str(args.evidence),
            str(args.train_pool),
            str(args.dev_pool),
            str(args.output_root),
        ]
    )
    if forbidden_hits:
        raise SystemExit(
            "STRICT GUARD FAILED: forbidden token(s) detected\n"
            + json.dumps(forbidden_hits, sort_keys=True)
        )

    train_family, dev_family = _ensure_family_contract(args.train_pool, args.dev_pool)
    if train_family != dev_family:
        raise SystemExit("Train/dev context family mismatch in strict mode.")

    train_claims, dev_claims, evidence, train_pool, dev_pool = _load_inputs(args)
    if not train_claims or not dev_claims:
        raise SystemExit("Input claim files must be non-empty.")
    claim_profiles_train = _build_profile_cache(train_claims)
    claim_profiles_dev = _build_profile_cache(dev_claims)

    summary_rows: list[dict[str, Any]] = []

    for context_k in args.context_ks:
        run_id = f"{args.run_id}_k{context_k}"
        run_dir = args.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest_path = run_dir / "run_manifest.json"
        run_record_path = run_dir / "run_record.json"

        train_contexts, train_diag = _build_contexts(
            train_claims,
            evidence,
            train_pool,
            top_k=context_k,
            source_family=train_family,
        )
        dev_contexts, dev_diag = _build_contexts(
            dev_claims,
            evidence,
            dev_pool,
            top_k=context_k,
            source_family=dev_family,
        )

        train_ids, train_texts, y_train, train_context_items = _rows_from_contexts(
            train_claims,
            train_contexts,
            evidence_token_budget=args.evidence_token_budget,
        )
        dev_ids, dev_texts, y_dev, dev_context_items = _rows_from_contexts(
            dev_claims,
            dev_contexts,
            evidence_token_budget=args.evidence_token_budget,
        )

        train_context_path = run_dir / f"{run_id}_train_context_top{context_k}.jsonl"
        dev_context_path = run_dir / f"{run_id}_dev_context_top{context_k}.jsonl"
        _write_jsonl(train_context_path, [train_contexts[cid] for cid in sorted(train_contexts)])
        _write_jsonl(dev_context_path, [dev_contexts[cid] for cid in sorted(dev_contexts)])

        train_side_matrix, _ = _build_shallow_matrix(
            train_ids,
            claim_profiles_train,
            train_context_items,
            evidence,
        )
        dev_side_matrix, _ = _build_shallow_matrix(
            dev_ids,
            claim_profiles_dev,
            dev_context_items,
            evidence,
        )

        model_outputs: dict[str, Any] = {}
        model_files: list[str] = [str(run_manifest_path), str(run_record_path), str(train_context_path), str(dev_context_path)]

        if args.run_tfidf_only:
            best, all_results, pred, prob, matrix = _grid_search(
                model_id="tfidf_only",
                train_texts=train_texts,
                y_train=y_train,
                dev_texts=dev_texts,
                y_dev=y_dev,
                args=args,
                use_side_features=False,
                train_side=None,
                dev_side=None,
            )
            variant_key = _variant_key(
                feature_mode="tfidf_only",
                use_side_features=False,
                tfidf_config={
                    "max_features": int(best["tfidf_max_features"]),
                    "ngram_range": list(best["tfidf_ngram_range"]),
                },
                c_value=float(best["model_C"]),
                n_side_features=0,
            )
            best["selection_trace"]["feature_mode"] = "tfidf_only"
            pred_payload = _build_predictions_payload(
                dev_ids,
                claim_lookup=dev_claims,
                contexts=dev_contexts,
                y_pred=pred,
                y_prob=prob,
            )
            artifact_paths, files = _write_model_artifacts(
                outdir=run_dir,
                model_id=variant_key,
                predictions=pred_payload,
                metrics=best,
                matrix=matrix,
            )
            grid_path = _write_grid_payload(run_dir, variant_key, all_results, best)
            best["selection_trace"]["feature_mode"] = "tfidf_only"
            best["side_features_enabled"] = False
            best["selection_trace"].setdefault("selection_mode", args.selection_mode)
            model_outputs["tfidf_only"] = {
                **artifact_paths,
                "selected_variant": variant_key,
                "grid_search": str(grid_path),
                "status": best["collapse_gate"]["status"],
                "family": train_family,
                "selection_trace": best.get("selection_trace", {}),
            }
            model_files.extend(files + [str(grid_path)])
            model_outputs["tfidf_only"]["metrics"] = best
            summary_rows.append(
                {
                    "run_id": run_id,
                    "context_k": context_k,
                    "mode": "tfidf_only",
                    "macro_f1": best.get("macro_f1"),
                    "accuracy": best.get("accuracy"),
                    "top_class_share": best.get("top_class_share"),
                    "acceptance": best.get("acceptance_gate", {}).get("status"),
                    "status": best.get("collapse_gate", {}).get("status"),
                    "metric_path": artifact_paths["metrics"],
                }
            )

        if args.run_tfidf_shallow:
            best, all_results, pred, prob, matrix = _grid_search(
                model_id="tfidf_with_shallow",
                train_texts=train_texts,
                y_train=y_train,
                dev_texts=dev_texts,
                y_dev=y_dev,
                args=args,
                use_side_features=True,
                train_side=train_side_matrix,
                dev_side=dev_side_matrix,
            )
            variant_key = _variant_key(
                feature_mode="tfidf_plus_shallow",
                use_side_features=True,
                tfidf_config={
                    "max_features": int(best["tfidf_max_features"]),
                    "ngram_range": list(best["tfidf_ngram_range"]),
                },
                c_value=float(best["model_C"]),
                n_side_features=train_side_matrix.shape[1],
            )
            pred_payload = _build_predictions_payload(
                dev_ids,
                claim_lookup=dev_claims,
                contexts=dev_contexts,
                y_pred=pred,
                y_prob=prob,
            )
            artifact_paths, files = _write_model_artifacts(
                outdir=run_dir,
                model_id=variant_key,
                predictions=pred_payload,
                metrics=best,
                matrix=matrix,
            )
            grid_path = _write_grid_payload(run_dir, variant_key, all_results, best)
            best["selection_trace"]["feature_mode"] = "tfidf_plus_shallow"
            best["side_features_enabled"] = True
            best["selection_trace"].setdefault("selection_mode", args.selection_mode)
            model_outputs["tfidf_plus_shallow"] = {
                **artifact_paths,
                "selected_variant": variant_key,
                "grid_search": str(grid_path),
                "status": best["collapse_gate"]["status"],
                "family": train_family,
                "selection_trace": best.get("selection_trace", {}),
            }
            model_files.extend(files + [str(grid_path)])
            model_outputs["tfidf_plus_shallow"]["metrics"] = best
            summary_rows.append(
                {
                    "run_id": run_id,
                    "context_k": context_k,
                    "mode": "tfidf_plus_shallow",
                    "macro_f1": best.get("macro_f1"),
                    "accuracy": best.get("accuracy"),
                    "top_class_share": best.get("top_class_share"),
                    "acceptance": best.get("acceptance_gate", {}).get("status"),
                    "status": best.get("collapse_gate", {}).get("status"),
                    "metric_path": artifact_paths["metrics"],
                }
            )

        manifest = manifest_base(
            run_id=run_id,
            status="strict-candidate",
            mode="STRICT",
            stage="o_c5_tfidf_side",
            command=args.command,
            working_directory=Path.cwd(),
            random_seed=args.random_seed,
        )
        manifest["input_files"] = [
            {"path": str(args.train_claims), "sha256": sha256_file(args.train_claims), "split": "train", "labels_used": True},
            {"path": str(args.dev_claims), "sha256": sha256_file(args.dev_claims), "split": "dev", "labels_used": True},
            {"path": str(args.evidence), "sha256": sha256_file(args.evidence), "split": "evidence", "labels_used": False},
            {"path": str(args.train_pool), "sha256": sha256_file(args.train_pool), "split": "current_run_train_artifact", "labels_used": False},
            {"path": str(args.dev_pool), "sha256": sha256_file(args.dev_pool), "split": "current_run_dev_artifact", "labels_used": False},
        ]
        manifest["forbidden_input_scan"] = {
            "passed": not bool(forbidden_hits),
            "notes": "Input path scan passed." if not forbidden_hits else str(forbidden_hits),
        }
        manifest["output_files"] = sorted(set(model_files))
        manifest["metrics"] = {
            "train_context_family": train_family,
            "dev_context_family": dev_family,
            "strict_family_match": train_family == dev_family,
            "train_context_source": str(args.train_pool),
            "dev_context_source": str(args.dev_pool),
            "train_context_k": context_k,
            "dev_context_k": context_k,
            "evidence_token_budget": args.evidence_token_budget,
            "train_context_diag": train_diag,
            "dev_context_diag": dev_diag,
            "models": model_outputs,
            "train_context_path": str(train_context_path),
            "dev_context_path": str(dev_context_path),
            "selection_mode": args.selection_mode,
            "selection_criterion": "train_holdout_macro_f1",
            "selection_trace_saved": True,
            "context_size": context_k,
            "baseline_macro_f1_reference": float(args.baseline_macro_f1),
            "strict_vs_diagnostic": "strict",
        }
        manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
        manifest["runtime"]["device"] = "cpu"
        manifest["data_flow_summary"] = (
            "Built explicit strict train/dev contexts from CE factual alias pools, compared "
            "TF-IDF-only against TF-IDF + shallow lexical/entity/number/year/shape/source features."
        )
        manifest["split_isolation_summary"] = (
            "Dev split used only for final, non-tuned scoring; model selection uses train-holdout."
        )
        write_json(run_manifest_path, manifest)

        record = {
            "run_id": run_id,
            "stage": "o_c5_tfidf_side",
            "mode": "STRICT",
            "status": "strict-candidate",
            "split": "train/dev",
            "command": args.command,
            "command_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "claims_train": len(train_claims),
            "claims_dev": len(dev_claims),
            "evidence_count": len(evidence),
            "train_pool": str(args.train_pool),
            "dev_pool": str(args.dev_pool),
            "train_context_family": train_family,
            "dev_context_family": dev_family,
            "selection_mode": args.selection_mode,
            "run_recorded_at": int(time.time()),
            "context_k": context_k,
            "wall_seconds": manifest["runtime"]["wall_seconds"],
            "files_written": [str(run_manifest_path), str(run_record_path), *sorted(set(model_files))],
            "model_results": model_outputs,
            "selection_trace": {
                "tfidf_only": model_outputs.get("tfidf_only", {}).get("selection_trace", {}),
                "tfidf_plus_shallow": model_outputs.get("tfidf_plus_shallow", {}).get("selection_trace", {}),
            },
        }
        write_json(run_record_path, record)

    summary_path = args.output_root / "summary.json"
    write_json(summary_path, {"rows": summary_rows})

    write_json(args.output_root / "selection_trace_summary.json", {"rows": summary_rows})

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
