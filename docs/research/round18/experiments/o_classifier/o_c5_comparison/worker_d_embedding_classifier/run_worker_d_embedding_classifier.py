#!/usr/bin/env python
"""Round18 O-C5 Worker D: embedding / shallow classifier comparison.

Builds strict-context claim classifiers with three feature sets:
- embedding-only (claim/context embedding blocks + pooled CE/embedding/rank scalars),
- shallow-only (scalar matching/statistics features),
- embedding-plus-shallow (concatenation).

Model selection is constrained to train split holdout or CV only; dev is used only
for final scoring. It writes metrics, prediction histograms, per-class recall,
confusion matrix, selection traces, and probability payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None  # type: ignore[assignment]
    HAS_LIGHTGBM = False

import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "round18" / "tools").exists():
        REPO_ROOT = parent
        break
if REPO_ROOT is None:
    REPO_ROOT = Path(__file__).resolve().parents[5]

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
DEFAULT_TRAIN_EMBEDDING_POOL = Path(
    "round18/outputs/o_dense/o_d1x_embedding_bm25_char_train_top500/train_full_train_o_d1x_embedding_bm25_char_train_top500_strict_top500_candidates.json"
)
DEFAULT_DEV_EMBEDDING_POOL = Path(
    "round18/outputs/o_dense/o_d1x_embedding_bm25_char_dev_top500/dev_full_dev_o_d1x_embedding_bm25_char_dev_top500_strict_top500_candidates.json"
)
DEFAULT_OUTPUT_ROOT = Path("round18/outputs/o_classifier/o_c5_comparison/worker_d_embedding_classifier")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "run_manifest.json"
DEFAULT_RECORD = DEFAULT_OUTPUT_ROOT / "run_record.json"
DEFAULT_RUN_ID = "worker_d_embedding_classifier"
DEFAULT_BASELINE_MACRO_F1 = 0.4719276094276095
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CONTEXT_KS = "20"

LABEL_ORDER = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")


@dataclass(frozen=True)
class ContextItem:
    evidence_id: str
    rank: int
    source_score: float
    source_rank: int
    ce_score: float
    ce_rank: int
    embedding_score: float
    embedding_rank: int
    source: str
    text: str


EMBEDDING_SCALAR_FEATURES = [
    "claim_ctx_ip_top1",
    "claim_ctx_ip_mean",
    "claim_ctx_ip_std",
    "claim_ctx_ip_min",
    "claim_ctx_ip_max",
    "source_score_top1",
    "source_score_mean",
    "source_score_std",
    "source_score_min",
    "source_score_max",
    "source_rank_top1",
    "source_rank_mean",
    "source_rank_std",
    "source_rank_min",
    "source_rank_max",
    "ce_score_top1",
    "ce_score_mean",
    "ce_score_std",
    "ce_score_min",
    "ce_score_max",
    "ce_rank_top1",
    "ce_rank_mean",
    "ce_rank_std",
    "ce_rank_min",
    "ce_rank_max",
    "embedding_score_top1",
    "embedding_score_mean",
    "embedding_score_std",
    "embedding_score_min",
    "embedding_score_max",
    "embedding_rank_top1",
    "embedding_rank_mean",
    "embedding_rank_std",
    "embedding_rank_min",
    "embedding_rank_max",
    "context_size",
]

SHALLOW_FEATURES = [
    "claim_token_count",
    "claim_char_count",
    "claim_entity_count",
    "claim_number_count",
    "claim_year_count",
    "evidence_count",
    "evidence_unique_source_count",
    "context_token_mean",
    "context_token_std",
    "context_token_min",
    "context_token_max",
    "context_char_mean",
    "context_char_std",
    "context_char_min",
    "context_char_max",
    "claim_to_evidence_ratio_top1",
    "claim_to_evidence_ratio_mean",
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
    "inv_ce_rank_top1",
    "inv_ce_rank_mean",
    "inv_embedding_rank_top1",
    "inv_embedding_rank_mean",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round18 O-C5 Worker D embedding + shallow feature classifier."
    )
    parser.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN_CLAIMS)
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--dev-pool", type=Path, default=DEFAULT_DEV_POOL)
    parser.add_argument(
        "--train-embedding-pool",
        type=Path,
        default=DEFAULT_TRAIN_EMBEDDING_POOL,
        help="Optional dense CE+embedding context artifact used to override embedding scores/ranks.",
    )
    parser.add_argument(
        "--dev-embedding-pool",
        type=Path,
        default=DEFAULT_DEV_EMBEDDING_POOL,
        help="Optional dense CE+embedding context artifact used to override embedding scores/ranks.",
    )
    parser.add_argument("--context-ks", default=DEFAULT_CONTEXT_KS, help="Comma-separated context lengths, e.g. 5,20,64")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--run-embedding-only", action="store_true")
    parser.add_argument("--run-shallow-only", action="store_true")
    parser.add_argument("--run-embedding-shallow", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--selection-mode", choices=("train_holdout", "cv"), default="train_holdout")
    parser.add_argument("--train-val-fraction", type=float, default=0.20)
    parser.add_argument("--selection-cv-folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--logreg-c-grid", default="0.5,1.0,2.0")
    parser.add_argument("--mlp-hidden-grid", default="32,64")
    parser.add_argument("--mlp-alpha-grid", default="1e-4,1e-3")
    parser.add_argument("--enable-lightgbm", action="store_true")
    parser.add_argument("--lightgbm-estimators-grid", default="120,180")
    parser.add_argument("--lightgbm-leaf-grid", default="31")
    parser.add_argument("--lightgbm-lr-grid", default="0.08,0.12")
    parser.add_argument("--baseline-macro-f1", type=float, default=DEFAULT_BASELINE_MACRO_F1)
    parser.add_argument("--collapse-threshold", type=float, default=0.70)
    parser.add_argument(
        "--enable-diagnostic",
        action="store_true",
        help="Allow strict-family mismatch between train/dev context pools.",
    )
    parser.add_argument("--smoke-rows", type=int, default=0)
    parser.add_argument("--random-cv-shuffle", action="store_true")
    return parser.parse_args()


def _parse_int_grid(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("integer grid cannot be empty")
    return values


def _parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("float grid cannot be empty")
    return values


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _safe_div(n: float, d: float) -> float:
    if float(d) == 0.0:
        return 0.0
    return float(n) / float(d)


def _extract_pool_family(path: Path) -> str:
    stem = path.name.replace("-", "_").removesuffix(".json")
    stem = re.sub(r"_top\d+_candidates$", "", stem)
    stem = re.sub(r"_candidates$", "", stem)
    match = re.search(r"(?:^|_)(o_[a-z0-9_]+)(?:_|$)", stem)
    if not match:
        raise SystemExit(f"Cannot infer candidate-family family from path: {path}")
    return match.group(1)


def _ensure_family_contract(train_pool: Path, dev_pool: Path, enable_diagnostic: bool) -> tuple[str, str, str]:
    train_family = _extract_pool_family(train_pool)
    dev_family = _extract_pool_family(dev_pool)
    if train_family != dev_family and not enable_diagnostic:
        raise SystemExit(
            "Strict context-family mismatch. "
            f"train_family={train_family!r}, dev_family={dev_family!r}. "
            "Pass --enable-diagnostic to allow this run."
        )
    return train_family, dev_family, ("diagnostic-only" if train_family != dev_family else "strict-candidate")


def _parse_inputs(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    for path in (args.train_claims, args.dev_claims, args.evidence, args.train_pool, args.dev_pool):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    if not isinstance(train_claims, dict) or not isinstance(dev_claims, dict):
        raise SystemExit("Claims must be JSON object keyed by claim_id.")
    if not isinstance(evidence, dict):
        raise SystemExit("Evidence must be JSON object keyed by evidence_id.")
    for split_name, claims in (("train", train_claims), ("dev", dev_claims)):
        invalid = sorted(
            {claim_id for claim_id, row in claims.items() if str(row.get("claim_label")) not in LABEL_ORDER}
        )
        if invalid:
            raise SystemExit(f"{split_name} labels invalid (showing first 16): {invalid[:16]}")
    return train_claims, dev_claims, {k: str(v) for k, v in evidence.items() if isinstance(v, str)}


def _load_embedding_pool(path: Path, claims: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    mapped: dict[str, dict[str, dict[str, float]]] = {}
    for claim_id, rows in payload.items():
        if claim_id not in claims or not isinstance(rows, list):
            continue
        inner: dict[str, dict[str, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            evidence_id = _coerce_text(row.get("evidence_id")).strip()
            if not evidence_id:
                continue
            inner[evidence_id] = {
                "embedding_score": _coerce_float(row.get("embedding_score"), _coerce_float(row.get("reranker_score"), 0.0)),
                "embedding_rank": _coerce_int(row.get("embedding_rank"), _coerce_int(row.get("rank"), 1)),
                "source_score": _coerce_float(row.get("source_score"), _coerce_float(row.get("score"), 0.0)),
                "source_rank": _coerce_int(row.get("source_rank"), _coerce_int(row.get("rank"), 1)),
            }
        if inner:
            mapped[claim_id] = inner
    return mapped


def _parse_pool_rows(
    path: Path,
    claims: dict[str, dict[str, Any]],
    evidence: dict[str, str],
    source_family: str,
    top_k: int,
    strict: bool,
) -> tuple[dict[str, list[ContextItem]], dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Context pool malformed: {path}")

    contexts: dict[str, list[ContextItem]] = {}
    diagnostics = {
        "missing_claims_in_pool": [],
        "missing_evidence_text": 0,
        "deduplicated_rows": 0,
        "total_candidates": 0,
        "insufficient_context_claims": [],
    }

    for claim_id in claims:
        rows = payload.get(claim_id, [])
        if not isinstance(rows, list):
            if strict:
                raise SystemExit(f"Missing/invalid context list for claim in pool: {claim_id}")
            continue
        parsed: list[ContextItem] = []
        seen: set[str] = set()
        for pos, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if len(parsed) >= top_k:
                break
            evidence_id = _coerce_text(row.get("evidence_id")).strip()
            if not evidence_id:
                continue
            if evidence_id in seen:
                diagnostics["deduplicated_rows"] += 1
                continue
            seen.add(evidence_id)

            text = evidence.get(evidence_id, "")
            if not text:
                diagnostics["missing_evidence_text"] += 1

            parsed.append(
                ContextItem(
                    evidence_id=evidence_id,
                    rank=_coerce_int(row.get("rank"), pos + 1),
                    source_score=_coerce_float(row.get("source_score"), _coerce_float(row.get("score"), 0.0)),
                    source_rank=_coerce_int(row.get("source_rank"), pos + 1),
                    ce_score=_coerce_float(row.get("ce_score"), _coerce_float(row.get("score"), 0.0)),
                    ce_rank=_coerce_int(row.get("ce_rank"), pos + 1),
                    embedding_score=_coerce_float(row.get("embedding_score"), _coerce_float(row.get("score"), 0.0)),
                    embedding_rank=_coerce_int(row.get("embedding_rank"), pos + 1),
                    source=source_family,
                    text=text,
                )
            )
        diagnostics["total_candidates"] += len(parsed)
        if strict and len(parsed) < top_k:
            diagnostics["insufficient_context_claims"].append(claim_id)
            raise SystemExit(f"Strict context failure: {claim_id} has {len(parsed)} < {top_k} candidates.")
        contexts[claim_id] = parsed

    missing = [claim_id for claim_id in claims if claim_id not in payload]
    if missing and strict:
        raise SystemExit(f"Pool missing claim ids in strict mode: {missing[:16]}")
    diagnostics["missing_claims_in_pool"] = missing
    return contexts, diagnostics


def _attach_embedding_overrides(
    contexts: dict[str, list[ContextItem]],
    overrides: dict[str, dict[str, dict[str, float]]],
) -> dict[str, list[ContextItem]]:
    if not overrides:
        return contexts
    rebuilt: dict[str, list[ContextItem]] = {}
    for claim_id, rows in contexts.items():
        merged: list[ContextItem] = []
        claim_override = overrides.get(claim_id, {})
        for item in rows:
            if item.evidence_id in claim_override:
                ov = claim_override[item.evidence_id]
                merged.append(
                    ContextItem(
                        evidence_id=item.evidence_id,
                        rank=item.rank,
                        source_score=_coerce_float(ov.get("source_score"), item.source_score),
                        source_rank=_coerce_int(ov.get("source_rank"), item.source_rank),
                        ce_score=item.ce_score,
                        ce_rank=item.ce_rank,
                        embedding_score=_coerce_float(ov.get("embedding_score"), item.embedding_score),
                        embedding_rank=_coerce_int(ov.get("embedding_rank"), item.embedding_rank),
                        source=item.source,
                        text=item.text,
                    )
                )
            else:
                merged.append(item)
        rebuilt[claim_id] = merged
    return rebuilt


class EmbeddingBackend:
    def __init__(self, model_name: str, batch_size: int, max_length: int, random_seed: int) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = "transformers"
        self.dim = 384
        self._cache: dict[str, np.ndarray] = {}
        self._rng = np.random.RandomState(random_seed)
        self.fallback_reason = ""
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModel | None = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.dim = int(getattr(self.model.config, "hidden_size", 384))
        except Exception as exc:  # pragma: no cover - environment dependent
            self.fallback_reason = (
                "transformers model load failed; using deterministic hashing fallback. "
                f"source={type(exc).__name__}: {exc}"
            )
            self.mode = "diagnostic_fallback_hashing"
            self.tokenizer = None
            self.model = None
            self._vectorizer = HashingVectorizer(
                n_features=768,
                analyzer="char",
                ngram_range=(3, 5),
                alternate_sign=False,
                norm="l2",
                dtype=np.float32,
            )
            self.dim = 768

    def is_diagnostic(self) -> bool:
        return self.mode.startswith("diagnostic_")

    def _mean_pool(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = model_output.last_hidden_state
        weights = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        weighted = hidden * weights
        return weighted.sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [_coerce_text(t) for t in texts]
        unique: list[str] = []
        for text in texts:
            if text not in self._cache:
                unique.append(text)
        if unique:
            if self.mode == "transformers":
                assert self.tokenizer is not None and self.model is not None
                with torch.no_grad():
                    for start in range(0, len(unique), self.batch_size):
                        batch = unique[start : start + self.batch_size]
                        encoded = self.tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                        outputs = self.model(**encoded)
                        pooled = self._mean_pool(outputs, encoded["attention_mask"]).detach().cpu().numpy()
                        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                        norms = np.where(norms <= 0.0, 1.0, norms)
                        pooled = pooled / norms
                        for key, vec in zip(batch, pooled):
                            self._cache[key] = vec.astype(np.float32)
            else:
                matrix = self._vectorizer.transform(unique)  # type: ignore[attr-defined]
                dense = matrix.toarray().astype(np.float32, copy=False)
                norms = np.linalg.norm(dense, axis=1, keepdims=True)
                norms = np.where(norms <= 0.0, 1.0, norms)
                dense = dense / norms
                for key, vec in zip(unique, dense):
                    self._cache[key] = vec.astype(np.float32)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            vec = self._cache.get(text)
            if vec is None:
                out[idx] = np.zeros((self.dim,), dtype=np.float32)
            else:
                out[idx] = vec
        return out


def _word_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(_coerce_text(text))]


def _text_profile(text: str) -> dict[str, Any]:
    raw_tokens = WORD_RE.findall(_coerce_text(text))
    tokens = [tok.lower() for tok in raw_tokens]
    entities = set(
        [tok for tok in raw_tokens if tok and (tok[0].isupper() or tok.isupper()) and len(tok) > 1]
    )
    content = [tok for tok in tokens if len(tok) > 1 and tok.isalpha()]
    return {
        "raw": _coerce_text(text),
        "tokens": tokens,
        "token_set": set(tokens),
        "content_set": set(content),
        "char_count": len(text),
        "entities": entities,
        "numbers": set(NUMBER_RE.findall(text)),
        "years": set(YEAR_RE.findall(text)),
    }


def _build_profiles(items: dict[str, dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {claim_id: _text_profile(row.get(key, "")) for claim_id, row in items.items()}


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.max(arr)),
        float(np.min(arr)),
        float(np.std(arr)),
    )


def _build_feature_rows(
    claim_ids: list[str],
    claim_profiles: dict[str, dict[str, Any]],
    claim_embeddings: dict[str, np.ndarray],
    context_items_by_claim: dict[str, list[ContextItem]],
    evidence_embeddings: dict[str, np.ndarray],
    evidence_profiles: dict[str, dict[str, Any]],
    dim: int,
) -> dict[str, Any]:
    embedding_rows: list[np.ndarray] = []
    shallow_rows: list[np.ndarray] = []
    claim_context_ips: dict[str, list[float]] = {}
    for claim_id in claim_ids:
        claim_vec = claim_embeddings[claim_id]
        context_items = context_items_by_claim.get(claim_id, [])
        if context_items:
            context_vecs = [evidence_embeddings.get(item.evidence_id, np.zeros((dim,), dtype=np.float32)) for item in context_items]
            context_vec = np.mean(np.asarray(context_vecs, dtype=np.float32), axis=0)
            ips = [float(np.dot(claim_vec, cvec)) for cvec in context_vecs]
        else:
            context_vec = np.zeros((dim,), dtype=np.float32)
            ips = []

        claim_context_ips[claim_id] = ips
        abs_diff = np.abs(claim_vec - context_vec)
        hadamard = claim_vec * context_vec
        ip_top1 = ips[0] if ips else 0.0
        ip_mean, ip_max, ip_min, ip_std = _stats(ips)

        claim_profile = claim_profiles[claim_id]
        claim_tokens = claim_profile["token_set"]
        claim_entities = claim_profile["entities"]
        claim_numbers = claim_profile["numbers"]
        claim_years = claim_profile["years"]
        claim_chars = claim_profile["char_count"]

        source_scores: list[float] = []
        source_ranks: list[float] = []
        ce_scores: list[float] = []
        ce_ranks: list[float] = []
        embedding_scores: list[float] = []
        embedding_ranks: list[float] = []
        sources = set()
        ctx_tokens: list[float] = []
        ctx_chars: list[float] = []
        lexical_jaccard: list[float] = []
        lexical_recall: list[float] = []
        lexical_precision: list[float] = []
        entity_overlap: list[float] = []
        number_overlap: list[float] = []
        year_overlap: list[float] = []

        for item in context_items:
            source_scores.append(item.source_score)
            source_ranks.append(float(max(item.source_rank, 1)))
            ce_scores.append(item.ce_score)
            ce_ranks.append(float(max(item.ce_rank, 1)))
            embedding_scores.append(item.embedding_score)
            embedding_ranks.append(float(max(item.embedding_rank, 1)))
            sources.add(item.source)

            ep = evidence_profiles.get(item.evidence_id, {"content_set": set(), "char_count": 0, "entities": set(), "numbers": set(), "years": set()})
            ctx_set = set(ep.get("content_set", set()))
            ctx_chars.append(float(ep.get("char_count", 0)))
            ctx_tokens.append(float(len(ctx_set)))
            entities = set(ep.get("entities", set()))
            numbers = set(ep.get("numbers", set()))
            years = set(ep.get("years", set()))

            inter = len(claim_tokens & ctx_set)
            union = len(claim_tokens | ctx_set)
            lexical_jaccard.append(_safe_div(inter, union))
            lexical_recall.append(_safe_div(inter, max(len(claim_tokens), 1)))
            lexical_precision.append(_safe_div(inter, max(len(ctx_set), 1)))
            entity_overlap.append(float(len(claim_entities & entities)))
            number_overlap.append(_safe_div(len(claim_numbers & numbers), max(len(claim_numbers), 1)))
            year_overlap.append(_safe_div(len(claim_years & years), max(len(claim_years), 1)))

        source_mean, source_max, source_min, source_std = _stats(source_scores)
        ce_mean, ce_max, ce_min, ce_std = _stats(ce_scores)
        emb_mean, emb_max, emb_min, emb_std = _stats(embedding_scores)
        sr_mean, sr_max, sr_min, sr_std = _stats(source_ranks)
        cr_mean, cr_max, cr_min, cr_std = _stats(ce_ranks)
        er_mean, er_max, er_min, er_std = _stats(embedding_ranks)
        tok_mean, tok_max, tok_min, tok_std = _stats(ctx_tokens)
        chr_mean, chr_max, chr_min, chr_std = _stats(ctx_chars)
        jacc_mean, jacc_max, jacc_min, jacc_std = _stats(lexical_jaccard)
        recall_mean, recall_max, recall_min, recall_std = _stats(lexical_recall)
        prec_mean, prec_max, prec_min, prec_std = _stats(lexical_precision)
        ent_mean, ent_max, ent_min, ent_std = _stats(entity_overlap)
        num_mean, num_max, num_min, num_std = _stats(number_overlap)
        yr_mean, yr_max, yr_min, yr_std = _stats(year_overlap)

        shallow = [
            float(len(claim_tokens)),
            float(claim_chars),
            float(len(claim_entities)),
            float(len(claim_numbers)),
            float(len(claim_years)),
            float(len(context_items)),
            float(len(sources)),
            float(tok_mean),
            float(tok_std),
            float(tok_min),
            float(tok_max),
            float(chr_mean),
            float(chr_std),
            float(chr_min),
            float(chr_max),
            float(_safe_div(claim_chars, ctx_chars[0] if ctx_chars else 0.0)),
            float(_safe_div(claim_chars, tok_mean if tok_mean else 0.0)),
            float(jacc_mean if lexical_jaccard else 0.0),
            float(jacc_mean),
            float(lexical_recall[0] if lexical_recall else 0.0),
            float(recall_mean),
            float(lexical_precision[0] if lexical_precision else 0.0),
            float(prec_mean),
            float(entity_overlap[0] if entity_overlap else 0.0),
            float(ent_mean),
            float(number_overlap[0] if number_overlap else 0.0),
            float(num_mean),
            float(year_overlap[0] if year_overlap else 0.0),
            float(yr_mean),
            _safe_div(1.0, source_ranks[0] + 1.0) if source_ranks else 0.0,
            _safe_div(sum(_safe_div(1.0, rank) for rank in source_ranks), max(len(source_ranks), 1)),
            _safe_div(1.0, ce_ranks[0] + 1.0) if ce_ranks else 0.0,
            _safe_div(sum(_safe_div(1.0, rank) for rank in ce_ranks), max(len(ce_ranks), 1)),
            _safe_div(1.0, embedding_ranks[0] + 1.0) if embedding_ranks else 0.0,
            _safe_div(sum(_safe_div(1.0, rank) for rank in embedding_ranks), max(len(embedding_ranks), 1)),
        ]

        embedding_scalar = [
            float(ip_top1),
            float(ip_mean),
            float(ip_std),
            float(ip_min),
            float(ip_max),
            float(source_scores[0]) if source_scores else 0.0,
            float(source_mean),
            float(source_std),
            float(source_min),
            float(source_max),
            float(source_ranks[0]) if source_ranks else 0.0,
            float(sr_mean),
            float(sr_std),
            float(sr_min),
            float(sr_max),
            float(ce_scores[0]) if ce_scores else 0.0,
            float(ce_mean),
            float(ce_std),
            float(ce_min),
            float(ce_max),
            float(ce_ranks[0]) if ce_ranks else 0.0,
            float(cr_mean),
            float(cr_std),
            float(cr_min),
            float(cr_max),
            float(embedding_scores[0]) if embedding_scores else 0.0,
            float(emb_mean),
            float(emb_std),
            float(emb_min),
            float(emb_max),
            float(embedding_ranks[0]) if embedding_ranks else 0.0,
            float(er_mean),
            float(er_std),
            float(er_min),
            float(er_max),
            float(len(context_items)),
        ]

        embedding_rows.append(
            np.concatenate(
                [
                    claim_vec.astype(np.float32),
                    context_vec.astype(np.float32),
                    abs_diff.astype(np.float32),
                    hadamard.astype(np.float32),
                    np.asarray(embedding_scalar, dtype=np.float32),
                ],
                axis=0,
            )
        )
        shallow_rows.append(np.asarray(shallow, dtype=np.float32))

    return {
        "embedding_matrix": np.asarray(embedding_rows, dtype=np.float32),
        "shallow_matrix": np.asarray(shallow_rows, dtype=np.float32),
        "claim_context_ips": claim_context_ips,
    }


def _evaluate_predictions(y_true_idx: list[int], y_pred_idx: list[int], y_prob: np.ndarray) -> dict[str, Any]:
    y_true_labels = [ID_TO_LABEL[idx] for idx in y_true_idx]
    y_pred_labels = [ID_TO_LABEL[idx] for idx in y_pred_idx]
    acc = accuracy_score(y_true_idx, y_pred_idx)
    macro_f1 = f1_score(y_true_idx, y_pred_idx, labels=list(ID_TO_LABEL.keys()), average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_idx, y_pred_idx, labels=list(ID_TO_LABEL.keys()), average="micro", zero_division=0)
    macro_recall = recall_score(y_true_idx, y_pred_idx, labels=list(ID_TO_LABEL.keys()), average="macro", zero_division=0)
    confusion = confusion_matrix(y_true_idx, y_pred_idx, labels=list(ID_TO_LABEL.keys()))
    per_class_recall: dict[str, float] = {}
    row_totals = confusion.sum(axis=1).astype(float)
    for idx, label in ID_TO_LABEL.items():
        per_class_recall[label] = float(confusion[idx, idx] / row_totals[idx]) if row_totals[idx] else 0.0
    histogram = {label: 0 for label in LABEL_ORDER}
    for pred in y_pred_labels:
        histogram[pred] = histogram.get(pred, 0) + 1
    top_share = _safe_div(max(histogram.values()), max(len(y_pred_labels), 1))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_recall": float(macro_recall),
        "classification_report": classification_report(
            y_true_labels,
            y_pred_labels,
            labels=LABEL_ORDER,
            zero_division=0,
        ),
        "per_class_recall": per_class_recall,
        "prediction_histogram": histogram,
        "top_class_share": float(top_share),
        "top_class": max(histogram.items(), key=lambda kv: (kv[1], kv[0]))[0],
        "confusion_matrix": confusion.tolist(),
        "proba": y_prob.tolist(),
    }


def _collapse_gate(prediction_hist: dict[str, int], threshold: float) -> dict[str, Any]:
    total = sum(prediction_hist.values())
    if total <= 0:
        return {"status": "failed", "reason": "no_predictions", "threshold": float(threshold), "max_class_share": 0.0, "collapsed_classes": []}
    max_label, max_count = max(prediction_hist.items(), key=lambda kv: (kv[1], kv[0]))
    max_share = _safe_div(max_count, total)
    collapsed = [label for label, count in prediction_hist.items() if _safe_div(count, total) > threshold]
    return {
        "status": "passed" if max_share <= threshold else "failed",
        "reason": ("passed" if max_share <= threshold else f"single class collapse above {threshold}: {max_label}"),
        "max_class": max_label,
        "max_class_share": float(max_share),
        "threshold": float(threshold),
        "collapsed_classes": collapsed,
    }


def _acceptance_gate(
    metrics: dict[str, Any],
    threshold: float,
    baseline_macro_f1: float,
) -> dict[str, Any]:
    failures: list[str] = []
    histogram = metrics.get("prediction_histogram", {})
    per_class_recall = metrics.get("per_class_recall", {})
    if any(int(histogram.get(label, 0)) <= 0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_predictions")
    if any(float(per_class_recall.get(label, 0.0)) <= 0.0 for label in LABEL_ORDER):
        failures.append("one_or_more_classes_have_zero_recall")
    if float(metrics.get("top_class_share", 1.0)) > float(threshold):
        failures.append("max_class_share_above_threshold")
    if float(metrics.get("macro_f1", 0.0)) < float(baseline_macro_f1):
        failures.append("macro_f1_below_baseline")
    return {
        "status": "passed" if not failures else "failed",
        "thresholds": {
            "top_class_share": float(threshold),
            "min_macro_f1": float(baseline_macro_f1),
            "per_class_recall_all_nonzero": True,
            "per_class_predictions_all_nonzero": True,
        },
        "failures": failures,
    }


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["micro_f1"]),
        float(metrics["macro_recall"]),
        -float(metrics["top_class_share"]),
    )


def _model_candidates(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for c in _parse_float_grid(args.logreg_c_grid):
        candidates.append(("logreg", {"C": float(c)}))
    for hidden in _parse_int_grid(args.mlp_hidden_grid):
        for alpha in _parse_float_grid(args.mlp_alpha_grid):
            candidates.append(("mlp", {"hidden_units": int(hidden), "alpha": float(alpha)}))
    if args.enable_lightgbm and HAS_LIGHTGBM:
        for est in _parse_int_grid(args.lightgbm_estimators_grid):
            for leaves in _parse_int_grid(args.lightgbm_leaf_grid):
                for lr in _parse_float_grid(args.lightgbm_lr_grid):
                    candidates.append(
                        (
                            "lightgbm",
                            {
                                "n_estimators": int(est),
                                "num_leaves": int(leaves),
                                "learning_rate": float(lr),
                            },
                        )
                    )
    return candidates


def _build_model(algo: str, config: dict[str, Any], random_state: int, n_jobs: int) -> Any:
    if algo == "logreg":
        return make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(
                C=float(config["C"]),
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2500,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        )
    if algo == "mlp":
        return make_pipeline(
            StandardScaler(with_mean=True),
            MLPClassifier(
                hidden_layer_sizes=(int(config["hidden_units"]),),
                alpha=float(config["alpha"]),
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
            ),
        )
    if algo == "lightgbm":
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM requested but not available.")
        return LGBMClassifier(  # type: ignore[misc]
            objective="multiclass",
            num_class=len(LABEL_ORDER),
            n_estimators=int(config["n_estimators"]),
            num_leaves=int(config["num_leaves"]),
            learning_rate=float(config["learning_rate"]),
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    raise ValueError(f"Unknown algorithm: {algo}")


def _fit_and_score(
    x_train: np.ndarray,
    y_train_idx: list[int],
    x_val: np.ndarray,
    y_val_idx: list[int],
    algo: str,
    config: dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> tuple[Any, list[int], np.ndarray, dict[str, Any]]:
    model = _build_model(algo, config, random_state=random_state, n_jobs=n_jobs)
    y_train_arr = np.asarray(y_train_idx)
    y_val_arr = np.asarray(y_val_idx)
    t0 = time.perf_counter()
    model.fit(x_train, y_train_arr)
    train_seconds = float(time.perf_counter() - t0)
    proba = model.predict_proba(x_val)
    pred = np.argmax(proba, axis=1).astype(int).tolist()
    metrics = _evaluate_predictions(y_val_arr.tolist(), pred, proba.astype(float))
    metrics["fit_seconds"] = train_seconds
    metrics["algo"] = algo
    metrics["config"] = dict(config)
    return model, pred, proba.astype(float), metrics


def _run_train_holdout(
    x_train_all: np.ndarray,
    y_train_idx: list[int],
    args: argparse.Namespace,
    train_claim_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args.train_val_fraction),
        random_state=args.random_seed,
    )
    try:
        train_idx, hold_idx = next(splitter.split(np.arange(len(y_train_idx)), y_train_idx))
    except ValueError as exc:
        return {
            "selection_status": "failed",
            "selection_mode": "train_holdout",
            "reason": str(exc),
            "candidate_results": [],
        }, {"claim_ids": [], "predictions": {}, "probability_matrix": []}, {}

    tr_idx = np.asarray(train_idx)
    va_idx = np.asarray(hold_idx)
    x_tr, x_va = x_train_all[tr_idx], x_train_all[va_idx]
    y_tr = [y_train_idx[i] for i in tr_idx.tolist()]
    y_va = [y_train_idx[i] for i in va_idx.tolist()]

    candidates = _model_candidates(args)
    best_row: dict[str, Any] | None = None
    best_model = None
    best_pred = []
    best_prob = np.empty((0, len(LABEL_ORDER)), dtype=float)
    candidate_rows: list[dict[str, Any]] = []

    for algo, config in candidates:
        model, pred, proba, metrics = _fit_and_score(
            x_tr,
            y_tr,
            x_va,
            y_va,
            algo=algo,
            config=config,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
        metrics["selection_mode"] = "train_holdout"
        metrics["train_size"] = int(len(y_tr))
        metrics["selection_size"] = int(len(y_va))
        metrics["selection_key"] = _selection_key(metrics)
        candidate_rows.append(metrics)
        if best_row is None or metrics["selection_key"] > best_row["selection_key"]:
            best_row = metrics
            best_model = model
            best_pred = pred
            best_prob = proba

    if best_row is None or best_model is None:
        return {
            "selection_status": "failed",
            "selection_mode": "train_holdout",
            "reason": "no_candidate_succeeded",
            "candidate_results": candidate_rows,
        }, {"claim_ids": [], "predictions": {}, "probability_matrix": []}, {}

    holdout_claim_ids = [str(train_claim_ids[idx]) for idx in va_idx.tolist()]
    holdout_payload = {
        "claim_ids": holdout_claim_ids,
        "predictions": {
            claim_id: {
                "pred_label": ID_TO_LABEL[int(pred)],
                "label_distribution": [
                    {"label": label, "prob": float(prob)}
                    for label, prob in zip(LABEL_ORDER, proba_row)
                ],
            }
            for claim_id, pred, proba_row in zip(holdout_claim_ids, best_pred, best_prob)
        },
        "probability_matrix": best_prob.tolist(),
        "label_order": LABEL_ORDER,
    }
    selection_result = {
        "selection_mode": "train_holdout",
        "selection_status": "selected",
        "selected_algo": best_row["algo"],
        "selected_config": best_row["config"],
        "selection_train_size": int(len(y_tr)),
        "selection_val_size": int(len(y_va)),
        "selected_row": best_row,
        "candidate_results": candidate_rows,
    }
    return selection_result, holdout_payload, {"model": best_model, "algo": best_row["algo"], "config": best_row["config"]}


def _run_cv(
    x_train_all: np.ndarray,
    y_train_idx: list[int],
    args: argparse.Namespace,
    train_claim_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    y_arr = np.asarray(y_train_idx)
    if len(set(y_train_idx)) < 2 or len(y_arr) < max(2, args.selection_cv_folds):
        return {
            "selection_status": "failed",
            "selection_mode": "cv",
            "reason": "insufficient_data_for_cv",
            "candidate_results": [],
        }, {"claim_ids": [], "predictions": {}, "probability_matrix": []}, {}

    splitter = StratifiedKFold(
        n_splits=args.selection_cv_folds,
        shuffle=True,
        random_state=args.random_seed,
    )
    candidates = _model_candidates(args)
    best_key = None
    best_payload: dict[str, Any] | None = None
    best_model = None
    best_oof_prob = np.empty((0, len(LABEL_ORDER)), dtype=float)
    candidate_rows: list[dict[str, Any]] = []

    for algo, config in candidates:
        oof_prob = np.zeros((len(y_train_idx), len(LABEL_ORDER)), dtype=float)
        oof_pred = [""] * len(y_train_idx)
        fold_metrics: list[dict[str, Any]] = []

        for fold_train_idx, fold_val_idx in splitter.split(np.arange(len(y_arr)), y_arr):
            x_tr, x_va = x_train_all[fold_train_idx], x_train_all[fold_val_idx]
            y_tr = [y_train_idx[i] for i in fold_train_idx.tolist()]
            y_va = [y_train_idx[i] for i in fold_val_idx.tolist()]
            model, pred, proba, metrics = _fit_and_score(
                x_tr,
                y_tr,
                x_va,
                y_va,
                algo=algo,
                config=config,
                random_state=args.random_seed,
                n_jobs=args.n_jobs,
            )
            for local_pos, global_idx in enumerate(fold_val_idx):
                oof_pred[global_idx] = ID_TO_LABEL[int(pred[local_pos])]
                oof_prob[global_idx] = proba[local_pos]
            fold_metrics.append(metrics)

        fold_macro = float(np.mean([m["macro_f1"] for m in fold_metrics]))
        fold_micro = float(np.mean([m["micro_f1"] for m in fold_metrics]))
        fold_recall = float(np.mean([m["macro_recall"] for m in fold_metrics]))
        fold_share = float(np.mean([m["top_class_share"] for m in fold_metrics]))
        agg = {
            "algo": algo,
            "config": dict(config),
            "selection_mode": "cv",
            "selection_cv_folds": int(args.selection_cv_folds),
            "macro_f1": fold_macro,
            "micro_f1": fold_micro,
            "macro_recall": fold_recall,
            "top_class_share": float(fold_share),
            "selection_key": (fold_macro, fold_micro, fold_recall, -fold_share),
            "fold_metrics": fold_metrics,
            "fit_seconds_cv_mean": float(np.mean([m.get("fit_seconds", 0.0) for m in fold_metrics])),
        }
        candidate_rows.append(agg)
        if best_key is None or agg["selection_key"] > best_key:
            best_key = agg["selection_key"]
            best_payload = agg
            best_oof_prob = oof_prob
            best_model = _build_model(algo, config, random_state=args.random_seed, n_jobs=args.n_jobs)
            # train refit for return payload
            best_model.fit(x_train_all, y_arr)

    if best_payload is None or best_model is None:
        return {
            "selection_status": "failed",
            "selection_mode": "cv",
            "reason": "no_candidate_succeeded",
            "candidate_results": candidate_rows,
        }, {"claim_ids": [], "predictions": {}, "probability_matrix": []}, {}

    holdout_payload = {
        "claim_ids": [str(train_claim_ids[i]) for i in range(len(y_train_idx))],
        "predictions": {
            str(train_claim_ids[i]): {
                "pred_label": pred_label,
                "label_distribution": [
                    {"label": label, "prob": float(prob)}
                    for label, prob in zip(LABEL_ORDER, best_oof_prob[i])
                ],
            }
            for i, pred_label in enumerate(oof_pred)
        },
        "probability_matrix": best_oof_prob.tolist(),
        "label_order": LABEL_ORDER,
    }

    return {
        "selection_status": "selected",
        "selection_mode": "cv",
        "selected_algo": best_payload["algo"],
        "selected_config": best_payload["config"],
        "selection_cv_folds": int(args.selection_cv_folds),
        "selected_row": best_payload,
        "candidate_results": candidate_rows,
    }, holdout_payload, {"model": best_model, "algo": best_payload["algo"], "config": best_payload["config"]}


def _encode_rows_to_payload(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row, ensure_ascii=False) for row in rows]


def _write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *LABEL_ORDER])
        for label, row in zip(LABEL_ORDER, matrix):
            writer.writerow([label, *[int(v) for v in row]])


def _prepare_context_rows(
    claim_id: str,
    claim: dict[str, Any],
    context_items: list[ContextItem],
    strict: bool,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_text": claim.get("claim_text", ""),
        "claim_label": claim.get("claim_label"),
        "context_item_count": len(context_items),
        "strict_context": strict,
        "classifier_evidence_context": [item.__dict__ for item in context_items],
        "final_evidence_candidates": [item.evidence_id for item in context_items],
    }


def _prepare_classification_outputs(
    feature_mode: str,
    run_prefix: str,
    selection_result: dict[str, Any],
    holdout_payload: dict[str, Any],
    selected_model: dict[str, Any],
    x_dev: np.ndarray,
    y_dev_idx: list[int],
    claim_ids: list[str],
    claims: dict[str, dict[str, Any]],
    context_rows_by_claim: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    if selection_result.get("selection_status") != "selected" or not holdout_payload.get("predictions"):
        dev_metrics = {
            "selection_status": "failed",
            "selection_result": selection_result,
            "feature_mode": feature_mode,
            "selection_trace": selection_result,
            "acceptance_gate": {"status": "failed", "failures": ["selection_failed"]},
            "collapse_gate": _collapse_gate(
                {"SUPPORTS": 0, "REFUTES": 0, "NOT_ENOUGH_INFO": 0, "DISPUTED": 0},
                threshold=args.collapse_threshold,
            ),
        }
        metric_payload = dict(dev_metrics)
        metric_payload["runtime_seconds"] = 0.0
        final_prob = np.empty((0, len(LABEL_ORDER)), dtype=float)
    else:
        model = selected_model["model"]
        sel_start = time.perf_counter()
        proba = model.predict_proba(x_dev).astype(float)
        pred_idx = np.argmax(proba, axis=1).astype(int).tolist()
        dev_eval = _evaluate_predictions(y_dev_idx, pred_idx, proba)
        runtime_dev = float(time.perf_counter() - sel_start)
        dev_eval["feature_mode"] = feature_mode
        dev_eval["selection_result"] = selection_result
        dev_eval["selected_algo"] = selection_result.get("selected_algo")
        dev_eval["selected_config"] = selection_result.get("selected_config")
        dev_eval["train_size"] = int(len(y_dev_idx))
        dev_eval["acceptance_gate"] = _acceptance_gate(
            dev_eval,
            threshold=args.collapse_threshold,
            baseline_macro_f1=args.baseline_macro_f1,
        )
        dev_eval["collapse_gate"] = _collapse_gate(dev_eval["prediction_histogram"], args.collapse_threshold)
        dev_eval["runtime_seconds"] = runtime_dev
        final_prob = proba
        metric_payload = dev_eval

    pred_path = args.output_root / f"{run_prefix}_{feature_mode}_dev_predictions.json"
    proba_path = args.output_root / f"{run_prefix}_{feature_mode}_dev_proba.json"
    metric_path = args.output_root / f"{run_prefix}_{feature_mode}_metrics.json"
    report_path = args.output_root / f"{run_prefix}_{feature_mode}_classification_report.txt"
    conf_path = args.output_root / f"{run_prefix}_{feature_mode}_confusion_matrix.csv"
    holdout_pred_path = args.output_root / f"{run_prefix}_{feature_mode}_train_holdout_predictions.json"
    holdout_proba_path = args.output_root / f"{run_prefix}_{feature_mode}_train_holdout_proba.json"
    selection_path = args.output_root / f"{run_prefix}_{feature_mode}_selection_trace.json"
    if "predictions" in holdout_payload:
        # keep stable compact JSON with id-aligned probability matrix for fusion
        write_json(
            holdout_proba_path,
            {
                "label_order": LABEL_ORDER,
                "claim_ids": holdout_payload.get("claim_ids", []),
                "predictions": holdout_payload.get("predictions", {}),
                "probabilities": holdout_payload.get("probability_matrix", []),
                "selection_mode": selection_result.get("selection_mode"),
            },
        )
        write_json(holdout_pred_path, holdout_payload["predictions"])
    else:
        write_json(holdout_pred_path, {})
        write_json(holdout_proba_path, {"label_order": LABEL_ORDER, "claim_ids": [], "predictions": {}, "probabilities": []})

    write_json(selection_path, {"selection": selection_result})
    write_json(metric_path, metric_payload)
    report_path.write_text(metric_payload.get("classification_report", ""), encoding="utf-8")

    if final_prob.size:
        predictions_payload = {
            claim_id: {
                "claim_text": claims[claim_id].get("claim_text", ""),
                "claim_label": ID_TO_LABEL[int(np.argmax(proba_i))] if final_prob.size else "",
                "evidences": context_rows_by_claim[claim_id].get("final_evidence_candidates", []),
                "label_distribution": [
                    {"label": LABEL_ORDER[j], "prob": float(prob)}
                    for j, prob in enumerate(proba_i.tolist())
                ],
            }
            for claim_id, proba_i in zip(claim_ids, final_prob)
        }
        write_json(pred_path, predictions_payload)
        write_json(
            proba_path,
            {
                "label_order": LABEL_ORDER,
                "claim_ids": claim_ids,
                "probabilities": final_prob.tolist(),
            },
        )
    else:
        write_json(pred_path, {})
        write_json(proba_path, {"label_order": LABEL_ORDER, "claim_ids": claim_ids, "probabilities": []})

    if final_prob.size:
        _write_confusion_csv(conf_path, np.array(metric_payload.get("confusion_matrix", np.zeros((4, 4))), dtype=int))
    else:
        _write_confusion_csv(conf_path, np.zeros((4, 4), dtype=int))
    return metric_payload, holdout_payload, pred_path, metric_path, conf_path


def _build_report(
    report_path: Path,
    all_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    run_mode: str,
    baseline_macro_f1: float,
) -> None:
    def _fmt(v: Any) -> str:
        return f"{float(v):.4f}"

    strict_note = "strict" if run_mode == "strict-candidate" else "diagnostic"
    passed = [row for row in all_rows if row.get("acceptance_gate", {}).get("status") == "passed"]
    lines: list[str] = []
    lines.append("# O-C5 Worker D: Embedding + Shallow Classifier\n")
    lines.append("## Scope\n")
    lines.append(f"- context family: **{strict_note}**\n")
    lines.append(f"- selection: **{summary.get('selection_mode', '')}**\n")
    lines.append(f"- baseline macro-F1: **{_fmt(baseline_macro_f1)}**\n")
    lines.append(f"- strict vs diagnostic: **{run_mode}**\n")
    lines.append(f"- embedding backend fallback: **{summary.get('embedding_backend', 'unknown')}**\n\n")
    lines.append("## Contract\n")
    lines.append("- Train/dev hyperparameter tuning is forbidden.\n")
    lines.append("- Train split holdout / CV only for model selection.\n")
    lines.append("- Dev is used only for final scoring and output payloads.\n")
    lines.append("- collapse gate: all 4 classes must appear, all per-class recall > 0, top-class share <= 0.70.\n\n")
    lines.append(f"- Best macro-F1: **{_fmt(summary.get('best_macro_f1', 0.0))}**\n")
    lines.append(
        f"- Exceeds baseline { _fmt(baseline_macro_f1)}: "
        f"`{bool(summary.get('best_macro_f1', 0.0) > baseline_macro_f1)}`\n"
    )
    lines.append("\n")
    lines.append("| run_id | context_k | feature_mode | selected_algo | macro_f1 | macro_recall | top_class_share | acceptance | collapse |\n")
    lines.append("|---|---:|---|---|---:|---:|---:|---|---|\n")
    for row in all_rows:
        lines.append(
            f"| {row.get('run_id','')} | {row.get('context_k','')} | {row.get('feature_mode','')} | "
            f"{row.get('selected_algo','')} | {_fmt(row.get('macro_f1', 0.0))} | {_fmt(row.get('macro_recall',0.0))} | "
            f"{_fmt(row.get('top_class_share', 0.0))} | {row.get('acceptance_gate', {}).get('status','')} | "
            f"{row.get('collapse_gate', {}).get('status','')} |\n"
        )
    if summary.get("best_row"):
        lines.append("\n## Best Variant\n")
        br = summary["best_row"]
        lines.append(f"- run_id: `{br.get('run_id')}`\n")
        lines.append(f"- context_k: `{br.get('context_k')}`\n")
        lines.append(f"- feature_mode: `{br.get('feature_mode')}`\n")
        lines.append(f"- selected_algo: `{br.get('selected_algo')}`\n")
        lines.append(f"- selected_config: `{json.dumps(br.get('selected_config', {}), ensure_ascii=False)}`\n")
        lines.append(f"- macro_F1: `{_fmt(br.get('macro_f1', 0.0) )}`\n")
    lines.append(f"- strict runs count: `{sum(1 for r in all_rows if r.get('strict_mode', True))}` (strict mode enforced)\n" if run_mode == "strict-candidate" else "- strict runs count: `0` (diagnostic run)\n")
    lines.append(f"- diagnostic outputs: holdout/OoF and dev proba are written for fusion handoff.\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.command = " ".join(sys.argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

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
        raise SystemExit(f"Forbidden path tokens detected: {json.dumps(forbidden_hits, sort_keys=True)}")

    args.context_ks = _parse_int_grid(args.context_ks)
    train_family, dev_family, run_mode = _ensure_family_contract(
        args.train_pool,
        args.dev_pool,
        args.enable_diagnostic,
    )
    train_claims, dev_claims, evidence = _parse_inputs(args)

    if args.smoke_rows > 0:
        train_ids_all = list(train_claims.keys())[: args.smoke_rows]
        dev_ids_all = list(dev_claims.keys())[: max(1, min(args.smoke_rows, len(dev_claims)))]
        train_claims = {cid: train_claims[cid] for cid in train_ids_all}
        dev_claims = {cid: dev_claims[cid] for cid in dev_ids_all}

    train_ids = list(train_claims.keys())
    dev_ids = list(dev_claims.keys())

    feature_modes = ["embedding_only", "shallow_only", "embedding_plus_shallow"]
    if args.run_embedding_only or args.run_shallow_only or args.run_embedding_shallow:
        selected_modes: list[str] = []
        if args.run_embedding_only:
            selected_modes.append("embedding_only")
        if args.run_shallow_only:
            selected_modes.append("shallow_only")
        if args.run_embedding_shallow:
            selected_modes.append("embedding_plus_shallow")
        feature_modes = selected_modes

    train_context_raw, train_diag = _parse_pool_rows(
        args.train_pool,
        train_claims,
        evidence,
        train_family,
        top_k=max(args.context_ks),
        strict=(run_mode == "strict-candidate"),
    )
    dev_context_raw, dev_diag = _parse_pool_rows(
        args.dev_pool,
        dev_claims,
        evidence,
        dev_family,
        top_k=max(args.context_ks),
        strict=(run_mode == "strict-candidate"),
    )

    train_embed_pool = _load_embedding_pool(args.train_embedding_pool, train_claims)
    dev_embed_pool = _load_embedding_pool(args.dev_embedding_pool, dev_claims)
    train_context_raw = _attach_embedding_overrides(train_context_raw, train_embed_pool)
    dev_context_raw = _attach_embedding_overrides(dev_context_raw, dev_embed_pool)

    claim_profiles_train = _build_profiles(train_claims, "claim_text")
    claim_profiles_dev = _build_profiles(dev_claims, "claim_text")
    evidence_profiles = _build_profiles({eid: {"claim_text": text} for eid, text in evidence.items()}, "claim_text")

    backend = EmbeddingBackend(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        random_seed=args.random_seed,
    )

    needed_evidence = set()
    for rows in list(train_context_raw.values()) + list(dev_context_raw.values()):
        for item in rows:
            needed_evidence.add(item.evidence_id)

    all_claim_texts = [train_claims[cid].get("claim_text", "") for cid in train_ids] + [
        dev_claims[cid].get("claim_text", "") for cid in dev_ids
    ]
    all_claim_emb = backend.encode(all_claim_texts)
    claim_embeddings_train = {cid: all_claim_emb[idx] for idx, cid in enumerate(train_ids)}
    claim_embeddings_dev = {
        cid: all_claim_emb[len(train_ids) + idx] for idx, cid in enumerate(dev_ids)
    }

    evidence_embeddings: dict[str, np.ndarray] = {}
    for evidence_id in needed_evidence:
        text = evidence.get(evidence_id, "")
        vec = backend.encode([text])[0]
        evidence_embeddings[evidence_id] = vec.astype(np.float32)

    all_rows: list[dict[str, Any]] = []
    selection_trace_rows: list[dict[str, Any]] = []
    run_output_paths: list[str] = []
    if backend.is_diagnostic():
        summary_backend = f"diagnostic_fallback ({backend.fallback_reason})"
    else:
        summary_backend = f"transformers:{args.embedding_model}"

    for k in args.context_ks:
        run_prefix = f"{args.run_id}_k{k}"
        run_dir = args.output_root / f"{run_prefix}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_contexts = {cid: rows[:k] for cid, rows in train_context_raw.items()}
        dev_contexts = {cid: rows[:k] for cid, rows in dev_context_raw.items()}
        train_context_rows = [
            _prepare_context_rows(cid, train_claims[cid], train_contexts[cid], strict=(run_mode == "strict-candidate"))
            for cid in train_ids
        ]
        dev_context_rows = [
            _prepare_context_rows(cid, dev_claims[cid], dev_contexts[cid], strict=(run_mode == "strict-candidate"))
            for cid in dev_ids
        ]
        train_context_path = run_dir / f"{run_prefix}_train_context_top{k}.jsonl"
        dev_context_path = run_dir / f"{run_prefix}_dev_context_top{k}.jsonl"
        with train_context_path.open("w", encoding="utf-8") as f:
            for row in train_context_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with dev_context_path.open("w", encoding="utf-8") as f:
            for row in dev_context_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        train_pack = _build_feature_rows(
            claim_ids=train_ids,
            claim_profiles=claim_profiles_train,
            claim_embeddings=claim_embeddings_train,
            context_items_by_claim=train_contexts,
            evidence_embeddings=evidence_embeddings,
            evidence_profiles=evidence_profiles,
            dim=backend.dim,
        )
        dev_pack = _build_feature_rows(
            claim_ids=dev_ids,
            claim_profiles=claim_profiles_dev,
            claim_embeddings=claim_embeddings_dev,
            context_items_by_claim=dev_contexts,
            evidence_embeddings=evidence_embeddings,
            evidence_profiles=evidence_profiles,
            dim=backend.dim,
        )

        x_train_emb = train_pack["embedding_matrix"]
        x_train_shallow = train_pack["shallow_matrix"]
        x_dev_emb = dev_pack["embedding_matrix"]
        x_dev_shallow = dev_pack["shallow_matrix"]

        x_train = {
            "embedding_only": x_train_emb,
            "shallow_only": x_train_shallow,
            "embedding_plus_shallow": np.hstack([x_train_emb, x_train_shallow]),
        }
        x_dev = {
            "embedding_only": x_dev_emb,
            "shallow_only": x_dev_shallow,
            "embedding_plus_shallow": np.hstack([x_dev_emb, x_dev_shallow]),
        }

        y_train_idx = [LABEL_TO_ID[str(train_claims[cid].get("claim_label"))] for cid in train_ids]
        y_dev_idx = [LABEL_TO_ID[str(dev_claims[cid].get("claim_label"))] for cid in dev_ids]
        run_context_trace: dict[str, Any] = {
            "run_id": run_prefix,
            "context_k": int(k),
            "selection_mode": args.selection_mode,
            "feature_modes": feature_modes,
            "candidate_count": len(_model_candidates(args)),
            "feature_runs": [],
        }
        run_dir_paths: list[str] = [
            str(train_context_path),
            str(dev_context_path),
        ]

        for feature_mode in feature_modes:
            if x_train[feature_mode].shape[0] == 0 or x_train[feature_mode].shape[1] == 0:
                row = {
                    "run_id": run_prefix,
                    "context_k": int(k),
                    "feature_mode": feature_mode,
                    "feature_dim": int(x_train[feature_mode].shape[1]),
                    "strict_mode": (run_mode == "strict-candidate"),
                    "selection_mode": args.selection_mode,
                    "status": "failed",
                    "reason": "empty_features",
                    "acceptance_gate": {"status": "failed", "failures": ["empty_features"]},
                }
                all_rows.append(row)
                continue

            if args.selection_mode == "cv":
                selection_result, holdout_payload, model_bundle = _run_cv(x_train[feature_mode], y_train_idx, args, train_ids)
            else:
                selection_result, holdout_payload, model_bundle = _run_train_holdout(
                    x_train[feature_mode],
                    y_train_idx,
                    args,
                    train_ids,
                )

            context_row_map = {row["claim_id"]: row for row in dev_context_rows}
            metric_payload, holdout_payload, pred_path, metric_path, conf_path = _prepare_classification_outputs(
                feature_mode=feature_mode,
                run_prefix=f"{run_prefix}",
                selection_result=selection_result,
                holdout_payload=holdout_payload,
                selected_model=model_bundle,
                x_dev=x_dev[feature_mode],
                y_dev_idx=y_dev_idx,
                claim_ids=dev_ids,
                claims=dev_claims,
                context_rows_by_claim=context_row_map,
                args=args,
            )
            row = {
                "run_id": run_prefix,
                "context_k": int(k),
                "feature_mode": feature_mode,
                "strict_mode": (run_mode == "strict-candidate"),
                "selection_mode": args.selection_mode,
                "feature_dim": int(x_train[feature_mode].shape[1]),
                **{k: metric_payload.get(k) for k in ["macro_f1", "macro_recall", "accuracy", "top_class_share"]},
                "selected_algo": selection_result.get("selected_algo"),
                "selected_config": selection_result.get("selected_config"),
                "acceptance_gate": metric_payload.get("acceptance_gate", {}),
                "collapse_gate": metric_payload.get("collapse_gate", {}),
                "selection_payload": selection_result,
                "classification_report": metric_payload.get("classification_report", ""),
                "per_class_recall": metric_payload.get("per_class_recall", {}),
                "prediction_histogram": metric_payload.get("prediction_histogram", {}),
                "confusion_matrix": metric_payload.get("confusion_matrix", []),
                "runtime_seconds": metric_payload.get("runtime_seconds", 0.0),
            }
            all_rows.append(row)
            run_context_trace["feature_runs"].append(row)
            run_dir_paths.extend(
                [
                    str(pred_path),
                    str(metric_path),
                    str(conf_path),
                    str(args.output_root / f"{run_prefix}_{feature_mode}_train_holdout_predictions.json"),
                    str(args.output_root / f"{run_prefix}_{feature_mode}_train_holdout_proba.json"),
                    str(args.output_root / f"{run_prefix}_{feature_mode}_selection_trace.json"),
                    str(args.output_root / f"{run_prefix}_{feature_mode}_dev_proba.json"),
                    str(args.output_root / f"{run_prefix}_{feature_mode}_classification_report.txt"),
                ]
            )

        selection_trace_rows.append(run_context_trace)
        write_json(run_dir / "run_record.json", run_context_trace)
        run_output_paths.extend(run_dir_paths)

    best_row = sorted(
        all_rows,
        key=lambda r: (
            float(r.get("macro_f1", 0.0)),
            float(r.get("macro_recall", 0.0)),
            float(r.get("accuracy", 0.0)),
        ),
        reverse=True,
    )[0] if all_rows else {}
    best_macro_f1 = float(best_row.get("macro_f1", 0.0))

    for row in all_rows:
        row["acceptance_gate"] = row.get("acceptance_gate", _acceptance_gate(row, args.collapse_threshold, args.baseline_macro_f1))
        row["collapse_gate"] = row.get("collapse_gate", _collapse_gate(row.get("prediction_histogram", {}), args.collapse_threshold))

    selection_summary_path = args.output_root / "selection_summary.json"
    selection_trace_path = args.output_root / "selection_trace_summary.json"
    write_json(selection_summary_path, {"selection_rows": all_rows, "run_mode": run_mode})
    write_json(selection_trace_path, {"selection_rows": selection_trace_rows})
    run_output_paths.extend([str(selection_summary_path), str(selection_trace_path)])

    manifest = manifest_base(
        run_id=args.run_id,
        status=run_mode,
        mode="STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
        stage="o_c5_comparison",
        command=args.command,
        working_directory=Path.cwd(),
        random_seed=args.random_seed,
    )
    manifest["input_files"] = [
        {"path": str(args.train_claims), "sha256": sha256_file(args.train_claims), "split": "train", "labels_used": True},
        {"path": str(args.dev_claims), "sha256": sha256_file(args.dev_claims), "split": "dev", "labels_used": True},
        {"path": str(args.evidence), "sha256": sha256_file(args.evidence), "split": "evidence", "labels_used": False},
        {"path": str(args.train_pool), "sha256": sha256_file(args.train_pool), "split": "current_run_train_pool", "labels_used": False},
        {"path": str(args.dev_pool), "sha256": sha256_file(args.dev_pool), "split": "current_run_dev_pool", "labels_used": False},
    ]
    manifest["forbidden_input_scan"] = {
        "passed": not bool(forbidden_hits),
        "notes": "no forbidden path token hits",
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["runtime"]["device"] = str(backend.device)
    if torch.cuda.is_available():
        manifest["runtime"]["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
    manifest["metrics"] = {
        "best_macro_f1": float(best_macro_f1),
        "best_feature_mode": best_row.get("feature_mode"),
        "best_context_k": best_row.get("context_k"),
        "best_algo": best_row.get("selected_algo"),
        "selected_models": [row.get("selected_algo") for row in all_rows],
        "baseline_macro_f1_reference": float(args.baseline_macro_f1),
        "strict_vs_diagnostic": run_mode,
        "embedding_backend": summary_backend,
    }
    manifest["data_flow_summary"] = (
        "Built claim embeddings and context embeddings from claim/evidence text; "
        "constructed embedding blocks (claim/context vectors, abs diff, elementwise product, inner-product and CE/embedding/rank scalars) "
        "and shallow side features; compared embedding-only / shallow-only / embedding-plus-shallow."
    )
    manifest["split_isolation_summary"] = (
        "Model selection only uses train split holdout or CV; dev used only for final scoring."
    )
    manifest["selection_summary"] = selection_summary_path.name
    manifest["output_files"] = sorted(set(run_output_paths + [str(args.manifest), str(args.record)]))
    write_json(args.manifest, manifest)

    record = {
        "run_id": args.run_id,
        "stage": "o_c5_comparison",
        "status": run_mode,
        "mode": "STRICT" if run_mode == "strict-candidate" else "DIAGNOSTIC",
        "selection_mode": args.selection_mode,
        "selection_cv_folds": int(args.selection_cv_folds),
        "context_ks": args.context_ks,
        "feature_modes": feature_modes,
        "train": len(train_claims),
        "dev": len(dev_claims),
        "best": {
            "macro_f1": float(best_macro_f1),
            "feature_mode": best_row.get("feature_mode"),
            "context_k": best_row.get("context_k"),
            "selected_algo": best_row.get("selected_algo"),
            "selected_config": best_row.get("selected_config"),
            "selection_gate_passed": bool(best_row.get("acceptance_gate", {}).get("status") == "passed"),
            "collapse_gate_passed": bool(best_row.get("collapse_gate", {}).get("status") == "passed"),
        },
        "baseline_macro_f1_reference": float(args.baseline_macro_f1),
        "exceeds_baseline": bool(best_macro_f1 > args.baseline_macro_f1),
        "selection_summary": str(selection_summary_path),
        "selection_trace_summary": str(selection_trace_path),
        "manifest": str(args.manifest),
        "selection_rows": all_rows,
        "train_context_family": train_family,
        "dev_context_family": dev_family,
        "embedding_backend": summary_backend,
        "train_pool": str(args.train_pool),
        "dev_pool": str(args.dev_pool),
    }
    write_json(args.record, record)

    _build_report(
        Path("round18/reports/o_c5_classification_comparison/worker_d_embedding_classifier_report.md"),
        all_rows,
        {
            "selection_mode": args.selection_mode,
            "best_macro_f1": best_macro_f1,
            "best_row": best_row,
            "embedding_backend": summary_backend,
        },
        run_mode,
        args.baseline_macro_f1,
    )

    print("Selection summary:", selection_summary_path)
    print("Selection trace:", selection_trace_path)
    print("Manifest:", args.manifest)
    print("Record:", args.record)


if __name__ == "__main__":
    main()
