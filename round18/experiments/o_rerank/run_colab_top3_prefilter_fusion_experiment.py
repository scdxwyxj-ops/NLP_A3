#!/usr/bin/env python
"""Dev-only top3 fusion experiment for the self-contained Colab pipeline.

This script does not modify the submission notebook. It tests a cheaper version
of the tutorial top3 idea: keep the full sparse top500 candidate pool for final
fusion, but only run the cross-encoder on a per-claim prefilter subset.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_DEV_CLAIMS = Path("colab_notebooks/dev-claims.json")
DEFAULT_EVIDENCE = Path("colab_notebooks/evidence.json")
DEFAULT_DEV_CANDIDATES = Path("round18/reports/colab_sklearn_candidate_experiment/fused_char_heavy_top500.json")
DEFAULT_OUTPUT_DIR = Path("round18/reports/colab_top3_prefilter_fusion")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def mean_pool(outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    vecs: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(**batch)
            emb = mean_pool(out, batch["attention_mask"])
        vecs.append(F.normalize(emb.float(), dim=1).cpu())
    return torch.cat(vecs, 0) if vecs else torch.empty((0, 384))


@torch.inference_mode()
def ce_score_pairs(
    pairs: list[dict[str, str]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        enc = tokenizer(
            [row["claim_text"] for row in batch_pairs],
            [row["evidence_text"] for row in batch_pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        logits = model(**enc).logits.detach().float().cpu().numpy()
        vals = logits[:, 0] if logits.ndim == 2 else logits
        scores.extend(float(x) for x in vals)
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(pairs):
            print("ce scored", min(start + batch_size, len(pairs)), "/", len(pairs))
    return scores


def minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def evaluate(claims: dict[str, Any], ranked: dict[str, list[dict[str, Any]]], k: int) -> dict[str, float]:
    recalls: list[float] = []
    hits: list[bool] = []
    fs: list[float] = []
    for cid, claim in claims.items():
        gold = set(claim.get("evidences", []))
        pred = [row["evidence_id"] for row in ranked.get(cid, [])[:k]]
        overlap = gold & set(pred)
        recall = len(overlap) / len(gold) if gold else 0.0
        precision = len(overlap) / len(pred) if pred else 0.0
        f_score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        recalls.append(recall)
        hits.append(bool(overlap))
        fs.append(f_score)
    return {
        f"macro_recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hit_any@{k}": float(np.mean(hits)) if hits else 0.0,
        f"evidence_f@{k}": float(np.mean(fs)) if fs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-claims", type=Path, default=DEFAULT_DEV_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--dev-candidates", type=Path, default=DEFAULT_DEV_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--ce-prefilter-k", type=int, default=128)
    parser.add_argument("--embed-batch", type=int, default=128)
    parser.add_argument("--ce-batch", type=int, default=96)
    args = parser.parse_args()

    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    candidates = load_json(args.dev_candidates)
    claim_ids = list(claims)

    candidate_eids: list[str] = []
    seen: set[str] = set()
    for cid in claim_ids:
        for row in candidates.get(cid, [])[: args.candidate_k]:
            eid = row["evidence_id"]
            if eid in evidence and eid not in seen:
                seen.add(eid)
                candidate_eids.append(eid)
    print("candidate evidence union", len(candidate_eids))

    embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    evidence_emb = encode_texts(
        [evidence[eid] for eid in candidate_eids],
        embed_tokenizer,
        embed_model,
        device,
        batch_size=args.embed_batch,
        max_length=128,
    ).contiguous()
    claim_emb = encode_texts(
        [claims[cid]["claim_text"] for cid in claim_ids],
        embed_tokenizer,
        embed_model,
        device,
        batch_size=args.embed_batch,
        max_length=128,
    ).contiguous()
    del embed_model, embed_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eid_to_idx = {eid: idx for idx, eid in enumerate(candidate_eids)}
    per_claim_rows: dict[str, list[dict[str, Any]]] = {}
    ce_pairs: list[dict[str, str]] = []
    ce_pair_keys: list[tuple[str, str]] = []
    for qi, cid in enumerate(claim_ids):
        base = [row for row in candidates.get(cid, [])[: args.candidate_k] if row["evidence_id"] in eid_to_idx]
        idxs = torch.as_tensor([eid_to_idx[row["evidence_id"]] for row in base], dtype=torch.long)
        emb_scores = (evidence_emb[idxs] @ claim_emb[qi]).numpy() if len(base) else np.asarray([], dtype=np.float32)
        emb_norm = minmax(emb_scores.astype(np.float32, copy=False))
        order = np.argsort(-emb_scores) if len(base) else np.asarray([], dtype=np.int64)
        emb_rank = np.empty(len(base), dtype=np.int32)
        if len(base):
            emb_rank[order] = np.arange(1, len(base) + 1)
        rows: list[dict[str, Any]] = []
        for pos, (row, emb_score, emb_norm_score) in enumerate(zip(base, emb_scores, emb_norm), start=1):
            rows.append(
                {
                    "evidence_id": row["evidence_id"],
                    "source_rank": int(row.get("rank", pos)),
                    "embedding_score": float(emb_score),
                    "embedding_norm": float(emb_norm_score),
                    "embedding_rank": int(emb_rank[pos - 1]),
                }
            )
        rows.sort(key=lambda row: (min(row["source_rank"], row["embedding_rank"]), row["source_rank"], row["evidence_id"]))
        selected = rows[: args.ce_prefilter_k]
        for row in selected:
            eid = row["evidence_id"]
            ce_pairs.append(
                {
                    "claim_text": claims[cid]["claim_text"],
                    "evidence_text": " ".join(evidence[eid].split()[:90]),
                }
            )
            ce_pair_keys.append((cid, eid))
        per_claim_rows[cid] = rows
    print("ce prefilter pairs", len(ce_pairs), "k", args.ce_prefilter_k)

    ce_tokenizer = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
    ce_model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL_NAME).to(device)
    ce_scores = ce_score_pairs(
        ce_pairs,
        ce_tokenizer,
        ce_model,
        device,
        batch_size=args.ce_batch,
        max_length=256,
    )
    del ce_model, ce_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ce_by_claim: dict[str, dict[str, float]] = {}
    for (cid, eid), score in zip(ce_pair_keys, ce_scores):
        ce_by_claim.setdefault(cid, {})[eid] = score

    ranked: dict[str, list[dict[str, Any]]] = {}
    weights = {"ce_score": 0.25, "embedding_score": 0.5, "ce_rank": 0.25}
    for cid, rows in per_claim_rows.items():
        ce_map = ce_by_claim.get(cid, {})
        ce_vals = np.asarray([ce_map[eid] for eid in ce_map], dtype=np.float32)
        ce_norm_map: dict[str, float] = {}
        ce_rank_map: dict[str, int] = {}
        if len(ce_vals):
            norm_vals = minmax(ce_vals)
            ce_items = list(ce_map.items())
            for (eid, _), norm_score in zip(ce_items, norm_vals):
                ce_norm_map[eid] = float(norm_score)
            sorted_ce = sorted(ce_map.items(), key=lambda item: (-item[1], item[0]))
            ce_rank_map = {eid: rank for rank, (eid, _) in enumerate(sorted_ce, start=1)}
        fused_rows = []
        for row in rows:
            eid = row["evidence_id"]
            ce_rank = ce_rank_map.get(eid, args.candidate_k + 1)
            score = (
                weights["ce_score"] * ce_norm_map.get(eid, 0.0)
                + weights["embedding_score"] * row["embedding_norm"]
                + weights["ce_rank"] * (1.0 / (60.0 + float(ce_rank)) if eid in ce_rank_map else 0.0)
            )
            fused_rows.append(
                {
                    "evidence_id": eid,
                    "score": float(score),
                    "fusion_score": float(score),
                    "embedding_score": row["embedding_score"],
                    "embedding_rank": row["embedding_rank"],
                    "source_rank": row["source_rank"],
                    "ce_score": float(ce_map.get(eid, 0.0)),
                    "ce_rank": int(ce_rank) if eid in ce_rank_map else None,
                }
            )
        fused_rows.sort(
            key=lambda row: (
                -row["fusion_score"],
                row["ce_rank"] if row["ce_rank"] is not None else 10**9,
                row["embedding_rank"],
                row["source_rank"],
                row["evidence_id"],
            )
        )
        for rank, row in enumerate(fused_rows, start=1):
            row["rank"] = rank
        ranked[cid] = fused_rows[: args.candidate_k]

    metrics = {str(k): evaluate(claims, ranked, k) for k in [1, 3, 5, 10, 64, 500]}
    summary = {
        "candidate_file": str(args.dev_candidates),
        "candidate_k": args.candidate_k,
        "ce_prefilter_k": args.ce_prefilter_k,
        "weights": weights,
        "metrics": metrics,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"top3_prefilter_k{args.ce_prefilter_k}_ranked.json", ranked)
    write_json(args.output_dir / f"top3_prefilter_k{args.ce_prefilter_k}_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
