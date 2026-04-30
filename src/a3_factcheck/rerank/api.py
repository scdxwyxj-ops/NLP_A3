import json
from pathlib import Path

import numpy as np


class EvidenceReranker:
    """Small inference API for claim-evidence reranking.

    This class wraps a HuggingFace sequence-classification model and exposes a
    task-level API: score claim/evidence pairs, rerank candidate ids, and write
    prediction JSON for the assignment evaluator.
    """

    def __init__(self, model, tokenizer, device=None, max_length=256, batch_size=16):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        device=None,
        max_length=256,
        batch_size=16,
    ):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "EvidenceReranker requires transformers and torch. Install the "
                "training extras or add them in Colab before loading a model."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )

    def score_pairs(self, claim_texts, evidence_texts):
        scores = []
        with self.torch.no_grad():
            for start in range(0, len(claim_texts), self.batch_size):
                batch_claims = claim_texts[start : start + self.batch_size]
                batch_evidence = evidence_texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_claims,
                    batch_evidence,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device) for key, value in encoded.items()
                }
                logits = self.model(**encoded).logits
                if logits.shape[-1] == 1:
                    batch_scores = self.torch.sigmoid(logits[:, 0])
                else:
                    batch_scores = self.torch.softmax(logits, dim=-1)[:, -1]
                scores.extend(batch_scores.detach().cpu().tolist())
        return np.asarray(scores, dtype=np.float32)

    def rerank_claim(self, claim_text, evidence_by_id, candidate_ids, top_k=None):
        candidate_ids = list(candidate_ids)
        if not candidate_ids:
            return []
        evidence_texts = [evidence_by_id[evidence_id] for evidence_id in candidate_ids]
        scores = self.score_pairs(
            [claim_text] * len(candidate_ids),
            evidence_texts,
        )
        order = np.argsort(scores)[::-1]
        if top_k is not None:
            order = order[:top_k]
        return [
            {
                "evidence_id": candidate_ids[idx],
                "score": float(scores[idx]),
                "rank": rank,
            }
            for rank, idx in enumerate(order, start=1)
        ]

    def rerank_claims(
        self,
        claims,
        evidence,
        candidate_pool,
        top_k=5,
        default_label="NOT_ENOUGH_INFO",
    ):
        predictions = {}
        for claim_id, claim in claims.items():
            candidate_ids = [
                candidate.evidence_id for candidate in candidate_pool.get(claim_id, [])
            ]
            ranked = self.rerank_claim(
                claim_text=claim["claim_text"],
                evidence_by_id=evidence,
                candidate_ids=candidate_ids,
                top_k=top_k,
            )
            predictions[claim_id] = {
                "claim_text": claim["claim_text"],
                "claim_label": default_label,
                "evidences": [item["evidence_id"] for item in ranked],
            }
        return predictions


def write_predictions(predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
