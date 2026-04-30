import argparse
import json
from pathlib import Path

import numpy as np

from a3_factcheck.data import load_json
from a3_factcheck.rerank.api import EvidenceReranker
from a3_factcheck.rerank.candidates import load_candidate_pool
from a3_factcheck.rerank.dataset import pair_record, write_jsonl


def score_candidates(claims, evidence, pool, reranker):
    ranked_by_claim = {}
    for claim_id, claim in claims.items():
        candidates = pool.get(claim_id, [])
        candidate_ids = [candidate.evidence_id for candidate in candidates]
        if not candidate_ids:
            ranked_by_claim[claim_id] = []
            continue

        scores = reranker.score_pairs(
            [claim["claim_text"]] * len(candidate_ids),
            [evidence[evidence_id] for evidence_id in candidate_ids],
        )
        order = np.argsort(scores)[::-1]
        ranked = []
        for minilm_rank, candidate_idx in enumerate(order, start=1):
            candidate = candidates[candidate_idx]
            ranked.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": candidate.evidence_id,
                    "bm25_rank": candidate.rank,
                    "bm25_score": candidate.score,
                    "minilm_rank": minilm_rank,
                    "minilm_score": float(scores[candidate_idx]),
                }
            )
        ranked_by_claim[claim_id] = ranked
    return ranked_by_claim


def build_pairs(claims, evidence, ranked_by_claim, negatives_per_claim):
    rows = []
    for claim_id, claim in claims.items():
        claim_text = claim["claim_text"]
        gold_ids = list(dict.fromkeys(claim.get("evidences", [])))
        gold_set = set(gold_ids)
        ranked = ranked_by_claim.get(claim_id, [])
        ranked_by_evidence = {item["evidence_id"]: item for item in ranked}

        for evidence_id in gold_ids:
            if evidence_id not in evidence:
                continue
            item = ranked_by_evidence.get(evidence_id, {})
            row = pair_record(
                claim_id=claim_id,
                evidence_id=evidence_id,
                claim_text=claim_text,
                evidence_text=evidence[evidence_id],
                label=1,
                source="gold_candidate"
                if evidence_id in ranked_by_evidence
                else "gold",
                bm25_rank=item.get("bm25_rank"),
                bm25_score=item.get("bm25_score"),
            )
            row["minilm_rank"] = item.get("minilm_rank")
            row["minilm_score"] = item.get("minilm_score")
            rows.append(row)

        added_negatives = 0
        for item in ranked:
            evidence_id = item["evidence_id"]
            if evidence_id in gold_set or evidence_id not in evidence:
                continue
            row = pair_record(
                claim_id=claim_id,
                evidence_id=evidence_id,
                claim_text=claim_text,
                evidence_text=evidence[evidence_id],
                label=0,
                source="minilm_mined_hard_negative",
                bm25_rank=item["bm25_rank"],
                bm25_score=item["bm25_score"],
            )
            row["minilm_rank"] = item["minilm_rank"]
            row["minilm_score"] = item["minilm_score"]
            rows.append(row)
            added_negatives += 1
            if added_negatives >= negatives_per_claim:
                break
    return rows


def write_ranked_candidates(ranked_by_claim, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ranked_by_claim, f, ensure_ascii=False, indent=2)


def write_stats(rows, ranked_by_claim, output_path):
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    stats = {
        "claims": len(ranked_by_claim),
        "pairs": len(rows),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "negative_source": "minilm_mined_hard_negative",
        "mean_negative_minilm_rank": float(
            np.mean([row["minilm_rank"] for row in negatives])
        )
        if negatives
        else None,
        "mean_negative_bm25_rank": float(np.mean([row["bm25_rank"] for row in negatives]))
        if negatives
        else None,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def write_examples(rows, output_path, limit=20):
    negatives = [row for row in rows if row["label"] == 0]
    negatives = sorted(
        negatives,
        key=lambda row: (row["minilm_rank"], row["bm25_rank"] or 999999),
    )
    lines = ["# MiniLM-Mined Hard Negative Examples", ""]
    for row in negatives[:limit]:
        lines.append(
            f"## {row['claim_id']} -> {row['evidence_id']} "
            f"(MiniLM rank {row['minilm_rank']}, BM25 rank {row['bm25_rank']})"
        )
        lines.append("")
        lines.append(f"Claim: {row['claim_text']}")
        lines.append("")
        lines.append(f"Negative evidence: {row['evidence_text']}")
        lines.append("")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Mine task-aware hard negatives with a zero-shot reranker."
    )
    parser.add_argument("--claims", default="data/train-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument(
        "--candidate-pool", default="outputs/round05_top100/train-bm25-top100.json"
    )
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--negatives-per-claim", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/round06")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    pool = load_candidate_pool(args.candidate_pool)
    reranker = EvidenceReranker.from_pretrained(
        args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    ranked_by_claim = score_candidates(
        claims=claims,
        evidence=evidence,
        pool=pool,
        reranker=reranker,
    )
    rows = build_pairs(
        claims=claims,
        evidence=evidence,
        ranked_by_claim=ranked_by_claim,
        negatives_per_claim=args.negatives_per_claim,
    )

    suffix = f"top{max(len(v) for v in pool.values())}-neg{args.negatives_per_claim}"
    pairs_path = output_dir / f"train-task-aware-negatives-{suffix}.jsonl"
    ranked_path = output_dir / f"train-minilm-ranked-candidates-{suffix}.json"
    stats_path = output_dir / f"task-aware-negative-stats-{suffix}.json"
    examples_path = output_dir / f"task-aware-negative-examples-{suffix}.md"

    write_jsonl(rows, pairs_path)
    write_ranked_candidates(ranked_by_claim, ranked_path)
    stats = write_stats(rows, ranked_by_claim, stats_path)
    write_examples(rows, examples_path)

    print(json.dumps({"pairs_path": str(pairs_path), **stats}, indent=2))


if __name__ == "__main__":
    main()
