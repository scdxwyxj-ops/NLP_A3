import argparse
import json
from pathlib import Path

from a3_factcheck.data import load_json
from a3_factcheck.metrics import load_predictions
from a3_factcheck.semantic import extract_semantic_features


def load_ranked_candidates(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ids_from_predictions(predictions, claim_id, top_k):
    return predictions.get(claim_id, {}).get("evidences", [])[:top_k]


def context_from_predictions(predictions, claim_id, evidence, top_k, include_semantic):
    context = []
    for rank, evidence_id in enumerate(
        ids_from_predictions(predictions, claim_id, top_k), start=1
    ):
        item = {
            "evidence_id": evidence_id,
            "rank": rank,
            "text": evidence[evidence_id],
        }
        if include_semantic:
            item["semantic_features"] = extract_semantic_features(evidence[evidence_id])
        context.append(item)
    return context


def context_from_ranked(ranked_candidates, claim_id, evidence, top_k, include_semantic):
    context = []
    for item in ranked_candidates.get(claim_id, [])[:top_k]:
        evidence_id = item["evidence_id"]
        record = {
            "evidence_id": evidence_id,
            "rank": item.get("minilm_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "bm25_score": item.get("bm25_score"),
            "reranker_score": item.get("minilm_score"),
            "text": evidence[evidence_id],
        }
        if include_semantic:
            record["semantic_features"] = extract_semantic_features(evidence[evidence_id])
        context.append(record)
    return context


def build_rows(
    claims,
    evidence,
    final_predictions=None,
    context_predictions=None,
    ranked_candidates=None,
    final_top_k=3,
    context_top_k=20,
    include_semantic=False,
):
    rows = []
    for claim_id, claim in claims.items():
        if ranked_candidates is not None:
            context = context_from_ranked(
                ranked_candidates,
                claim_id,
                evidence,
                context_top_k,
                include_semantic,
            )
            final_ids = [item["evidence_id"] for item in context[:final_top_k]]
        else:
            context = context_from_predictions(
                context_predictions,
                claim_id,
                evidence,
                context_top_k,
                include_semantic,
            )
            final_ids = ids_from_predictions(final_predictions, claim_id, final_top_k)

        row = {
            "claim_id": claim_id,
            "claim_text": claim["claim_text"],
            "claim_label": claim.get("claim_label"),
            "gold_evidences": claim.get("evidences"),
            "final_evidence_candidates": final_ids,
            "classifier_context_top_k": context_top_k,
            "classifier_evidence_context": context,
        }
        if include_semantic:
            row["claim_semantic_features"] = extract_semantic_features(
                claim["claim_text"]
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build classifier-ready claim/evidence context JSONL."
    )
    parser.add_argument("--claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--final-predictions", default="")
    parser.add_argument("--context-predictions", default="")
    parser.add_argument("--ranked-candidates", default="")
    parser.add_argument("--final-top-k", type=int, default=3)
    parser.add_argument("--context-top-k", type=int, default=20)
    parser.add_argument("--include-semantic", action="store_true")
    parser.add_argument("--output", default="outputs/round06/dev-classifier-context.jsonl")
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    ranked_candidates = (
        load_ranked_candidates(args.ranked_candidates)
        if args.ranked_candidates
        else None
    )
    final_predictions = (
        load_predictions(args.final_predictions) if args.final_predictions else None
    )
    context_predictions = (
        load_predictions(args.context_predictions) if args.context_predictions else None
    )

    if ranked_candidates is None and (
        final_predictions is None or context_predictions is None
    ):
        raise SystemExit(
            "Provide either --ranked-candidates or both --final-predictions "
            "and --context-predictions."
        )

    rows = build_rows(
        claims=claims,
        evidence=evidence,
        final_predictions=final_predictions,
        context_predictions=context_predictions,
        ranked_candidates=ranked_candidates,
        final_top_k=args.final_top_k,
        context_top_k=args.context_top_k,
        include_semantic=args.include_semantic,
    )
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
