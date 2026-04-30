import argparse
from itertools import islice

from a3_factcheck.data import load_json
from a3_factcheck.embeddings.ppmi_svd import (
    fit_ppmi_svd,
    nearest_neighbors,
    tokenize,
)


def iter_training_texts(claims, evidence, max_evidence_docs):
    for claim in claims.values():
        yield claim["claim_text"]
    yield from islice(evidence.values(), max_evidence_docs)


def main():
    parser = argparse.ArgumentParser(
        description="Demo classical unsupervised word embeddings using PPMI + SVD."
    )
    parser.add_argument("--claims", default="data/train-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--max-evidence-docs", type=int, default=5000)
    parser.add_argument("--max-vocab", type=int, default=5000)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument(
        "--queries",
        default="climate,temperature,carbon,energy,sea",
        help="Comma-separated query words for nearest-neighbor inspection.",
    )
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    texts = list(iter_training_texts(claims, evidence, args.max_evidence_docs))
    tokenized_docs = [tokenize(text) for text in texts]

    vocab, id_to_token, embeddings, svd = fit_ppmi_svd(
        tokenized_docs=tokenized_docs,
        max_vocab=args.max_vocab,
        min_count=args.min_count,
        window_size=args.window_size,
        dim=args.dim,
    )

    print(f"documents: {len(tokenized_docs)}")
    print(f"vocab size: {len(vocab)}")
    print(f"embedding dim: {embeddings.shape[1]}")
    explained = svd.explained_variance_ratio_.sum()
    print(f"svd explained variance ratio sum: {explained:.4f}")

    for query in [item.strip().lower() for item in args.queries.split(",") if item.strip()]:
        print(f"\nnearest neighbors for '{query}':")
        neighbors = nearest_neighbors(
            query=query,
            vocab=vocab,
            id_to_token=id_to_token,
            embeddings=embeddings,
            top_n=10,
        )
        if not neighbors:
            print("  <not in vocabulary>")
            continue
        for token, score in neighbors:
            print(f"  {token:20s} {score:.4f}")


if __name__ == "__main__":
    main()

