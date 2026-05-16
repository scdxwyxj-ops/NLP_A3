import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from experiments.retrieval.evaluate_candidate_recall import evaluate_pool, write_pool
from experiments.retrieval.train_recall_compressor import (
    FEATURE_NAMES,
    build_rows,
    pool_index,
)


class FeatureMLP(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_examples(y, claim_ids, negatives_per_positive, seed):
    rng = random.Random(seed)
    by_claim = {}
    for idx, claim_id in enumerate(claim_ids):
        by_claim.setdefault(claim_id, {"pos": [], "neg": []})
        if y[idx]:
            by_claim[claim_id]["pos"].append(idx)
        else:
            by_claim[claim_id]["neg"].append(idx)

    examples = []
    for claim_id, parts in by_claim.items():
        negatives = parts["neg"]
        if len(negatives) < negatives_per_positive:
            continue
        hard_head = negatives[: max(negatives_per_positive * 4, negatives_per_positive)]
        tail = negatives[max(negatives_per_positive * 4, negatives_per_positive) :]
        rng.shuffle(tail)
        ordered_negatives = hard_head + tail
        for pos_idx in parts["pos"]:
            examples.append((pos_idx, ordered_negatives[:negatives_per_positive]))
    rng.shuffle(examples)
    return examples


def train_model(model, X, examples, mean, std, device, epochs, batch_size, lr, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        shuffled = list(examples)
        rng.shuffle(shuffled)
        total_loss = 0.0
        steps = 0
        model.train()
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            candidate_indexes = [[pos, *negs] for pos, negs in batch]
            values = np.asarray([X[indexes] for indexes in candidate_indexes], dtype=np.float32)
            values = (values - mean) / std
            tensor = torch.from_numpy(values).to(device)
            scores = model(tensor)
            labels = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"epoch={epoch} loss={avg_loss:.4f}")
    return history


def score_dev(model, X_dev, mean, std, device, batch_size):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, X_dev.shape[0], batch_size):
            values = (X_dev[start : start + batch_size] - mean) / std
            tensor = torch.from_numpy(values.astype(np.float32)).to(device)
            scores.append(model(tensor).detach().cpu().numpy())
    return np.concatenate(scores)


def rank_predictions(claims, claim_ids, evidence_ids, scores, top_k):
    grouped = {claim_id: [] for claim_id in claims}
    for claim_id, evidence_id, score in zip(claim_ids, evidence_ids, scores):
        grouped[claim_id].append((evidence_id, float(score)))
    pool = {}
    for claim_id, items in grouped.items():
        ranked = sorted(items, key=lambda item: (-item[1], item[0]))[:top_k]
        pool[claim_id] = [
            Candidate(claim_id=claim_id, evidence_id=evidence_id, rank=rank, score=score)
            for rank, (evidence_id, score) in enumerate(ranked, start=1)
        ]
    return pool


def main():
    parser = argparse.ArgumentParser(
        description="Train a claim-level listwise compressor for topN-to-topK recall."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--train-rrf-pool", required=True)
    parser.add_argument("--dev-rrf-pool", required=True)
    parser.add_argument("--train-bm25-pool", required=True)
    parser.add_argument("--dev-bm25-pool", required=True)
    parser.add_argument("--train-char-pool", required=True)
    parser.add_argument("--dev-char-pool", required=True)
    parser.add_argument("--train-dense-pool", required=True)
    parser.add_argument("--dev-dense-pool", required=True)
    parser.add_argument("--train-dense2-pool", default="")
    parser.add_argument("--dev-dense2-pool", default="")
    parser.add_argument("--output-dir", default="outputs/round12/listwise_compressor")
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--output-k", type=int, default=500)
    parser.add_argument("--negatives-per-positive", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pools = {
        "rrf": load_candidate_pool(args.train_rrf_pool),
        "bm25": load_candidate_pool(args.train_bm25_pool),
        "char": load_candidate_pool(args.train_char_pool),
        "dense": load_candidate_pool(args.train_dense_pool),
    }
    dev_pools = {
        "rrf": load_candidate_pool(args.dev_rrf_pool),
        "bm25": load_candidate_pool(args.dev_bm25_pool),
        "char": load_candidate_pool(args.dev_char_pool),
        "dense": load_candidate_pool(args.dev_dense_pool),
    }
    if args.train_dense2_pool and args.dev_dense2_pool:
        train_pools["dense2"] = load_candidate_pool(args.train_dense2_pool)
        dev_pools["dense2"] = load_candidate_pool(args.dev_dense2_pool)

    train_indexes = {name: pool_index(pool) for name, pool in train_pools.items()}
    dev_indexes = {name: pool_index(pool) for name, pool in dev_pools.items()}
    X_train, y_train, train_row_claims, _train_eids = build_rows(
        train_claims,
        evidence,
        train_pools["rrf"],
        train_indexes,
        args.max_candidates,
        train_mode=True,
    )
    X_dev, _y_dev, dev_row_claims, dev_eids = build_rows(
        dev_claims,
        evidence,
        dev_pools["rrf"],
        dev_indexes,
        args.max_candidates,
        train_mode=False,
    )
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    examples = make_examples(
        y=y_train,
        claim_ids=train_row_claims,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    print(f"train_rows={X_train.shape[0]} positives={int(y_train.sum())} examples={len(examples)} dev_rows={X_dev.shape[0]}")
    model = FeatureMLP(dim=len(FEATURE_NAMES), hidden=args.hidden, dropout=args.dropout).to(device)
    history = train_model(
        model=model,
        X=X_train,
        examples=examples,
        mean=mean,
        std=std,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    scores = score_dev(
        model=model,
        X_dev=X_dev,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.score_batch_size,
    )
    pool = rank_predictions(dev_claims, dev_row_claims, dev_eids, scores, args.output_k)
    candidate_path = candidate_dir / f"listwise_mlp_top{args.output_k}.json"
    write_pool(pool, candidate_path)
    row = evaluate_pool(dev_claims, pool, "listwise_mlp", args.output_k, 0.0)
    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "mean": mean,
            "std": std,
            "args": vars(args),
            "history": history,
        },
        output_dir / "model.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "history": history,
                "metrics": row,
                "candidate_path": str(candidate_path),
                "examples": len(examples),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"listwise_mlp top{args.output_k} macro={row['macro_recall']:.4f} "
        f"hit={row['hit_any']:.4f} all={row['all_gold']:.4f} "
        f"refutes={row['refutes_macro_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
