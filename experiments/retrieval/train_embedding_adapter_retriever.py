import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from a3_factcheck.data import load_json
from a3_factcheck.rerank.candidates import Candidate, load_candidate_pool
from a3_factcheck.retrieval.dense import (
    device_from_arg,
    encode_texts,
    load_encoder,
    read_ids,
)
from experiments.retrieval.evaluate_candidate_recall import (
    LABELS,
    evaluate_pool,
    parse_ints,
)


class LowRankResidualAdapter(nn.Module):
    def __init__(self, dim, rank, alpha):
        super().__init__()
        self.down = nn.Parameter(torch.empty(dim, rank))
        self.up = nn.Parameter(torch.zeros(rank, dim))
        self.alpha = float(alpha)
        nn.init.normal_(self.down, mean=0.0, std=0.02)

    def forward(self, x):
        adapted = x + self.alpha * ((x @ self.down) @ self.up)
        return F.normalize(adapted, p=2, dim=-1)


def load_embedding_memmap(cache_dir):
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    ids = read_ids(cache_dir / "evidence_ids.json")
    shape = (int(meta["count"]), int(meta["dim"]))
    embeddings = np.memmap(
        cache_dir / "evidence_embeddings.dat",
        dtype=np.float32,
        mode="r",
        shape=shape,
    )
    return ids, embeddings, meta


def write_pool(pool, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        claim_id: [asdict(candidate) for candidate in candidates]
        for claim_id, candidates in pool.items()
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_examples(claims, negative_pool, evidence_id_to_index, negatives_per_positive, seed):
    rng = random.Random(seed)
    examples = []
    for claim_offset, (claim_id, claim) in enumerate(claims.items()):
        gold_ids = [eid for eid in claim.get("evidences", []) if eid in evidence_id_to_index]
        if not gold_ids:
            continue
        gold = set(gold_ids)
        negatives = [
            candidate.evidence_id
            for candidate in negative_pool.get(claim_id, [])
            if candidate.evidence_id not in gold and candidate.evidence_id in evidence_id_to_index
        ]
        # Keep the highest-ranked hard negatives and add light shuffling so the
        # adapter does not see exactly the same tail for every positive.
        head = negatives[: max(negatives_per_positive * 4, negatives_per_positive)]
        tail = negatives[max(negatives_per_positive * 4, negatives_per_positive) :]
        rng.shuffle(tail)
        negatives = head + tail
        if len(negatives) < negatives_per_positive:
            continue
        for pos_id in gold_ids:
            chosen = negatives[:negatives_per_positive]
            examples.append(
                (
                    claim_offset,
                    evidence_id_to_index[pos_id],
                    [evidence_id_to_index[eid] for eid in chosen],
                )
            )
    rng.shuffle(examples)
    return examples


def batch_iter(examples, batch_size, seed):
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def train_adapter(
    adapter,
    query_embeddings,
    evidence_embeddings,
    examples,
    device,
    epochs,
    batch_size,
    lr,
    seed,
):
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    query_tensor = torch.from_numpy(query_embeddings).to(device)
    adapter.train()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        start_time = time.perf_counter()
        for batch in batch_iter(examples, batch_size, seed + epoch):
            claim_indexes = torch.tensor([item[0] for item in batch], dtype=torch.long, device=device)
            candidate_indexes = [
                [item[1], *item[2]]
                for item in batch
            ]
            candidate_np = np.asarray(
                [
                    np.asarray(evidence_embeddings[indexes], dtype=np.float32)
                    for indexes in candidate_indexes
                ],
                dtype=np.float32,
            )
            candidates = torch.from_numpy(candidate_np).to(device)
            queries = query_tensor[claim_indexes]

            q = adapter(queries)
            e = adapter(candidates)
            scores = torch.einsum("bd,bnd->bn", q, e)
            labels = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(scores / 0.05, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        elapsed = time.perf_counter() - start_time
        history.append({"epoch": epoch, "loss": avg_loss, "seconds": elapsed})
        print(f"epoch={epoch} loss={avg_loss:.4f} seconds={elapsed:.1f}")
    return history


def search_with_adapter(adapter, query_embeddings, evidence_ids, evidence_embeddings, claims, top_k, device, chunk_size):
    adapter.eval()
    query_tensor = torch.from_numpy(query_embeddings).to(device)
    with torch.no_grad():
        query_adapted = adapter(query_tensor).detach()
    n_queries = query_adapted.shape[0]
    best_scores = torch.full((n_queries, 0), -1e9, dtype=torch.float32, device=device)
    best_indices = torch.empty((n_queries, 0), dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, evidence_embeddings.shape[0], chunk_size):
            chunk_np = np.asarray(evidence_embeddings[start : start + chunk_size], dtype=np.float32)
            chunk = torch.from_numpy(chunk_np).to(device)
            chunk_adapted = adapter(chunk)
            scores = query_adapted @ chunk_adapted.T
            local_k = min(top_k, scores.shape[1])
            local_scores, local_idx = torch.topk(scores, k=local_k, dim=1)
            local_idx = local_idx + start

            merged_scores = torch.cat([best_scores, local_scores], dim=1)
            merged_indices = torch.cat([best_indices, local_idx], dim=1)
            keep_k = min(top_k, merged_scores.shape[1])
            keep_scores, keep_pos = torch.topk(merged_scores, k=keep_k, dim=1)
            keep_indices = torch.gather(merged_indices, 1, keep_pos)
            best_scores = keep_scores
            best_indices = keep_indices
            if start == 0 or (start // chunk_size) % 5 == 0:
                print(f"searched {min(start + chunk_size, evidence_embeddings.shape[0])}/{evidence_embeddings.shape[0]}")

    claim_ids = list(claims.keys())
    pool = {}
    scores_cpu = best_scores.cpu().numpy()
    indices_cpu = best_indices.cpu().numpy()
    for row, claim_id in enumerate(claim_ids):
        pool[claim_id] = [
            Candidate(
                claim_id=claim_id,
                evidence_id=evidence_ids[int(index)],
                rank=rank,
                score=float(score),
            )
            for rank, (index, score) in enumerate(
                zip(indices_cpu[row].tolist(), scores_cpu[row].tolist()),
                start=1,
            )
        ]
    return pool


def main():
    parser = argparse.ArgumentParser(
        description="Train a lightweight low-rank embedding adapter and evaluate full-corpus dense recall."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence-cache-dir", default="outputs/round11/dense_bge_small_top2000/cache")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--train-negative-pool", default="outputs/round11/train_sparse_dualdense_rrf_top2000/candidates/rrf_bm25_char_dense_plain_prefix_bge_top2000.json")
    parser.add_argument("--output-dir", default="outputs/round12/embedding_adapter_bge")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--negatives-per-positive", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pool-top-k", type=int, default=2000)
    parser.add_argument("--top-k-values", default="50,100,200,500,1000,2000")
    parser.add_argument("--search-chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_from_arg(args.device)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence_ids, evidence_embeddings, meta = load_embedding_memmap(args.evidence_cache_dir)
    evidence_id_to_index = {evidence_id: idx for idx, evidence_id in enumerate(evidence_ids)}
    dim = int(meta["dim"])

    tokenizer, encoder = load_encoder(args.model, device)
    train_query_embeddings = encode_texts(
        [claim["claim_text"] for claim in train_claims.values()],
        tokenizer=tokenizer,
        model=encoder,
        device=device,
        batch_size=args.query_batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
    )
    dev_query_embeddings = encode_texts(
        [claim["claim_text"] for claim in dev_claims.values()],
        tokenizer=tokenizer,
        model=encoder,
        device=device,
        batch_size=args.query_batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    negative_pool = load_candidate_pool(args.train_negative_pool)
    examples = build_examples(
        train_claims,
        negative_pool,
        evidence_id_to_index,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    print(f"training examples={len(examples)} dim={dim} device={device}")
    adapter = LowRankResidualAdapter(dim=dim, rank=args.rank, alpha=args.alpha).to(device)
    history = train_adapter(
        adapter=adapter,
        query_embeddings=train_query_embeddings,
        evidence_embeddings=evidence_embeddings,
        examples=examples,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    pool_top_k = max([args.pool_top_k, *parse_ints(args.top_k_values)])
    start = time.perf_counter()
    pool = search_with_adapter(
        adapter=adapter,
        query_embeddings=dev_query_embeddings,
        evidence_ids=evidence_ids,
        evidence_embeddings=evidence_embeddings,
        claims=dev_claims,
        top_k=pool_top_k,
        device=device,
        chunk_size=args.search_chunk_size,
    )
    search_seconds = time.perf_counter() - start

    method = f"adapter_bge_rank{args.rank}_alpha{str(args.alpha).replace('.', 'p')}_top{pool_top_k}"
    candidate_path = candidate_dir / f"{method}.json"
    write_pool(pool, candidate_path)

    rows = []
    for retained_k in parse_ints(args.top_k_values):
        row = evaluate_pool(
            claims=dev_claims,
            pool=pool,
            method=method,
            retained_k=retained_k,
            build_seconds=search_seconds,
        )
        row["search_seconds"] = search_seconds
        rows.append(row)
        label_bits = " ".join(
            f"{label.lower()}={row[f'{label.lower()}_macro_recall']:.4f}"
            for label in LABELS
        )
        print(
            f"{method} k={retained_k} macro={row['macro_recall']:.4f} "
            f"hit_any={row['hit_any']:.4f} all_gold={row['all_gold']:.4f} {label_bits}"
        )

    with (output_dir / "candidate_recall_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "dim": dim,
            "rank": args.rank,
            "alpha": args.alpha,
            "history": history,
            "args": vars(args),
        },
        output_dir / "adapter.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": method,
                "training_examples": len(examples),
                "history": history,
                "candidate_path": str(candidate_path),
                "search_seconds": search_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary: {output_dir / 'candidate_recall_summary.csv'}")


if __name__ == "__main__":
    main()
