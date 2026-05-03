import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def device_from_arg(device):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def pool_outputs(outputs, attention_mask, pooling):
    if pooling == "cls":
        embeddings = outputs.last_hidden_state[:, 0]
    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def encode_texts(texts, tokenizer, model, device, batch_size, max_length, pooling):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            embeddings = pool_outputs(
                outputs=outputs,
                attention_mask=encoded["attention_mask"],
                pooling=pooling,
            )
            chunks.append(embeddings.cpu().numpy().astype(np.float32))
    return np.vstack(chunks) if chunks else np.empty((0, model.config.hidden_size))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_ids(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_ids(path, ids):
    write_json(path, ids)


def build_embedding_cache(
    evidence,
    model_name,
    cache_dir,
    device,
    batch_size=128,
    max_length=128,
    pooling="cls",
    text_prefix="",
    dtype=np.float32,
):
    dtype = np.dtype(dtype)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path = cache_dir / "evidence_ids.json"
    embeddings_path = cache_dir / "evidence_embeddings.dat"
    meta_path = cache_dir / "metadata.json"

    evidence_ids = list(evidence.keys())
    if ids_path.exists() and embeddings_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model_name") == model_name
            and meta.get("pooling") == pooling
            and meta.get("max_length") == max_length
            and meta.get("text_prefix", "") == text_prefix
            and meta.get("dtype") == str(dtype)
            and meta.get("count") == len(evidence_ids)
        ):
            shape = (meta["count"], meta["dim"])
            return read_ids(ids_path), np.memmap(
                embeddings_path, dtype=dtype, mode="r", shape=shape
            )

    tokenizer, model = load_encoder(model_name, device)
    dim = int(model.config.hidden_size)
    embeddings = np.memmap(
        embeddings_path, dtype=dtype, mode="w+", shape=(len(evidence_ids), dim)
    )

    for start in range(0, len(evidence_ids), batch_size):
        batch_ids = evidence_ids[start : start + batch_size]
        batch_texts = [
            f"{text_prefix}{evidence[evidence_id]}" for evidence_id in batch_ids
        ]
        batch_embeddings = encode_texts(
            texts=batch_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
        )
        embeddings[start : start + len(batch_ids)] = batch_embeddings
        if start == 0 or (start // batch_size) % 100 == 0:
            print(f"Encoded {start + len(batch_ids)}/{len(evidence_ids)} evidence")

    embeddings.flush()
    write_ids(ids_path, evidence_ids)
    write_json(
        meta_path,
        {
            "model_name": model_name,
            "pooling": pooling,
            "max_length": max_length,
            "text_prefix": text_prefix,
            "count": len(evidence_ids),
            "dim": dim,
            "dtype": str(np.dtype(dtype)),
        },
    )
    return evidence_ids, np.memmap(embeddings_path, dtype=dtype, mode="r", shape=(len(evidence_ids), dim))


def search_dense_topk(
    query_embeddings,
    evidence_embeddings,
    top_k,
    chunk_size=100_000,
    device=None,
    query_batch_size=32,
):
    if device is not None and torch.device(device).type == "cuda":
        return search_dense_topk_torch(
            query_embeddings=query_embeddings,
            evidence_embeddings=evidence_embeddings,
            top_k=top_k,
            chunk_size=chunk_size,
            device=torch.device(device),
            query_batch_size=query_batch_size,
        )

    results = []
    for query in query_embeddings:
        best_scores = np.empty((0,), dtype=np.float32)
        best_indices = np.empty((0,), dtype=np.int64)
        for start in range(0, evidence_embeddings.shape[0], chunk_size):
            chunk = np.asarray(evidence_embeddings[start : start + chunk_size])
            scores = chunk @ query
            local_k = min(top_k, scores.shape[0])
            if local_k == scores.shape[0]:
                selected = np.argsort(scores)[::-1]
            else:
                selected = np.argpartition(scores, -local_k)[-local_k:]
                selected = selected[np.argsort(scores[selected])[::-1]]
            candidate_scores = scores[selected]
            candidate_indices = selected + start
            best_scores = np.concatenate([best_scores, candidate_scores])
            best_indices = np.concatenate([best_indices, candidate_indices])
            keep_k = min(top_k, best_scores.shape[0])
            keep = np.argpartition(best_scores, -keep_k)[-keep_k:]
            keep = keep[np.argsort(best_scores[keep])[::-1]]
            best_scores = best_scores[keep]
            best_indices = best_indices[keep]
        results.append((best_indices.tolist(), best_scores.tolist()))
    return results


def search_dense_topk_torch(
    query_embeddings,
    evidence_embeddings,
    top_k,
    chunk_size,
    device,
    query_batch_size,
):
    """Chunked GPU top-k search without materializing the full score matrix."""
    results = []
    n_queries = query_embeddings.shape[0]
    n_evidence = evidence_embeddings.shape[0]
    query_batch_size = max(1, int(query_batch_size))

    with torch.no_grad():
        for query_start in range(0, n_queries, query_batch_size):
            query_np = query_embeddings[
                query_start : query_start + query_batch_size
            ].astype(np.float32, copy=False)
            queries = torch.from_numpy(query_np).to(device=device, dtype=torch.float32)
            best_scores = None
            best_indices = None

            for evidence_start in range(0, n_evidence, chunk_size):
                evidence_np = np.asarray(
                    evidence_embeddings[
                        evidence_start : evidence_start + chunk_size
                    ]
                ).astype(np.float32, copy=False)
                evidence = torch.from_numpy(evidence_np).to(
                    device=device, dtype=torch.float32
                )
                scores = queries @ evidence.T
                local_k = min(top_k, scores.shape[1])
                candidate_scores, candidate_indices = torch.topk(
                    scores, k=local_k, dim=1
                )
                candidate_indices = candidate_indices + evidence_start

                if best_scores is None:
                    best_scores = candidate_scores
                    best_indices = candidate_indices
                else:
                    merged_scores = torch.cat([best_scores, candidate_scores], dim=1)
                    merged_indices = torch.cat([best_indices, candidate_indices], dim=1)
                    keep_k = min(top_k, merged_scores.shape[1])
                    best_scores, keep = torch.topk(merged_scores, k=keep_k, dim=1)
                    best_indices = torch.gather(merged_indices, 1, keep)

                del evidence, scores, candidate_scores, candidate_indices

            batch_scores = best_scores.cpu().numpy()
            batch_indices = best_indices.cpu().numpy()
            for indices, scores in zip(batch_indices, batch_scores):
                results.append((indices.tolist(), scores.tolist()))
            del queries, best_scores, best_indices

    return results
