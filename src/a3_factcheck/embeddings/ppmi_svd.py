import math
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?")


def tokenize(text):
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_vocabulary(tokenized_docs, max_vocab=5000, min_count=2):
    counts = Counter(token for doc in tokenized_docs for token in doc)
    kept = [
        token
        for token, count in counts.most_common(max_vocab)
        if count >= min_count
    ]
    return {token: idx for idx, token in enumerate(kept)}


def build_cooccurrence_matrix(tokenized_docs, vocab, window_size=2):
    counts = defaultdict(float)
    for doc in tokenized_docs:
        ids = [vocab[token] for token in doc if token in vocab]
        for center_pos, center_id in enumerate(ids):
            start = max(0, center_pos - window_size)
            end = min(len(ids), center_pos + window_size + 1)
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                distance = abs(context_pos - center_pos)
                counts[(center_id, ids[context_pos])] += 1.0 / distance

    if not counts:
        size = len(vocab)
        return coo_matrix((size, size), dtype=np.float32).tocsr()

    rows, cols, values = zip(
        *((row, col, value) for (row, col), value in counts.items())
    )
    size = len(vocab)
    return coo_matrix((values, (rows, cols)), shape=(size, size), dtype=np.float32).tocsr()


def ppmi_matrix(cooccurrence):
    coo = cooccurrence.tocoo()
    total = float(coo.data.sum())
    if total == 0:
        return coo_matrix(cooccurrence.shape, dtype=np.float32).tocsr()

    row_sums = np.asarray(cooccurrence.sum(axis=1)).ravel()
    col_sums = np.asarray(cooccurrence.sum(axis=0)).ravel()

    values = []
    rows = []
    cols = []
    for row, col, count in zip(coo.row, coo.col, coo.data):
        denominator = row_sums[row] * col_sums[col]
        if denominator <= 0:
            continue
        pmi = math.log((count * total) / denominator)
        if pmi > 0:
            rows.append(row)
            cols.append(col)
            values.append(pmi)

    return coo_matrix(
        (values, (rows, cols)),
        shape=cooccurrence.shape,
        dtype=np.float32,
    ).tocsr()


def fit_ppmi_svd(tokenized_docs, max_vocab=5000, min_count=2, window_size=2, dim=50):
    vocab = build_vocabulary(tokenized_docs, max_vocab=max_vocab, min_count=min_count)
    cooccurrence = build_cooccurrence_matrix(
        tokenized_docs=tokenized_docs,
        vocab=vocab,
        window_size=window_size,
    )
    ppmi = ppmi_matrix(cooccurrence)

    n_components = min(dim, max(1, min(ppmi.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(ppmi)
    embeddings = normalize(embeddings)
    id_to_token = {idx: token for token, idx in vocab.items()}
    return vocab, id_to_token, embeddings, svd


def nearest_neighbors(query, vocab, id_to_token, embeddings, top_n=10):
    if query not in vocab:
        return []
    query_id = vocab[query]
    scores = embeddings @ embeddings[query_id]
    order = np.argsort(scores)[::-1]
    neighbors = []
    for idx in order:
        if idx == query_id:
            continue
        neighbors.append((id_to_token[idx], float(scores[idx])))
        if len(neighbors) >= top_n:
            break
    return neighbors

