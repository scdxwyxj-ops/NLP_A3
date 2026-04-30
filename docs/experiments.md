# Experiments

## Naming

Use descriptive names for experiment scripts and outputs:

```txt
experiments/retrieval/tfidf_baseline.py
outputs/round04/dev-tfidf-top3.json
```

## Running TF-IDF Baseline

```bash
PYTHONPATH=src python experiments/retrieval/tfidf_baseline.py \
  --top-k-values 1,3,5,10,20 \
  --output outputs/round04/dev-tfidf.json
```

The legacy wrapper still works:

```bash
PYTHONPATH=. python src/tfidf_retrieval_baseline.py
```

## Running Round05 Reranker

Build BM25 top-50 candidate pools and supervised hard-negative training pairs:

```bash
PYTHONPATH=src python experiments/rerank/build_hard_negative_dataset.py \
  --output-dir outputs/round05 \
  --top-k 50 \
  --negatives-per-claim 5
```

Train the first cross-encoder reranker in Colab or an environment with PyTorch
and Transformers installed:

```bash
PYTHONPATH=src python experiments/rerank/train_cross_encoder.py \
  --model-name cross-encoder/ms-marco-MiniLM-L6-v2 \
  --train-pairs outputs/round05/train-reranker-pairs-top50-neg5.jsonl \
  --output-dir models/round05/minilm-reranker
```

Rerank dev BM25 candidates and evaluate top-k evidence selection:

```bash
PYTHONPATH=src python experiments/rerank/rerank_with_cross_encoder.py \
  --candidate-pool outputs/round05/dev-bm25-top50.json \
  --model models/round05/minilm-reranker \
  --top-k-values 1,3,5,10
```

Current dataset-build stats:

- train pairs: `10262`
- positives: `4122`
- hard negatives: `6140`
- train recall@50: `0.2889`
- dev recall@50: `0.3249`

## Recording Results

Record stable results in `agent_docs/rounds/` first. Promote report-ready summaries into `docs/` when they become useful for teammates.
