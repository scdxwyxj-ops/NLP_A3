# Round07 报告 C：Rerank 结果

## 目的

Stage C 使用稳定的 zero-shot MiniLM cross-encoder，对 Stage A/B 中最好的
candidate pools 做 reranking。

## 脚本

```txt
experiments/rerank/rerank_pool_once.py
```

该脚本只对每个 candidate pool 打分一次，然后输出多个 top-k prediction 文件，
避免为不同 top-k 重复跑 cross-encoder。

## 测试的 Candidate Pools

| Candidate Pool | Candidate Recall@500 | 选择原因 |
|---|---:|---|
| `rrf_bm25_char_tfidf_top500` | 0.6579 | recall、all-gold、REFUTES recall 最均衡 |
| `bm25_top500` | 0.5861 | 检查扩大 BM25 alone 是否足够 |
| `tfidf_char_top500` | 0.6610 | candidate macro recall 最高 |

## Final Evidence 结果

| 方法 | Output k | Evidence F-score | Macro Recall | REFUTES Recall | Hit-any |
|---|---:|---:|---:|---:|---:|
| 旧最佳：BM25 top50 -> zero-shot MiniLM | 3 | 0.1642 | 0.1877 | 0.0494 | 0.3961 |
| BM25 top500 -> zero-shot MiniLM | 3 | 0.1945 | 0.2286 | 0.1605 | 0.4416 |
| char TF-IDF top500 -> zero-shot MiniLM | 3 | 0.1920 | 0.2276 | 0.1605 | 0.4481 |
| RRF(BM25, char TF-IDF) top500 -> zero-shot MiniLM | 3 | 0.1987 | 0.2331 | 0.1790 | 0.4545 |

## Classifier Context 结果

同一个最佳 pipeline 在 top-20 时能提供更宽的 classifier context：

| 方法 | Context k | Evidence F-score | Macro Recall | REFUTES Recall | Hit-any |
|---|---:|---:|---:|---:|---:|
| RRF(BM25, char TF-IDF) top500 -> zero-shot MiniLM | 20 | 0.1106 | 0.4540 | 0.3951 | 0.7143 |

top-20 不适合作为 final evidence output，因为 precision 会下降；但它更适合作为
classifier 输入，因为能覆盖更多可能相关的证据。

## 决策

Round07 当前默认 retrieval pipeline：

```txt
candidate generation:
  RRF(BM25 top500, char TF-IDF top500)

reranker:
  zero-shot cross-encoder/ms-marco-MiniLM-L6-v2

final evidence output:
  top-3

classifier context:
  top-20 first, top-50 as a wider ablation
```

## 下一步

Stage D 从以下 ranked candidates 构造 classifier-ready JSONL：

```txt
outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json
```

比较的 classifier context：

- top-3
- top-10
- top-20
- top-50

