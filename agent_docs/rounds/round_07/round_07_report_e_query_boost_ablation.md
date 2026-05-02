# Round07 报告 E：Query-Boost Candidate Ablation

日期：2026-05-01

## 问题

显式增强 claim 中的重要片段，是否能提升 candidate recall？

重点关注：

- 数字
- 百分比
- 化学形式
- 命名实体

这个实验的动机来自之前的 hard misses：很多 false positives 在主题上相关，
但没有匹配 claim 中具体的数量关系、实体关系或语义关系。

## 方法

新增一个轻量 candidate-generation ablation：

```txt
claim text
+ repeated salient fragments
-> BM25 count-query scoring
-> top-500 candidates
```

salient fragments 包括：

- 数字表达，例如 percentage、ppm、tonnes、years；
- 气候/科学缩写，例如 `CO2`、`GHG`、`IPCC`、`UNEP`；
- capitalized phrases；
- 较长的非停用词 content terms。

然后测试该 pool 本身，以及它作为 RRF supplement 合并到当前最佳 sparse pool：

```txt
RRF(BM25 top500, char TF-IDF top500)
```

## 文件

实现：

```txt
experiments/retrieval/evaluate_query_boost_recall.py
```

输出：

```txt
outputs/round07_query_boost/query_boost_recall_summary.csv
outputs/round07_query_boost/candidates/
outputs/round07_query_boost/dev-query-boost-minilm-top3.json
outputs/round07_query_boost_analysis/
```

## Candidate Recall 结果

query-boost pool 本身弱于当前最佳 sparse pool。

| Candidate Source | k | Macro Recall | Hit-any | All-gold | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| RRF(BM25,char TF-IDF) | 500 | 0.6579 | 0.8896 | 0.4091 | 0.5926 |
| salient-boost BM25 | 500 | 0.5495 | 0.8117 | 0.2727 | 0.4012 |
| RRF(current best, salient-boost BM25) | 500 | 0.6419 | 0.8701 | 0.3701 | 0.6111 |

这不是一个干净的整体提升。唯一有价值的信号是 RRF supplement 把 `REFUTES`
candidate recall 从 `0.5926` 提高到 `0.6111`，但同时降低了 overall macro recall
和 all-gold rate。

## Rerank 结果

我用同一个 zero-shot MiniLM cross-encoder rerank 了这个偏 `REFUTES` 的 supplement pool。

| Candidate Pool | Reranker | Output k | Evidence F-score | Hit-any |
|---|---|---:|---:|---:|
| current best RRF(BM25,char) | MiniLM | 3 | 0.1987 | 0.4545 |
| query-boost supplement pool | MiniLM | 3 | 0.1927 | 0.4481 |
| query-boost supplement pool | MiniLM | 20 | 0.1079 | 0.7143 |

query-boost supplement 没有超过当前默认 final-evidence pipeline。

## 决策

不把 query-boost BM25 提升为 Round07 默认 pipeline。

当前默认仍然是：

```txt
RRF(BM25 top500, char TF-IDF top500)
-> zero-shot cross-encoder/ms-marco-MiniLM-L6-v2
```

query-boost 作为负结果记录是有价值的，也可能在未来作为 `REFUTES` class-specific
supplement，但现在不应该阻塞主线。

## 下一步

更有意义的提升应该转向 supervised label prediction：

1. 保留当前 retrieval pipeline 作为 evidence output；
2. 使用 top-20/top-50 reranked evidence 作为 classifier context；
3. 测试比 bag-of-words logistic regression 更强的 classifier。

