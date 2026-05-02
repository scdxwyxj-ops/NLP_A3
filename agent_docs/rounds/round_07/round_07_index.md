# Round07 索引

## 目标

Round07 是 retrieval rescue 阶段。核心目标不是继续在弱候选池上调
reranker，而是先把第一阶段 candidate recall 拉起来，让后续 reranking
和 claim classification 有足够的证据上下文。

## 核心假设

当前瓶颈是第一阶段候选召回不足：

- BM25 top-50 macro recall：`0.3249`
- BM25 top-100 macro recall：`0.4188`
- Round05/Round06 旧最佳 final evidence F-score：MiniLM top-3 约 `0.1642`

如果 gold evidence 没有进入候选池，后面的 reranker 和 classifier 都无法恢复。

## Stage 划分

| Stage | 名称 | 目的 | 主要输出 | 状态 |
|---|---|---|---|---|
| A | Candidate recall curve | 测 BM25/TF-IDF 在更大 k 下的候选召回 | `outputs/round07/candidate_recall_summary.csv` | 完成 |
| B | Sparse candidate union | 用 RRF 合并 BM25、word TF-IDF、char TF-IDF | candidate pools + recall table | 完成 |
| C | Rerank best pools | 用 zero-shot MiniLM rerank 最好的候选池 | evidence F-score 和 context recall 表 | 完成 |
| D | Classifier context package | 从最佳 rerank 结果生成 top-k 分类上下文 | classifier-ready JSONL | 完成 |
| E | Classifier baseline | 跑一个快速 concat classifier baseline | label accuracy / macro-F1 表 | 部分完成 |
| F | Query-boost ablation | 测数字、实体、百分比 query expansion 是否补召回 | query-boost recall/rerank 报告 | 完成 |
| G | Acceptance summary | 总结可验收成果和后续方向 | 中文验收报告 | 完成 |

## 验收标准

Round07 可验收条件：

1. 有 BM25 top-50/top-100/top-200/top-500 的 candidate recall 表。
2. 至少实现一个 sparse union candidate pool，例如 BM25 + TF-IDF variants。
3. 有 class-specific recall，尤其是 `REFUTES`。
4. 明确决定哪些 candidate pools 值得进入 rerank。
5. 文档说明 recall 是否足够支撑后续 reranking/classification。

这些条件目前已经满足。

## 当前最佳结果

```txt
candidate pool:
  RRF(BM25 top500, char TF-IDF top500)

reranker:
  zero-shot cross-encoder/ms-marco-MiniLM-L6-v2

final evidence:
  top-3

dev evidence F-score:
  0.1987
```

相比旧最佳 `0.1642`，这是明确提升。

## 当前分类器 baseline

当前最好的快速分类器：

```txt
retrieval:
  RRF(BM25 top500, char TF-IDF top500) -> MiniLM -> final top3

classifier:
  TF-IDF logistic regression over claim + top50 evidence context

dev label accuracy:
  0.4610

dev macro-F1:
  0.4242

assignment harmonic mean:
  0.2778
```

这个 baseline 只是用于定位问题。Retrieval 已经有实际提升，但分类器还需要更强的监督模型。

## Query-Boost Ablation

我测试了一个 salient-query BM25 变体，把数字、百分比、化学缩写和实体片段重复加入 query。

结果没有超过当前默认 retrieval：

```txt
current default top3 F-score:
  0.1987

query-boost supplement top3 F-score:
  0.1927
```

它在一个候选池中把 `REFUTES` candidate recall 从 `0.5926` 提到 `0.6111`，
但整体 macro recall 和 all-gold rate 下降。因此它只作为 ablation 记录，不进入默认方案。

## 当前结论

Round07 的 retrieval rescue 可以验收。下一步应该固定当前 retrieval default，
然后重点做 supervised claim classifier：

```txt
RRF(BM25 top500, char TF-IDF top500)
-> zero-shot MiniLM
-> top3 final evidence
-> top20/top50 classifier context
```

