# Round08 索引

## 状态

Round08 本轮已完成验收报表。当前阶段是根据验收结果决定是否扩展 MiniLM scoring
scope 到 top100/top200。

Round07 已达成目标并可关闭：

```txt
RRF(BM25 top500, char TF-IDF top500)
-> zero-shot MiniLM
-> top3 evidence F-score = 0.1987
```

## 当前核心判断

Round08 的主要瓶颈不再是第一阶段 candidate hit-any，而是：

```txt
top500 candidate pool 中已有 gold evidence，
但 reranker / evidence selection 没有稳定把它排进 top3/top20。
```

## 当前 Discussion

- `discussions/discussion_2026-05-01_requirements_response.md`

## 当前报告

- `round_08_report_a_execution_and_acceptance.md`

## 建议优先级

| Priority | 方向 | 当前判断 |
|---:|---|---|
| 1 | Error layer analysis | 必做，用来定位损失层 |
| 2 | Feature table construction | 必做，是 fusion reranker 前置步骤 |
| 3 | Feature fusion reranker | 主线，预计收益最高 |
| 4 | Dense supplement ablation | 可做，但需要安装依赖和缓存 embeddings |
| 5 | Evidence-wise verifier planning | 分类主线，但需谨慎定义 pair labels |
| 6 | Field-aware / decomposition | 只能做降级 ablation |

## 当前最佳结果

```txt
Fusion GBDT top3 evidence F-score:
  0.2011

Round07 MiniLM-only top3 evidence F-score:
  0.1987
```

Classifier context 最佳结果：

```txt
Fusion GBDT top50 + TF-IDF logistic regression
accuracy:
  0.4675
macro-F1:
  0.4376
harmonic mean:
  0.2812
```

## 数据审计结论

`data/evidence.json` 是扁平结构：

```txt
evidence_id -> evidence text
```

没有 title、document id、section、page、sentence index、previous sentence、
next sentence 等字段。连续 `evidence-N` 也不是同一文档相邻句。

因此：

```txt
title/body/section field-aware BM25:
  当前数据不支持

previous/current/next evidence expansion:
  当前数据不支持

document-level diversity based on source id:
  当前数据不支持
```

当前可直接做的是从已有 candidate/rerank 产物构造 feature table：

```txt
claim_id
evidence_id
label_is_gold
minilm_rank/score
rrf_rank/score
bm25_rank/score
char_tfidf_rank/score
semantic overlap features
```

## 暂不作为第一优先级

- 继续大幅打磨 sparse retrieval；
- 复杂 FAISS ANN；
- 直接上 transformer concat classifier；
- 复杂 evidence set selection objective。
- evidence expansion with previous/next sentence。
