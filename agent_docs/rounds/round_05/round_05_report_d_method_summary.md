# Round05 Report D：Current method summary

报告时间：`2026-04-30`

## 一、当前是否还是用 BM25？

是。当前已经实现的 Round05 方案仍然使用 BM25，但 BM25 的角色已经变了。

Round04 中，BM25 是一个完整的 unsupervised retrieval baseline：

```txt
claim -> BM25 score all evidence -> top-k evidence -> eval.py
```

Round05 中，BM25 是 supervised reranker 前面的 candidate generator：

```txt
claim
-> BM25 top-50 candidate evidence
-> supervised cross-encoder reranker
-> reranked top-k evidence
-> later classifier
```

也就是说，BM25 负责快速从大 evidence corpus 里召回一小批候选；监督模型负责在这批候选里重新排序。

## 二、已经做过的方法

### 1. TF-IDF lexical retrieval

位置：

```txt
src/a3_factcheck/retrieval/tfidf.py
experiments/retrieval/tfidf_baseline.py
```

方法：
- 把 evidence 和 claim 转成 TF-IDF bag-of-words / n-gram vectors。
- 用向量相似度排序 evidence。
- 不使用训练标签，是无监督 lexical baseline。

作用：
- 最早的可运行 retrieval baseline。
- 验证数据读取、prediction JSON、`eval.py` 流程。

当前结果：
- dev best top-3 retrieval F-score 约 `0.0533`。

局限：
- 主要依赖词面重合。
- 对 paraphrase、语义关系、否定关系不敏感。

### 2. BM25 lexical retrieval

位置：

```txt
src/a3_factcheck/retrieval/bm25.py
experiments/retrieval/bm25_baseline.py
```

方法：
- 用 CountVectorizer 建立 unigram/bigram 词频矩阵。
- 对 evidence term frequency 做 BM25 saturation 和 document length normalization。
- claim 作为 query，与 evidence BM25 matrix 做 sparse similarity。

作用：
- 比 TF-IDF 更强的 lexical baseline。
- 当前作为 Round05 supervised reranker 的 candidate generator。

当前结果：
- dev top-5 retrieval F-score 约 `0.0772`。
- Round05 dev recall@50 约 `0.3249`。

局限：
- 仍然主要靠词面匹配。
- top-50 没召回的 gold evidence，后续 reranker 无法恢复。

### 3. Distributional semantics / PPMI-SVD demo

位置：

```txt
src/a3_factcheck/embeddings/ppmi_svd.py
experiments/embeddings/ppmi_svd_demo.py
```

方法：
- 从共现矩阵构造 PPMI 表征。
- 用 SVD 降维得到无监督词向量。
- 不使用深度学习，不使用标签。

作用：
- 解释和演示无监督词向量如何从 distributional semantics 得到。
- 目前不是主 retrieval pipeline。

当前判断：
- 可以作为报告中的 exploratory baseline / conceptual demonstration。
- 不建议继续投入太多，因为核心任务需要提升 evidence ranking 和后续 claim classification。

### 4. Supervised cross-encoder reranker

位置：

```txt
src/a3_factcheck/rerank/
experiments/rerank/build_hard_negative_dataset.py
experiments/rerank/train_cross_encoder.py
experiments/rerank/rerank_with_cross_encoder.py
configs/rerank/minilm_reranker.json
```

方法：
- 先用 BM25 为每个 claim 召回 top-50 candidates。
- 构造训练样本：
  - positive = claim + gold evidence
  - hard negative = claim + BM25 top-50 中非 gold evidence
- 用 open-source pretrained transformer cross-encoder 做二分类 relevance scoring。
- 推理时对 BM25 candidates 打分并重新排序。

当前推荐模型：

```txt
cross-encoder/ms-marco-MiniLM-L6-v2
```

保守备选：

```txt
distilbert-base-uncased
```

作用：
- 这是 Round05 的核心方法。
- 它把 retrieval 从纯无监督词面排序推进到 supervised claim-evidence relevance ranking。
- 也是后续 classifier 的 evidence preprocessing component。

当前实现状态：
- candidate pool builder 已完成。
- hard-negative dataset builder 已完成。
- training script 已完成。
- rerank inference API 已完成。
- 还未正式训练模型，因为当前环境缺 `transformers`。

## 三、方法之间的关系

当前系统不是四个互相独立的最终方案，而是逐步推进的 pipeline：

```txt
TF-IDF
  -> early lexical baseline

BM25
  -> stronger lexical baseline
  -> Round05 candidate generator

PPMI-SVD
  -> unsupervised representation demo
  -> not main path

Supervised cross-encoder reranker
  -> current main path
  -> improves BM25 candidate ordering
  -> feeds evidence into later classifier
```

## 四、为什么不直接跳过 BM25？

因为 evidence corpus 很大，cross-encoder 不能直接对每个 claim 和所有 evidence 做 pairwise scoring。

如果直接暴力计算：

```txt
num_claims * num_evidence passages
```

成本会很高，不适合免费 Colab，也不适合最终 notebook 复现。

所以合理架构是 two-stage retrieval：

```txt
fast lexical retrieval -> slower neural reranking
```

BM25 快、稳定、可解释；cross-encoder 慢但更懂 claim-evidence 语义关系。

## 五、当前最大风险

当前最大风险不是 reranker API，而是 BM25 top-50 candidate recall 偏低。

dev recall@50 约 `0.3249` 意味着很多 gold evidence 没进入候选池。reranker 只能在候选池内部重排，不能找回没召回的 evidence。

如果 MiniLM reranker 训练后提升有限，下一步应该优先改 candidate generator：
- BM25 + TF-IDF union。
- 增大 BM25 candidate top-k。
- query expansion。
- dense retriever 作为辅助候选源。

## 六、当前下一步

1. 安装 `transformers`。
2. 训练 MiniLM cross-encoder reranker。
3. 对 dev BM25 top-50 candidates rerank。
4. 比较 top-1 / top-3 / top-5 / top-10 retrieval F-score。
5. 如果候选 recall 成为瓶颈，进入 hybrid candidate generator。
