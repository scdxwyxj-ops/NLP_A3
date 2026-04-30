# Discussion：无监督、非深度学习词向量如何得到

讨论时间：`2026-04-30`

## 问题

如何演示“无监督、无深度学习”地得到有效的词向量表征？

## 简短回答

可以用经典 distributional semantics：

```txt
文本语料 -> word-context 共现矩阵 -> PPMI 加权 -> SVD 降维 -> dense word vectors
```

这条路线不需要人工标签，也不需要神经网络训练。它只依赖一个假设：

```txt
语义相近的词，会出现在相似上下文中。
```

也就是 distributional hypothesis。

## 方法原理

### 1. 建 word-context 共现矩阵

先把语料 tokenized。对每个词，统计它窗口附近出现过哪些上下文词。

例如窗口大小为 2：

```txt
global temperature has increased
```

`temperature` 的上下文可以包括：

```txt
global, has
```

最后得到一个矩阵：

```txt
rows: target words
cols: context words
values: co-occurrence counts
```

### 2. 用 PPMI 重新加权

原始共现次数会被高频词污染。因此使用 PPMI：

```txt
PPMI(w, c) = max(PMI(w, c), 0)
PMI(w, c) = log P(w, c) / (P(w) P(c))
```

直觉：
- 如果一个词和上下文一起出现的概率高于随机独立出现，PMI 为正。
- 如果没有显著关联，值置为 0。

这样可以突出有意义的上下文关系。

### 3. 用 SVD 降维

PPMI 矩阵很大且稀疏。SVD 把它分解并压缩成低维 dense vectors：

```txt
PPMI matrix -> SVD -> 50d / 100d word vectors
```

降维后，相似上下文的词会在向量空间里靠近。

## 为什么这算有效词向量

得到向量后可以做 cosine similarity：

```txt
nearest_neighbors("temperature")
```

如果训练语料足够相关，邻居应该倾向于出现：

```txt
warming, climate, surface, global, atmospheric, ...
```

这说明词向量捕捉到了统计语义相似性。

## 代码位置

Reusable code：

```txt
src/a3_factcheck/embeddings/ppmi_svd.py
```

Demo experiment：

```txt
experiments/embeddings/ppmi_svd_demo.py
```

运行示例：

```bash
PYTHONPATH=src python experiments/embeddings/ppmi_svd_demo.py \
  --max-evidence-docs 3000 \
  --max-vocab 3000 \
  --dim 50 \
  --queries climate,temperature,carbon,energy,sea
```

## 小规模演示结果

验证命令：

```bash
PYTHONPATH=src python experiments/embeddings/ppmi_svd_demo.py \
  --max-evidence-docs 1000 \
  --max-vocab 1500 \
  --dim 30 \
  --queries climate,temperature,carbon,energy,sea
```

输出摘要：

```txt
documents: 2228
vocab size: 1500
embedding dim: 30
svd explained variance ratio sum: 0.1231
```

部分 nearest neighbors：

```txt
climate -> change, paris, global, models, predicted
temperature -> surface, temperatures, ocean, data, measurements, satellite
carbon -> dioxide, co2, emissions, greenhouse, atmospheric, anthropogenic
energy -> renewable, wind, power, thermal, solar
sea -> level, arctic, ice, levels, rise, melting, rising
```

这个结果说明，即使不用标签、不用深度学习，PPMI+SVD 也能从共现统计中学到一定语义结构。

## 和当前 retrieval 的关系

当前 TF-IDF retrieval 也是 lexical/statistical 方法，但它主要做 claim-evidence 相似度。

PPMI + SVD 可以提供另一种无监督词向量：
- 可以用于 query expansion。
- 可以用于 evidence clustering。
- 可以用于 cluster-based multicell-level selection 的候选相似度特征。

但它仍不是深度语义模型，通常不如 transformer reranker 强。
