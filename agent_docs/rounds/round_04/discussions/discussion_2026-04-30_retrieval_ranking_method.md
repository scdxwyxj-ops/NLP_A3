# Discussion：当前排序问题用什么做，原理是什么

讨论时间：`2026-04-30`

## 问题

我们现在做了排序问题。当前到底是用什么来做排序？它的原理是什么？

## 简短回答

当前已经实现的是 **TF-IDF lexical retrieval baseline**。

也就是说，我们暂时不是用神经网络或监督 ranker 排序，而是用 claim 和每条 evidence 的词面相似度排序：

```txt
claim text -> TF-IDF vector
evidence text -> TF-IDF vector
score = claim vector · evidence vector
按 score 从高到低排序 evidence ids
取 top-k 作为预测 evidence set
```

当前代码位置：

```txt
experiments/retrieval/tfidf_baseline.py
src/a3_factcheck/retrieval/tfidf.py
```

旧兼容入口：

```txt
src/tfidf_retrieval_baseline.py
```

## TF-IDF 排序的原理

TF-IDF 把文本转换成稀疏向量。每个维度对应一个 unigram 或 bigram token。

当前配置大致是：
- lowercase。
- unicode accent stripping。
- English stopword removal。
- unigram + bigram。
- max features = `200000`。

一个 token 的权重由两部分组成：
- TF：这个 token 在当前文本里出现得多不多。
- IDF：这个 token 在整个 evidence corpus 里稀不稀有。

直觉：
- 常见词如 `the`, `is`, `and` 不应该影响太大。
- 更有区分度的词，如 climate/science entity、数字、术语，应有更高权重。

排序分数使用 claim vector 和 evidence vector 的点积：

```txt
similarity(claim, evidence) = TFIDF(claim) · TFIDF(evidence)
```

分数越高，说明 claim 和 evidence 在加权词面空间里越接近，因此 evidence 排名越靠前。

## 当前为什么叫 baseline

这是 baseline，不是最终方案。原因：
- 它不使用 train set 的 gold evidence 训练排序模型。
- 它只看词面相似，不真正理解语义、否定、数字关系或多证据组合。
- 当前 dev 结果较弱，TF-IDF top-3 retrieval F-score 约 `0.0533`。

因此它的价值主要是：
- 跑通 evidence retrieval pipeline。
- 确认 prediction JSON 和 `eval.py` 连接正确。
- 给 BM25、reranker、classifier 提供比较起点。

## 这个排序有没有监督

当前 TF-IDF baseline 本身是 **无监督** 的。它没有用 gold evidence 去学习参数。

但这个任务本身是可以做 **有监督排序** 的，因为 train/dev labelled claims 里有：

```txt
claim_text
claim_label
evidences: [gold evidence ids]
```

其中 `evidences` 就是 retrieval 的监督信号。

后续有监督 reranker 可以这样构造训练样本：

```txt
positive: claim + gold evidence
negative: claim + non-gold evidence
```

推荐负例来自 BM25/TF-IDF top-k 里排名靠前但不是 gold 的 hard negatives。

## 下一步应该怎么排序

推荐路线：

1. 先做 **BM25 baseline**。
   - 仍然是 lexical retrieval。
   - 通常比普通 TF-IDF 更适合大规模文本检索。
   - 比较 top-k：`1, 3, 5, 10, 20, 50`。

2. 再做 **supervised reranker**。
   - 先用 BM25 取 top-50 candidates。
   - 用 train gold evidence 训练 claim-evidence relevance model。
   - 输入：`[claim] [SEP] [evidence]`。
   - 输出：相关性分数。
   - 对 BM25 candidates 重新排序。

3. 如果做 cluster-based variant，则使用 **multicell-level masks**。
   - 先聚类 top-N candidates。
   - 在 cluster 内选择多个 evidence cells。
   - mask 控制哪些 cells 进入 aggregation/reranker/final output。
   - 不再使用 `context-level masks` 这个表述。

## 当前结论

当前排序方法是：

```txt
TF-IDF lexical similarity ranking
```

当前排序原理是：

```txt
把 claim 和 evidence 都转成 TF-IDF 向量，用向量点积作为相似度分数，再按分数排序 evidence。
```

它是一个无监督 baseline。真正更强的方案应该是：

```txt
BM25 high-recall candidate retrieval -> supervised transformer/cross-encoder reranking -> top-k evidence selection
```

