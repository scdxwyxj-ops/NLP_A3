# Discussion：Topic relevance 不等于 evidence relevance

日期：`2026-04-30`

## 问题

用户指出 `claim-375` 的 hard miss：

```txt
Claim:
when 3 per cent of total annual global emissions of carbon dioxide are from humans
and Australia produces 1.3 per cent of this 3 per cent, then no amount of emissions
reduction here will have any effect on global climate.
```

模型预测的 FP evidence 看起来都和 carbon dioxide / Australia / emissions 有关，但 gold evidence 是另一组更能解释 claim 判断的证据。

## 结论

这个观察是对的。当前 reranker 的问题不是完全“不相关”，而是它排序的是 **topic relevance**，而 assignment 需要的是 **fact-checking evidence relevance**。

两者不同：

```txt
topic relevance:
这段 evidence 是否和 claim 主题相似？

fact-checking evidence relevance:
这段 evidence 是否足以支持、反驳、或帮助判断 claim label？
```

在 `claim-375` 里，FP evidence 确实和主题相关：
- Australia emissions
- carbon dioxide emissions
- global annual averages
- electricity generation

但它们没有覆盖 claim 的关键推理链：
- human emissions 是否只是 total annual CO2 的 3%？
- Australia 在其中的比例如何？
- emission reduction 是否对 global climate 没有效果？
- anthropogenic emissions 与 warming 的因果关系是什么？

因此它们是 topical FP，而不是完全无关 FP。

## 为什么预训练排序模型会这样

`cross-encoder/ms-marco-MiniLM-L6-v2` 是 passage reranker，主要训练目标是 query-passage relevance。

它很擅长：
- 找主题相似文本。
- 判断 query 和 passage 是否语义相关。
- 把明显不相关 passage 排低。

但它不一定擅长：
- 判断某段 evidence 是否能完成 fact verification。
- 区分“只是同主题”与“能支撑标签判断”。
- 做多跳证据组合。
- 区分 claim 中数值、比例、因果关系的关键点。

所以 zero-shot MiniLM 能明显超过 BM25，但仍然会把 topical evidence 排到 gold evidence 前面。

## 对当前 pipeline 的影响

这说明当前 retrieval 还有两个层次：

```txt
Stage 1: lexical / semantic candidate retrieval
  目标：不要漏掉可能相关证据。

Stage 2: fact-checking-aware evidence selection
  目标：选出真正支持分类判断的证据。
```

MiniLM zero-shot 更像 Stage 1.5：比 BM25 更懂语义，但还不是完全 task-aware evidence selector。

## 改进方向

### 1. 训练 task-aware reranker

用 train gold evidence 做监督：

```txt
positive = gold evidence
negative = topical but non-gold evidence
```

重点不是随机负例，而是 hard negative：
- BM25/MiniLM 排很高但不是 gold。
- 与 claim 主题高度相关但缺少关键判断信息。

这比普通 binary relevance 更接近 assignment 目标。

### 2. 增加 claim decomposition

把 claim 拆成关键约束：

```txt
human emissions = 3% total annual CO2?
Australia = 1.3% of that 3%?
emission reduction has no effect on global climate?
```

然后 retrieval 不只按整句相似度，而是覆盖这些 sub-claims。

### 3. Reranker 加入 label-aware signal

后续 classifier 可以反过来帮助 evidence selection：

```txt
claim + evidence -> label logits
```

如果某个 evidence 对 SUPPORTS/REFUTES/NEI/DISPUTED 判断更有区分度，就应比纯主题相关 evidence 更靠前。

### 4. 对 REFUTES 做专项改进

当前 REFUTES retrieval 最弱。可能原因是反驳证据经常不是复述 claim，而是提供相反数值、相反因果、或限制条件。

这类 evidence 词面可能不如 topical FP 相似，所以需要：
- contradiction-aware negative sampling。
- 数值/实体/因果 cue 的特征。
- classifier-aware reranking。

## 当前判断

用户指出的问题是当前系统的核心瓶颈之一：

```txt
我们的 reranker 现在更像“相关文本排序器”，
但最终需要的是“证据价值排序器”。
```

下一步如果目标是分类最好，不能只追求 passage relevance。应该让 reranker 或 classifier 学到：

```txt
这段 evidence 是否能改变或支撑 label decision？
```
