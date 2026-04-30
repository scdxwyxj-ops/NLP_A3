# Round06 Report A：Improvement plan after Round05

报告时间：`2026-04-30`

## 一、背景

Round05 已经确认：
- BM25 是可用 candidate generator，但 recall 偏低。
- zero-shot `cross-encoder/ms-marco-MiniLM-L6-v2` 明显提升 evidence ranking。
- 直接 fine-tuning 还没有超过 zero-shot reranker。
- 错误案例显示，模型经常选中 topic-relevant evidence，但这些 evidence 不一定能支撑 fact-checking label decision。

因此 Round06 不应只是继续跑更多普通 reranker，而要让排序更贴近最终分类任务。

## 二、两个最有机会的提升方向

### 方向一：用模型排序挖更有效的 negatives

当前 negatives 的问题：

```txt
BM25 top candidates 中非 gold evidence
```

这已经比随机负例好，但还不够精细。更好的方式是：

```txt
BM25 top-50/top-100
-> zero-shot MiniLM rerank
-> 取 MiniLM 排名前列但不是 gold 的 evidence
-> 作为 task-aware hard negatives
```

这些 negatives 的价值更高，因为它们正是模型真实会犯错的类型：

```txt
topic-relevant but not evidence-relevant
```

训练目标变成：

```txt
gold evidence score higher than high-scoring topical false positives
```

这比普通 binary relevance 更接近 assignment 的 evidence selection 目标。

建议实验：
- `negatives_per_claim = 5 / 10 / 20`
- candidate source：BM25 top-50 vs top-100
- model：single-logit MiniLM BCE
- learning rate：`5e-6`、`1e-5`
- epochs：`0.3`、`0.5`、`1.0`

判断指标：
- retrieval F-score top-3/top-5。
- recall top-10/top-20。
- REFUTES hit-any。
- downstream classifier accuracy。

### 方向二：更细粒度的语义提取作为辅助

用户提出的主语/宾语一致是一个有价值的方向，但不建议做成手写规则分类器。

原因：
- evidence relevance 不只由 subject/object 决定。
- 还涉及数值、比例、因果、否定、比较和范围限制。
- 手写规则容易脆弱，也不符合课程对 hand-crafted prediction logic 的限制精神。

更好的使用方式是把语义提取作为辅助信息：

```txt
claim_text
retrieved evidence text
semantic feature summary
-> classifier / reranker input
```

可提取的信息：
- entities：主语、对象、地名、组织、物理量名。
- quantities：百分比、温度、年份、排放量。
- relations：produce, contribute, cause, reduce, affect。
- negation/polarity：no effect, not, less, without。
- comparison：more than, less than, highest, lowest。
- causality：because, due to, result in, lead to。

这些信息可以用于：
- hard negative mining：选择同实体但数值/关系不匹配的 negatives。
- classifier input formatting：显式提示模型关注关键约束。
- error analysis：解释为什么 topic relevant passage 不是 gold evidence。

## 三、推荐优先级

优先级最高：

```txt
MiniLM-mined task-aware hard negatives
```

原因：
- 直接针对 Round05 的失败模式。
- 工程成本低。
- 很容易和现有 reranker pipeline 接上。
- 有明确 dev 指标可比较。

第二优先级：

```txt
semantic extraction prototype
```

原因：
- 有助于后续 classifier。
- 能改善报告分析质量。
- 但短期效果不如 hard negatives 稳定。

暂不建议：

```txt
大型手写 SVO rule system
```

它可能能解释个别例子，但很难覆盖整个 climate fact-checking dataset。

## 四、Round06 建议 pipeline

```txt
BM25 top-100 candidates
-> zero-shot MiniLM rerank
-> mine top-ranked non-gold hard negatives
-> fine-tune single-logit MiniLM reranker
-> output:
   - top-3 evidence for submission
   - top-20/top-50 evidence for classifier
-> classifier with optional semantic feature summary
```

## 五、核心假设

Round06 的核心假设是：

```txt
分类效果不是只靠更强 classifier，
而是靠 classifier 能看到更高 recall、更 task-relevant 的 evidence context。
```

因此排序模型和分类模型之间要明确分工：
- reranker：提高 evidence ordering 和 context quality。
- classifier：基于更可靠的 evidence context 判断 label。

## 六、第一批待办

1. 写 `mine_task_aware_negatives.py`。
2. 生成 MiniLM-mined hard-negative dataset。
3. 重训 MiniLM single-logit reranker。
4. 做 top-k context ablation。
5. 写 semantic feature extraction prototype。
