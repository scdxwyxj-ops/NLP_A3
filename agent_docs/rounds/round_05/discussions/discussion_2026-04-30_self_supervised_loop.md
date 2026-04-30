# Discussion：是否可以训练一个自监督循环？

日期：`2026-04-30`

## 问题

我们或许可以训练一个自监督的循环吗？

## 简短回答

可以，但不应该作为当前第一优先级。

更准确地说，我们可以做一个 **pseudo-label / self-training loop**，但它不是严格意义上纯自监督，因为我们已经有 train gold evidence labels。它更像：

```txt
BM25 / cross-encoder teacher
-> 给未标注或弱标注 candidate pairs 打 pseudo labels
-> 训练 student reranker
-> student rerank candidates
-> 选高置信样本继续训练
```

这个思路有潜力，但风险也很高：如果 teacher 的错误被反复强化，student 会学到更自信的错误排序。

## 可以怎么做

一个合理的 self-training loop：

1. 用 BM25 为 train/dev/test-unlabelled claims 召回 top-N candidates。
2. 用当前最强 teacher reranker 打分，例如 zero-shot `cross-encoder/ms-marco-MiniLM-L6-v2`。
3. 选高置信 pseudo positives 和 pseudo negatives：
   - pseudo positive：teacher 分数最高且明显高于第二名。
   - pseudo negative：BM25 排名高但 teacher 分数很低。
4. 把 gold supervised pairs 和 pseudo pairs 混合训练 student reranker。
5. 在 dev 上评估，不直接相信训练 loss。
6. 只保留 dev retrieval F-score 真正提升的版本。

## 为什么现在不优先做

当前 Round05 刚跑出的结果说明：

- BM25 top-5 F-score：约 `0.0772`
- zero-shot MS MARCO MiniLM reranker top-3 F-score：约 `0.1642`
- 单 logit supervised fine-tuning top-5 F-score：约 `0.1044`
- 错误的 2-class head fine-tuning top-10 F-score：约 `0.0372`

这说明当前最大收益来自一个强 teacher reranker，而不是我们的 fine-tuning loop。

在还没有把 supervised fine-tuning 做稳之前，直接上 self-training loop 可能会把噪声放大。

## 更推荐的顺序

当前推荐顺序：

1. 先把 zero-shot cross-encoder reranker 作为强 baseline 固定下来。
2. 改 supervised fine-tuning：
   - 保留单 logit reranking head。
   - 用 BCE loss。
   - 调低 learning rate。
   - 尝试更少 epoch 或 early stopping。
   - 改 hard-negative sampling。
3. 提升 candidate generator recall：
   - BM25 + TF-IDF union。
   - 更大 top-k。
   - query expansion。
4. 如果 supervised reranker 稳定超过 zero-shot baseline，再尝试 self-training。

## 如果要做 self-training，应该怎么约束

为了避免跑偏，self-training 需要这些约束：

- pseudo labels 只用于 train 或 test-unlabelled 的 claim text，不用 dev gold 做训练。
- dev set 只用于评估和调参。
- 每轮只加入高置信 pseudo pairs。
- pseudo pairs 权重低于 gold pairs。
- 每一轮都必须和上一轮在 dev F-score 上比较。
- 如果 dev F-score 下降，立刻回滚。

## 当前结论

可以把 self-training loop 作为 Round05 后半段或 Round06 的扩展实验，但当前主线仍然应该是：

```txt
BM25 candidate generator
-> zero-shot / supervised cross-encoder reranker
-> stable top-k evidence API
-> downstream classifier
```

原因很直接：我们现在已经看到 zero-shot reranker 明显提升，而自训练循环还没有必要性和稳定性保证。
