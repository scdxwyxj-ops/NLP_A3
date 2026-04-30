# Discussion：retrieval priority discovery

讨论时间：`2026-04-30`

## 背景

用户希望调用 discovery-planner 来重新聚焦任务：无监督方案只作为 baseline，真正重要的是把 evidence ranking/retrieval 做好，并最终回到有监督问题。

官方 discovery-planner launcher 运行失败，原因是当前目录不是 git repository：

```txt
Error: discovery_planner must run inside a git repository.
```

因此本讨论记录采用 discovery-planner 的思路手动整理 mission、requirements 和 backlog。

## 当前判断

我们现在不是已经有两个等价的最终模型，而是有：

1. **TF-IDF retrieval baseline**
   - 已用于 claim-evidence 排序。
   - 无监督、非深度学习。
   - 直接输出 top-k evidence predictions。

2. **PPMI+SVD distributional embeddings demo**
   - 展示无监督词向量可行。
   - 当前还不是 evidence retrieval model。
   - 可以作为 query expansion、evidence clustering、multicell-level selection 的辅助特征。

因此项目重心不应继续深挖无监督词向量本身，而应把它作为辅助 baseline/feature，主要推进 retrieval ranking quality。

## Mission

建立可靠的 evidence retrieval/ranking pipeline：

```txt
unsupervised lexical baseline -> stronger BM25 baseline -> supervised reranker -> top-k evidence selection
```

最终目标是提升 dev retrieval F-score，并为后续 classifier 提供更可信 evidence candidates。

## Requirements

- 必须能读取课程 `train/dev/evidence` JSON。
- 每条 claim 至少返回 1 个 evidence id。
- 输出必须能被 `eval.py` 评估。
- 无监督 baseline 只作为起点，不作为最终主贡献。
- 后续 supervised reranker 必须只使用 train/dev labelled claims 中的 gold evidence，不引入外部训练数据。
- 实验脚本放在 `experiments/`，可复用代码放在 `src/a3_factcheck/`。

## Prioritized backlog

### P0：BM25 unsupervised baseline

实现 BM25 retrieval：
- 输入 claim 和 evidence corpus。
- 输出 top-k evidence ids。
- 比较 top-k：`1, 3, 5, 10, 20, 50`。
- 目标：超过当前 TF-IDF top-3 F-score `0.0533`。

### P1：retrieval error analysis

抽样查看成功/失败：
- claim text。
- gold evidence text。
- predicted top-k evidence text。
- overlap。
- failure reason。

### P2：supervised reranker dataset

构造训练样本：
- positive：claim + gold evidence。
- hard negative：BM25 top-k 中非 gold evidence。

### P3：supervised reranker

训练 sequence model / cross-encoder reranker：
- 输入 `[claim] [SEP] [evidence]`。
- 输出 relevance score。
- 只 rerank BM25 top-50/top-100 candidates。

## Top 3 next tasks

1. 实现并验证 BM25 baseline。
2. 生成 dev top-k 指标表。
3. 写 retrieval error analysis 脚本，为 supervised reranker 设计 hard negatives。

