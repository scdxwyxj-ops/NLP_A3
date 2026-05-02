# Round07 报告 A：Discovery Stage Plan

## Requirement 理解

Round07 requirement 的核心判断是：当前系统不是 reranker 没调好，而是第一阶段
candidate recall 太低。已有证据是 BM25 top-100 candidate macro recall 只有
约 `0.4188`，而 MiniLM rerank 后的旧最佳 final evidence F-score 也只有
约 `0.1642`。

因此正确做法是把任务拆成两个不同目标：

```txt
final evidence output:
  小集合，高 precision，用于最终提交和 evidence F-score

classifier evidence context:
  宽上下文，高 recall，用于后续 claim classification
```

下一阶段实验应先测量和提高 candidate recall，再继续投入 reranker fine-tuning。

## Discovery Planner 状态

我按要求调用了 `discovery-planner` 做本地规划。工具启动后成功写入了
`.agent/discovery/runs/0002/` 的 brief，但本地 discovery tool 内部引用的
Codex binary 路径已经失效：

```txt
/home/xj/.vscode-server/extensions/openai.chatgpt-26.422.30944-linux-x64/bin/linux-x86_64/codex: No such file or directory
```

因此我保留这个失败记录，并在 `agent_docs` 中手动继续同一套 discovery-planning
流程。

## Stage A：Candidate Recall Curve

目的：

先在 rerank 之前测第一阶段 retrieval recall。

方法：

- BM25 top-50/top-100/top-200/top-500
- word TF-IDF top-50/top-100/top-200/top-500
- char TF-IDF top-50/top-100/top-200/top-500

指标：

- macro recall
- micro recall
- hit-any
- all-gold
- label-specific macro recall
- average candidate count
- runtime

验收：

`outputs/round07/candidate_recall_summary.csv` 存在，并包含上述方法和 k 值。

## Stage B：Sparse Candidate Union

目的：

检验多个弱 retriever 的多样性是否能提高候选召回。

方法：

- BM25 + word TF-IDF RRF union
- BM25 + char TF-IDF RRF union
- BM25 + word TF-IDF + char TF-IDF RRF union

指标：

同 Stage A。

验收：

至少一个 union candidate pool 在相同 retained k 下超过 BM25 的 candidate macro recall。

## Stage C：Rerank Best Pools

目的：

用稳定的 zero-shot MiniLM reranker 评估最佳 candidate pools。

方法：

- 从 Stage A/B 选择最好的 2-3 个 candidate pools
- 使用 `cross-encoder/ms-marco-MiniLM-L6-v2` rerank
- 比较 output top-1/top-3/top-5/top-10
- 额外比较 classifier context top-20/top-50

验收：

能够判断新 candidate pool 是否提升：

- final evidence F-score，目标超过旧最佳 `0.1642`
- 或 classifier context recall / hit-any

## Stage D：Classifier Context Package

目的：

为 label classification 构造更宽的 ranked evidence context。

方法：

- 生成 claim + evidence context top-20/top-50
- 可选加入 semantic summaries

验收：

为选定 context variants 生成 classifier-ready JSONL 文件。

## Stage E：Classifier Baseline

目的：

开始测量 retrieval 改进对最终 claim label prediction 的影响。

方法：

- concat baseline：claim + top-k evidence -> 4-way label
- 如果时间允许，继续做 evidence-wise verifier + aggregation

验收：

得到 label accuracy、macro-F1、confusion matrix 和 class-specific scores。

## Stage F：Query-Boost Ablation

目的：

测试数字、百分比、化学缩写和实体片段 query expansion 是否能补充 candidate recall，
尤其是 `REFUTES`。

验收：

得到 query-boost candidate recall 表和 rerank 结果，并决定是否纳入默认 pipeline。

## Immediate Backlog

1. 实现 candidate recall 脚本，支持多个 sparse retrieval 方法和 RRF union。
2. 跑 BM25/TF-IDF candidate recall top-50/top-100/top-200/top-500。
3. 保存最佳 sparse candidate pools。
4. 将 Stage A/B 结果写入 Round07 报告。
5. 选择 candidate pools 进入 MiniLM reranking。
6. 跑 classifier context baseline。

## Stop Conditions

如果出现以下情况，需要停止 reranker 工作并回到 candidate generation：

- BM25/TF-IDF unions 不提升 recall；
- top-500 recall 明显低于目标 `0.55`；
- `REFUTES` class-specific recall 长期明显低于其他标签。

