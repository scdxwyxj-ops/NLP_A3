# Round06 索引：Task-aware reranking and semantic evidence features

报告时间：`2026-04-30`

## 一、本轮总目标

Round06 的目标是解决 Round05 暴露出的核心问题：

```txt
当前 reranker 更擅长 topic relevance，
但最终任务需要 fact-checking evidence relevance。
```

本轮重点不是再单纯提高 passage ranking 分数，而是让 retrieval/reranking 更服务于最终 claim classification。

## 二、本轮当前状态

- 状态：`规划中`
- 当前 stage：`Stage A-E 已完成初版；等待 final synthesis 或进入 classifier baseline`
- 依赖：
  - Round05 已完成 zero-shot MiniLM reranker baseline。
  - 当前最佳 evidence retrieval：MiniLM zero-shot top-3 dev F-score `0.1642`。
  - 当前候选召回瓶颈：BM25 top-50 dev recall `0.3249`，BM25 top-100 dev recall `0.4188`。
- 当前判断：下一步最有机会的提升来自 task-aware hard negatives 和更细粒度语义辅助特征。
- 最终产物：`round_06_report_final_comparison.md` + `outputs/round06/comparison_table.csv`。

## 三、Stage 划分

### Stage A：Baseline consolidation and table scaffold

状态：`已完成初版`

目标：
- 建立 Round06 final comparison table schema。
- 把 Round05 baseline rows 整理进去。
- 明确每个指标来源和 pending 项。

验收指标：
- 表格 schema 固定。
- BM25、zero-shot MiniLM、Round05 BCE MiniLM baseline 已入表。
- 后续实验只追加行，不临时改指标口径。

### Stage B：Task-aware hard negatives

状态：`已完成初版`

目标：
- 用 zero-shot MiniLM 在 train candidate pool 上排序。
- 选出排名靠前但不是 gold evidence 的 passages 作为 hard negatives。
- 这些 negatives 更接近真实错误：topic-relevant but not evidence-relevant。

验收指标：
- 生成新的 reranker training JSONL。
- 记录 hard negative 来源、rank、score。
- 重新训练 single-logit BCE reranker。
- 和 zero-shot MiniLM、Round05 BCE fine-tuning 对比。

### Stage C：Task-aware reranker training comparison

状态：`已完成初版`

目标：
- 用 task-aware negatives 重新训练 single-logit MiniLM reranker。
- 和 zero-shot MiniLM、Round05 BCE MiniLM 对比。

验收指标：
- 至少一个 conservative variant 完成。
- 记录 top-3/top-5 F-score、top-10/top-20 recall、REFUTES recall。
- 结果无论好坏都写入 comparison table。

### Stage D：Fine-grained semantic extraction

状态：`已完成初版`

目标：
- 从 claim 和 evidence 中提取更细粒度语义信息。
- 不做脆弱的手写规则预测，而是作为 reranker/classifier 的辅助输入或分析特征。

候选特征：
- entities：Australia, humans, CO2, emissions。
- quantities：3%, 1.3%, 29 billion tonnes。
- relations：produce, cause, reduce, affect。
- polarity/negation：no effect, not, less than。
- comparison/causality cues：more than, because, contribute, lead to。

验收指标：
- 有可运行的 extraction script。
- 输出 claim/evidence 的 semantic feature JSONL。
- 至少用于 error analysis 或 classifier input formatting。

### Stage E：Classifier-oriented evidence packaging

状态：`已完成初版`

目标：
- 区分 final evidence output 和 classifier input。

建议：

```txt
final evidence output: MiniLM top-3
classifier input: MiniLM top-10/top-20/top-50 + semantic feature summary
```

验收指标：
- 生成 classifier-ready dataset。
- 对 top-k context 做 ablation。
- 记录 label accuracy 和 harmonic mean。

### Stage F：Final synthesis

状态：`待开始`

目标：
- 生成最终 Round06 对比报告。
- 明确 Round07 classifier 的推荐输入。

验收指标：
- `round_06_report_final_comparison.md` 完成。
- `outputs/round06/comparison_table.csv` 完成。
- 有下一步 classifier task handoff。

## 四、本轮成功标准

最低成功：
- 新 hard-negative reranker 训练完成。
- 分类输入数据格式稳定。
- 明确 top-k context 对 classifier 的影响。

理想成功：
- task-aware fine-tuning 超过 zero-shot MiniLM top-3 F-score `0.1642`，或至少提升 classifier accuracy。
- REFUTES 的 retrieval hit-any / recall 有改善。
- 语义特征能帮助解释或改善分类错误。

## 五、当前下一步

1. 生成 MiniLM-mined hard negatives。
2. 重训 single-logit BCE reranker。
3. 做 claim/evidence semantic feature extraction prototype。
4. 进入 classifier baseline。

## 六、Discovery planning outputs

- `.agent/mission.md`
- `.agent/requirements.md`
- `.agent/plan.md`
- `.agent/backlog.md`
- `.agent/handoff_to_agent_loop.md`
- Report B：`agent_docs/rounds/round_06/round_06_report_b_stage_plan_and_table_schema.md`
- Report C：`agent_docs/rounds/round_06/round_06_report_c_execution_results.md`

## 七、当前结果快照

- `outputs/round06/comparison_table.csv` 已生成，当前 `10` 行。
- task-aware negatives：`16402` pairs，其中 positive `4122`，negative `12280`。
- task-aware MiniLM top-3 F-score：`0.1599`，接近但未超过 zero-shot MiniLM top-3 `0.1642`。
- classifier-ready packages 已生成：
  - `outputs/round06/train-classifier-context-minilm-top20-semantic.jsonl`
  - `outputs/round06/dev-classifier-context-minilm-top20-semantic.jsonl`
