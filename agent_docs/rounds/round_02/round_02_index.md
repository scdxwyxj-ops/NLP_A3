# Round02 索引：COMP90042 A3 问题定义与长期规划

报告时间：`2026-04-28`

## 一、本轮总目标
- 把 Assignment 3 从“课程 README 要求”解释成清楚的技术问题：输入是什么、输出是什么、系统要学习什么、为什么 retrieval 和 classification 都关键。
- 明确本项目必须完成的交付物、禁止事项、评估口径、评分导向和当前缺口。
- 把后续工作拆成多个 round，每个 round 给出 stage、产出物和验收指标。

## 二、本轮对问题的定义
- 本项目是一个 automated fact-checking pipeline，而不是单独的文本分类任务。
- 对每条 claim，系统必须先从 `evidence.json` 检索 evidence ids，再基于 claim + evidence 预测四分类 label。
- Retrieval 子问题衡量系统能否找到 gold evidence；classification 子问题衡量系统能否根据证据判断 claim 状态。
- 最终主指标是 retrieval F-score 和 classification accuracy 的 harmonic mean，因此任何只优化单一模块的方案都不完整。
- 最终报告要解释系统设计为什么合理，结果为什么可信，以及错误主要来自 retrieval、classification 还是 evidence aggregation。

## 三、全局约束
- 最终系统必须包含至少一个 sequence modelling component：RNN、LSTM、GRU 或 Transformer。
- 最终系统不得使用闭源 API 或 proprietary models，包括 OpenAI GPT、Claude、Gemini、Copilot。
- 只允许使用课程提供的 train/dev/test/evidence 数据；不得额外引入外部训练或评测数据。
- 最终代码必须放进课程指定的 `.ipynb` template，并能在免费版 Google Colab 跑通。
- 报告必须使用 ACL LaTeX template，正文不超过 7 页，不允许 appendix。
- report、notebook logs、reported results 必须与最终代码一致。
- leaderboard 是 optional，不计入最终分数；如参加，不能人工检查 test labels 或手动改预测。

## 四、Stage 划分
### Stage A：具体问题定义、任务范围与必须完成内容
报告：

```txt
agent_docs/rounds/round_02/round_02_report_a_task_scope_and_requirements.md
```

状态：`已达标`

### Stage B：长期 round/stage 规划与验收指标
报告：

```txt
agent_docs/rounds/round_02/round_02_report_b_long_term_round_plan.md
```

状态：`已达标`

## 五、跨 Stage 依赖
- Stage B 依赖 Stage A 的具体问题定义和课程要求抽取。
- 后续实现 round 应以 Stage B 的验收指标为准，但可以在和队友沟通后调整优先级。

## 六、关闭条件
- 两份规划报告均完成。
- Report A 能回答：我们具体要解决什么问题、每条数据代表什么、系统输出什么、retrieval/classification/joint metric 各自意味着什么。
- Report B 能回答：后续分成哪些 round、每个 round 有哪些 stage、每个 stage 怎样验收。
- 报告覆盖：必须交付物、课程约束、推荐 round 划分、每个 stage 的验收指标。

## 七、下一步
- 等 Round01 的队友沟通渠道完成后，把本规划发给队友讨论。
- 获取数据和 notebook template，开启 Round03：数据资产与 baseline 可运行化。
