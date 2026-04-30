# Round02 Report A：任务介绍与必须完成内容

报告时间：`2026-04-28`

## 一、任务一句话介绍

COMP90042 Assignment 3 要求三人小组实现一个自动事实核查系统：给定一条 claim，从 `evidence.json` 中检索相关证据，并基于检索证据将 claim 分类为 `SUPPORTS`、`REFUTES`、`NOT_ENOUGH_INFO`、`DISPUTED` 四类之一。

项目评分重点不是单纯追求 leaderboard 高分，而是研究过程、方法合理性、设计解释、实验分析和报告质量。

## 二、我们具体要解决的问题

这个作业不是普通文本分类，也不是普通搜索。它是一个两阶段 fact-checking 问题：系统必须先在 evidence corpus 里找到能判断 claim 真假的证据，再根据这些证据做四分类判断。

对每一条 claim，系统面对的是下面这个具体问题：

```txt
输入：
  claim_id: claim-2967
  claim_text: "[South Australia] has the most expensive electricity in the world."
  evidence corpus:
    evidence-0 -> "..."
    evidence-1 -> "..."
    ...

系统必须输出：
  claim_id: claim-2967
  claim_label: SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED
  evidences: [evidence-id-1, evidence-id-2, ...]
```

也就是说，我们要做的是一个端到端 pipeline：

```txt
claim text
  -> evidence retrieval: 从 evidence.json 里召回候选证据
  -> evidence selection / reranking: 排出最可能有用的证据
  -> claim verification: 根据 claim + evidence 判断 label
  -> prediction JSON: 输出 evidence ids + label
  -> eval.py: 计算 retrieval F-score、classification accuracy、harmonic mean
```

### 1. Retrieval 子问题：找证据

Retrieval 的目标是从 `evidence.json` 中找出 gold evidence ids。难点不只是“找相似句子”，而是找到足以支持判断的证据：

- claim 和 evidence 可能词面不完全匹配，需要处理同义表达、实体别名、上下文差异。
- 一个 claim 可能需要多条 evidence 才能判断。
- evidence corpus 里会有大量看似相关但不能证明 claim 的干扰项。
- retrieval 结果会直接影响 classifier；如果证据找错，后面的分类模型即使很强也会被错误输入误导。

因此 retrieval 不是辅助模块，而是主任务的一半。它需要独立评估、调参和误差分析。

### 2. Classification 子问题：判标签

Classification 的目标是在已检索 evidence 的基础上预测四类 label：

- `SUPPORTS`：证据支持 claim。
- `REFUTES`：证据反驳 claim。
- `NOT_ENOUGH_INFO`：证据不足以判断 claim。
- `DISPUTED`：证据之间存在争议或互相冲突，不能简单归为支持/反驳。

难点是 label 不是只由 claim 本身决定，而是由 claim 与 evidence 的关系决定。一个合理的 classifier 需要建模：

- claim 和 evidence 是否讲同一个实体或事件。
- evidence 是否能推出 claim，而不只是主题相关。
- 多条 evidence 之间是互补、无关，还是冲突。
- retrieved evidence 不完美时，模型如何避免被噪声带偏。

课程还硬性要求系统包含至少一个 sequence modelling component，所以最终 classifier 或 reranker 里必须有 RNN、LSTM、GRU 或 Transformer 之一。

### 3. Joint 子问题：检索和分类要一起优化

最终主指标是 retrieval F-score 和 classification accuracy 的 harmonic mean。这个指标会惩罚偏科：

- 如果 retrieval 很差，classification 再高也不能说明系统真的会 fact-check。
- 如果 classification 很差，retrieval 找到证据也不能形成正确判断。
- 如果 retrieval 返回太多证据，recall 可能升高，但 precision 下降，还可能干扰 classifier。
- 如果 retrieval 返回太少证据，precision 可能高，但漏掉关键 evidence，classification 也会受影响。

因此长期规划必须把系统当作 retrieval + classification 的联合系统，而不是两个互不相关的小作业。

## 三、数据、输入、输出与评估对象

### 输入
- `train-claims.json`：带 label 和 evidence ids 的训练集。
- `dev-claims.json`：带 label 和 evidence ids 的开发集，用于调参、评估和误差分析。
- `test-claims-unlabelled.json`：无标签测试集，用于 optional leaderboard 或最终测试输出。
- `evidence.json`：证据库，key 是 evidence id，value 是 evidence passage。
- `dev-claims-baseline.json`：开发集示例预测文件。
- `eval.py`：课程提供的评估脚本。

### 每条 labelled claim 的含义

`train-claims.json` 和 `dev-claims.json` 中的每个样本包含：

- `claim_text`：需要核查的自然语言声明。
- `claim_label`：gold label，是四分类监督信号。
- `evidences`：gold evidence ids，是 retrieval 监督信号。

这意味着训练时可以同时学习两个目标：

- 用 `claim_text` 和 `evidences` 训练/调试 evidence retrieval。
- 用 `claim_text`、gold 或 retrieved evidence、`claim_label` 训练/调试 classifier。

`test-claims-unlabelled.json` 没有 label 和 gold evidence，只能作为最终输出预测的输入。不能人工检查或手动修正 test predictions。

### 系统输出
每个 claim 至少输出：
- 一个预测 label：`SUPPORTS`、`REFUTES`、`NOT_ENOUGH_INFO`、`DISPUTED`。
- 至少一个 retrieved evidence id。

输出格式必须能被 `eval.py` 接受。也就是说，最终 notebook 不只是训练模型，还必须生成符合课程 schema 的 prediction JSON。

### 评估对象

`eval.py` 会分别评估：

- Evidence Retrieval F-score：预测 evidence ids 与 gold evidence ids 的匹配程度。
- Claim Classification Accuracy：预测 label 是否等于 gold label。
- Harmonic Mean：综合 retrieval F-score 和 classification accuracy，是 leaderboard 排名主指标。

Report 中必须同时解释这三个指标，因为它们分别对应系统的不同失败模式。

## 四、这个项目真正要求我们产出的能力

完成作业不等于“跑一个模型”。最终系统应该展示以下能力：

1. 数据理解能力：知道 claim、evidence、label、prediction schema 各自承担什么角色。
2. 检索能力：能从 evidence corpus 中找出对 claim 有判断价值的 evidence。
3. 关系建模能力：能判断 claim 和 evidence 之间是支持、反驳、信息不足还是争议。
4. 工程复现能力：代码可以在 Colab 从头跑通，并生成和 report 一致的结果。
5. 研究分析能力：能比较 baseline、改进方法、ablation，并解释错误来源。
6. 写作表达能力：能用 7 页 ACL report 讲清楚系统设计、实验、结果和限制。

## 五、必须完成的东西

### 1. 可运行系统
- 一个完整 pipeline：读取数据、构建 evidence index、检索 evidence、预测 label、导出 prediction JSON。
- 系统必须包含至少一个 sequence modelling component：RNN、LSTM、GRU 或 Transformer。
- 代码必须能在免费版 Google Colab 运行，符合内存和算力限制。
- 所有最终实验代码必须在课程指定 `.ipynb` template 中，核心结构不能破坏。

### 2. Retrieval 组件
- 至少实现一个可复现的 evidence retrieval 方法。
- 必须确保每条 claim 至少返回一个 evidence id。
- 建议保留多个 candidate retrieval variants，便于报告比较，例如 lexical baseline、dense or transformer-based reranker、hybrid retrieval。

### 3. Classification 组件
- 至少实现一个 claim/evidence label classifier。
- classifier 不能是 hand-crafted if-then classification rules。
- 如果使用 pretrained open-source transformer，需要说明 fine-tuning、input construction、evidence aggregation 和设计贡献。

### 4. Evaluation 与实验记录
- 使用 `eval.py` 在 dev set 上报告：
  - Evidence Retrieval F-score
  - Claim Classification Accuracy
  - Harmonic Mean of F and A
- 记录运行日志、配置、随机种子、训练时间、主要超参数。
- 做 error analysis：区分 retrieval error、classification error、evidence aggregation error、label confusion。

### 5. 最终报告
- 使用 ACL LaTeX template。
- 正文最多 7 页，不包括 team contribution 和 references。
- 必须包含 group number。
- final 版本要用 `\usepackage[final]{acl}`。
- 报告要忠实描述最终代码和结果，不得报告无法由 notebook 复现的结果。
- 需要引用相关论文或技术资料来支持设计选择。

### 6. 最终提交包
- `COMP90042_teamname.pdf`：ACL template 生成的 PDF report。
- `COMP90042_teamname_resource.zip`，至少包含：
  - 课程指定 template 中的 `.ipynb` file(s)。
  - 可选 README：说明如何运行代码。
  - 可选 shell scripts：安装依赖或辅助运行。
- 不得上传数据文件或 trained model checkpoints。

### 7. 小组协作与 peer review
- Assignment 3 项目本身是 35 marks。
- Peer review 是单独 8 marks，时间为 2026-05-24 至 2026-05-29。
- 需要保留每个成员贡献记录，报告中的 team contributions 不计入 7 页正文限制。

## 六、硬性禁止事项

- 不得使用 closed-source APIs 或 proprietary models：OpenAI GPT、Claude、Gemini、Copilot 等。
- 不得直接复制外部开源项目代码或完整 solution。
- 不得使用外部训练或评测数据。
- 不得手工检查 test dataset 或 test labels。
- 不得 post-hoc 手动修改 predictions。
- 不得提交无法由最终代码生成的结果。
- 不得依赖付费算力或免费 Colab 跑不动的模型。
- 不得上传数据文件或模型 checkpoint。

## 七、评分导向

最终 35 marks 主要来自：
- Writing clarity：3
- Tables/Figures：4
- Method soundness：8
- Work substance：5
- Novelty/ambition：6
- Results and analysis：6
- Citation：3

因此项目策略不应只追排行榜分数。更稳的目标是：
- 有清晰的 retrieval/classification pipeline。
- 有一个合理强的 sequence model。
- 有比较实验和 ablation。
- 有可信 error analysis。
- 有简洁但信息密度高的报告叙事。

## 八、当前缺口

截至本报告，仓库中只有 `README.md` 和 `agent_docs/`，还缺：
- 课程数据文件。
- 课程 notebook template。
- `eval.py`。
- baseline prediction file。
- 小组分工确认。
- 方案选型和实验记录。

这些缺口应在后续 round 中优先补齐。
