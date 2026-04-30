# Round02 Report B：长期 Round/Stage 规划与验收指标

报告时间：`2026-04-28`

## 一、总体节奏

当前日期为 2026-04-28，最终报告与代码提交截止为 2026-05-22 23:59 AEST。可用时间约 24 天。建议把工作拆成 8 个后续 round，其中 Round03 到 Round07 解决系统和实验，Round08 到 Round10 解决报告、提交和 peer review 准备。

Round01 已用于建立队友沟通渠道；Round02 是本规划 round。

## 二、Round 总览

| Round | 主题 | 建议时间 | 关闭状态目标 |
| --- | --- | --- | --- |
| Round03 | 数据资产、template、baseline 可运行化 | 04-28 至 04-30 | 能在 Colab/local 跑 `eval.py` 和 baseline |
| Round04 | Retrieval baseline 与诊断 | 04-30 至 05-03 | 有可复现 retrieval baseline 和 dev F-score |
| Round05 | Sequence classifier baseline | 05-03 至 05-07 | 有 Transformer/RNN 类 sequence model 的 dev accuracy |
| Round06 | Hybrid system 与 joint tuning | 05-07 至 05-12 | 有 end-to-end pipeline 和 harmonic mean 提升 |
| Round07 | Error analysis、ablation、最终模型锁定 | 05-12 至 05-16 | 选定 final system，结果可复现 |
| Round08 | Report 结构、图表、引用与初稿 | 05-16 至 05-19 | ACL report 初稿完整 |
| Round09 | Final notebook、submission package、reproducibility | 05-19 至 05-21 | zip/pdf 候选包可跑通 |
| Round10 | 最终检查、提交、peer review 准备 | 05-21 至 05-29 | 05-22 前提交，05-24 后处理 peer review |

## 三、Round03：数据资产、template、baseline 可运行化

### Stage A：收集课程文件
验收指标：
- 项目目录或约定路径中存在 `train-claims.json`、`dev-claims.json`、`test-claims-unlabelled.json`、`evidence.json`、`dev-claims-baseline.json`、`eval.py`。
- 记录数据文件来源和存放位置。
- 不把数据文件提交到最终 zip，除非课程明确要求。

### Stage B：建立 notebook template 工作副本
验收指标：
- 课程指定 `.ipynb` template 已复制或保存为小组工作版本。
- notebook 可以在 Colab 打开。
- notebook 中保留课程要求的核心结构。

### Stage C：跑通 baseline evaluation
验收指标：
- `python eval.py --predictions dev-claims-baseline.json --groundtruth dev-claims.json` 能跑通。
- 记录 baseline 的 F-score、accuracy、harmonic mean。
- 明确 dev prediction JSON 的 required schema。

## 四、Round04：Retrieval baseline 与诊断

### Stage A：数据读取与 evidence indexing
验收指标：
- 能稳定读取所有 JSON。
- 建立 claim/evidence 数据结构。
- evidence id 到 text 的映射无丢失。
- 有基本统计：claim 数量、evidence 数量、label 分布、每条 claim evidence 数量分布。

### Stage B：Lexical retrieval baseline
验收指标：
- 实现一个不依赖训练的 retrieval baseline，例如 TF-IDF 或 BM25。
- 每个 claim 至少返回 top-k evidence。
- 在 dev 上报告 retrieval F-score。
- 保存不同 k 值的 dev retrieval 表现。

### Stage C：Retrieval 误差分析
验收指标：
- 抽样分析 retrieval 成功和失败案例。
- 记录常见失败原因：词面不匹配、实体歧义、证据太长/太短、需要多证据组合。
- 产出至少一张表或一段报告可用分析。

## 五、Round05：Sequence classifier baseline

### Stage A：构造 classification 输入
验收指标：
- 明确 classifier 输入格式：claim + top-k evidence，或 claim/evidence pair 后聚合。
- 对 train/dev 构造可复现 dataset。
- 防止 dev/test 信息泄露。

### Stage B：实现 sequence modelling component
验收指标：
- 至少实现一个 RNN、LSTM、GRU 或 Transformer component。
- 若使用 HuggingFace Transformer，模型为 open-source 且免费 Colab 可运行。
- 能在 train 上训练，在 dev 上评估 accuracy。
- 记录主要超参数、随机种子、训练日志。

### Stage C：Classification 诊断
验收指标：
- 报告 dev classification accuracy。
- 输出 confusion matrix 或 label-wise accuracy。
- 分析 `NOT_ENOUGH_INFO` 与其他类、`DISPUTED` 与 `SUPPORTS/REFUTES` 的混淆。

## 六、Round06：Hybrid system 与 joint tuning

### Stage A：End-to-end prediction pipeline
验收指标：
- 输入 claim file 和 evidence corpus，输出课程格式 prediction JSON。
- 对每条 claim 同时给出 evidence ids 和 label。
- 能用 `eval.py` 一键评估 dev predictions。

### Stage B：Retriever/classifier 联合调参
验收指标：
- 比较至少 3 个关键设置，例如 retrieval top-k、evidence concatenation 长度、classifier model size、reranking strategy。
- 报告 F-score、accuracy、harmonic mean 三个指标。
- 选择主指标更均衡的配置，而不是只优化单项。

### Stage C：改进方案
验收指标：
- 至少尝试一个有技术贡献的改进，例如 hybrid retrieval、reranker、evidence aggregation、hard negative sampling、confidence calibration。
- 改进必须能在免费 Colab 运行。
- 报告改进相对 baseline 的变化和原因。

## 七、Round07：Error analysis、ablation、最终模型锁定

### Stage A：Ablation study
验收指标：
- 至少完成 2 到 4 个 ablation，例如无 reranker、不同 top-k、不同 evidence aggregation、不同 classifier backbone。
- 每个 ablation 有 dev metrics。
- 结果能支撑报告中的设计选择。

### Stage B：错误分析与风险归因
验收指标：
- 至少抽样 20 个 dev 错误，分类归因。
- 明确哪些错误来自 retrieval，哪些来自 classification。
- 产出报告可用的 error taxonomy。

### Stage C：Final system freeze
验收指标：
- 明确 final system 配置。
- notebook 中结果可以从干净运行复现。
- final dev metrics 固定并记录。
- 如果参加 leaderboard，生成 test predictions，但不人工检查 test 内容或手动编辑结果。

## 八、Round08：Report 结构、图表、引用与初稿

### Stage A：Report outline
验收指标：
- ACL LaTeX project 创建完成。
- 标题、abstract、introduction、approach、experiments、results、conclusion、team contributions、references 结构齐全。
- 正文预计不超过 7 页。

### Stage B：图表与结果叙事
验收指标：
- 至少准备核心结果表：baseline vs final、ablation、error analysis。
- 图表编号、caption、引用位置完整。
- 报告清楚解释为何方法有效或失败。

### Stage C：Citation 与 scholarship
验收指标：
- BibTeX 引用相关 retrieval、fact checking、sequence classification 或 transformer 文献。
- 引用用于支持设计选择，不只是堆参考文献。
- 所有非原创方法都清楚标注来源。

## 九、Round09：Final notebook、submission package、reproducibility

### Stage A：Notebook 清理与 Colab 复现
验收指标：
- notebook 从头运行能复现 report 中 dev results。
- 包含 running logs 和 reported results。
- 安装依赖步骤清楚，免费 Colab 可完成。
- 无密钥、无私有路径、无本机绝对路径依赖。

### Stage B：Submission package
验收指标：
- 生成 `COMP90042_teamname.pdf`。
- 生成 `COMP90042_teamname_resource.zip`。
- zip 包含 required notebook，必要时包含 README 或 shell script。
- zip 不包含数据文件或 trained model checkpoints。

### Stage C：Consistency audit
验收指标：
- report 中的方法、参数、结果与 notebook 一致。
- prediction 文件由 notebook 生成，而不是手工编辑。
- 贡献说明与实际小组分工一致。

## 十、Round10：最终检查、提交、peer review 准备

### Stage A：Pre-submit checklist
验收指标：
- 05-22 23:59 AEST 前完成提交。
- 文件名符合要求。
- PDF 能打开，zip 能解压，notebook 能运行。
- 没有 appendix，没有超过 7 页正文。

### Stage B：After-submit archive
验收指标：
- 保存最终提交版本的本地副本。
- 记录提交时间、文件名、版本 hash 或校验信息。
- 记录最终 dev/test metrics。

### Stage C：Peer review readiness
验收指标：
- 05-24 至 05-29 peer review 时间被记录。
- 准备阅读其他报告的评价维度：clarity、soundness、results、analysis、citation。
- 记录本组贡献，方便回应 Part 2。

## 十一、优先级建议

最高优先级：
- 先拿到数据、template、`eval.py` 并跑通 baseline。
- 尽快建立 retrieval baseline，因为 retrieval F-score 直接限制 harmonic mean。
- 尽早确定 sequence classifier，保证满足硬性模型要求。

中等优先级：
- 做 hybrid retrieval/classification 改进。
- 做 ablation 和 error analysis，为报告拿分。

低优先级：
- Leaderboard；它不计入最终 marks，可以作为 final system 的额外检查，但不应牺牲报告和复现。

## 十二、总体完成判定

整个 Assignment 3 可视为完成，当且仅当：
- final notebook 在免费 Colab 跑通并复现结果。
- final PDF report 用 ACL template 生成，正文不超过 7 页。
- final zip 不含禁止提交的数据或 checkpoint。
- dev evaluation metrics、实验表、报告叙事全部一致。
- 小组贡献记录明确。
- 在 2026-05-22 23:59 AEST 前提交。
