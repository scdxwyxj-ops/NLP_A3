# COMP90042 A3 Agent Memory

一句话概括：本项目是 COMP90042 Assignment 3 的团队 NLP fact-checking project；当前最重要事项是进入 Round06，把 Round05 的 reranker baseline 改造成更 task-aware 的 evidence preprocessing，再服务最终 classifier。

## 当前状态
- 项目名称：`COMP90042 A3 NLP Project`
- 当前阶段：`Round06 / Task-aware reranking and semantic evidence features`
- 当前目标：用 MiniLM-mined hard negatives 和细粒度语义特征，提高 evidence context 对最终 classifier 的价值。
- 当前第一优先级：生成 task-aware hard negatives，重新 fine-tune single-logit MiniLM reranker，并准备 classifier-ready top-k evidence context。
- 当前主要阻塞：沟通侧等待 Kaitlyn 给 `jevorianx@gmail.com` 的 Drive access；技术侧需要避免和 Kaitlyn 已做内容重复。

## 当前范围
需要完成：
- 做 task-aware reranker 改进、semantic feature prototype 和 classifier baseline。
- 将后续核心实验代码迁移或同步到课程 notebook template。
- 等待 Kaitlyn 对 `jevorianx@gmail.com` 授权或发送已创建/完成内容的链接/文件。

当前不做：
- 不通过姓名、邮箱或课程信息反向搜索未主动提供的私人社交账号。
- 不在未确认身份或未获授权的账号上发送额外联系请求。
- 不开始最终 classifier 训练，直到 reranker API 和 top-k evidence 输出稳定。

## 固定约束
- 课程规则：项目代码需遵守 README 中的 COMP90042 Assignment 3 规则，尤其是禁止闭源 API/模型用于最终系统。
- 登录流程：需要网站登录时优先使用 Edge；到登录、MFA、CAPTCHA 等位置停下来等待用户手动完成。
- 隐私边界：仅使用队友明确发送给用户的社交账号 handle。
- 文档约束：`agent_docs/codex.md` 和 `agent_docs/rounds/` 是长期记忆区。

## 关键目录
- `agent_docs/`：Codex 项目记忆。
- `agent_docs/rounds/round_01/`：Round01 添加队友 Instagram 的过程记录。
- `agent_docs/rounds/round_02/`：COMP90042 A3 长期规划、任务范围和 round/stage 验收指标。
- `agent_docs/rounds/round_03/`：课程文件资产审计、template 和 baseline 可运行化记录。
- `agent_docs/rounds/round_04/`：TF-IDF retrieval baseline 和后续 retrieval 诊断记录。
- `agent_docs/rounds/round_05/`：BM25 + MiniLM reranker、recall 和错误分析记录。
- `agent_docs/rounds/round_06/`：task-aware negatives、semantic evidence features 和 classifier-oriented evidence packaging 计划。
- `data/`：课程数据文件；最终 zip 不应包含这些数据文件。
- `configs/`：实验配置。
- `docs/`：面向队友和最终报告的稳定项目文档。
- `experiments/`：可运行实验脚本。
- `notebooks/`：课程指定 notebook template 工作副本。
- `outputs/round04/`：Round04 dev prediction 输出。
- `src/a3_factcheck/`：可复用 pipeline 代码。
- `src/a3_factcheck/rerank/`：supervised evidence reranker API、candidate pool 和 hard-negative dataset 构造。
- `tests/`：轻量测试目录。
- `README.md`：课程项目说明和规则来源。

## 当前 Round
### Round01：添加 NLP 队友 Instagram 并进入沟通渠道
- Index：`agent_docs/rounds/round_01/round_01_index.md`
- Stage 报告：`agent_docs/rounds/round_01/round_01_stage_a_instagram_requests_report.md`
- 当前状态：`已达标`
- 关闭条件：Kaitlyn 和 Om 均通过 Instagram 请求，且可以进入或继续推进 group chat 沟通。

### Round02：COMP90042 A3 长期规划
- Index：`agent_docs/rounds/round_02/round_02_index.md`
- 报告：
  - `agent_docs/rounds/round_02/round_02_report_a_task_scope_and_requirements.md`
  - `agent_docs/rounds/round_02/round_02_report_b_long_term_round_plan.md`
- 当前状态：`已达标`
- 关闭条件：已完成任务介绍、必须交付物、课程约束、round/stage 划分和验收指标。

### Round03：数据资产、template、baseline 可运行化
- Index：`agent_docs/rounds/round_03/round_03_index.md`
- 报告：
  - `agent_docs/rounds/round_03/round_03_report_a_data_assets_audit.md`
- 当前状态：`已达标`
- 关闭条件：课程数据、template、`eval.py` 已放置到项目中，baseline evaluation 可运行并已记录指标。

### Round04：Retrieval baseline 与诊断
- Index：`agent_docs/rounds/round_04/round_04_index.md`
- 报告：
  - `agent_docs/rounds/round_04/round_04_report_a_tfidf_retrieval_baseline.md`
  - `agent_docs/rounds/round_04/round_04_report_b_retrieval_strategy_plan.md`
  - `agent_docs/rounds/round_04/round_04_report_c_repository_structure.md`
  - `agent_docs/rounds/round_04/round_04_report_d_bm25_baseline.md`
- 当前状态：`进行中`
- 当前结果：TF-IDF top-3 dev retrieval F-score 约 `0.0533`；BM25 top-5 dev retrieval F-score 约 `0.0772`。
- 下一关闭条件：完成 retrieval 误差分析，并用 BM25 top-50 构造 supervised reranker candidate pool。

### Round05：Supervised reranker 对比实验
- Index：`agent_docs/rounds/round_05/round_05_index.md`
- 报告：
  - `agent_docs/rounds/round_05/round_05_report_a_discovery_plan.md`
  - `agent_docs/rounds/round_05/round_05_report_b_literature_and_model_choice.md`
- 当前状态：`已结轮`
- 当前目标：用 BM25 top-50 candidates 构造 hard negatives，训练 open-source pretrained transformer cross-encoder reranker。
- 当前建议模型：`cross-encoder/ms-marco-MiniLM-L6-v2`；保守备选 `distilbert-base-uncased`。
- 当前实现：已完成 BM25 top-50 candidate pool、hard-negative pairs、`EvidenceReranker` API 和训练/推理入口。
- 当前数据结果：train pairs `10262`，positive `4122`，hard negative `6140`，dev recall@50 `0.3249`。
- 当前最佳结果：zero-shot `cross-encoder/ms-marco-MiniLM-L6-v2` reranker top-3 dev F-score `0.1642`；single-logit BCE fine-tuning top-5 dev F-score `0.1044`。

### Round06：Task-aware reranking and semantic evidence features
- Index：`agent_docs/rounds/round_06/round_06_index.md`
- 报告：
  - `agent_docs/rounds/round_06/round_06_report_a_improvement_plan.md`
- 当前状态：`规划中`
- 当前目标：用模型排序挖更有效 hard negatives，并探索更细粒度语义提取作为 reranker/classifier 辅助输入。
- 当前判断：短期优先 MiniLM-mined task-aware hard negatives；中期加入 entities/quantities/relations/negation 等语义摘要。

## 当前全局判断
- 当前最关键的问题：zero-shot reranker 学到的是 topic relevance，最终任务需要 fact-checking evidence relevance；Round06 要让 negatives 和 classifier input 更贴近 label decision。
- 当前最可信的事实来源：README、Gmail 线程 `Re: NLP Group assignment`、Instagram 页面状态、Round01/Round02 报告。
- 下一步建议：先固定 zero-shot MiniLM reranker 作为强 baseline；用它挖 high-ranked non-gold negatives，再训练 task-aware reranker，并同步准备 classifier baseline。
