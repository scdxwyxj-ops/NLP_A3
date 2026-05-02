# COMP90042 A3 Agent Memory

一句话概括：本项目是 COMP90042 Assignment 3 的团队 NLP fact-checking project；Round07 retrieval rescue 已达成目标，当前进入 Round08：定位 top500/top20/top3 的损失层，并优化 reranking / evidence selection / classifier aggregation。

## 当前状态
- 项目名称：`COMP90042 A3 NLP Project`
- 当前阶段：`Round10 / Neural verification baselines`
- 当前目标：补齐 neural baseline，验证当前瓶颈是否来自 relevance-style reranking，而不是 fact verification / stance detection。
- 当前第一优先级：Round10 完整实验已完成；下一步应将 final decoupled pipeline 固化到最终 notebook / submission pipeline。
- 当前主要阻塞：Verifier pair-level 有信号，但直接用于 top3 evidence rerank 很弱；final evidence 仍应使用 Round09 blend。

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
- `agent_docs/rounds/round_07/`：candidate recall rescue、sparse RRF union、top500 MiniLM reranking 和 classifier baseline。
- `agent_docs/rounds/round_08/`：Round08 requirements 消化、模糊建议筛选和 reranker/classifier 下一步计划。
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

### Round07：Retrieval recall rescue and classifier context
- Index：`agent_docs/rounds/round_07/round_07_index.md`
- Requirement：`agent_docs/rounds/round_07/requirement.md`
- 报告：
  - `agent_docs/rounds/round_07/round_07_report_a_discovery_stage_plan.md`
  - `agent_docs/rounds/round_07/round_07_report_b_candidate_recall_results.md`
  - `agent_docs/rounds/round_07/round_07_report_c_rerank_results.md`
  - `agent_docs/rounds/round_07/round_07_report_d_classifier_context_and_baseline.md`
  - `agent_docs/rounds/round_07/round_07_report_e_query_boost_ablation.md`
  - `agent_docs/rounds/round_07/round_07_report_f_acceptance_summary.md`
- 当前状态：`已结轮；Stage A-D 完成，Stage E 部分完成，Stage F query-boost ablation 完成，Stage G 验收总结完成`
- 关键结果：
  - BM25 top500 dev candidate macro recall `0.5861`
  - char TF-IDF top500 dev candidate macro recall `0.6610`
  - RRF(BM25, char TF-IDF) top500 dev candidate macro recall `0.6579`
  - RRF(BM25, char TF-IDF) top500 + zero-shot MiniLM top3 dev evidence F-score `0.1987`
  - RRF(BM25, char TF-IDF) top500 + zero-shot MiniLM top20 dev macro recall `0.4540`
  - best quick concat classifier baseline: top50 plain TF-IDF logistic regression accuracy `0.4610`, macro-F1 `0.4242`
  - query-boost supplement top3 dev evidence F-score `0.1927`，低于当前默认；仅在一个候选池中把 `REFUTES` candidate recall 从 `0.5926` 提到 `0.6111`
- 当前判断：retrieval rescue 有效，已经超过旧 best F-score `0.1642`；query-boost 不应替换默认方案；下一步不要再做简单 concat classifier，应做 evidence-wise verifier/aggregation 或 transformer classifier。

### Round08：Reranking and classifier aggregation planning
- Index：`agent_docs/rounds/round_08/round_08_index.md`
- Requirement：`agent_docs/rounds/round_08/requirements.md`
- Updated requirement：`agent_docs/rounds/round_08/requirements_updated.md`
- 报告：
  - `agent_docs/rounds/round_08/round_08_report_a_execution_and_acceptance.md`
- Discussions：
  - `agent_docs/rounds/round_08/discussions/discussion_2026-05-01_requirements_response.md`
- 当前状态：`本轮验收完成`
- 当前判断：
  - Round08 主瓶颈不是 first-stage candidate hit-any，而是 reranking / evidence selection。
  - 优先做 error layer analysis，把错误拆成 `gold not in top500`、`gold in top500 but not top20`、`gold in top20 but not top3`、`gold in context but label wrong`。
  - 已完成 error layer analysis、feature table construction、feature fusion reranker、classifier context update。
  - Fusion GBDT top3 evidence F-score `0.2011`，略高于 MiniLM-only `0.1987`。
  - Fusion GBDT top20 macro recall `0.4669`，高于 MiniLM-only `0.4540`。
  - Fusion GBDT top50 context + TF-IDF logreg classifier accuracy `0.4675`，macro-F1 `0.4376`，harmonic mean `0.2812`。
  - Dense retrieval 可以做，但只能作为 supplement ablation，不替换当前 sparse RRF，也不应阻塞主线。
  - Field-aware BM25、claim decomposition、evidence expansion、relation/comparison parsing 需要先做数据审计或降级实现。

### Round09：Alpha-blend and REFUTES calibration
- Index：`agent_docs/rounds/round_09/round_09_index.md`
- Requirement：`agent_docs/rounds/round_09/requirements.md`
- 报告：
  - `agent_docs/rounds/round_09/round_09_report_a_execution_and_acceptance.md`
- 当前状态：`本轮验收完成`
- 当前判断：
  - 最佳 final evidence selector 是 `MiniLM top100 scope + REFUTES x2 Fusion GBDT + alpha blend alpha_minilm=0.4`。
  - top3 evidence F-score 达到 `0.2105`，高于 Round07 MiniLM-only `0.1987` 和 Round08 pure Fusion GBDT `0.2011`。
  - best blend 的 REFUTES recall `0.1420`，仍低于 MiniLM-only `0.1790`。
  - classifier macro-F1 仍然是 Round08 Fusion GBDT top50 更好：`0.4376`；Round09 blend top50 macro-F1 是 `0.4142`。
  - assignment harmonic mean 当前 Round09 blend 更好：`0.2865`。

### Round10：Neural verification baselines planning
- Index：`agent_docs/rounds/round10/round10_index.md`
- Requirement：`agent_docs/rounds/round10/requirements.md`
- 报告：
  - `agent_docs/rounds/round10/round10_report_a_stage_plan_and_acceptance.md`
  - `agent_docs/rounds/round10/round10_report_b_execution_results.md`
- 当前状态：`本轮最低验收完成`
- 当前判断：
  - 必须补 neural baseline，否则项目会太像 retrieval engineering。
  - Round10 不应跑大量模型网格，应集中在 transformer claim classifier、claim-evidence verifier、pairwise neural reranker、hybrid scoring。
  - DistilRoBERTa concat classifier over Fusion GBDT top10 达到 accuracy `0.5260`、macro-F1 `0.4543`，超过 TF-IDF baseline `0.4675` / `0.4376`。
  - 完整 final-system 组合已补齐：Round09 blend evidence top3 + DistilRoBERTa classifier label 达到 evidence F `0.2105`、accuracy `0.5260`、harmonic mean `0.3007`。
  - Neural verifier epoch4 pair-level macro-F1 `0.5027`，但 verifier top3 evidence F-score 只有 `0.1029`。
  - Verifier top50 REFUTES recall `0.5309`，说明它有 REFUTES/context signal，但不适合直接控制 final top3。

## 当前全局判断
- 当前最关键的问题：final evidence 与 claim label 最优组件不同；final evidence 用 Round09 blend，claim label 用 DistilRoBERTa concat classifier；当前组合 dev harmonic mean `0.3007`。
- 当前最可信的事实来源：README、Gmail 线程 `Re: NLP Group assignment`、Instagram 页面状态、Round01/Round02 报告。
- 下一步建议：形成最终系统组合：Round09 blend alpha0.4 负责 evidence top3，DistilRoBERTa concat classifier 负责 claim label；verifier 暂作为 REFUTES/context diagnostic。
