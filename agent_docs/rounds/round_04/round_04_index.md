# Round04 索引：Retrieval baseline 与诊断

报告时间：`2026-04-29`

## 一、本轮总目标
- 建立可复现的 lexical evidence retrieval baseline。
- 在 dev set 上报告 retrieval F-score，并观察 top-k 对主指标的影响。
- 为后续 sequence classifier 提供候选 evidence 输入。

## 二、本轮当前状态
- 状态：`进行中`
- 当前 stage：`Stage C：Retrieval 误差分析`
- 当前实现：`src/tfidf_retrieval_baseline.py`
- 当前报告：`agent_docs/rounds/round_04/round_04_report_a_tfidf_retrieval_baseline.md`
- 策略报告：`agent_docs/rounds/round_04/round_04_report_b_retrieval_strategy_plan.md`
- 结构整理报告：`agent_docs/rounds/round_04/round_04_report_c_repository_structure.md`
- BM25 报告：`agent_docs/rounds/round_04/round_04_report_d_bm25_baseline.md`
- Discussion：`agent_docs/rounds/round_04/discussions/discussion_2026-04-30_retrieval_ranking_method.md`
- Discussion：`agent_docs/rounds/round_04/discussions/discussion_2026-04-30_unsupervised_word_vectors.md`
- Discussion：`agent_docs/rounds/round_04/discussions/discussion_2026-04-30_retrieval_priority_discovery.md`
- Discussion：`agent_docs/rounds/round_04/discussions/discussion_2026-04-30_pretrained_model_rules.md`
- 当前输出目录：`outputs/round04/`

## 三、Stage 划分
### Stage A：数据读取与 evidence indexing
状态：`已达标`

依据：
- Round03 已下载并验证 `train/dev/test/evidence` JSON。
- `evidence.json` 可解析，包含 1,208,827 条 evidence。

### Stage B：Lexical retrieval baseline
状态：`已达标`

验收指标：
- 实现不依赖训练的 TF-IDF 或 BM25 retrieval baseline。
- 每个 dev claim 至少返回 top-k evidence。
- 在 dev 上报告 retrieval F-score。
- 保存不同 k 值的 dev retrieval 表现。

### Stage C：Retrieval 误差分析
状态：`待开始`

验收指标：
- 抽样分析 retrieval 成功和失败案例。
- 记录常见失败原因。
- 产出报告可用分析素材。

## 四、下一步
1. 做 retrieval 误差分析，抽样查看 TF-IDF/BM25 成功和失败案例。
2. 用 BM25 top-50 构造 supervised reranker candidate pool 和 hard negatives。
3. 进入二阶段 reranker：BM25 top-50 -> transformer/cross-encoder rerank。
4. 如果采用 cluster-based variant，统一使用 `multicell-level masks`，不再使用 `context-level masks`。
5. 新实验放入 `experiments/`，可复用代码放入 `src/a3_factcheck/`，稳定说明放入 `docs/`。
6. 把候选 evidence 输出格式稳定下来，供 Round05 sequence classifier 使用。
