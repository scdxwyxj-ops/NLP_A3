# Round05 索引：Supervised reranker 对比实验

报告时间：`2026-04-30`

## 一、本轮总目标

建立一个有监督 evidence reranker 对比实验，把 Round04 的无监督 retrieval baseline 转化为可训练的 claim-evidence relevance ranking pipeline。

核心路线：

```txt
BM25 top-50 candidates
-> construct positive / hard negative claim-evidence pairs
-> train supervised transformer cross-encoder reranker
-> rerank candidates
-> select top-k evidence ids
-> eval.py dev retrieval F-score
```

## 二、本轮当前状态

- 状态：`已结轮`
- 当前 stage：`Round05 complete；Round06 将继续 task-aware negatives 和 semantic feature exploration`
- 依赖：
  - Round04 TF-IDF baseline 已完成。
  - Round04 BM25 baseline 已完成，最佳 dev F-score：top-5 `0.0772`。
  - 课程规则允许 open-source pretrained models，但禁止 closed-source APIs/proprietary models 和外部训练/评测数据。
- 当前重点：不要继续深挖无监督词向量；无监督方案只作为 baseline 和 candidate generator。
- 当前模型建议：第一版使用 `cross-encoder/ms-marco-MiniLM-L6-v2`；保守备选 `distilbert-base-uncased`。
- 当前工程结果：已新增 `src/a3_factcheck/rerank/` API 和 `experiments/rerank/` 实验入口。

## 三、Stage 划分

### Stage A：Discovery and experiment design

状态：`已完成初版`

验收指标：
- 明确 model/backbone 选择。
- 明确训练样本构造：
  - positive = claim + gold evidence
  - hard negative = claim + BM25 top-k non-gold evidence
- 明确 dev evaluation protocol。
- 明确 Colab/free compute 风险。

### Stage B：Candidate pool and hard negatives

状态：`已完成初版`

验收指标：
- 从 BM25 top-50 生成 train/dev candidate pool。
- 每个 train claim 有 positive pairs 和 hard negative pairs。
- 输出格式稳定，供 reranker training 使用。
- 不检查或使用 test labels。

### Stage C：Supervised cross-encoder baseline

状态：`已完成初版`

验收指标：
- 使用 open-source pretrained transformer。
- fine-tune claim-evidence binary relevance classifier。
- 保存训练日志和 dev metrics。
- 能在免费 Colab 可运行规模内完成。

### Stage D：Rerank and evaluate evidence selection

状态：`已完成初版`

验收指标：
- 用 reranker 对 BM25 candidates 重新排序。
- 比较 top-k：`1, 3, 5, 10`。
- 报告 retrieval F-score、classification placeholder accuracy、harmonic mean。
- 与 BM25 top-5 F-score `0.0772` 对比。

## 四、Round05 成功标准

最低成功：
- 有一个可运行的 supervised reranker training/eval pipeline。
- dev retrieval F-score 超过 BM25 top-5 `0.0772`。

理想成功：
- reranker 明显提升 gold evidence 排名前列命中。
- 可以为 Round06 classifier 提供更好的 evidence candidates。
- 报告中能清楚说明：
  - 为什么 BM25 只作为 candidate generator。
  - 为什么 cross-encoder 适合 claim-evidence relevance scoring。
  - hard negatives 如何提高 reranker 学习难度和有效性。

## 五、结轮结论

Round05 已完成：

1. 实现 BM25 candidate pool + hard negative dataset。
2. 实现 reranker dataset loader。
3. 训练 MiniLM cross-encoder reranker。
4. rerank dev candidates，并与 BM25 top-5 F-score `0.0772` 对比。
5. 完成 top-k recall、TP/FP/FN/TN 和 topic relevance vs evidence relevance 错误分析。

Round06 接续方向：

```txt
task-aware hard negatives
+ fine-grained semantic extraction
+ classifier-oriented evidence packaging
```

## 六、当前实现记录

- Report C：`agent_docs/rounds/round_05/round_05_report_c_reranker_candidate_api.md`
- Report D：`agent_docs/rounds/round_05/round_05_report_d_method_summary.md`
- Report E：`agent_docs/rounds/round_05/round_05_report_e_reranker_training_results.md`
- Report F：`agent_docs/rounds/round_05/round_05_report_f_recall_analysis.md`
- Report G：`agent_docs/rounds/round_05/round_05_report_g_error_analysis_for_classifier.md`
- 生成 train hard-negative pairs：`10262`
- positive pairs：`4122`
- hard negatives：`6140`
- dev recall@50：`0.3249`
- dev recall@100：`0.4188`
- 当前最佳 retrieval：zero-shot `cross-encoder/ms-marco-MiniLM-L6-v2` reranker top-3，dev F-score `0.1642`
- 当前分类前处理判断：final evidence output 用 MiniLM top-3；classifier input 应尝试 MiniLM top-10/top-20/top-50。
- 当前重要错误类型：MiniLM 常把 topic-relevant evidence 排高，但这些 evidence 不一定是 fact-checking label decision 所需的 gold evidence。
