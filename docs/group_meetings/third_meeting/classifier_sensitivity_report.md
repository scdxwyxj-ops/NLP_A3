# Classifier Sensitivity Report

本次实验是**classifier-only**敏感性矩阵：仅基于现有 top64 检索候选上下文，不重新跑 retrieval / embedding / CE。
固定参数采用你给定的最终 classifier 配置（`context_mode=enhanced`, `top_budget=120`, `tail_budget=50`, `max_features=60000`, `class_weight=balanced`, `ngram_range=(1,2)`, `stop_words=True`, `min_df=1`, `sublinear_tf=False`, `solver=liblinear`, `OneVsRest(LogisticRegression)`）。

## Selected point
- `C=0.125`, `context_k=20`
- dev accuracy: 0.519481
- dev macro-F1: 0.484141
- confusion matrix (labels 顺序: SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED):
  - SUPPORTS: [39, 10, 17, 2]
  - REFUTES: [3, 15, 8, 1]
  - NOT_ENOUGH_INFO: [10, 8, 20, 3]
  - DISPUTED: [4, 7, 1, 6]

- per-class recall:
  - SUPPORTS: 0.573529
  - REFUTES: 0.555556
  - NOT_ENOUGH_INFO: 0.487805
  - DISPUTED: 0.333333

`prediction_histogram`（预测分布）: {"SUPPORTS": 56, "REFUTES": 40, "NOT_ENOUGH_INFO": 46, "DISPUTED": 12}
