# O-C5 Worker F：Train-only Selection 审计报告（只读复算，基于 Worker A 成果）

## 输入与范围
- 严格上下文池（严格复用 worker A 的上下文来源）：
  - `train`: `round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json`
  - `dev`: `round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json`
- 比较集合：`context_k ∈ {5,10,20,32,64}`，`feature_mode ∈ {tfidf_only, tfidf_plus_shallow}`
- 仅做审计，不新增实验。数据源来自：
  - `round18/outputs/o_classifier/o_c5_comparison/worker_a_tfidf_side/summary.json`
  - 每条行对应 `metric_path` 下的 `selection_meta.selection_results[0].macro_f1`（train-holdout）

## 结果汇总（来自已有 selection_trace）
列含义：
- `train_holdout_macro_f1`: Worker A 在 `selection_meta.selection_results[0].macro_f1`（严格意义上的 train-holdout）
- `dev_macro_f1`: 对应 `metric_path` 的最终 dev 评分（也是 `macro_f1`）
- `top_class_share`: 预测端四类频次最大比例
- `collapse_gate`: 是否 `max_class_share<=0.70` 且 4 类都有预测

| k | mode | train_holdout_macro_f1 | dev_macro_f1 | top_class_share | collapse_gate |
|---:|---|---:|---:|---:|---|
| 5 | tfidf_only | 0.349606 | 0.404692 | 0.370130 | passed |
| 5 | tfidf_plus_shallow | 0.330461 | 0.495421 | 0.376623 | passed |
| 10 | tfidf_only | 0.346651 | 0.376912 | 0.402597 | passed |
| 10 | tfidf_plus_shallow | 0.360101 | 0.438768 | 0.409091 | passed |
| 20 | tfidf_only | 0.352128 | 0.469567 | 0.389610 | passed |
| 20 | tfidf_plus_shallow | 0.367250 | 0.462597 | 0.415584 | passed |
| 32 | tfidf_only | 0.371479 | 0.423615 | 0.402597 | passed |
| 32 | tfidf_plus_shallow | 0.363632 | 0.475738 | 0.422078 | passed |
| 64 | tfidf_only | 0.391631 | 0.379670 | 0.389610 | passed |
| 64 | tfidf_plus_shallow | **0.408942** | **0.382842** | **0.344156** | **passed** |

## Train-only 选择结论（复用现有 A 产物重算）
- 按 `train_holdout_macro_f1` 在上述 10 个候选中排名，**winner = `tfidf_plus_shallow, k=64`**
  - 训练内 holdout 宏指标：`0.4089419183`
  - 该候选对应 dev 最终宏指标：`0.38284172098`
- O-C4 baseline 宏指标（给定）：`0.4719276094276095`
- `0.382842 < 0.4719276`，故 **winner 不满足 strict promote**

## k=5 tfidf_plus_shallow 的 strict 情况
- `train_holdout_macro_f1 = 0.3304607514`（低于 winner）
- `dev_macro_f1 = 0.4954212454`，高于 O-C4 baseline `+0.023494`
- `collapse_gate`: passed
  - 四类都出现了预测，预测直方图中每类均>0
  - `top_class_share = 0.376623 <= 0.70`
  - 四类 recall 均非零（`DISPUTED/NI/REFUTES/SUPPORTS` 全部 > 0）
- 因此：`k=5 tfidf_plus_shallow` 在自身 dev 对比中可超过 baseline，但若执行严格的 train-only winner 选择（跨 k/mode），并不会被选为 final winner。

## 输出与命令
- 未新增实验，仅做只读审计；未创建新的 `outputs` 文件。
- 本次复算仅涉及：
  - `round18/outputs/o_classifier/o_c5_comparison/worker_a_tfidf_side/summary.json`
  - 各 `metric_path` 文件中的 `selection_meta.selection_results[0].macro_f1` 与 `collapse_gate`
