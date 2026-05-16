# Worker G Final Candidate Robustness Report

固定对比配置（严禁重新筛参数）:
- 候选: `k=5` TF-IDF + shallow, `max_features=30000`, ngram `1-2`, `C=1.0`
- Baseline: `k=20` TF-IDF only, `max_features=60000`, ngram `1-2`, `C=4.0`（O-C4 类比）

## 结论
- 结论：**不支持 strict promote**。train-only 重复抽样与 5-Fold CV 的配对平均表现均为负。
- Train-only repeated holdout 平均 `Δmacro-F1 = -0.010880`（分开 seed: holdout_seed_1337:-0.0290, holdout_seed_2027:-0.0089, holdout_seed_42:-0.0579, holdout_seed_7:+0.0184, holdout_seed_99:+0.0230）。
- 5-Fold CV 平均 `Δmacro-F1 = -0.006603`（分开 fold: cv_fold_1:-0.0067, cv_fold_2:+0.0043, cv_fold_3:-0.0495, cv_fold_4:+0.0312, cv_fold_5:-0.0123）。

## Final dev（仅复核，不参与候选选择）
- candidate: macro-F1=0.495421, accuracy=0.545455, macro-recall=0.502461, top-class-share=0.3766, collapse=passed
- baseline: macro-F1=0.471928, accuracy=0.519481, macro-recall=0.470067, top-class-share=0.4351, collapse=passed
- dev Δ: macro-F1=+0.023494, accuracy=+0.025974, macro-recall=+0.032394, top-class-share=-0.058442
- candidate per-class recall: SUPPORTS:0.574, REFUTES:0.444, NOT_ENOUGH_INFO:0.659, DISPUTED:0.333
- baseline per-class recall: SUPPORTS:0.603, REFUTES:0.407, NOT_ENOUGH_INFO:0.537, DISPUTED:0.333
- 产物: `worker_g_final_candidate_candidate_dev_metrics.json`, `worker_g_final_candidate_candidate_dev_predictions.json`, `worker_g_final_candidate_candidate_dev_confusion_matrix.csv`，以及 baseline 对应文件。

## 训练内 repeated holdout（seeds: 1337,2027,42,7,99）
| 方法 | macro-F1 | accuracy | macro-recall | top-class-share | collapse pass rate |
|---|---|---|---|---|---|
| k=5 tfidf+shallow (candidate) | 0.3624 ± 0.0369 | 0.4195 ± 0.0327 | 0.3675 ± 0.0401 | 0.3642 ± 0.0105 | 100% |
| k=20 tfidf-only (baseline) | 0.3733 ± 0.0302 | 0.4211 ± 0.0347 | 0.3730 ± 0.0305 | 0.4179 ± 0.0235 | 100% |

### repeated holdout paired deltas
| split | Δmacro-F1 | Δaccuracy | Δmacro-recall | Δtop-class-share |
|---|---:|---:|---:|---:|
| holdout_seed_1337 | -0.0290 | -0.0447 | -0.0259 | -0.0650 |
| holdout_seed_2027 | -0.0089 | -0.0041 | +0.0010 | -0.1016 |
| holdout_seed_42 | -0.0579 | -0.0285 | -0.0585 | -0.0447 |
| holdout_seed_7 | +0.0184 | +0.0407 | +0.0298 | -0.0488 |
| holdout_seed_99 | +0.0230 | +0.0285 | +0.0258 | -0.0081 |

## 5-Fold CV
| 方法 | macro-F1 | accuracy | macro-recall | top-class-share | collapse pass rate |
|---|---|---|---|---|---|
| k=5 tfidf+shallow (candidate) | 0.3668 ± 0.0427 | 0.4308 ± 0.0330 | 0.3717 ± 0.0412 | 0.3828 ± 0.0332 | 100% |
| k=20 tfidf-only (baseline) | 0.3734 ± 0.0424 | 0.4291 ± 0.0356 | 0.3724 ± 0.0444 | 0.4145 ± 0.0408 | 100% |

### CV paired deltas
| fold | Δmacro-F1 | Δaccuracy | Δmacro-recall | Δtop-class-share |
|---|---:|---:|---:|---:|
| cv_fold_1 | -0.0067 | +0.0041 | -0.0005 | -0.0407 |
| cv_fold_2 | +0.0043 | +0.0122 | +0.0060 | -0.0244 |
| cv_fold_3 | -0.0495 | -0.0203 | -0.0400 | +0.0081 |
| cv_fold_4 | +0.0312 | +0.0367 | +0.0408 | -0.0776 |
| cv_fold_5 | -0.0123 | -0.0245 | -0.0101 | -0.0245 |

## per-class recall 与 collapse
CSV 已保留每个 split 的 `per_class_recall_*` 与 `collapse_gate_*` 字段。两组配置 20 条 split 中全部通过 `collapse_gate (top_class_share <= 0.70)`。
- repeated_holdout 平均 per-class recall（候选）: SUPPORTS 0.4404, REFUTES 0.3550, NOT_ENOUGH_INFO 0.5065, DISPUTED 0.1680
- repeated_holdout 平均 per-class recall（基线）: SUPPORTS 0.4962, REFUTES 0.3300, NOT_ENOUGH_INFO 0.4260, DISPUTED 0.2400
- CV 平均 per-class recall（候选）: SUPPORTS 0.4700, REFUTES 0.3617, NOT_ENOUGH_INFO 0.5023, DISPUTED 0.1527
- CV 平均 per-class recall（基线）: SUPPORTS 0.5144, REFUTES 0.3310, NOT_ENOUGH_INFO 0.4352, DISPUTED 0.2090

## 推荐更新到 tutorial 的措辞
- 该候选在 train-only 评估上并未稳健超越 O-C4 类 baseline，不能作为 final promote，若 tutorial 只做稳定性可列为“non-promotion”案例。