# O-C5 TF-IDF vs TF-IDF+Shallow (Strict) Report

## 任务与设置
- 严格模式（strict）：只使用固定 CE factual context pool 的前置候选，不引入诊断集
- Train pool: `round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json`
- Dev pool: `round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json`
- Claims/evidence: `data/train-claims.json` / `data/dev-claims.json` / `data/evidence.json`
- 模型:
  - `tfidf_only`：TF-IDF + LogisticRegression
  - `tfidf_plus_shallow`：TF-IDF + sparse shallow side features（实体/数字/年份重叠、长度形态、source/ce 排序与评分统计等）
- 选择准则：只在 train-holdout 上做 `train_val_fraction=0.2` 选择；dev 仅做最终确认评估
- 目标 k: 5, 10, 20, 32, 64
- Collapse gate: `top_class_share <= 0.70`；四类预测都要非零；每类召回都要非零

## 运行命令
1. Smoke（确认脚本闭环）
```bash
python round18/experiments/o_classifier/o_c5_comparison/worker_a_tfidf_side/run_worker_a_tfidf_side.py \
  --context-ks 5 --run-tfidf-only --run-tfidf-shallow --n-jobs 4
```
2. 全量
```bash
python round18/experiments/o_classifier/o_c5_comparison/worker_a_tfidf_side/run_worker_a_tfidf_side.py \
  --context-ks 5,10,20,32,64 --run-tfidf-only --run-tfidf-shallow --n-jobs 1
```

输出根目录：`round18/outputs/o_classifier/o_c5_comparison/worker_a_tfidf_side`

## 结果汇总（dev）
指标：Macro-F1 / Accuracy / Top-class share / 每类 recall（最小值）

| k | model | Macro-F1 | Accuracy | Top-class share | Min per-class recall |
|---:|---|---:|---:|---:|---:|
| 5 | tfidf_only | 0.404692 | 0.435065 | 0.3701 | 0.3333 |
| 5 | tfidf_plus_shallow | 0.495421 | 0.545455 | 0.3766 | 0.3333 |
| 10 | tfidf_only | 0.376912 | 0.435065 | 0.4026 | 0.2222 |
| 10 | tfidf_plus_shallow | 0.438768 | 0.506494 | 0.4091 | 0.2222 |
| 20 | tfidf_only | 0.469567 | 0.506494 | 0.3896 | 0.3333 |
| 20 | tfidf_plus_shallow | 0.462597 | 0.512987 | 0.4156 | 0.3333 |
| 32 | tfidf_only | 0.423615 | 0.474026 | 0.4026 | 0.2778 |
| 32 | tfidf_plus_shallow | 0.475738 | 0.512987 | 0.4221 | 0.3889 |
| 64 | tfidf_only | 0.379670 | 0.428571 | 0.3896 | 0.2222 |
| 64 | tfidf_plus_shallow | 0.382842 | 0.422078 | 0.3442 | 0.2778 |

全部 10 行均通过 collapse gate：`status=passed`（四类均有预测且无空召回；top_class_share 均 < 0.7）。

## 与当前 baseline 对比
- 当前 baseline（用户给定）`macro-F1 = 0.4719276094`
- 超过 baseline 的仅：
  - `k=5`: `tfidf_plus_shallow=0.495421 (+0.02349)`
  - `k=32`: `tfidf_plus_shallow=0.475738 (+0.00381)`
- 其余 strict 条件下未超过 baseline

## 严格/诊断说明
- **strict**：本次实验均为 strict 跑法，train/dev 使用不同 pool（同族一致）且无 dev 参与参数选择
- **diagnostic**：未额外重跑独立诊断 runner（按你的要求仅补充了 strict baseline 对比）；selection 使用固定 `train_holdout_macro_f1`
- 每个 run 都写了：
  - `run_manifest.json`
  - `run_record.json`
  - `*_selection_trace.json`
  - `*_metrics.json`
  - `*_confusion_matrix.csv`
  - `*_dev_predictions.json`
  - `selection_trace_summary.json`
  - `summary.json`

## 结论
在你给定的 baseline 对比目标下，`tfidf_plus_shallow` 在 k=5 和 k=32 时有提升（超过 0.4719），但多数 k 下优势不稳定；建议在后续若继续扩大实验，优先复用 **k=5/32 strict** 配置并进一步检验是否可重复到别的 seed 或更稳健的 side-feature 变体。

