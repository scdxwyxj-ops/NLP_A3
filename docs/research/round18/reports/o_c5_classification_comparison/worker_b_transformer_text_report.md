## Worker B Transformer Text Report

### 结论
- 在严格上下文（`train`/`dev` 都使用 `o_ce_factual_context`）下，Transformer tokenized classifier 的最佳全量配置为：
  - `distilroberta-base`, `max_length=256`, `lr=3e-5`, `epochs=3`, `k_train=20`, `k_dev=20`
- 与 TF-IDF LogReg 基线 `macro_f1=0.4719` 比较，当前严格任务下 dev `macro_f1=0.4178`，**未超越基线**。
- 结论：不能支持“TF-IDF LogReg 只因没比较 transformer”的猜测；在这条严格上下文链路上，Transformer tokenizer + 序列分类器尚未明显改善该任务性能。

### 运行与复现命令
#### Smoke（严格候选上下文）
```bash
python round18/experiments/o_classifier/o_c5_comparison/worker_b_transformer_text/run_worker_b_transformer_text.py \
  --train-pool round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json \
  --dev-pool round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json \
  --smoke-only --smoke-rows 80 \
  --smoke-lr-grid 2e-5 --smoke-epoch-grid 1 --smoke-max-length-grid 128 \
  --train-val-fraction 0.2 --batch-size 8 --model-name distilroberta-base \
  --output-dir round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text \
  --run-id worker_b_transformer_text_smoke
```

#### Full（严格）
```bash
python round18/experiments/o_classifier/o_c5_comparison/worker_b_transformer_text/run_worker_b_transformer_text.py \
  --train-pool round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json \
  --dev-pool round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json \
  --train-context-k 20 --dev-context-k 20 \
  --train-val-fraction 0.2 \
  --batch-size 8 --model-name distilroberta-base \
  --lr-grid 2e-5,3e-5 --epoch-grid 2,3 --max-length-grid 256,384 \
  --output-dir round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text \
  --run-id worker_b_transformer_text_full_k20
```

### 关键结果
- 运行模式：`STRICT`
- 训练候选数：8（`max_length in {256,384}`, `lr in {2e-5,3e-5}`, `epoch in {2,3}`，均为 train-holdout 选参）
- 选中配置：`{max_length:256, lr:3e-5, epochs:3}`（基于 holdout `macro_f1`）
- 关键 dev 指标（`worker_b_transformer_text_full_k20_k20_ml256_lr3em05_ep3_metrics.json`）：
  - `accuracy=0.538961038961039`
  - `macro_f1=0.41781166914160695`
  - `macro_recall=0.43615827089643444`
  - `top_class=SUPPORTS`
  - `top_class_share=0.6103896103896104`
  - `selection_gate.status=failed`，`selection_gate.failures=["macro_f1_below_baseline"]`
  - `per_class_recall`
    - SUPPORTS: 0.7794117647058824
    - REFUTES: 0.5925925925925926
    - NOT_ENOUGH_INFO: 0.3170731707317073
    - DISPUTED: 0.05555555555555555
  - `prediction_histogram`: SUPPORTS=94, REFUTES=38, NOT_ENOUGH_INFO=21, DISPUTED=1
- runtime（秒）：
  - `full_train_wall_seconds=23.103532065000763`
  - `total_wall_seconds=195.03452947299957`

### 输出产物（本工作目录下）
- 选择与上下文：
  - `round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/worker_b_transformer_text_full_k20_selection_trace.json`
  - `round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/worker_b_transformer_text_full_k20_train_context_top20.jsonl`
  - `round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/worker_b_transformer_text_full_k20_dev_context_top20.jsonl`
- dev 结果：
  - `round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/worker_b_transformer_text_full_k20_k20_ml256_lr3em05_ep3_dev_predictions.json`
  - `..._dev_proba.json`
  - `..._metrics.json`
  - `..._confusion_matrix.csv`
  - `..._classification_report.txt`
- train-holdout（fusion）：
  - `..._train_holdout_predictions.json`
  - `..._train_holdout_proba.json`
- manifest/record：
  - `round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/run_manifest.json`
  - `round18/outputs/o_classifier/o_c5_comparison/worker_b_transformer_text/run_record.json`

### 与 TF-IDF 对比结论（针对问题）
- 当前严格 context 下，Transformer 在全量训练与验证表现上未突破 TF-IDF LogReg 的 0.4719 baseline，且仍被 `macro_f1_below_baseline` gate 拦截；因此“仅仅因为没比对 Transformer 才导致问题”这条解释**不成立**。  
- 可能的下一步是扩大 grid（如 384/512、更多 epoch、加权采样、类别重采样）或在同一严格上下文内切换其他轻量模型继续对比。
