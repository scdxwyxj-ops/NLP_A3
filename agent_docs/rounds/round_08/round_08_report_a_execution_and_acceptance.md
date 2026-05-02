# Round08 报告 A：执行结果与验收报表

日期：2026-05-01

## 一句话结论

Round08 已完成本轮验收目标：在不依赖不存在的 document/title/neighbor metadata 的前提下，
完成了错误分层、feature table、lightweight feature fusion reranker 和 classifier context
对比实验。

最好的 Round08 结果是：

```txt
Fusion GBDT reranker top3 evidence F-score = 0.2011
Round07 MiniLM-only top3 evidence F-score = 0.1987
```

提升不大，但说明 `MiniLM score + sparse ranks + shallow semantic overlap` 的融合有正向信号。

## Discovery Planner 状态

按要求调用了 `discovery-planner`：

```txt
python /mnt/a/code/agent_dev/discovery_planner/scripts/discovery_planner.py ...
```

但本地 launcher 失败：

```txt
Codex discovery failed with exit code 127
```

因此本轮继续沿用 discovery-planner 的结构手动执行，并将计划、审计和结果记录在
`agent_docs/rounds/round_08/`。

## Stage B：Error Layer Analysis

脚本：

```txt
experiments/analysis/error_layer_analysis.py
```

输出：

```txt
outputs/round08/error_layer_summary.csv
outputs/round08/error_layer_by_claim.csv
```

结果：

| Layer | Count | Claim % | Gold Evidence % | Main Fix |
|---|---:|---:|---:|---|
| gold not in top500 | 180 | 0.5909 | 0.3666 | dense supplement / candidate retrieval |
| gold in top500 but not top50 | 58 | 0.3052 | 0.1181 | score more candidates / fusion reranker |
| gold in top50 but not top20 | 52 | 0.2468 | 0.1059 | feature fusion reranker |
| gold in top20 but not top3 | 108 | 0.4610 | 0.2200 | final evidence selection |
| coverage top500 | 311 | 0.8896 | 0.6334 | diagnostic |
| coverage top50 | 253 | 0.8052 | 0.5153 | diagnostic |
| coverage top20 | 201 | 0.7143 | 0.4094 | diagnostic |
| coverage top3 | 93 | 0.4545 | 0.1894 | diagnostic |

解释：

- top500 hit-any 已经高，但仍有 `36.66%` gold evidence 不在 top500。
- 从 top20 到 top3 又丢掉 `21.996%` gold evidence。
- 所以 Round08 不能只做 dense，也不能只做 classifier；rerank/final selection 是明确瓶颈。

## Stage C：Feature Table Construction

脚本：

```txt
experiments/rerank/build_feature_table.py
```

输出：

```txt
outputs/round08/train_feature_table_top50.csv
outputs/round08/dev_feature_table_top50.csv
```

规模：

| Split | Rows | Positive Rows | Scope |
|---|---:|---:|---|
| train | 61,400 | 1,908 | MiniLM top50 over RRF top500 |
| dev | 7,700 | 253 | MiniLM top50 over RRF top500 |

Feature table 包含：

```txt
minilm_rank / minilm_score
rrf_rank / rrf_score
bm25_rank / bm25_score
char_tfidf_rank / char_tfidf_score
source_count
entity overlap
percentage / quantity / year overlap
negation features
comparison / causality / relation verb overlap
```

限制：

```txt
当前是 cheap fusion，只重排 MiniLM top50。
它不能救 gold in top500 but not top50 的证据。
```

## Stage D：Feature Fusion Reranker

脚本：

```txt
experiments/rerank/train_feature_fusion_reranker.py
```

输出：

```txt
outputs/round08/fusion_logreg/
outputs/round08/fusion_gbdt/
outputs/round08/fusion_comparison.csv
```

Pair-level diagnostics：

| Model | Dev AP | Dev ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.1616 | 0.8104 |
| GBDT | 0.1902 | 0.8192 |

Evidence reranking comparison：

| Method | Output k | Evidence F-score | Macro Recall | Hit-any | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| MiniLM only | 3 | 0.1987 | 0.2331 | 0.4545 | 0.1790 |
| Fusion logreg | 3 | 0.1957 | 0.2188 | 0.4416 | 0.0802 |
| Fusion GBDT | 3 | 0.2011 | 0.2321 | 0.4545 | 0.0864 |
| MiniLM only | 20 | 0.1106 | 0.4540 | 0.7143 | 0.3951 |
| Fusion logreg | 20 | 0.1138 | 0.4649 | 0.7338 | 0.4136 |
| Fusion GBDT | 20 | 0.1153 | 0.4669 | 0.7403 | 0.4136 |

结论：

- GBDT fusion 是本轮最佳 final evidence top3：`0.2011`，略高于 MiniLM-only `0.1987`。
- Fusion 对 top20 context 更稳定：macro recall 从 `0.4540` 提到 `0.4669`，hit-any 从 `0.7143` 提到 `0.7403`。
- 但 Fusion GBDT 的 top3 `REFUTES Recall` 下降，说明第一版 fusion 更偏向整体 F-score/context recall，不适合直接声称解决了 REFUTES。

## Stage F：Classifier Context Update

使用 Fusion GBDT ranked top50 重新构造 classifier context：

```txt
outputs/round08/dev-classifier-context-fusion-gbdt-top20-semantic.jsonl
outputs/round08/dev-classifier-context-fusion-gbdt-top50-semantic.jsonl
outputs/round08/train-classifier-context-fusion-gbdt-top20-semantic.jsonl
outputs/round08/train-classifier-context-fusion-gbdt-top50-semantic.jsonl
```

然后复用现有 TF-IDF logistic regression classifier。

Classifier 对比：

| Context | Semantic | Accuracy | Macro-F1 | Evidence F-score | Harmonic Mean |
|---|---|---:|---:|---:|---:|
| Round07 MiniLM top50 | no | 0.4610 | 0.4242 | 0.1987 | 0.2778 |
| Fusion GBDT top20 | no | 0.4481 | 0.3930 | 0.2011 | 0.2776 |
| Fusion GBDT top20 | yes | 0.4481 | 0.3972 | 0.2011 | 0.2776 |
| Fusion GBDT top50 | no | 0.4675 | 0.4376 | 0.2011 | 0.2812 |
| Fusion GBDT top50 | yes | 0.4351 | 0.3895 | 0.2011 | 0.2751 |

结论：

- 最好的 classifier context 是 `Fusion GBDT top50 without semantic summary`。
- 它把 classifier macro-F1 从 `0.4242` 提到 `0.4376`。
- Harmonic mean 从 `0.2778` 提到 `0.2812`。
- semantic summary 在当前 TF-IDF classifier 里仍然没有稳定帮助。

## Round08 验收判断

Round08 本轮目标已满足：

1. 已完成错误分层表，明确损失发生在 candidate miss、top500->top50、top50->top20、top20->top3。
2. 已完成 train/dev feature table，成功 join MiniLM、RRF、BM25、char TF-IDF 和 semantic overlap features。
3. 已完成 fusion reranker 对比，GBDT top3 F-score 超过 MiniLM-only。
4. 已完成 classifier context update，Fusion GBDT top50 context 提升 classifier macro-F1 和 harmonic mean。

## 下一步建议

不要继续做基于不存在 metadata 的 evidence expansion。

下一步更现实的方向：

```txt
1. 扩展 MiniLM scoring scope，从 top50 扩到 top100/top200。
2. 让 fusion reranker 能救 gold in top500 but not top50。
3. 加入 class-aware objective 或 REFUTES-specific calibration。
4. 如果资源允许，再做 dense supplement，但不要阻塞主线。
```

