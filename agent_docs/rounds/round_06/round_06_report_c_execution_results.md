# Round06 Report C：Execution results for Stages A-E

报告时间：`2026-04-30`

## 一、本次执行范围

本次执行覆盖 Round06 的前五个 stage 初版：

- Stage A：baseline rows + comparison table scaffold。
- Stage B：MiniLM-mined task-aware hard negatives。
- Stage C：task-aware MiniLM reranker conservative fine-tuning。
- Stage D：fine-grained semantic feature prototype。
- Stage E：classifier-oriented evidence packaging。

## 二、Stage A：comparison table

已生成：

```txt
outputs/round06/comparison_table.csv
```

当前表格有 `10` 行，包含：
- BM25 top-5。
- zero-shot MiniLM top-3/top-5/top-20。
- Round05 BCE MiniLM top-5。
- Round06 task-aware MiniLM top-3/top-5/top-20。
- BM25 top-50/top-100 candidate recall ceiling rows。

当前最强 final evidence output 仍是：

```txt
R05-MINILM-ZS-T3
retrieval_f_score = 0.164193
macro_recall = 0.187662
precision = 0.168831
```

## 三、Stage B：MiniLM-mined task-aware negatives

命令产物：

```txt
outputs/round06/train-task-aware-negatives-top100-neg10.jsonl
outputs/round06/train-minilm-ranked-candidates-top100-neg10.json
outputs/round06/task-aware-negative-stats-top100-neg10.json
outputs/round06/task-aware-negative-examples-top100-neg10.md
```

统计：

```txt
claims = 1228
pairs = 16402
positive_pairs = 4122
negative_pairs = 12280
mean_negative_minilm_rank = 6.1536
mean_negative_bm25_rank = 35.3335
```

解释：
- negatives 来自 BM25 top-100。
- 再用 zero-shot MiniLM 排序。
- 每个 claim 取 MiniLM 排名前列但不是 gold 的 evidence。
- mean MiniLM rank 约 `6.15`，说明这些 negatives 是模型真正容易排高的 topical false positives。

## 四、Stage C：task-aware reranker result

训练设置：

```txt
model = cross-encoder/ms-marco-MiniLM-L6-v2
loss = single-logit BCE
negatives = MiniLM-mined top100 non-gold, 10 per claim
learning_rate = 5e-6
epochs = 0.5
batch_size = 32
```

训练结果：

```txt
train_loss = 0.9281
runtime = 14.92s
```

dev results：

| method | top-k | F-score | precision | macro recall | hit-any | REFUTES recall |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot MiniLM | 3 | 0.1642 | 0.1688 | 0.1877 | 0.3961 | 0.0494 |
| task-aware MiniLM | 3 | 0.1599 | 0.1667 | 0.1790 | 0.3766 | 0.0494 |
| zero-shot MiniLM | 5 | 0.1578 | 0.1325 | 0.2332 | 0.4545 | 0.0679 |
| task-aware MiniLM | 5 | 0.1526 | 0.1273 | 0.2256 | 0.4351 | 0.0864 |
| zero-shot MiniLM | 20 | 0.0777 | 0.0461 | 0.3035 | 0.5519 | 0.1327 |
| task-aware MiniLM | 20 | 0.0766 | 0.0455 | 0.3000 | 0.5519 | 0.1327 |

判断：
- task-aware fine-tuning 接近 zero-shot，但没有超过。
- top-5 REFUTES recall 有提升：`0.0679 -> 0.0864`。
- 这说明 hard negatives 可能改善部分类别，但当前训练设置还不足以全面超过 zero-shot。

## 五、Stage D：semantic features

已生成：

```txt
outputs/round06/semantic_features_minilm_top5.jsonl
outputs/round06/semantic_feature_examples_minilm_top5.md
```

当前提取字段：

```txt
entities
percentages
quantities
years
negation_cues
comparison_cues
causality_cues
relation_verbs
```

当前定位：
- 不作为手写规则预测器。
- 作为 classifier input 或 error analysis 的辅助信息。
- 对 `claim-375` 这类错误，可以显式展示百分比、实体、否定和因果 cue 是否被 evidence 覆盖。

## 六、Stage E：classifier evidence packages

已生成：

```txt
outputs/round06/train-classifier-context-minilm-top20-semantic.jsonl
outputs/round06/dev-classifier-context-minilm-top20-semantic.jsonl
```

格式特征：
- `final_evidence_candidates`：当前 top-3 evidence。
- `classifier_evidence_context`：top-20 evidence context。
- 每个 evidence 可带 `semantic_features`。
- train package 来自 MiniLM-ranked BM25 top-100 candidates。
- dev package 来自 zero-shot MiniLM top-20 predictions。

## 七、当前结论

当前最稳策略：

```txt
final evidence output:
  zero-shot MiniLM top-3

classifier input:
  MiniLM top-20 context + optional semantic features
```

task-aware fine-tuning 暂时不是新的 best row，但它提供了一个重要发现：

```txt
用模型挖 negatives 是合理的，
但需要进一步调训练目标/采样策略，不能只做简单 BCE 微调。
```

## 八、下一步建议

优先进入 Round07 classifier baseline：
- 用 `train-classifier-context-minilm-top20-semantic.jsonl`。
- 对比 with vs without semantic features。
- 对比 top-3/top-10/top-20 context。

如果继续优化 reranker，则优先尝试：
- pairwise ranking loss。
- 更低 learning rate 或 partial layer freezing。
- negatives per claim `5` vs `10` vs `20`。
- 对 REFUTES 做 label-aware sampling。
