# Round10 报告 A：Stage Plan 与验收指标

日期：2026-05-01

## 总体目标

Round10 的目标是补齐 neural baseline，并回答一个关键问题：

```txt
当前系统剩余瓶颈，是 retrieval/reranking pipeline 的工程问题，
还是模型没有学到 fact verification / stance detection 的本质？
```

Round07-Round09 已经完成：

```txt
Round07:
  sparse retrieval rescue

Round08:
  feature fusion reranker

Round09:
  REFUTES-calibrated alpha blend
```

当前最强 evidence selector：

```txt
Round09 top100 REFUTES x2 blend alpha=0.4
top3 evidence F-score = 0.2105
REFUTES recall = 0.1420
```

当前最强 classifier context：

```txt
Round08 Fusion GBDT top50
classifier macro-F1 = 0.4376
```

Round10 需要补的不是更多 sparse tricks，而是 neural verification baselines。

## Discovery Planner 记录

按要求调用了 discovery-planner：

```txt
python /mnt/a/code/agent_dev/discovery_planner/scripts/discovery_planner.py \
  --goal "Round10 COMP90042 A3 ..."
```

但本地 launcher 失败：

```txt
Codex discovery failed with exit code 127
```

因此本报告手动完成 discovery-planner 应输出的 stage plan、backlog 和验收指标。

## Stage A：Neural Baseline Planning

### 目的

明确 Round10 只跑必要 neural baselines，不乱开模型坑。

必要 baseline 分三类：

```txt
1. Transformer concat claim classifier
2. Neural claim-evidence verifier
3. Pairwise neural reranker
```

### 输入

已有产物：

```txt
outputs/round08/dev-classifier-context-fusion-gbdt-top50-semantic.jsonl
outputs/round08/train-classifier-context-fusion-gbdt-top50-semantic.jsonl
outputs/round09/blend_top100_refutes_x2/alpha_0.4-top3.json
outputs/round09/blend_top100_refutes_x2/alpha_0.4_ranked.json
outputs/round09/train_feature_table_top100.csv
outputs/round09/dev_feature_table_top100.csv
```

### 输出

```txt
agent_docs/rounds/round10/round10_report_a_stage_plan_and_acceptance.md
```

### 验收指标

Stage A 验收条件：

- 明确每类 neural baseline 的目标、输入、输出；
- 明确不做过大模型网格；
- 明确与 Round09 baseline 的比较指标。

## Stage B：Transformer Concat Claim Classifier

### 目的

补一个课程层面容易解释的 neural classifier baseline：

```txt
claim + top-k evidence -> SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED
```

### 推荐模型

按资源从小到大：

```txt
distilroberta-base
roberta-base
microsoft/deberta-v3-small
```

第一版只需要跑一个稳定模型，推荐：

```txt
distilroberta-base
```

如果 Colab/GPU 资源允许，再跑 `deberta-v3-small`。

### Context 设置

不要直接塞 top50，因为 512 tokens 会截断严重。

推荐：

```txt
Round08 Fusion GBDT top10
Round08 Fusion GBDT top20 compressed
Round09 blend top10
Round09 blend top20 compressed
```

### 输出

```txt
outputs/round10/claim_classifier/
outputs/round10/claim_classifier_summary.csv
```

### 对比基线

当前非神经 baseline：

```txt
Round08 Fusion GBDT top50 + TF-IDF logreg
accuracy = 0.4675
macro-F1 = 0.4376
```

### 验收指标

| Metric | Target |
|---|---:|
| dev accuracy | report |
| dev macro-F1 | compare against `0.4376` |
| REFUTES F1 | report |
| confusion matrix | required |
| training curve | required |

判断规则：

```txt
如果 macro-F1 > 0.4376:
  neural classifier 有实际收益

如果 macro-F1 <= 0.4376:
  concat classifier 不适合噪声 evidence context，但仍可作为 neural baseline 写进报告
```

## Stage C：Neural Claim-Evidence Verifier

### 目的

这是 Round10 最重要的 neural model。

它直接学习：

```txt
claim + single evidence -> SUPPORT / REFUTE / NEUTRAL
```

这比 MS MARCO MiniLM relevance score 更接近 fact verification。

### 数据构造

Positive：

```txt
SUPPORTS claim + gold evidence -> SUPPORT
REFUTES claim + gold evidence -> REFUTE
```

Neutral：

```txt
retrieved non-gold evidence -> NEUTRAL
random evidence -> NEUTRAL
```

暂不把 `DISPUTED` gold evidence 强行变成 pair-level positive，因为 DISPUTED 更像
claim-level aggregation 结果。

### 采样建议

```txt
SUPPORT positives:
  all

REFUTE positives:
  all, optionally upsample x2

NEUTRAL:
  max 5-10 per claim
  include MiniLM high-score non-gold
  include RRF high-rank non-gold
  include random non-gold
```

### Claim-level 聚合

对 top-k evidence 计算：

```txt
p_support, p_refute, p_neutral
```

然后：

```txt
S_sup = max_j p_support_j
S_ref = max_j p_refute_j
```

规则：

```txt
if S_sup high and S_ref low:
    SUPPORTS
elif S_ref high and S_sup low:
    REFUTES
elif S_sup high and S_ref high:
    DISPUTED
else:
    NOT_ENOUGH_INFO
```

Evidence score：

```txt
score = max(p_support, p_refute)
```

或：

```txt
score = 1 - p_neutral
```

### 输出

```txt
outputs/round10/verifier/pair_dataset_train.jsonl
outputs/round10/verifier/pair_dataset_dev.jsonl
outputs/round10/verifier/pair_metrics.json
outputs/round10/verifier/claim_aggregation_metrics.json
outputs/round10/verifier/verifier_rerank_predictions.json
```

### 验收指标

| Metric | Target |
|---|---:|
| pair-level macro-F1 | report |
| pair-level REFUTE F1 | report |
| claim-level accuracy | compare against `0.4675` |
| claim-level macro-F1 | compare against `0.4376` |
| verifier top3 evidence F | compare against `0.2105` |
| verifier REFUTES recall | target `> 0.1420`, ideally close to `0.1790` |

判断规则：

```txt
如果 verifier improves REFUTES recall:
  它解决了 relevance-only 的关键缺陷

如果 verifier improves claim macro-F1:
  它适合作为 classifier aggregation 主线

如果 verifier pair-F1 高但 retrieval F 不涨:
  aggregation/ranking score 设计仍需调
```

## Stage D：Pairwise Neural Reranker

### 目的

直接检验之前 neural reranker 失败是否因为：

```txt
BCE objective 不适合 ranking
epoch 不够
hard negatives 不够好
```

### 训练目标

输入：

```txt
claim, positive evidence, negative evidence
```

目标：

```txt
s(claim, positive) > s(claim, negative)
```

推荐 loss：

```txt
RankNet:
  -log sigmoid(s_pos - s_neg)

or margin ranking:
  max(0, margin - s_pos + s_neg)
```

### 负样本

每个 gold positive 配：

```txt
1 random negative
2 MiniLM high-score non-gold
2 RRF high-rank non-gold
1 entity/number-overlap hard negative
```

REFUTES positive 可 x2/x3 采样。

### 输出

```txt
outputs/round10/pairwise_reranker/train_pairs.jsonl
outputs/round10/pairwise_reranker/dev_metrics_by_epoch.csv
outputs/round10/pairwise_reranker/dev_predictions_top3.json
```

### 验收指标

| Metric | Target |
|---|---:|
| dev top3 evidence F | compare against `0.2105` |
| dev top20 macro recall | compare against Round09 blend context |
| REFUTES recall | target `>= 0.1420`, ideally `>= 0.1790` |
| pos-neg score gap | should increase across epochs |
| dev metric curve | required |

判断规则：

```txt
如果 epoch 1->3 持续上涨:
  之前可能训练不够

如果 train loss 降但 dev 不涨:
  objective / negative sampling 仍有问题

如果 REFUTES 不涨:
  reranker 仍只是学 relevance，不是 verification
```

## Stage E：Hybrid Verifier/Reranker Scoring

### 目的

将 verifier signal 注入当前最强 ranking pipeline。

当前最强：

```txt
Round09 blend alpha=0.4
top3 evidence F = 0.2105
REFUTES recall = 0.1420
```

Hybrid score：

```txt
score =
  alpha * MiniLM_score
  beta  * GBDT_score
  gamma * verifier_score
```

其中：

```txt
verifier_score = max(p_support, p_refute)
```

或：

```txt
verifier_score = 1 - p_neutral
```

### 权重搜索

不要大网格。

固定 Round09 最优附近：

```txt
MiniLM/GBDT blend alpha_minilm = 0.4
```

只扫：

```txt
gamma = 0.2, 0.4, 0.6
```

### 输出

```txt
outputs/round10/hybrid/hybrid_summary.csv
outputs/round10/hybrid/best_hybrid_predictions.json
```

### 验收指标

| Metric | Target |
|---|---:|
| top3 evidence F | `>= 0.2105` preferred |
| REFUTES recall | `> 0.1420`, ideally `>= 0.1790` |
| top20 recall | report |
| assignment harmonic mean | report |

判断规则：

```txt
如果 hybrid improves REFUTES without hurting F:
  verifier signal 应进入最终系统

如果 hybrid hurts F but improves REFUTES:
  报告 tradeoff，保留为 REFUTES-oriented ablation

如果 hybrid no improvement:
  verifier signal 当前训练不足
```

## Stage F：Epoch / Objective Diagnostic

### 目的

回答用户关心的问题：

```txt
之前 neural model 不好，是 epoch 不够，还是 objective 错？
```

### 实验

对 neural verifier 或 pairwise reranker 跑学习曲线：

```txt
epochs = 1, 2, 3, 5
lr = 2e-5
```

每个 epoch 记录：

```txt
train loss
dev loss
dev top3 evidence F
dev top20 recall
REFUTES recall
pair-level macro-F1
positive-negative score gap
```

### 输出

```txt
outputs/round10/learning_curve.csv
```

### 验收指标

必须能分类判断：

| Pattern | Interpretation |
|---|---|
| train loss 降，dev F 涨 | epoch 不够，可以继续训练 |
| train loss 降，dev F 不涨 | overfit / objective mismatch |
| train loss 不降 | LR / batch / label construction 有问题 |
| REFUTES 不涨 | 缺 stance/contradiction supervision |

## Stage G：Round10 Final Report

### 输出

```txt
agent_docs/rounds/round10/round10_report_b_execution_results.md
docs/round10_neural_baseline_report.md
```

### 最终验收表

Round10 final report 至少要包含：

| System | Evidence F | REFUTES Recall | Accuracy | Macro-F1 | Harmonic Mean |
|---|---:|---:|---:|---:|---:|
| Round09 blend alpha0.4 | 0.2105 | 0.1420 | current | current | current |
| Neural concat classifier | new | new | new | new | new |
| Neural verifier aggregation | new | new | new | new | new |
| Neural pairwise reranker | new | new | new | new | new |
| Hybrid verifier reranker | new | new | new | new | new |

## Round10 本阶段验收指标

Round10 不是必须所有 neural 模型都赢；它的验收标准是：

```txt
1. 至少完成一个 transformer claim classifier baseline。
2. 至少完成一个 claim-evidence verifier 或 pairwise neural reranker。
3. 产出和 Round09 blend 的直接对比表。
4. 明确 neural model 是否解决 REFUTES。
5. 明确 epoch/objective/negative sampling 哪个更可能是瓶颈。
6. 形成可写进最终报告的 neural baseline section。
```

最低验收：

```txt
Transformer concat classifier + verifier dataset/first verifier run + final comparison table
```

理想验收：

```txt
Verifier + hybrid scoring 在 REFUTES recall 或 top3 F-score 上超过 Round09 blend
```

## 当前执行建议

优先级排序：

```txt
1. Build verifier pair dataset
2. Train neural verifier
3. Use verifier score for evidence reranking and claim aggregation
4. Run lightweight transformer concat classifier
5. If time remains, train pairwise neural reranker
6. Hybrid score sweep
```

不要优先做：

```txt
large model grid
huge epoch/LR grid
top50 full concat classifier
dense retrieval
more GBDT alpha tuning
```

