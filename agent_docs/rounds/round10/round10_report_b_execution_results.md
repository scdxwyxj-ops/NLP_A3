# Round10 报告 B：Neural Baseline 执行结果

日期：2026-05-01

## 一句话结论

Round10 已完成本轮最低验收目标：补齐 transformer claim classifier baseline，
构造并训练 claim-evidence neural verifier，并测试 verifier hybrid scoring。

最重要结果：

```txt
DistilRoBERTa concat classifier top10
accuracy = 0.5260
macro-F1 = 0.4543
```

它超过了当前 TF-IDF logreg classifier baseline：

```txt
Round08 Fusion GBDT top50 + TF-IDF logreg
accuracy = 0.4675
macro-F1 = 0.4376
```

但 neural verifier 直接作为 final evidence reranker 很弱：

```txt
verifier top3 evidence F-score = 0.1029
```

说明 verifier 学到了一些 pair-level signal，但还不能直接替代 retrieval/reranking。

## Stage B：Transformer Concat Claim Classifier

脚本：

```txt
experiments/neural/train_concat_transformer_classifier.py
```

输入：

```txt
outputs/round08/train-classifier-context-fusion-gbdt-top50-semantic.jsonl
outputs/round08/dev-classifier-context-fusion-gbdt-top50-semantic.jsonl
```

设置：

```txt
model = distilroberta-base
context = top10 evidence from Round08 Fusion GBDT top50
max_length = 512
epochs = 5
batch_size = 8
lr = 2e-5
```

输出：

```txt
outputs/round10/claim_classifier/distilroberta_fusion_top10_e5/
```

学习曲线：

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 1.2673 | 1.2907 | 0.4481 | 0.1658 |
| 2 | 1.2411 | 1.2442 | 0.4416 | 0.1532 |
| 3 | 1.1807 | 1.2973 | 0.3961 | 0.2321 |
| 4 | 1.0770 | 1.2019 | 0.4805 | 0.3425 |
| 5 | 0.8771 | 1.3339 | 0.5260 | 0.4543 |

最终结果：

| Model | Context | Accuracy | Macro-F1 | Evidence F-score | Harmonic Mean |
|---|---|---:|---:|---:|---:|
| TF-IDF logreg | Fusion GBDT top50 | 0.4675 | 0.4376 | 0.2011 | 0.2812 |
| DistilRoBERTa | Fusion GBDT top10 | 0.5260 | 0.4543 | 0.2011 | 0.2910 |

结论：

- neural concat classifier 是有效 baseline；
- 它超过了 TF-IDF logreg 的 accuracy 和 macro-F1；
- 训练到 5 epoch 才明显超过 baseline，说明“epoch 不够”对 classifier baseline 是成立的。

## Stage C：Neural Claim-Evidence Verifier

### Dataset

脚本：

```txt
experiments/neural/build_verifier_dataset.py
```

输出：

```txt
outputs/round10/verifier/pair_dataset_train.jsonl
outputs/round10/verifier/pair_dataset_dev.jsonl
```

构造规则：

```txt
SUPPORTS claim + gold evidence -> SUPPORT
REFUTES claim + gold evidence -> REFUTE
retrieved non-gold / random evidence -> NEUTRAL
DISPUTED gold evidence 暂不作为 pair-level positive
```

数据规模：

| Split | Rows | SUPPORT | REFUTE | NEUTRAL |
|---|---:|---:|---:|---:|
| train | 14,537 | 1,343 | 914 | 12,280 |
| dev | 1,825 | 171 | 114 | 1,540 |

### Training

脚本：

```txt
experiments/neural/train_pair_verifier.py
```

设置：

```txt
model = distilroberta-base
epochs = 5
batch_size = 16
max_length = 256
lr = 2e-5
```

输出：

```txt
outputs/round10/verifier_distilroberta_e5/
```

Pair-level 学习曲线：

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.4993 | 0.5031 | 0.8395 | 0.3244 |
| 2 | 0.3911 | 0.4737 | 0.8367 | 0.4091 |
| 3 | 0.2778 | 0.5973 | 0.8318 | 0.4635 |
| 4 | 0.2057 | 0.6133 | 0.8230 | 0.5027 |
| 5 | 0.1572 | 0.7176 | 0.8247 | 0.4404 |

结论：

- pair-level verifier 在 epoch 4 达到最佳 macro-F1 `0.5027`；
- epoch 5 train loss 继续下降但 dev loss 变差，说明已经开始过拟合；
- 对 verifier 来说，不是简单“越多 epoch 越好”，需要 early stopping。

## Stage C.2：Verifier as Reranker / Aggregator

使用 epoch 5 verifier 对 dev top50 evidence 打分，score 定义为：

```txt
score = max(p_support, p_refute)
```

结果：

| k | Evidence F-score | Claim Accuracy | Macro Recall | REFUTES Recall | Harmonic Mean |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.1029 | 0.4545 | 0.1048 | 0.0340 | 0.1678 |
| 20 | 0.1028 | 0.4481 | 0.4105 | 0.4074 | 0.1672 |
| 50 | 0.0611 | 0.4481 | 0.5475 | 0.5309 | 0.1076 |

解读：

- Verifier top3 很弱，不能替代 Round09 final evidence selector。
- Verifier top20/top50 的 REFUTES recall 明显高，说明它更适合宽 context 或 REFUTES-oriented supplement。
- 直接用 `max(p_support, p_refute)` 排 top3 不够好，需要更好的 hybrid scoring 或 threshold calibration。

## Stage E：Hybrid Verifier Scoring

Hybrid score：

```txt
score = (1 - gamma) * Round09_blend_score + gamma * verifier_score
```

输出：

```txt
outputs/round10/hybrid_verifier_e5/hybrid_summary.csv
```

结果：

| gamma | k | Evidence F-score | Macro Recall | REFUTES Recall | Harmonic Mean |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 3 | 0.2038 | 0.2352 | 0.1235 | 0.2789 |
| 0.4 | 3 | 0.1987 | 0.2272 | 0.1173 | 0.2741 |
| 0.6 | 3 | 0.1788 | 0.1975 | 0.0679 | 0.2546 |
| 0.8 | 3 | 0.1312 | 0.1437 | 0.0340 | 0.2022 |
| 0.2 | 20 | 0.1220 | 0.4890 | 0.4074 | 0.1911 |
| 0.6 | 20 | 0.1206 | 0.4926 | 0.4568 | 0.1895 |
| 0.8 | 20 | 0.1180 | 0.4873 | 0.4938 | 0.1862 |

结论：

- verifier hybrid 没有超过 Round09 top3 F-score `0.2105`；
- verifier signal 会提高 top20 macro recall / REFUTES recall；
- gamma 越大，top3 越差，说明 verifier score 不适合直接控制 final top3；
- verifier 更适合作为 classifier context 或 REFUTES supplement。

## Round10 验收表

| System | Evidence F | REFUTES Recall | Accuracy | Macro-F1 | Harmonic Mean |
|---|---:|---:|---:|---:|---:|
| Round09 blend alpha0.4 | 0.2105 | 0.1420 | 0.4481 | 0.4142 | 0.2865 |
| Round08 TF-IDF classifier baseline | 0.2011 | 0.0864 | 0.4675 | 0.4376 | 0.2812 |
| DistilRoBERTa concat classifier | 0.2011 | n/a | 0.5260 | 0.4543 | 0.2910 |
| DistilRoBERTa verifier top3 | 0.1029 | 0.0340 | 0.4545 | n/a | 0.1678 |
| Hybrid gamma0.2 top3 | 0.2038 | 0.1235 | 0.4416 | n/a | 0.2789 |
| Hybrid gamma0.6 top20 | 0.1206 | 0.4568 | 0.4416 | n/a | 0.1895 |

## Round10 验收判断

Round10 最低验收已达成：

1. 已完成 transformer concat claim classifier baseline。
2. 已完成 claim-evidence verifier dataset 和 first verifier run。
3. 已产出与 Round09 blend 的直接对比表。
4. 已明确 neural verifier 没有解决 top3 REFUTES，但能提高 top20/top50 REFUTES recall。
5. 已通过 learning curve 判断：
   - concat classifier 需要更多 epoch，5 epoch 才超过 TF-IDF baseline；
   - verifier epoch 4 最好，epoch 5 过拟合；
   - verifier objective 学到 pair signal，但 ranking score 设计不够好。

## 当前建议

### Final Evidence Selector

继续使用：

```txt
Round09 blend alpha0.4
```

因为它仍是当前最强 top3 evidence F-score：

```txt
0.2105
```

### Claim Classifier

使用：

```txt
DistilRoBERTa concat classifier over Fusion GBDT top10
```

因为它目前有最高 classifier macro-F1：

```txt
0.4543
```

### Verifier

不要直接用 verifier 替代 final reranker。

更合理用途：

```txt
1. 作为 top20/top50 classifier context supplement；
2. 作为 REFUTES-oriented diagnostic；
3. 后续用 early stopping epoch 4，并重新设计 aggregation / hybrid score。
```

## 下一步

如果继续推进，最值得做：

```txt
1. 保存/复用 verifier epoch4 checkpoint，而不是 epoch5。
2. 用 verifier probabilities 训练一个 aggregation classifier，而不是手写 max-threshold。
3. 对 concat classifier 做 seed repeat 或 DeBERTa-v3-small 对比。
4. 最终系统可拆分：Round09 blend 负责 evidence top3，DistilRoBERTa concat classifier 负责 claim label。
```

