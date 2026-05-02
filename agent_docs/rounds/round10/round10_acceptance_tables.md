# Round10 验收表格汇总

日期：2026-05-01

## 表 1：Round10 总体验收对比

| System | Evidence F | REFUTES Recall | Accuracy | Macro-F1 | Harmonic Mean | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Round09 blend alpha0.4 | 0.2105 | 0.1420 | 0.4481 | 0.4142 | 0.2865 | 当前最强 final evidence selector |
| Round08 TF-IDF classifier baseline | 0.2011 | 0.0864 | 0.4675 | 0.4376 | 0.2812 | 旧 classifier baseline |
| DistilRoBERTa concat classifier | 0.2011 | n/a | 0.5260 | 0.4543 | 0.2910 | 当前最强 claim classifier |
| Final decoupled system | 0.2105 | 0.1420 | 0.5260 | 0.4543 | 0.3007 | 当前推荐最终组合 |
| DistilRoBERTa verifier top3 | 0.1029 | 0.0340 | 0.4545 | n/a | 0.1678 | 不适合作为 final top3 reranker |
| Hybrid verifier gamma0.2 top3 | 0.2038 | 0.1235 | 0.4416 | n/a | 0.2789 | 未超过 Round09 blend |
| Hybrid verifier gamma0.6 top20 | 0.1206 | 0.4568 | 0.4416 | n/a | 0.1895 | REFUTES context recall 高，但 final precision 低 |

## 表 2：Transformer Claim Classifier 学习曲线

模型：

```txt
distilroberta-base
claim + Fusion GBDT top10 evidence -> 4-way claim label
```

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro-F1 | 判断 |
|---:|---:|---:|---:|---:|---|
| 1 | 1.2673 | 1.2907 | 0.4481 | 0.1658 | 还没学好 |
| 2 | 1.2411 | 1.2442 | 0.4416 | 0.1532 | 仍弱 |
| 3 | 1.1807 | 1.2973 | 0.3961 | 0.2321 | 波动 |
| 4 | 1.0770 | 1.2019 | 0.4805 | 0.3425 | 开始有效 |
| 5 | 0.8771 | 1.3339 | 0.5260 | 0.4543 | 超过 TF-IDF baseline |

结论：

```txt
concat classifier 需要训练到 5 epoch 才超过 TF-IDF baseline。
这里“epoch 不够”是一个合理解释。
```

## 表 3：Claim Classifier Baseline 对比

| Classifier | Evidence Context | Accuracy | Macro-F1 | Evidence F | Harmonic Mean | 结论 |
|---|---|---:|---:|---:|---:|---|
| TF-IDF Logistic Regression | Fusion GBDT top50 | 0.4675 | 0.4376 | 0.2011 | 0.2812 | 强非神经 baseline |
| DistilRoBERTa concat classifier | Fusion GBDT top10 | 0.5260 | 0.4543 | 0.2011 | 0.2910 | 最好 classifier |
| Final decoupled system | Round09 blend evidence + Fusion-label DistilRoBERTa | 0.5260 | 0.4543 | 0.2105 | 0.3007 | 最好 official harmonic |

结论：

```txt
Round10 成功补齐 neural classifier baseline，并且超过 TF-IDF baseline。
```

## 表 4：Verifier Dataset 规模

Pair label 构造：

```txt
SUPPORTS claim + gold evidence -> SUPPORT
REFUTES claim + gold evidence -> REFUTE
retrieved non-gold / random evidence -> NEUTRAL
DISPUTED gold evidence 暂不作为 pair-level positive
```

| Split | Rows | SUPPORT | REFUTE | NEUTRAL | 说明 |
|---|---:|---:|---:|---:|---|
| train | 14,537 | 1,343 | 914 | 12,280 | REFUTE 已上采样 |
| dev | 1,825 | 171 | 114 | 1,540 | 用于 pair-level 验证 |

## 表 5：Neural Verifier 学习曲线

模型：

```txt
distilroberta-base
claim + single evidence -> SUPPORT / REFUTE / NEUTRAL
```

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro-F1 | 判断 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.4993 | 0.5031 | 0.8395 | 0.3244 | neutral 主导，macro-F1 低 |
| 2 | 0.3911 | 0.4737 | 0.8367 | 0.4091 | 明显改善 |
| 3 | 0.2778 | 0.5973 | 0.8318 | 0.4635 | pair signal 继续变强 |
| 4 | 0.2057 | 0.6133 | 0.8230 | 0.5027 | 最佳 macro-F1 |
| 5 | 0.1572 | 0.7176 | 0.8247 | 0.4404 | 开始过拟合 |

结论：

```txt
Verifier 最好在 epoch 4。
epoch 5 train loss 继续下降，但 dev loss 上升、macro-F1 下降，说明过拟合。
```

## 表 6：Verifier 作为 Evidence Reranker 的结果

Verifier score：

```txt
score = max(p_support, p_refute)
```

| k | Evidence F | Claim Accuracy | Macro Recall | REFUTES Recall | Harmonic Mean | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.1029 | 0.4545 | 0.1048 | 0.0340 | 0.1678 | top3 很弱 |
| 20 | 0.1028 | 0.4481 | 0.4105 | 0.4074 | 0.1672 | REFUTES context recall 有用 |
| 50 | 0.0611 | 0.4481 | 0.5475 | 0.5309 | 0.1076 | 宽召回强，但 precision 低 |

结论：

```txt
Verifier 不适合直接替代 final top3 reranker。
它更适合做 REFUTES/context diagnostic 或后续 aggregation feature。
```

## 表 7：Verifier Hybrid Scoring

Hybrid score：

```txt
score = (1 - gamma) * Round09_blend_score + gamma * verifier_score
```

| gamma | k | Evidence F | Macro Recall | REFUTES Recall | Harmonic Mean | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 0.2 | 3 | 0.2038 | 0.2352 | 0.1235 | 0.2789 | 没超过 Round09 top3 |
| 0.4 | 3 | 0.1987 | 0.2272 | 0.1173 | 0.2741 | 下降 |
| 0.6 | 3 | 0.1788 | 0.1975 | 0.0679 | 0.2546 | 明显下降 |
| 0.8 | 3 | 0.1312 | 0.1437 | 0.0340 | 0.2022 | 不可用 |
| 0.2 | 20 | 0.1220 | 0.4890 | 0.4074 | 0.1911 | context recall 提高 |
| 0.6 | 20 | 0.1206 | 0.4926 | 0.4568 | 0.1895 | REFUTES top20 recall 高 |
| 0.8 | 20 | 0.1180 | 0.4873 | 0.4938 | 0.1862 | REFUTES 高但 precision 低 |

结论：

```txt
Verifier signal 注入越多，top20/REFUTES recall 越强，但 top3 precision 越差。
Hybrid 当前不适合作为 final evidence selector。
```

## 表 8：Round10 验收项完成情况

| 验收项 | 状态 | 证据 |
|---|---|---|
| 完成 transformer concat classifier baseline | 完成 | DistilRoBERTa top10，macro-F1 `0.4543` |
| 完成 claim-evidence verifier dataset | 完成 | train `14,537` rows，dev `1,825` rows |
| 完成 first verifier run | 完成 | verifier epoch4 pair macro-F1 `0.5027` |
| 与 Round09 blend 直接对比 | 完成 | Round10 验收表已列出 |
| 判断 neural model 是否解决 REFUTES | 完成 | top3 未解决；top20/top50 REFUTES recall 有帮助 |
| 判断 epoch/objective 问题 | 完成 | classifier 需要更多 epoch；verifier epoch5 过拟合；ranking score 设计不够好 |

## 表 9：当前推荐系统组合

| 组件 | 推荐方案 | 理由 |
|---|---|---|
| Final evidence selector | Round09 blend alpha0.4 | 当前最高 evidence F `0.2105` |
| Claim classifier | DistilRoBERTa concat classifier over Fusion GBDT top10 | 当前最高 accuracy `0.5260` / macro-F1 `0.4543` |
| Verifier | 暂作 diagnostic / REFUTES context supplement | top3 弱，但 top50 REFUTES recall 高 |

## 交流用简短结论

```txt
Round10 added neural baselines. The transformer concat classifier worked:
DistilRoBERTa over top10 evidence improved macro-F1 from 0.4376 to 0.4543.

The neural verifier learned pair-level SUPPORT/REFUTE/NEUTRAL signals, with
best pair macro-F1 around 0.5027, but it is not a good final top3 reranker yet.
It improves wide-context REFUTES recall, so it is useful as a diagnostic or
future aggregation feature.

For now, the best system should separate evidence selection and label prediction:
Round09 blend for final evidence, DistilRoBERTa concat classifier for claim label.
```
