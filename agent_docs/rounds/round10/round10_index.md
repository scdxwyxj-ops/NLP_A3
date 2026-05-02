# Round10 索引

## 状态

Round10 本轮已完成最低验收：已补齐 transformer concat classifier baseline，
并完成 claim-evidence verifier dataset、verifier first run 和 hybrid scoring diagnostic。
后续补充了完整实验报告：将 Round09 最强 evidence selector 与 Round10
DistilRoBERTa classifier 组合评估，当前 final dev harmonic mean 提升到 `0.3007`。

Round09 当前最强结果：

```txt
final evidence:
  MiniLM top100 scope
  + REFUTES x2 Fusion GBDT
  + alpha blend alpha_minilm = 0.4

top3 evidence F-score:
  0.2105

REFUTES recall:
  0.1420
```

关键问题：

```txt
Round09 提升了 overall evidence F-score，
但 REFUTES recall 仍低于 MiniLM-only 的 0.1790。
```

## Discovery Planner 状态

已按要求调用 discovery-planner：

```txt
python /mnt/a/code/agent_dev/discovery_planner/scripts/discovery_planner.py ...
```

结果：

```txt
Codex discovery failed with exit code 127
```

因此本轮继续按 discovery-planner 的结构手动规划，并记录在 Round10 文档中。

## 当前报告

- `round10_report_a_stage_plan_and_acceptance.md`
- `round10_report_b_execution_results.md`
- `round10_acceptance_tables.md`
- `round10_report_c_full_experiment_paper.md`

## Stage 总览

| Stage | 名称 | 目的 | 主要输出 | 验收状态 |
|---|---|---|---|---|
| A | Neural baseline planning | 明确神经模型实验边界和数据构造 | stage plan + acceptance metrics | done |
| B | Transformer claim classifier | claim + top-k evidence -> 4-way label | classifier baseline table | done |
| C | Neural pair verifier | claim + evidence -> SUPPORT/REFUTE/NEUTRAL | pair-level + aggregation results | done |
| D | Pairwise neural reranker | claim, gold evidence, hard negative -> ranking objective | neural reranker table | deferred |
| E | Verifier/reranker hybrid | MiniLM + GBDT + verifier score 融合 | hybrid scoring table | done |
| F | Epoch/objective diagnostic | 判断 epoch 不够还是 objective 错 | learning curve table | done |
| G | Final Round10 report | 总结神经 baseline 是否有效 | acceptance report | done |
| H | Full experiment paper report | 组合最佳 evidence + classifier 并可视化完整故事 | final system table + figures + paper report | done |

## 当前最佳神经结果

```txt
DistilRoBERTa concat classifier over Fusion GBDT top10
accuracy = 0.5260
macro-F1 = 0.4543
```

对比原 TF-IDF classifier baseline：

```txt
accuracy = 0.4675
macro-F1 = 0.4376
```

Verifier 结论：

```txt
pair-level verifier epoch4 macro-F1 = 0.5027
verifier top3 evidence F-score = 0.1029
verifier top50 REFUTES recall = 0.5309
```

因此 verifier 不适合直接替代 final reranker，但适合做 REFUTES/context diagnostic。

## 当前最终组合结果

```txt
Evidence selector:
  Round09 blend alpha=0.4 top3

Claim classifier:
  DistilRoBERTa concat classifier

Dev:
  Evidence F-score = 0.2105
  Claim accuracy = 0.5260
  Harmonic mean = 0.3007
```

## 本轮核心验收指标

Round10 不以“必须超过所有旧结果”为唯一验收标准，而以是否回答以下问题为验收：

```txt
1. neural classifier 是否超过 TF-IDF logreg macro-F1 0.4376？
2. neural verifier 是否改善 REFUTES recall 或 claim macro-F1？
3. pairwise neural reranker 是否超过 Round09 top3 F-score 0.2105？
4. hybrid score 是否在不牺牲 REFUTES 的情况下提高 evidence F-score？
5. epoch curve 是否能判断之前训练失败是 epoch 不够、objective 错，还是负样本构造问题？
```
