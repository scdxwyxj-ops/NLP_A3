# Round09 索引

## 状态

Round09 已完成本轮执行，目标是验证 Round08 后续建议：

```txt
MiniLM / GBDT alpha-blend
MiniLM scoring scope top100
gain/loss analysis
REFUTES-specific calibration
classifier context check
```

## 当前最佳结果

最佳 final evidence output：

```txt
MiniLM top100 scope
+ REFUTES x2 Fusion GBDT
+ alpha blend with alpha_minilm = 0.4/0.5/0.6

top3 evidence F-score:
  0.2105
```

对比：

```txt
Round07 MiniLM-only:
  0.1987

Round08 pure Fusion GBDT:
  0.2011

Round09 best blend:
  0.2105
```

## 重要限制

Round09 best blend 的 `REFUTES recall = 0.1420`，仍低于 MiniLM-only 的 `0.1790`。

因此：

```txt
best blend improves overall evidence F-score,
but does not fully solve REFUTES.
```

## 当前报告

- `round_09_report_a_execution_and_acceptance.md`

