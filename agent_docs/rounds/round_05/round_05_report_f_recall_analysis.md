# Round05 Report F：Top-k recall analysis

报告时间：`2026-04-30`

## 一、问题

如果后续 claim classifier 要做得准，retrieval 不能只看 final evidence F-score，还要看 top-k recall。

原因：
- final submission 需要少量高精度 evidence。
- classifier 需要看到足够多可能相关的 evidence。
- 如果 gold evidence 没进入候选集，classifier 后面很难补救。

## 二、指标定义

本报告记录四个 recall 相关指标：

- macro recall：每个 claim 的 gold evidence 命中比例，再对 claim 平均。
- micro recall：所有 gold evidence 作为整体的命中比例。
- hit-any：至少命中一个 gold evidence 的 claim 比例。
- all-gold：全部 gold evidence 都被命中的 claim 比例。

## 三、BM25 recall

dev set：

| method | macro recall | micro recall | hit-any | all-gold |
|---|---:|---:|---:|---:|
| BM25 top-1 | 0.0513 | 0.0387 | 0.1234 | 0.0195 |
| BM25 top-3 | 0.0763 | 0.0672 | 0.2013 | 0.0195 |
| BM25 top-5 | 0.1156 | 0.1018 | 0.2792 | 0.0325 |
| BM25 top-10 | 0.1687 | 0.1507 | 0.3701 | 0.0519 |
| BM25 top-20 | 0.2346 | 0.2159 | 0.4675 | 0.0779 |
| BM25 top-50 | 0.3249 | 0.3096 | 0.5844 | 0.1364 |
| BM25 top-100 | 0.4188 | - | - | - |

观察：
- BM25 top-50 只能覆盖约三分之一 gold evidence。
- BM25 top-100 的 macro recall 提升到 `0.4188`，说明扩大候选池是有意义的。

## 四、Zero-shot MiniLM reranker recall

基于 BM25 top-50 candidates：

| method | macro recall | micro recall | hit-any | all-gold |
|---|---:|---:|---:|---:|
| MiniLM zero-shot top-1 | 0.0979 | 0.0774 | 0.2468 | 0.0325 |
| MiniLM zero-shot top-3 | 0.1877 | 0.1589 | 0.3961 | 0.0714 |
| MiniLM zero-shot top-5 | 0.2332 | 0.2077 | 0.4545 | 0.0909 |
| MiniLM zero-shot top-10 | 0.2758 | 0.2607 | 0.5065 | 0.1104 |
| MiniLM zero-shot top-20 | 0.3035 | 0.2892 | 0.5519 | 0.1234 |
| MiniLM zero-shot top-50 | 0.3249 | 0.3096 | 0.5844 | 0.1364 |

观察：
- Reranker 对 top-3/top-5 recall 明显好于 BM25。
- top-50 recall 与 BM25 top-50 相同，因为 reranker 只是重排同一个候选池。
- 这证明 reranker 改善的是候选内部排序，不改善候选池覆盖率。

## 五、和 final F-score 的关系

Zero-shot MiniLM 的 final evidence F-score：

```txt
top-1 F = 0.1299
top-3 F = 0.1642
top-5 F = 0.1578
top-10 F = 0.1218
top-20 F = 0.0777
top-50 F = 0.0367
```

F-score 在 top-3 最好，top-k 继续变大后会下降，因为 evaluator 会惩罚多输出的无关 evidence。

但这不代表 classifier 只能看 top-3。比较合理的做法是：

```txt
classifier input: top-10 / top-20 / top-50 evidence candidates
final evidence output: reranker top-3 or top-5
```

## 六、当前结论

用户的判断是对的：为了最终分类更准，需要提高 recall。

当前建议：

1. 对 final evidence selection，先用 zero-shot MiniLM top-3，因为 dev F-score 当前最好。
2. 对 downstream classifier，输入不应只用 top-3；应尝试 top-10、top-20 或 top-50。
3. 下一阶段优先提升 candidate recall：
   - BM25 top-100。
   - BM25 + TF-IDF union。
   - query expansion。
   - dense retrieval 作为辅助候选源。

核心瓶颈：

```txt
reranker 能把召回到的 evidence 排得更好，
但不能恢复 candidate generator 没召回的 evidence。
```
