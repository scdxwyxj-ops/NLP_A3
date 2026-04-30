# Round05 Report G：Retrieval error analysis for downstream classifier

报告时间：`2026-04-30`

## 一、目的

当前 retrieval 指标虽然比 BM25 baseline 好，但还不足以保证最终 claim classification 做好。

因此本阶段输出 TP/FP/FN/TN 细分结果，帮助判断：
- 哪些方法适合作为 final evidence output。
- 哪些方法适合作为 classifier input。
- retrieval 的主要失败类型是什么。

## 二、输出目录

新增分析目录：

```txt
outputs/round05_analysis/
```

主要文件：

```txt
outputs/round05_analysis/summary.csv
outputs/round05_analysis/bm25_top5/claim_confusion.csv
outputs/round05_analysis/bm25_top5/examples.md
outputs/round05_analysis/minilm_zero_top3/claim_confusion.csv
outputs/round05_analysis/minilm_zero_top3/examples.md
outputs/round05_analysis/minilm_zero_top5/claim_confusion.csv
outputs/round05_analysis/minilm_zero_top5/examples.md
outputs/round05_analysis/minilm_zero_top20/claim_confusion.csv
outputs/round05_analysis/minilm_zero_top20/examples.md
```

TN 说明：

```txt
TN 是在 BM25 top-50 candidate pool + gold evidence 内计算的。
```

如果按全 evidence corpus 计算 TN，数值会非常大，对错误分析帮助不大。

## 三、整体 TP/FP/FN/TN

| method | TP | FP | FN | TN | precision | recall | hit-any |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 top-5 | 50 | 720 | 441 | 6828 | 0.0649 | 0.1018 | 0.2792 |
| MiniLM zero-shot top-3 | 78 | 384 | 413 | 7164 | 0.1688 | 0.1589 | 0.3961 |
| MiniLM zero-shot top-5 | 102 | 668 | 389 | 6880 | 0.1325 | 0.2077 | 0.4545 |
| MiniLM zero-shot top-20 | 142 | 2938 | 349 | 4610 | 0.0461 | 0.2892 | 0.5519 |
| MiniLM BCE top-5 | 67 | 703 | 424 | 6845 | 0.0870 | 0.1365 | 0.3377 |

## 四、核心观察

MiniLM zero-shot top-3 适合 final evidence output：
- precision 明显高于 BM25 top-5。
- final `eval.py` F-score 当前最高。

MiniLM zero-shot top-20 更适合 classifier input：
- recall 和 hit-any 更高。
- 但 FP 太多，不适合直接作为 final evidence output。

因此后续系统应分离两个接口：

```txt
final evidence output: MiniLM top-3
classifier input context: MiniLM top-10 / top-20 / top-50
```

## 五、按 label 的 retrieval 弱点

MiniLM zero-shot top-20：

| label | precision | recall | hit-any |
|---|---:|---:|---:|
| DISPUTED | 0.061 | 0.379 | 0.722 |
| NOT_ENOUGH_INFO | 0.062 | 0.249 | 0.659 |
| REFUTES | 0.019 | 0.175 | 0.259 |
| SUPPORTS | 0.043 | 0.345 | 0.559 |

最弱的是 `REFUTES`。

这说明当前 lexical retrieval + reranker 更容易找到主题相关 evidence，但不一定能找到真正构成反驳的 evidence。后续 classifier 如果 label prediction 想提升，必须特别关注 REFUTES 的 evidence recall。

## 六、对最终分类任务的建议

最终目标是 classification 最好，而不是 retrieval 独立最好。

建议下一步：

1. 先把 retrieval API 输出两份：
   - `evidence_for_submission`: MiniLM top-3
   - `evidence_for_classifier`: MiniLM top-20 or top-50
2. 训练 classifier 时不要只拼 top-3 evidence。
3. 对 classifier 输入做 ablation：
   - top-3
   - top-5
   - top-10
   - top-20
4. 单独分析 REFUTES 分类错误，判断是 retrieval miss 还是 classifier reasoning miss。

## 七、当前判断

现在效果“不够好”的根本原因不是单一模型坏，而是 pipeline 的 recall ceiling 偏低。

当前最重要的工程方向：

```txt
提高 candidate recall
-> 给 classifier 更多有用证据
-> final evidence output 仍保持高 precision
```

## 八、补充观察：topic relevance 不等于 evidence relevance

`claim-375` 的错误案例显示，MiniLM 选出的 FP evidence 并不是完全离题；它们经常和 claim 有相同主题，例如 Australia、carbon dioxide、emissions。

问题是这些 evidence 只是 topic relevant，不一定能支持 fact-checking label decision。

因此后续需要区分：

```txt
topic relevance:
文本主题和 claim 相似。

evidence relevance:
文本能支持、反驳或帮助判断 claim label。
```

这说明 zero-shot reranker 只是强 baseline，不是最终 task-aware evidence selector。下一步应改 hard negative 构造，让模型学习把“主题相关但证据价值不足”的 passage 排低。
