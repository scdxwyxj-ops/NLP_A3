# Discussion: Candidate Recall 表格怎么理解

日期：2026-05-01

## 这张表在回答什么问题？

这张表不是 final score 表，也不是 classifier 结果表。

它只回答一个问题：

```txt
在 reranker/classifier 之前，第一阶段 retrieval 能不能把 gold evidence 找进候选池？
```

也就是说，这张表评估的是 candidate generation 的召回能力。它的核心价值是判断：

- gold evidence 有没有机会进入后续 reranking；
- 哪个 candidate source 更适合作为 MiniLM reranker 的输入；
- 哪个 candidate source 更适合作为 classifier 的宽 evidence context。

如果 gold evidence 在这个阶段就没进候选池，后面的 reranker 再强也救不回来。

## 每一列是什么意思？

| 列名 | 含义 |
|---|---|
| Candidate Source | 第一阶段候选证据来自哪里，例如 BM25 或 BM25+TFIDF |
| N | 每个 claim 保留多少条候选 evidence |
| Macro Recall | 对每个 claim 单独算 recall 后再平均，衡量平均每个 claim 找回多少 gold evidence |
| Hit-any | 有至少一条 gold evidence 被找回的 claim 比例 |
| All-gold | 所有 gold evidence 都被找回的 claim 比例 |
| SUPPORTS Recall | SUPPORTS 类 claim 的平均 candidate recall |
| REFUTES Recall | REFUTES 类 claim 的平均 candidate recall |
| NEI Recall | NOT_ENOUGH_INFO 类 claim 的平均 candidate recall |
| DISPUTED Recall | DISPUTED 类 claim 的平均 candidate recall |

## 为什么 Macro Recall 很重要？

Macro Recall 是这张表的主指标。

原因是每个 claim 的 gold evidence 数量不一样。Macro Recall 先对每个 claim 算：

```txt
找回的 gold evidence 数量 / 该 claim 的 gold evidence 总数
```

然后再对所有 claim 平均。

这比只看总 TP 更公平，因为它不会让 gold evidence 特别多的 claim 主导结果。

## 为什么 Hit-any 也重要？

Hit-any 表示：

```txt
这个 claim 至少找回了一条正确 evidence 吗？
```

这个指标对 classifier 特别重要。哪怕没有找全 gold evidence，只要找回一条关键 evidence，
分类器就可能获得有用信号。

例如：

```txt
BM25 top500 hit-any = 0.8377
BM25+TFIDF top500 hit-any = 0.8896
```

这说明 BM25+TFIDF 能让更多 claim 至少看到一条正确证据。

## 为什么 All-gold 也要看？

All-gold 表示：

```txt
这个 claim 的所有 gold evidence 是否都进入候选池？
```

这个指标更严格。它对 final evidence output 更重要，因为最终评分会看 evidence set。

当前结果：

```txt
BM25 top500 all-gold = 0.3247
BM25+TFIDF top500 all-gold = 0.4091
```

说明 BM25+TFIDF 不只是找回“一点相关证据”，而是更经常把完整 gold evidence set
放进候选池。

## 为什么要分 label 看 recall？

因为这个任务里不同 label 的 retrieval 难度不一样。

尤其是 `REFUTES`，之前一直比较弱。很多 false positive 是主题相关，但不能真正反驳 claim。

当前结果很关键：

```txt
BM25 top100 REFUTES recall = 0.2716
BM25 top500 REFUTES recall = 0.5123
BM25+TFIDF top500 REFUTES recall = 0.5926
```

这说明扩大候选池和加入 char TF-IDF 对 `REFUTES` 有明显帮助。

## 为什么 BM25+TFIDF 比 BM25 更好？

BM25 擅长词级 lexical matching，但它容易漏掉：

- 拼写变体；
- 数字、单位、百分比附近的局部匹配；
- claim 和 evidence 表达方式不完全一致的情况；
- 一些 entity 或 phrase 的碎片匹配。

char TF-IDF 能补一部分这类问题。用 RRF 合并 BM25 和 char TF-IDF 后，候选池更丰富。

最关键对比：

| 方法 | N | Macro Recall | Hit-any | All-gold | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| BM25 | 500 | 0.5861 | 0.8377 | 0.3247 | 0.5123 |
| BM25+TFIDF | 500 | 0.6579 | 0.8896 | 0.4091 | 0.5926 |

所以第一天最重要的结论是：

```txt
BM25+TFIDF 的候选召回明显优于 BM25 alone。
```

## 为什么 BM25+TFIDF 的 N=200 也有意义？

N=500 recall 更高，但 reranking 成本也更高。

N=200 是一个中间点：

```txt
BM25 top200 macro recall = 0.4878
BM25+TFIDF top200 macro recall = 0.5361
```

这说明在候选数不扩大到 500 的情况下，BM25+TFIDF 仍然比 BM25 更强。

但如果目标是 classifier context 高召回，N=500 更有价值。

## dense 那一行为什么是 not run？

`BM25+TFIDF+dense` 目前没有真实实验结果，所以不能填数字。

Dense retrieval 需要额外运行：

```txt
encode all evidence passages
encode each claim
nearest-neighbor search
RRF(BM25, TFIDF, dense)
evaluate recall
```

当前 evidence corpus 有约 120 万条 evidence。第一次跑 dense embedding 会比较慢，
而且需要安装和缓存模型。因此 dense 不能作为第一天“已完成结果”。

正确写法是：

```txt
BM25+TFIDF+dense: not run
```

不要用估计值替代真实实验结果。

## 最终解释给队友可以怎么说？

可以这样简短解释：

```txt
This table evaluates the first-stage candidate retrieval, not the final model.
The goal is to check whether the gold evidence is included before reranking.

BM25 improves as we increase N, but BM25+TFIDF performs better overall.
At N=500, BM25+TFIDF reaches 0.6579 macro recall, 0.8896 hit-any,
0.4091 all-gold, and 0.5926 REFUTES recall.

So our current decision is to use BM25+TFIDF as the candidate pool for MiniLM
reranking and for building wider classifier contexts. Dense retrieval has not
been run yet, so that row should remain unfilled for now.
```

## 当前决策

第一天可交付结论：

```txt
Candidate generation:
  use RRF(BM25 top500, char TF-IDF top500)

Reason:
  better macro recall, hit-any, all-gold, and REFUTES recall than BM25 alone

Next:
  use this pool for reranking and classifier context
```

