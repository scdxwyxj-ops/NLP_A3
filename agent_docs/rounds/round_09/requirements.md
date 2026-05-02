可以验收，但别把 Round08 说成“性能大提升”。它真正的价值是：**你终于把瓶颈定位清楚了**。`Fusion GBDT top3 F=0.2011` 只比 `MiniLM-only 0.1987` 高 `+0.0024`，这个提升太小，不能强行吹；但 `top20 context` 的 macro recall 从 `0.4540 -> 0.4669`、hit-any 从 `0.7143 -> 0.7403`，classifier macro-F1 从 `0.4242 -> 0.4376`，这些说明 feature fusion 对“给分类器提供更好上下文”是有价值的。

## 我更新后的判断

Round08 后，系统应该拆成两个分支：

```txt
final evidence output:
  仍然不要完全相信 Fusion GBDT，需要和 MiniLM / blend 对比

classifier context:
  Fusion GBDT top50 without semantic summary 是当前默认选择
```

最关键的问题是：**Fusion GBDT top3 的 REFUTES recall 从 MiniLM 的 0.1790 掉到 0.0864**。这很严重。它说明 GBDT 学到的排序信号对 overall F-score 有一点帮助，但会牺牲 REFUTES evidence。也就是说，GBDT 现在更像一个 context reranker，而不是可靠的 final evidence selector。

---

# 1. Round08 哪些结论成立？

成立的结论：

```txt
1. Error layer analysis 是有用的。
2. Cheap fusion top50 有正向信号。
3. Fusion 对 top20/top50 context 比对 top3 final output 更稳定。
4. Classifier 目前更喜欢 top50，而不是 top20。
5. Semantic summary 对 TF-IDF classifier 没帮助，甚至有害。
```

不应该过度声称的结论：

```txt
1. 不要说 GBDT 明显超过 MiniLM。
2. 不要说 fusion 解决了 REFUTES。
3. 不要说 semantic features 已经提升 classifier。
4. 不要把 top3 F-score +0.0024 当成强证据。
```

你现在最诚实的表述应该是：

```txt
Feature fusion provides a small but positive reranking signal, especially for wider classifier context.
However, its final top-3 improvement is marginal and it reduces REFUTES recall, so further calibration is needed before replacing MiniLM as the final evidence selector.
```

---

# 2. 一个需要修正的报告细节

你表里的 `Harmonic Mean` 看起来不是：

[
H(\text{Macro-F1}, \text{Evidence-F})
]

而是：

[
H(\text{Accuracy}, \text{Evidence-F})
]

比如 `0.4675` 和 `0.2011` 的 harmonic mean 约等于 `0.2812`。如果报告里写 harmonic mean，必须明确：

```txt
Harmonic Mean = harmonic mean of label accuracy and evidence F-score
```

否则别人会以为你用了 macro-F1 和 evidence F-score。这个小地方很容易被 marker 抓住。

---

# 3. 当前默认系统应该怎么选？

我建议当前系统暂定为：

```txt
retrieval:
  RRF(BM25 top500, char TF-IDF top500)

final evidence candidates:
  compare MiniLM-only top3 vs Fusion GBDT top3 vs alpha-blend top3

classifier context:
  Fusion GBDT top50 without semantic summary

classifier:
  TF-IDF LogisticRegression baseline
```

其中 final evidence 不要立刻锁死为 GBDT。原因很简单：

| Method      | top3 F | top3 REFUTES Recall |
| ----------- | -----: | ------------------: |
| MiniLM-only | 0.1987 |              0.1790 |
| Fusion GBDT | 0.2011 |              0.0864 |

如果最终指标只看 micro evidence F，也许 GBDT 可以用；但如果 REFUTES 影响 label classification 或人工评价，MiniLM 更稳。下一轮必须做 **MiniLM 与 GBDT 的 rank blending**。

---

# 4. Round09 第一优先级：不要直接继续 dense，先做 alpha-blend

现在最值得做的是：

[
s_{\text{blend}}(q,e)
=====================

\alpha \cdot \hat{s}*{\text{MiniLM}}(q,e)
+
(1-\alpha)\cdot \hat{s}*{\text{GBDT}}(q,e)
]

其中 (\alpha) 扫：

```txt
0.0, 0.1, 0.2, ..., 1.0
```

评估：

```txt
top3 evidence F
top3 macro recall
top3 hit-any
top3 REFUTES recall
top20 macro recall
classifier macro-F1
```

目标不是只最大化 top3 F，而是找一个不牺牲 REFUTES 的点：

```txt
top3 F >= 0.2011
REFUTES recall 接近或超过 MiniLM-only 0.1790
top20 context recall 不低于 Fusion GBDT
```

如果 alpha-blend 能做到，那它比纯 GBDT 更适合作为最终 reranker。

---

# 5. Round09 第二优先级：MiniLM scoring scope 从 top50 扩到 top100/top200

你现在的 fusion 是 cheap fusion：

```txt
RRF top500 -> MiniLM top50 -> fusion rerank top50
```

这个上限太低。Error layer 里有：

```txt
gold in top500 but not top50 = 58
gold evidence % = 0.1181
```

这部分 cheap fusion 永远救不了。

下一轮应该跑：

```txt
RRF top500 -> MiniLM score top100 -> fusion rerank top100
RRF top500 -> MiniLM score top200 -> fusion rerank top200
```

不要一上来 top500，成本可能太大。先 top100，再 top200。

验收表：

| Scope                | top3 F | top3 Hit-any | top3 REFUTES Recall | top20 Macro Recall | Runtime |
| -------------------- | -----: | -----------: | ------------------: | -----------------: | ------: |
| MiniLM top50 + GBDT  | 0.2011 |       0.4545 |              0.0864 |             0.4669 | current |
| MiniLM top100 + GBDT |    new |          new |                 new |                new |     new |
| MiniLM top200 + GBDT |    new |          new |                 new |                new |     new |

如果 top100 就能提升，top200 可以作为 ablation；如果 top100 没提升，别浪费时间继续 top500。

---

# 6. Round09 第三优先级：做 gain/loss analysis

现在你只知道 GBDT 总体略涨，但不知道它赢在哪、输在哪。必须做对比：

```txt
MiniLM top3 有 gold，GBDT top3 没有 gold
GBDT top3 有 gold，MiniLM top3 没有 gold
两者都有
两者都没有
```

按 label 分开：

```txt
SUPPORTS
REFUTES
NEI
DISPUTED
```

尤其看 REFUTES：

```txt
哪些 REFUTES 被 GBDT 从 top3 踢掉？
它们的 feature pattern 是什么？
GBDT 是否过度依赖 source_count / sparse rank / entity overlap？
```

同时输出 GBDT feature importance。你需要知道模型到底学了什么。否则下一轮调参就是盲调。

建议脚本：

```txt
experiments/analysis/compare_minilm_vs_fusion_errors.py
```

输出：

```txt
outputs/round09/minilm_vs_gbdt_gain_loss_by_label.csv
outputs/round09/minilm_vs_gbdt_refutes_lost_examples.jsonl
outputs/round09/gbdt_feature_importance.csv
```

---

# 7. Round09 第四优先级：REFUTES-specific calibration

现在 REFUTES 是明显被 GBDT 伤害的类别。你有三个现实做法。

## 做法 A：训练时给 REFUTES positives 加权

训练 fusion reranker 时：

```txt
if claim_label == REFUTES and evidence_is_gold:
    sample_weight *= 2 or 3
```

然后比较：

| Model           | top3 F | REFUTES Recall | Macro Recall |
| --------------- | -----: | -------------: | -----------: |
| GBDT default    | 0.2011 |         0.0864 |       0.2321 |
| GBDT REFUTES x2 |    new |            new |          new |
| GBDT REFUTES x3 |    new |            new |          new |

这很简单，值得做。

## 做法 B：最终排序时加 REFUTES-sensitive bonus

如果 claim 有明显的 refutation cues，比如 negation、number mismatch、comparison mismatch，可以给相关 evidence 加一点分。但第一版不要写复杂规则，容易炸。

## 做法 C：先用 MiniLM 保底 REFUTES

例如 alpha-blend 或 rank ensemble：

```txt
final_score = 0.7 * MiniLM + 0.3 * GBDT
```

这通常比纯 GBDT 更稳。

我最推荐先做 A + C。

---

# 8. Classifier 下一轮怎么改？

现在最好的 classifier 是：

```txt
Fusion GBDT top50
without semantic summary
TF-IDF LogisticRegression
Accuracy = 0.4675
Macro-F1 = 0.4376
```

这个结论很重要：**semantic summary 作为文本拼进去没用**。但是这不代表 semantic features 没用，而是 TF-IDF 不会正确利用这种 summary。

下一轮 classifier 不要急着上 transformer。先做一个更强的 sklearn baseline：

```txt
text features:
  word TF-IDF 1-2 grams over claim + top50 evidence
  char TF-IDF 3-5 grams over claim + top50 evidence

numeric features:
  max MiniLM score
  mean top5 MiniLM score
  max GBDT score
  mean top5 GBDT score
  top1/top3 entity overlap
  top1/top3 number/year overlap
  count of evidence with high score
```

然后：

```txt
LogisticRegression
LinearSVC
RidgeClassifier
```

这比“semantic summary 文本化”更合理。

建议新脚本：

```txt
experiments/classification/train_tfidf_plus_numeric_features.py
```

对比表：

| Classifier | Context      | Features              | Accuracy | Macro-F1 |
| ---------- | ------------ | --------------------- | -------: | -------: |
| logreg     | Fusion top50 | word TF-IDF           |   0.4675 |   0.4376 |
| logreg     | Fusion top50 | word + char TF-IDF    |      new |      new |
| logreg     | Fusion top50 | word + char + numeric |      new |      new |
| LinearSVC  | Fusion top50 | word + char + numeric |      new |      new |

这个比 transformer 更现实，而且很可能涨一点。

---

# 9. Dense 现在怎么处理？

Dense 仍然有价值，但不要让它阻塞主线。Error layer 显示：

```txt
gold not in top500 = 180
Gold Evidence % = 0.3666
```

这说明 candidate retrieval 仍然有缺口，dense 可能补。但它不是 Round09 的唯一主线，因为 top20 -> top3 也丢了 `108` 个 gold evidence。

dense 跑完后只做这张表：

| Pool               | Macro Recall | Hit-any | All-gold | REFUTES Recall |
| ------------------ | -----------: | ------: | -------: | -------------: |
| sparse RRF         |       0.6579 |  0.8896 |   0.4091 |         0.5926 |
| dense top500       |          new |     new |      new |            new |
| sparse + dense RRF |          new |     new |      new |            new |

判断规则：

```txt
如果 sparse+dense 提升 all-gold 或 REFUTES recall，保留；
如果只提升很小，作为 ablation；
如果下降，不进主线。
```

dense 不要在报告里写成救世主。现在真正可控的收益来自 reranker scope、blend 和 classifier features。

---

# 10. Round09 最小可行计划

按优先级做：

```txt
1. compare MiniLM vs GBDT gain/loss，尤其 REFUTES lost cases
2. alpha-blend MiniLM and GBDT scores
3. expand MiniLM scoring scope to top100, optionally top200
4. train REFUTES-weighted GBDT fusion
5. rebuild classifier context using best reranker
6. train TF-IDF + numeric-feature classifier
7. dense supplement as ablation if finished
```

Round09 验收目标建议：

```txt
top3 evidence F:
  0.2011 -> 0.205+

top3 REFUTES recall:
  recover toward 0.1790, 不要低于 0.12

top20 macro recall:
  0.4669 -> 0.48+

classifier macro-F1:
  0.4376 -> 0.45+

accuracy/evidence harmonic:
  0.2812 -> 0.285+
```

---

# 11. 当前最重要的决策

我的建议很明确：

```txt
不要把 pure Fusion GBDT top3 直接当最终答案。
```

更合理的是：

```txt
final evidence:
  先做 MiniLM/GBDT alpha-blend，再决定

classifier context:
  用 Fusion GBDT top50 no semantic

classifier:
  先升级 sklearn 特征，不急着 transformer

dense:
  跑完就做 ablation，不要阻塞
```

Round08 是一个合格的工程推进轮，但不是性能突破轮。下一轮如果你能把 **GBDT 的 context 优势** 和 **MiniLM 的 REFUTES 稳定性** 融合起来，比继续堆新 retrieval 特征更有用。
