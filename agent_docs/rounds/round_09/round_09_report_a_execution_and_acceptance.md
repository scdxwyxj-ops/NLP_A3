# Round09 报告 A：Alpha Blend、Top100 Scope 与 REFUTES Calibration

日期：2026-05-01

## 一句话结论

Round09 成功把 final evidence top3 F-score 从 Round08 的 `0.2011` 继续提高到
`0.2105`。最好的方法是：

```txt
MiniLM top100 scope
+ REFUTES x2 weighted Fusion GBDT
+ alpha blend
```

但它仍没有完全解决 `REFUTES`：best blend 的 `REFUTES recall = 0.1420`，
仍低于 MiniLM-only 的 `0.1790`。

所以 Round09 的结论是：

```txt
alpha-blend 是当前最好的 final evidence selector；
Fusion 对 overall F-score 有帮助；
但 REFUTES 仍需要单独处理。
```

## Stage A：Alpha Blend on Top50

目的：

检查 Round08 的 MiniLM-only 和 Fusion GBDT 是否可以通过 rank/score blending 互补。

脚本：

```txt
experiments/rerank/blend_ranked_candidates.py
```

输入：

```txt
outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json
outputs/round08/fusion_gbdt/dev-fusion-ranked-top50.json
```

输出：

```txt
outputs/round09/blend_top50/blend_summary.csv
outputs/round09/blend_top50/blend_top3_label_metrics.csv
```

关键结果：

| alpha_minilm | top3 F-score | Macro Recall | REFUTES Recall |
|---:|---:|---:|---:|
| 0.0 | 0.2011 | 0.2321 | 0.0864 |
| 0.5 | 0.2078 | 0.2416 | 0.1235 |
| 0.7 | 0.2104 | 0.2448 | 0.1235 |
| 1.0 | 0.1987 | 0.2331 | 0.1790 |

解读：

- `alpha=0.7` 明显超过 MiniLM-only 和 pure GBDT。
- 但 REFUTES recall 仍没有回到 MiniLM-only。
- 这证明 alpha-blend 有用，但不能单独解决 REFUTES。

## Stage B：Gain/Loss Analysis

脚本：

```txt
experiments/analysis/compare_ranked_outputs.py
```

输出：

```txt
outputs/round09/gain_loss_minilm_vs_gbdt_top3/gain_loss_by_label.csv
outputs/round09/gain_loss_minilm_vs_gbdt_top3/examples.jsonl
```

MiniLM vs Round08 Fusion GBDT top3：

| Label | Bucket | Count |
|---|---|---:|
| REFUTES | both_hit | 4 |
| REFUTES | both_miss | 20 |
| REFUTES | minilm_only | 3 |
| REFUTES | fusion_gbdt_only | 0 |
| SUPPORTS | fusion_gbdt_only | 3 |
| SUPPORTS | minilm_only | 1 |

解读：

- GBDT 对 SUPPORTS 有收益。
- REFUTES 中没有 `fusion_gbdt_only`，但有 3 个 `minilm_only`。
- 这解释了为什么 Fusion GBDT overall F-score 小涨，但 REFUTES recall 大跌。

## Stage C：MiniLM Top100 Scope

目的：

Round08 的 cheap fusion 只重排 MiniLM top50，救不了：

```txt
gold in top500 but not top50
```

所以 Round09 把 MiniLM scoring scope 扩到 top100。

脚本：

```txt
experiments/rerank/rerank_pool_once.py
```

输出：

```txt
outputs/round09/dev-rrf-bm25-char-minilm-ranked-top100.json
outputs/round09/train-rrf-bm25-char-minilm-ranked-top100.json
```

MiniLM-only top100 scope 本身不会改变 top3/top20，因为 top50 内排序保持一致：

| Method | top3 F-score | top20 F-score |
|---|---:|---:|
| MiniLM top50 scope | 0.1987 | 0.1106 |
| MiniLM top100 scope | 0.1987 | 0.1106 |

top100 的价值在于给 fusion reranker 更多候选。

## Stage D：Top100 Fusion + REFUTES Weighting

构造 top100 feature table：

```txt
outputs/round09/train_feature_table_top100.csv
outputs/round09/dev_feature_table_top100.csv
```

规模：

| Split | Rows | Positive Rows |
|---|---:|---:|
| train | 122,800 | 2,131 |
| dev | 15,400 | 279 |

训练三个 GBDT：

| Model | top3 F-score | top20 F-score | Note |
|---|---:|---:|---|
| GBDT top100 default | 0.1975 | 0.1160 | overall top3 下降 |
| GBDT top100 REFUTES x2 | 0.2037 | 0.1154 | 最好的 pure fusion top100 |
| GBDT top100 REFUTES x3 | 0.1922 | 0.1154 | 权重过强，top3 下降 |

结论：

- top100 scope 不自动提升。
- REFUTES x2 是合理校准；x3 过强。
- pure fusion 仍不如后续 alpha-blend。

## Stage E：Top100 REFUTES x2 Alpha Blend

输入：

```txt
MiniLM top100 ranked
Fusion GBDT top100 REFUTES x2 ranked
```

输出：

```txt
outputs/round09/blend_top100_refutes_x2/blend_summary.csv
outputs/round09/blend_top100_refutes_x2/blend_top3_label_metrics.csv
```

关键结果：

| alpha_minilm | top3 F-score | Macro Recall | REFUTES Recall | Harmonic Mean |
|---:|---:|---:|---:|---:|
| 0.0 | 0.2037 | 0.2393 | 0.1420 | 0.2788 |
| 0.3 | 0.2071 | 0.2417 | 0.1420 | 0.2819 |
| 0.4 | 0.2105 | 0.2446 | 0.1420 | 0.2851 |
| 0.5 | 0.2105 | 0.2446 | 0.1420 | 0.2851 |
| 0.6 | 0.2105 | 0.2446 | 0.1420 | 0.2851 |
| 0.8 | 0.2074 | 0.2444 | 0.1605 | 0.2822 |
| 1.0 | 0.1987 | 0.2331 | 0.1790 | 0.2741 |

推荐选择：

```txt
alpha_minilm = 0.4
```

原因：

- top3 F-score 达到本轮最高 `0.2105`；
- macro recall 达到 `0.2446`；
- harmonic mean 达到 `0.2851`；
- REFUTES 比 pure GBDT 好，但仍未达到 MiniLM-only。

如果更重视 REFUTES，可以考虑：

```txt
alpha_minilm = 0.8
```

它的 REFUTES recall 是 `0.1605`，但 top3 F-score 降到 `0.2074`。

## Stage F：Top100 Blend Gain/Loss

MiniLM vs top100 REFUTES x2 Fusion GBDT top3：

| Label | Bucket | Count |
|---|---|---:|
| REFUTES | both_hit | 5 |
| REFUTES | both_miss | 19 |
| REFUTES | fusion_top100_x2_only | 1 |
| REFUTES | minilm_only | 2 |
| SUPPORTS | fusion_top100_x2_only | 3 |
| SUPPORTS | minilm_only | 5 |

解读：

- REFUTES x2 比 Round08 pure GBDT 好，因为它至少新增了 1 个 REFUTES hit。
- 但仍有 2 个 REFUTES 是 MiniLM-only。
- SUPPORTS 的 tradeoff 更复杂：fusion 新增 3 个，但也丢 5 个。

## Stage G：Feature Importance

Top100 REFUTES x2 GBDT 的前几个重要特征：

| Feature | Importance |
|---|---:|
| minilm_rank | 0.5036 |
| minilm_score | 0.2041 |
| rrf_score | 0.1179 |
| char_tfidf_score | 0.0419 |
| bm25_score | 0.0362 |
| rrf_rank | 0.0362 |
| char_tfidf_rank | 0.0152 |
| percentage_overlap_count | 0.0097 |
| bm25_rank | 0.0092 |
| entity_jaccard | 0.0062 |

解读：

- GBDT 主要仍依赖 MiniLM rank/score。
- Sparse RRF/BM25/char features 提供了第二层信号。
- Semantic overlap 有信号，但权重很小。
- 这说明当前 feature fusion 更像 reranker calibration，而不是真正的 semantic verifier。

## Stage H：Classifier Context Check

使用 best blend：

```txt
top100 REFUTES x2 blend
alpha_minilm = 0.4
context top50
TF-IDF LogisticRegression
```

结果：

| Context | Accuracy | Macro-F1 | Evidence F-score | Harmonic Mean |
|---|---:|---:|---:|---:|
| Round07 MiniLM top50 | 0.4610 | 0.4242 | 0.1987 | 0.2778 |
| Round08 Fusion GBDT top50 | 0.4675 | 0.4376 | 0.2011 | 0.2812 |
| Round09 Blend top100 x2 alpha0.4 top50 | 0.4481 | 0.4142 | 0.2105 | 0.2865 |

解读：

- Round09 blend 提高了 evidence F-score 和 assignment harmonic mean。
- 但 classifier macro-F1 低于 Round08 Fusion GBDT top50。
- 如果目标是 final assignment harmonic mean，Round09 blend 更好。
- 如果目标是 classifier macro-F1，Round08 Fusion GBDT top50 更好。

## 当前推荐

### Final Evidence Output

推荐：

```txt
Round09 top100 REFUTES x2 blend
alpha_minilm = 0.4
top3 evidence
```

理由：

```txt
best top3 evidence F-score = 0.2105
```

### Classifier Context

保守推荐仍是：

```txt
Round08 Fusion GBDT top50 without semantic summary
```

理由：

```txt
best classifier macro-F1 = 0.4376
```

如果最终优化 assignment harmonic mean，则可改用：

```txt
Round09 blend top100 x2 alpha0.4
```

它的 harmonic mean 是 `0.2865`，高于 Round08 的 `0.2812`。

## Round09 验收判断

Round09 本轮已完成：

1. alpha-blend sweep；
2. MiniLM top100 scoring scope；
3. gain/loss analysis；
4. REFUTES x2/x3 calibration；
5. feature importance analysis；
6. classifier context check。

Round09 可以验收。

## 下一步建议

不要继续把 GBDT 当作最终 verifier。它主要学到的是 MiniLM 和 sparse rank calibration。

下一步更有价值的是：

```txt
1. 保留 Round09 blend 作为 final evidence selector。
2. 分类器继续用 Round08 Fusion GBDT top50 context，或以 assignment harmonic mean 为目标改用 Round09 blend context。
3. 针对 REFUTES 单独做 error examples，设计 verifier/aggregation，而不是继续只调 alpha。
4. 如时间允许，再试 top200 scope，但 top100 已经说明 scope 扩展不是自动收益。
```

