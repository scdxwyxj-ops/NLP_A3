# Round04 Report D：BM25 retrieval baseline

报告时间：`2026-04-30`

## 一、目标

基于 discovery 结论，先建立一个更强的无监督 lexical retrieval baseline。无监督方案只作为 baseline，后续重点仍是 supervised reranker。

## 二、实现

Reusable code：

```txt
src/a3_factcheck/retrieval/bm25.py
```

Experiment entry：

```txt
experiments/retrieval/bm25_baseline.py
```

Config：

```txt
configs/retrieval/bm25_baseline.json
```

实现方式：
- 用 `CountVectorizer` 构建 evidence term-frequency matrix。
- 用 BM25 公式把 evidence term-frequency 转为 BM25 weight。
- 用 claim query terms 与 BM25 evidence matrix 点积得到 claim-evidence score。
- 对每个 claim 按 score 排序 evidence ids。
- 输出 top-k evidence prediction JSON。

当前参数：

```txt
max_features = 200000
k1 = 1.5
b = 0.75
ngram_range = (1, 2)
stop_words = english
```

## 三、运行命令

```bash
PYTHONPATH=src python experiments/retrieval/bm25_baseline.py \
  --top-k-values 1,3,5,10,20,50 \
  --output outputs/round04/dev-bm25.json
```

## 四、Dev results

| top-k | Evidence F-score | Accuracy占位 | Harmonic mean |
| ---: | ---: | ---: | ---: |
| 1 | 0.06688311688311689 | 0.44155844155844154 | 0.11616990927335755 |
| 3 | 0.0682282003710575 | 0.44155844155844154 | 0.1181935160645014 |
| 5 | 0.07721603793032365 | 0.44155844155844154 | 0.13144591617317444 |
| 10 | 0.07085187539732994 | 0.44155844155844154 | 0.1221101240420117 |
| 20 | 0.05809892911530406 | 0.44155844155844154 | 0.10268665730588816 |
| 50 | 0.03669627046682843 | 0.44155844155844154 | 0.06776116404466775 |

## 五、结论

BM25 明显超过当前 TF-IDF baseline：
- TF-IDF 最佳：top-3 F-score 约 `0.0533`。
- BM25 最佳：top-5 F-score 约 `0.0772`。

但 BM25 仍然很弱，不适合作为最终方案。它的主要价值是：
- 作为 stronger unsupervised retrieval baseline。
- 为 supervised reranker 提供 candidate pool。
- 为 error analysis 提供更合理的 starting point。

## 六、下一步

1. 用 BM25 top-50 生成 candidate pool。
2. 做 retrieval error analysis。
3. 从 BM25 top-k 中构造 supervised reranker hard negatives。
4. 开启 supervised reranker dataset / model round。

