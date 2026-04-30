# Round04 Report A：TF-IDF retrieval baseline

报告时间：`2026-04-29`

## 一、目标

建立一个不依赖训练的 lexical retrieval baseline，用于确认 pipeline 可以从 `evidence.json` 中为每条 dev claim 召回候选 evidence，并通过课程 `eval.py` 评估。

## 二、实现

脚本：

```txt
src/tfidf_retrieval_baseline.py
```

当前方法：
- 使用 `TfidfVectorizer` 对 1,208,827 条 evidence 建立 TF-IDF 表示。
- 使用 claim TF-IDF 与 evidence TF-IDF 点积排序。
- 每条 claim 输出 top-k evidence ids。
- classification label 暂用 train set majority label `SUPPORTS` 作为占位；此报告只把 retrieval F-score 作为有效结论。

输出：

```txt
outputs/round04/dev-tfidf-top1.json
outputs/round04/dev-tfidf-top3.json
outputs/round04/dev-tfidf-top5.json
outputs/round04/dev-tfidf-top10.json
outputs/round04/dev-tfidf-top20.json
```

## 三、初始结果

命令：

```bash
python src/tfidf_retrieval_baseline.py --top-k-values 1,3,5,10,20 --output outputs/round04/dev-tfidf.json
```

结果：

| top-k | Evidence F-score | Accuracy占位 | Harmonic mean |
| ---: | ---: | ---: | ---: |
| 1 | 0.04761904761904762 | 0.44155844155844154 | 0.08596713021491784 |
| 3 | 0.05326221397649969 | 0.44155844155844154 | 0.09505819910444187 |
| 5 | 0.04744382601525459 | 0.44155844155844154 | 0.0856814917477186 |
| 10 | 0.05001708248461496 | 0.44155844155844154 | 0.0898558366436168 |
| 20 | 0.04516465371631945 | 0.44155844155844154 | 0.08194735077137329 |

解释：
- Retrieval F-score 很低；当前最佳是 top-3，F-score 约 0.0533。
- 简单 word TF-IDF 对 climate claim/evidence 匹配不足，增加 top-k 也没有稳定提升。
- Accuracy 只是 majority label 占位结果，不代表系统分类能力。
- 后续需要比较 top-k，并考虑 BM25、query expansion、dense reranker 或 hybrid retrieval。

## 四、下一步

- 抽样查看 retrieval 成功/失败案例，定位词面不匹配、实体歧义和多证据需求。
- 实现 BM25 或改进 TF-IDF 配置，目标是明显超过当前 top-3 F-score。
- 将 best retrieval candidates 接入后续 sequence classifier baseline。
