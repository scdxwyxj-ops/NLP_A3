# Round07 报告 F：可验收成果总结

日期：2026-05-01

## 一句话结论

Round07 的 retrieval rescue 已经可以验收：我们把第一阶段 candidate recall 明显拉高，
并且把 final evidence F-score 从旧最佳 `0.1642` 提升到 `0.1987`。

## 可交付成果

### 0. 第一天交付表

这是第一天应该交付的核心表格。当前已经有 BM25 和 BM25+TFIDF 的真实结果；
`BM25+TFIDF+dense` 还没有跑 dense retrieval，因此暂不填假数。

| Candidate Source | N | Macro Recall | Hit-any | All-gold | SUPPORTS Recall | REFUTES Recall | NEI Recall | DISPUTED Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 50 | 0.3249 | 0.5844 | 0.1364 | 0.4172 | 0.1327 | 0.2634 | 0.4046 |
| BM25 | 100 | 0.4188 | 0.6753 | 0.2013 | 0.5027 | 0.2716 | 0.3366 | 0.5102 |
| BM25 | 200 | 0.4878 | 0.7727 | 0.2403 | 0.5716 | 0.3858 | 0.4000 | 0.5241 |
| BM25 | 500 | 0.5861 | 0.8377 | 0.3247 | 0.6797 | 0.5123 | 0.4927 | 0.5565 |
| BM25+TFIDF | 200 | 0.5361 | 0.8247 | 0.2468 | 0.6172 | 0.4259 | 0.4634 | 0.5611 |
| BM25+TFIDF | 500 | 0.6579 | 0.8896 | 0.4091 | 0.7306 | 0.5926 | 0.5610 | 0.7019 |
| BM25+TFIDF+dense | 300/500 | not run | not run | not run | not run | not run | not run | not run |

这里的 `BM25+TFIDF` 指：

```txt
RRF(BM25 top500, char TF-IDF top500)
```

### 1. 更强的 Candidate Generation

最终推荐的候选池：

```txt
RRF(BM25 top500, char TF-IDF top500)
```

关键指标：

| Candidate Pool | k | Macro Recall | Hit-any | All-gold | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| BM25 | 500 | 0.5861 | 0.8377 | 0.3247 | 0.5123 |
| char TF-IDF | 500 | 0.6610 | 0.8896 | 0.4026 | 0.5556 |
| RRF(BM25, char TF-IDF) | 500 | 0.6579 | 0.8896 | 0.4091 | 0.5926 |

结论：RRF(BM25, char TF-IDF) 不是 macro recall 最高，但它的 all-gold 和
`REFUTES` recall 更好，所以更适合作为 reranker 输入。

### 2. 更好的 Final Evidence Pipeline

当前默认 pipeline：

```txt
RRF(BM25 top500, char TF-IDF top500)
-> zero-shot cross-encoder/ms-marco-MiniLM-L6-v2
-> top-3 evidence
```

关键指标：

| 方法 | Output k | Evidence F-score | Macro Recall | REFUTES Recall | Hit-any |
|---|---:|---:|---:|---:|---:|
| 旧最佳：BM25 top50 -> MiniLM | 3 | 0.1642 | 0.1877 | 0.0494 | 0.3961 |
| 当前最佳：RRF(BM25,char) top500 -> MiniLM | 3 | 0.1987 | 0.2331 | 0.1790 | 0.4545 |

结论：扩大并融合候选池后，zero-shot MiniLM 的 rerank 效果明显提升。

### 3. Classifier Context 已准备好

final evidence output 应使用 top-3；classifier context 不应只用 top-3。

当前推荐：

```txt
final evidence:
  top-3

classifier context:
  top-20 first
  top-50 as wider ablation
```

top-20 context 指标：

| 方法 | Context k | Evidence F-score | Macro Recall | REFUTES Recall | Hit-any |
|---|---:|---:|---:|---:|---:|
| RRF(BM25,char) top500 -> MiniLM | 20 | 0.1106 | 0.4540 | 0.3951 | 0.7143 |

结论：top-20 不适合 final output，但适合作为 classifier 输入，因为 recall 更高。

### 4. 快速 Classifier Baseline

当前 baseline：

```txt
TF-IDF over claim + evidence context
LogisticRegression(class_weight="balanced")
```

最好结果：

| 方法 | Context k | Semantic Summary | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|
| concat TF-IDF logreg | 50 | no | 0.4610 | 0.4242 |

结论：这个模型只是 baseline。它证明 retrieval 已经改善，但最终分类还需要更强的监督模型。

### 5. Query-Boost Ablation 已排除

测试过数字、实体、百分比 query expansion：

```txt
query-boost supplement top3 evidence F-score:
  0.1927
```

低于当前默认 `0.1987`，因此不进入主线。

## 代码与输出

关键脚本：

```txt
experiments/retrieval/evaluate_candidate_recall.py
experiments/retrieval/evaluate_query_boost_recall.py
experiments/rerank/rerank_pool_once.py
experiments/classification/build_evidence_package.py
experiments/classification/train_concat_classifier.py
```

关键输出：

```txt
outputs/round07/candidate_recall_summary.csv
outputs/round07/candidates/
outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json
outputs/round07/dev-classifier-context-rrf-bm25-char-minilm-top20-semantic.jsonl
outputs/round07/dev-classifier-context-rrf-bm25-char-minilm-top50-semantic.jsonl
outputs/round07_classifier/summary.csv
outputs/round07_query_boost/query_boost_recall_summary.csv
```

## 验收判断

Round07 满足验收条件：

- 已覆盖 BM25 top-50/top-100/top-200/top-500 recall。
- 已实现 sparse union candidate pool。
- 已记录 class-specific recall，尤其是 `REFUTES`。
- 已明确选择 `RRF(BM25 top500, char TF-IDF top500)` 进入 rerank。
- 已证明 recall rescue 能提高 final evidence F-score。
- 已构造 classifier-ready context。
- 已跑出 classifier baseline，并指出下一阶段瓶颈。

## 下一阶段建议

下一阶段不要继续细调 sparse retrieval。更重要的是做 supervised claim classifier。

建议路线：

```txt
固定 retrieval:
  RRF(BM25 top500, char TF-IDF top500) -> MiniLM

final evidence:
  top-3

classifier input:
  top-20/top-50 evidence context

next model:
  evidence-wise verifier + aggregation
  或 transformer classifier over top-k context
```
