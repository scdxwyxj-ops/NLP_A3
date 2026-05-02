# Round07 报告 B：Candidate Recall 结果

## 目的

Stage A/B 用来验证 Round07 的核心假设：第一阶段 candidate recall 是当前主要瓶颈。
目标是确认更大的 sparse candidate pools 和 sparse retriever union 是否能在 rerank
之前提高 gold evidence 的召回。

## 脚本

```txt
experiments/retrieval/evaluate_candidate_recall.py
```

输出：

```txt
outputs/round07/candidate_recall_summary.csv
outputs/round07/candidates/*.json
```

## 核心结果

### 第一天交付表

| Candidate Source | N | Macro Recall | Hit-any | All-gold | SUPPORTS Recall | REFUTES Recall | NEI Recall | DISPUTED Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 50 | 0.3249 | 0.5844 | 0.1364 | 0.4172 | 0.1327 | 0.2634 | 0.4046 |
| BM25 | 100 | 0.4188 | 0.6753 | 0.2013 | 0.5027 | 0.2716 | 0.3366 | 0.5102 |
| BM25 | 200 | 0.4878 | 0.7727 | 0.2403 | 0.5716 | 0.3858 | 0.4000 | 0.5241 |
| BM25 | 500 | 0.5861 | 0.8377 | 0.3247 | 0.6797 | 0.5123 | 0.4927 | 0.5565 |
| BM25+TFIDF | 200 | 0.5361 | 0.8247 | 0.2468 | 0.6172 | 0.4259 | 0.4634 | 0.5611 |
| BM25+TFIDF | 500 | 0.6579 | 0.8896 | 0.4091 | 0.7306 | 0.5926 | 0.5610 | 0.7019 |
| BM25+TFIDF+dense | 300/500 | not run | not run | not run | not run | not run | not run | not run |

说明：`BM25+TFIDF` 对应当前主线的 `RRF(BM25 top500, char TF-IDF top500)`。
`BM25+TFIDF+dense` 还没有实际运行，因此不在第一天表格中填估计值。

### 详细结果

| 方法 | Retained k | Macro Recall | Hit-any | All-gold | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| BM25 | 100 | 0.4188 | 0.6753 | 0.2013 | 0.2716 |
| BM25 | 200 | 0.4878 | 0.7727 | 0.2403 | 0.3858 |
| BM25 | 500 | 0.5861 | 0.8377 | 0.3247 | 0.5123 |
| char TF-IDF | 100 | 0.4101 | 0.7208 | 0.1818 | 0.2809 |
| char TF-IDF | 200 | 0.5126 | 0.8182 | 0.2338 | 0.3611 |
| char TF-IDF | 500 | 0.6610 | 0.8896 | 0.4026 | 0.5556 |
| RRF(BM25, char TF-IDF) | 100 | 0.4637 | 0.7727 | 0.2143 | 0.3272 |
| RRF(BM25, char TF-IDF) | 200 | 0.5361 | 0.8247 | 0.2468 | 0.4259 |
| RRF(BM25, char TF-IDF) | 500 | 0.6579 | 0.8896 | 0.4091 | 0.5926 |
| RRF(BM25, word TF-IDF, char TF-IDF) | 500 | 0.6478 | 0.8766 | 0.3831 | 0.5617 |

## 解读

Requirement 里的短期目标是把 candidate macro recall 推到 `0.55` 以上。
这个目标已经达成：

- BM25 top-500 达到 `0.5861`
- char TF-IDF top-500 达到 `0.6610`
- RRF(BM25, char TF-IDF) top-500 达到 `0.6579`

单看 overall macro recall，char TF-IDF top-500 最高；但
`RRF(BM25, char TF-IDF)` 的 all-gold rate 和 `REFUTES` recall 更好。
这说明 RRF union 在整体召回和类别平衡之间更适合进入 reranking 和 classifier context。

## 决策

Stage C reranking 选择以下 candidate pools：

1. `rrf_bm25_char_tfidf_top500`
2. `bm25_top500`
3. `tfidf_char_top500`

在这些更强的 candidate pools 被 zero-shot MiniLM 评估之前，不继续做 reranker fine-tuning。
