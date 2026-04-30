# Round05 Report B：Literature and pretrained model choice

报告时间：`2026-04-30`

## 一、调研目标

为 Round05 supervised evidence reranker 选择合理技术路线和 open-source pretrained backbone。

核心问题：
- 文献是否支持 `BM25 candidate retrieval -> supervised cross-encoder reranker`？
- 在免费 Colab 约束下，哪个 open-source pretrained model 最适合作为第一版 reranker？

## 二、文献结论

### 1. FEVER-style fact verification requires evidence retrieval

参考：
- [FEVER Shared Task](https://aclanthology.org/W18-5501/)
- [FEVER dataset paper](https://arxiv.org/abs/1803.05355)

用途：
- 支持本项目不是单纯 classification，而是 retrieval + verification pipeline。
- Evidence 不能假设已给定，系统必须先找证据。

### 2. Evidence selection commonly uses coarse retrieval then reranking

参考：
- [Evidence Selection as a Token-Level Prediction Task](https://aclanthology.org/2021.fever-1.2.pdf)
- [Evidence Retrieval for Fact Verification using Multi-stage Reranking](https://aclanthology.org/2024.findings-emnlp.428.pdf)

用途：
- 支持先用 TF-IDF/BM25 做 coarse retrieval，再做 fine-grained evidence selection/reranking。
- 这和当前 Round04 -> Round05 计划一致。

### 3. BERT/cross-encoder reranking is a standard supervised ranking method

参考：
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- [How Does BERT Rerank Passages?](https://aclanthology.org/2021.blackboxnlp-1.39.pdf)

用途：
- 支持把 claim 和 evidence 拼接成 pair，让 transformer 同时编码两者。
- Cross-encoder 比单独 embedding similarity 更能建模 token-level interaction。

### 4. Fact-checking-specific reranking has precedent

参考：
- [Article Reranking by Memory-Enhanced Key Sentence Matching for Detecting Previously Fact-Checked Claims](https://aclanthology.org/2021.acl-long.425/)

用途：
- 支持 reranking 在 fact-checking/claim matching 场景中是合理技术路线。

## 三、模型选择

约束：
- 必须 open-source。
- 不能用 closed-source APIs/proprietary models。
- 必须能在免费 Google Colab 跑。
- fine-tuning 和 evaluation 只能使用课程 train/dev 数据。

候选比较：

| Model | Approx size | 优点 | 风险 |
| --- | ---: | --- | --- |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22.7M | 已经是 cross-encoder reranker，轻量，适合 reranking 起点 | 预训练目标来自 MS MARCO，需在报告中说明只作为 open-source pretrained initialization |
| [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) | 67M | 最稳、低内存、容易 fine-tune | 效果上限可能较低 |
| [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) | 110M | 经典标准 baseline | 较慢，显存占用更高 |
| [`microsoft/deberta-v3-small`](https://huggingface.co/microsoft/deberta-v3-small) | ~142M total | 强 NLU backbone | 免费 Colab 上 batch size 可能很小 |

## 四、推荐选择

第一版 supervised reranker：

```txt
cross-encoder/ms-marco-MiniLM-L6-v2
```

理由：
- 它已经按 query-passage pair reranking 形式训练过，和 claim-evidence relevance scoring 形式最接近。
- 体积小，Colab 风险低。
- 可以快速完成 Round05 对比实验。

备选：

```txt
distilbert-base-uncased
```

用途：
- 如果担心 MS MARCO reranker initialization 的报告解释，DistilBERT 是更普通、更保守的 pretrained transformer baseline。

更强但风险更高：

```txt
microsoft/deberta-v3-small
```

用途：
- 如果第一版 pipeline 跑通，再作为第二个 comparison backbone。

## 五、Round05 实验矩阵

最低实验：

| Experiment | Candidate generator | Reranker | Supervision |
| --- | --- | --- | --- |
| BM25 baseline | BM25 | none | none |
| MiniLM reranker | BM25 top-50 | MiniLM cross-encoder | train gold evidence + hard negatives |

可选扩展：

| Experiment | Candidate generator | Reranker | Supervision |
| --- | --- | --- | --- |
| DistilBERT reranker | BM25 top-50 | DistilBERT pair classifier | train gold evidence + hard negatives |
| DeBERTa-small reranker | BM25 top-50 | DeBERTa pair classifier | train gold evidence + hard negatives |

## 六、关键设计决定

1. 无监督 baseline 到 BM25 为止即可。
2. Distributional semantics / PPMI+SVD 不作为主模型，只作为可能的 query expansion 或 clustering feature。
3. Round05 主线是 supervised reranker。
4. 第一版训练目标用 binary relevance classification。
5. Pair 输入格式：

```txt
[CLS] claim [SEP] evidence [SEP]
```

6. 先限制 max length 到 `256`，避免 Colab 显存风险。

## 七、下一步 implementation

1. 生成 BM25 top-50 train/dev candidate pool。
2. 构造 hard negative JSONL：
   - positives：gold evidence。
   - negatives：BM25 top-50 中非 gold evidence。
3. 实现 reranker dataset loader。
4. 训练 MiniLM cross-encoder binary relevance classifier。
5. rerank dev candidates，输出 top-k evidence，使用 `eval.py` 比较 BM25 baseline。

