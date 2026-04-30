# Round05 Report A：Supervised reranker discovery plan

报告时间：`2026-04-30`

## 一、Discovery 目标

Round05 的目标不是继续堆无监督 baseline，而是用有监督方式改善 evidence ranking。

当前无监督结果：
- TF-IDF best：top-3 retrieval F-score `0.0533`
- BM25 best：top-5 retrieval F-score `0.0772`

BM25 比 TF-IDF 更好，但仍然弱。因此 BM25 应作为 candidate generator，而不是最终 retrieval model。

## 二、任务定义

有监督 reranking 问题定义为 binary relevance classification：

```txt
input: claim_text + evidence_text
label: 1 if evidence_id in gold evidences else 0
score: relevance probability/logit
```

训练时：
- positive pairs 来自 train gold evidence。
- negative pairs 来自 BM25 top-k 中非 gold evidence。

推理时：
- BM25 为每个 claim 召回 top-50 candidates。
- reranker 对每个 claim-evidence pair 打分。
- 按 reranker score 重新排序。
- 取 top-k evidence ids 输出。

## 三、为什么这是正确下一步

课程主任务要求 retrieval + classification。当前 retrieval 太弱，会限制后续 classifier。

无监督 lexical baseline 的作用已经足够：
- 确认数据读取、prediction JSON、`eval.py` 管线。
- 给 dev F-score baseline。
- 给 supervised reranker 提供候选池。

真正重要的是让模型学习：
- 哪些 evidence 虽然词面相似但不相关。
- 哪些 evidence 和 claim 存在语义支持/反驳关系。
- 如何把 gold evidence 排到 top-k 前列。

## 四、模型选择原则

必须满足：
- open-source pretrained model。
- HuggingFace/PyTorch 可加载。
- 免费 Google Colab 可运行。
- 只用课程 train/dev 数据做 fine-tuning 和 evaluation。
- 不能使用 closed-source APIs/proprietary models。

优先候选：
- `distilbert-base-uncased`：速度和内存最稳。
- `bert-base-uncased`：经典 baseline，参数更多。
- `microsoft/deberta-v3-small`：可能效果更好，但训练成本和依赖风险略高。
- `cross-encoder/ms-marco-MiniLM-L-6-v2` 或 similar MiniLM reranker：适合 reranking，但需要确认 license、依赖和是否允许用 MS MARCO reranking pretraining。

初始建议：
- 第一版 supervised reranker 用 `distilbert-base-uncased` 或 `microsoft/deberta-v3-small`。
- 如果 Colab 资源或训练时间紧，优先 DistilBERT。

## 五、实验设计

### Dataset

For each train claim:
- Add all gold evidence as positives.
- Add top BM25 non-gold candidates as hard negatives.
- Cap negatives per claim, e.g. `5` or `10`, to control class imbalance.

For dev:
- Use BM25 top-50 candidates.
- Score each candidate with reranker.
- Select top-k evidence ids.
- Evaluate with `eval.py`.

### Metrics

Primary for this round:
- Evidence Retrieval F-score.

Secondary:
- Candidate recall@50.
- Gold evidence rank distribution.
- Pair classification dev accuracy/AUC, if useful.

### Baselines

Compare against:
- TF-IDF best: `0.0533`
- BM25 best: `0.0772`

## 六、Implementation backlog

P0:
- Create candidate pool builder from BM25 outputs.
- Create hard negative dataset JSONL.

P1:
- Create `src/a3_factcheck/rerank/` reusable code.
- Create `experiments/rerank/` training/eval script.

P2:
- Train DistilBERT cross-encoder relevance classifier.
- Rerank dev BM25 top-50.
- Evaluate top-k.

P3:
- Try DeBERTa-small if DistilBERT works and compute allows.

## 七、Risks

- BM25 top-50 may not contain enough gold evidence; reranker cannot recover missing gold evidence.
- Data size may be small; overfitting risk is real.
- Class imbalance: many more negatives than positives.
- Free Colab resource limit may constrain model/backbone and batch size.
- The final notebook must reproduce reported results, so scripts need to be portable into notebook cells.

## 八、Open questions

- Which backbone gives best tradeoff under Colab constraints?
- Should reranker optimize binary cross entropy or pairwise ranking loss?
- How many hard negatives per claim are optimal?
- Should gold evidence not in BM25 top-50 be injected into train candidates for positives? Current answer: yes for training positives, but dev evaluation must use real BM25 candidates.

