下面我按“救火优先级”给你做一个完整规划。先说结论：你现在最危险的点不是 MiniLM 没调好，而是你在 **candidate recall 不够的情况下提前优化 reranker/classifier**。BM25 top-100 macro recall 只有 0.4188，意味着很多 gold evidence 在第一阶段已经丢了；后面的 reranker、classifier 再强也不可能恢复不存在于候选池里的证据。你当前最好的证据输出是 `BM25 top-50 -> zero-shot MiniLM -> top-3`，dev evidence F-score 约 0.1642，但这只能说明 reranker 比 BM25 强，不能说明系统已经足够支撑分类器。你的任务应该被拆成两个目标：**final evidence output 要高精度小集合；classifier evidence context 要高召回宽上下文**。这点你文档里已经判断对了，但后续实验还没有彻底围绕这个判断展开。 

---

# 1. 重新定义你的任务：不要把它当成单一 retrieval 任务

你的完整任务其实是：

[
\text{claim} \rightarrow \text{candidate evidence retrieval} \rightarrow \text{reranking} \rightarrow \text{claim classification}
]

其中 evidence retrieval 又有两个不同用途：

[
\text{Final Output Evidence}: \text{small, precise, evaluator-facing}
]

[
\text{Classifier Context}: \text{larger, recall-oriented, model-facing}
]

你现在的实验容易混淆这两个目标。`top-3` 可能 evidence F-score 最好，因为 precision 高；但 `top-3` 对 classifier 很可能太窄，因为 zero-shot MiniLM top-3 macro recall 只有 0.1877，而 top-20 macro recall 才到 0.3035，BM25 top-100 candidate recall 也只有 0.4188。也就是说，**最终提交 evidence 可能用 top-3，但分类器输入不应该只用 top-3**。

所以你后续所有实验都应该分两张表：

**表 A：Final Evidence Table**

| candidate source | reranker | output top-k | precision | recall | F-score |
| ---------------- | -------- | -----------: | --------: | -----: | ------: |

**表 B：Classifier Context Table**

| candidate source | reranker | context top-k | hit-any | all-gold | label accuracy / macro-F1 |
| ---------------- | -------- | ------------: | ------: | -------: | ------------------------: |

如果你只看 evidence F-score，会被误导；如果只看 classifier accuracy，也不知道是不是证据问题导致分类错。

---

# 2. 当前系统的真实瓶颈排序

我会把瓶颈排成这个顺序：

[
\text{Candidate Recall} > \text{Evidence-vs-Topic Reranking} > \text{Classifier Context Design} > \text{Fine-tuning}
]

你现在做了 TF-IDF、BM25、BM25+MiniLM rerank、BCE fine-tuning、task-aware hard negatives，但 fine-tuned MiniLM 没有超过 zero-shot MiniLM。这个结果不奇怪，因为你现在的 hard negative training 建立在一个 recall 不够的候选池上，而且 BCE pair classification 不一定匹配 final evidence-set F-score。你文档里 task-aware MiniLM top-3/top-5/top-20 都没有超过 zero-shot MiniLM，说明继续盲目微调 reranker 是低收益路线。

一句话：**先扩大候选池，后训练 reranker；先保证 gold 出现，后讨论 gold 排第几。**

---

# 3. 第一阶段：Candidate Generation，先把 recall 顶上去

这是你马上要做的主线。目标不是提升 final F-score，而是提升：

[
\text{CandidateRecall@N} =
\frac{1}{M}\sum_{i=1}^{M}
\frac{|C_i^N \cap G_i|}{|G_i|}
]

同时看：

[
\text{HitAny@N} =
\frac{1}{M}\sum_{i=1}^{M}
\mathbb{1}[C_i^N \cap G_i \neq \emptyset]
]

[
\text{AllGold@N} =
\frac{1}{M}\sum_{i=1}^{M}
\mathbb{1}[G_i \subseteq C_i^N]
]

其中 (C_i^N) 是第 (i) 个 claim 的 top-N candidates，(G_i) 是 gold evidence set。

## 3.1 先做 BM25 扩池实验

你已经有：

```txt
BM25 top-50 macro recall  = 0.3249
BM25 top-100 macro recall = 0.4188
```

下一步直接测：

```txt
BM25 top-200
BM25 top-500
BM25 top-1000
```

不要一开始就 rerank。先只测 candidate recall、hit-any、all-gold、runtime。你需要知道 BM25 的 recall 曲线是否还在涨。如果 top-100 到 top-500 能从 0.42 涨到 0.60+，那说明 BM25 还有可榨空间；如果涨得很慢，说明 lexical matching 本身已经不够。

实验表：

| Method |   N | Macro Recall |  Hit-any | All-gold | Avg candidates | Runtime |
| ------ | --: | -----------: | -------: | -------: | -------------: | ------: |
| BM25   |  50 |     existing | existing |        ? |             50 |       ? |
| BM25   | 100 |       0.4188 |        ? |        ? |            100 |       ? |
| BM25   | 200 |          new |      new |      new |            200 |     new |
| BM25   | 500 |          new |      new |      new |            500 |     new |

这张表比你现在继续调 learning rate 有价值得多。

## 3.2 做 sparse union：BM25 + TF-IDF + BM25 variants

不要只用一个 BM25 ranking。你应该构造多个弱 retriever，然后 union：

```txt
candidate_pool =
  BM25_default_topN
  ∪ TFIDF_word_topN
  ∪ TFIDF_char_ngram_topN
  ∪ BM25_no_stopword_topN
  ∪ BM25_entity_query_topN
```

推荐先做这些：

| Retriever                        | 目的                          |
| -------------------------------- | --------------------------- |
| BM25 default                     | 当前主力                        |
| TF-IDF word unigram/bigram       | 补 BM25 漏掉的 lexical match    |
| TF-IDF char 3-5 gram             | 补数字、变体、拼写、实体片段              |
| BM25 no stemming / with stemming | 看 preprocessing 是否影响 recall |
| Query entity-only BM25           | 强化 claim 里的 named entities  |
| Query number/year-only BM25      | 强化年份、百分比、数量型证据              |

合并时不要直接拼接排名，先去重，然后保存每个 candidate 的来源特征：

```json
{
  "evidence_id": "...",
  "bm25_rank": 12,
  "tfidf_rank": 83,
  "char_tfidf_rank": null,
  "entity_bm25_rank": 4,
  "in_sources": ["bm25", "entity_bm25"]
}
```

这些 source features 后面可以给 reranker 或 classifier 用。

## 3.3 做 dense retrieval，但要控制规模

官方 Sentence Transformers 的 retrieve-and-rerank 思路就是：第一阶段用快速 retriever 找较大候选，第二阶段用更慢但更准的 CrossEncoder rerank；CrossEncoder 是对 query-document pair 联合编码打分，不能拿来扫整个 corpus，只适合 rerank 已经筛出的候选。([SentenceTransformers][1])

你的 dense retrieval 建议用轻量 bi-encoder：

```txt
sentence-transformers/all-MiniLM-L6-v2
sentence-transformers/multi-qa-MiniLM-L6-cos-v1
```

流程：

```txt
encode all evidence passages -> evidence embedding matrix
encode claim -> claim embedding
cosine similarity search -> dense top-100/top-200
```

如果 corpus 不大，直接矩阵乘法就够；如果 corpus 大，用 FAISS。然后做：

```txt
hybrid_candidates =
  BM25_top200
  ∪ TFIDF_top200
  ∪ dense_top200
```

注意：dense retrieval 不一定直接提升 top-3 F-score，但它很可能补上 BM25 漏掉的 paraphrase evidence。它的核心价值是 **recall diversification**。

## 3.4 用 RRF 合并候选，而不是乱拼分数

不同 retriever 的分数不可比，BM25 score、TF-IDF cosine、dense cosine 不在同一尺度。不要直接加原始分数。用 Reciprocal Rank Fusion：

[
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
]

一般 (k=60)。这样每个 retriever 只贡献排名信号。你可以生成：

```txt
BM25 only
BM25 + TFIDF union
BM25 + dense union
BM25 + TFIDF + dense union
BM25 + TFIDF + dense with RRF ranking
```

然后比较 candidate recall。

---

# 4. 第二阶段：Reranking，暂时保留 zero-shot MiniLM 做主线

你现在最强的是 zero-shot MiniLM，不要因为“fine-tuned 听起来更高级”就强行换掉它。你应该先把它作为 stable reranker：

```txt
hybrid candidate pool top-200/top-500
-> zero-shot MiniLM rerank
-> final top-3/top-5
-> classifier context top-20/top-50
```

官方文档里的 retrieve-and-rerank 框架本来就是这个逻辑：retriever 负责高效候选召回，cross-encoder 负责在较小候选集上精排。([SentenceTransformers][1])

## 4.1 Rerank 实验矩阵

你应该跑下面这张表：

| Candidate Source |  Pool N | Reranker         | Output k | Precision | Macro Recall | F-score | Hit-any |
| ---------------- | ------: | ---------------- | -------: | --------: | -----------: | ------: | ------: |
| BM25             |      50 | MiniLM zero-shot |        3 |  existing |       0.1877 |  0.1642 |  0.3961 |
| BM25             |     100 | MiniLM zero-shot |        3 |       new |          new |     new |     new |
| BM25             |     200 | MiniLM zero-shot |        3 |       new |          new |     new |     new |
| BM25+TFIDF       |     200 | MiniLM zero-shot |        3 |       new |          new |     new |     new |
| BM25+dense       |     200 | MiniLM zero-shot |        3 |       new |          new |     new |     new |
| BM25+TFIDF+dense | 300/500 | MiniLM zero-shot |        3 |       new |          new |     new |     new |

同时每个 candidate source 都测：

```txt
output top-1
output top-3
output top-5
output top-10
context top-20
context top-50
```

但报告时不要全塞进去，最后选关键配置。

## 4.2 加一个轻量 feature fusion，不要直接改神经模型

因为你的错误是“topic-relevant distractors”，即 passage 与 claim 同主题但不能支持/反驳 claim。你已经提取了 entities、percentages、years、negation、comparison、causality、relation verbs 等 semantic features；这些不应该替代神经模型，但可以做 reranking 后处理或 feature fusion。

给每个 claim-evidence pair 计算：

```txt
entity_overlap
year_overlap
number_overlap
percentage_overlap
negation_mismatch
comparison_cue_match
causal_cue_match
relation_verb_overlap
```

然后用 dev set 训练一个轻量 rank fusion model：

```txt
features =
[
  minilm_score,
  bm25_score_norm,
  rrf_score,
  entity_overlap,
  number_overlap,
  year_overlap,
  negation_mismatch,
  relation_overlap
]

model = LogisticRegression / LightGBM / small MLP
target = evidence_is_gold
```

如果不能用 LightGBM，就用 sklearn LogisticRegression。这个方法比继续 full fine-tune MiniLM 更稳，因为它不破坏 zero-shot MiniLM 的强排序能力，只是在上面加任务特征。

排序分数：

[
s(q,e) =
\alpha s_{\text{MiniLM}}(q,e)

* \beta s_{\text{RRF}}(q,e)
* \gamma f_{\text{entity/number}}(q,e)

- \delta f_{\text{mismatch}}(q,e)
  ]

不要一开始手写权重；先用 logistic regression 学。

---

# 5. 第三阶段：Hard Negatives，不要再用粗糙 BCE 随机训练

你之前的 hard negatives 是：

```txt
BM25 top-100 -> zero-shot MiniLM rerank -> high-ranked non-gold as negatives
```

这个方向是对的，但有三个问题：

1. 非 gold 不一定真 negative，可能是未标注但有效 evidence；
2. BCE 独立 pair classification 不直接优化 claim 内 ranking；
3. negatives 类型不够结构化，模型学到的可能还是 topic relevance。

## 5.1 重新设计 negative taxonomy

你要把 negatives 分层，而不是全叫 negative：

| Negative Type                | 来源                         | 目的                    |
| ---------------------------- | -------------------------- | --------------------- |
| Easy random negative         | random corpus              | 保持基本区分能力              |
| BM25 hard negative           | BM25 高排名非 gold             | 学 lexical distractor  |
| MiniLM hard negative         | MiniLM 高排名非 gold           | 学 semantic distractor |
| Entity-overlap negative      | 共享实体但非 gold                | 学“同实体不等于证据”           |
| Number/year-overlap negative | 共享数字/年份但非 gold             | 学“数字相似不等于支持”          |
| Relation-mismatch negative   | 实体相同但谓词/关系不同               | 学事实关系                 |
| REFUTES-aware negative       | refute claims 的高混淆 passage | 强化反驳证据                |

训练数据每个 claim 建议：

```txt
positive: all gold evidence
negative:
  2 easy random
  2 BM25 hard
  2 MiniLM hard
  1 entity/number hard
  1 class-aware hard
```

不要让 hard negatives 占比过高。过高会导致模型过度惩罚所有 topic-related passages，反而把真正 evidence 也压下去。

## 5.2 训练目标从 BCE 改成 pairwise ranking

BCE 目标是：

[
\mathcal{L}_{BCE}
= -y\log \sigma(s) - (1-y)\log(1-\sigma(s))
]

但你的实际目标是：同一个 claim 下 gold evidence 排在 non-gold evidence 前面。因此更合适的是 pairwise margin loss：

[
\mathcal{L}_{rank}
==================

\max(0, m - s(q,e^+) + s(q,e^-))
]

或者 RankNet loss：

[
\mathcal{L}_{ranknet}
=====================

-\log \sigma(s(q,e^+) - s(q,e^-))
]

这比 BCE 更贴近 reranking。你的训练样本应该是 triplet：

```txt
(claim, positive_evidence, negative_evidence)
```

而不是独立 pair。

## 5.3 暂时不要把 fine-tuned reranker 作为默认

fine-tuned 模型只有在下面条件满足后才值得重新上主线：

```txt
hybrid candidate recall 明显高于 BM25
+
fine-tuned reranker 在 dev top-3/top-5 F-score 超过 zero-shot
+
REFUTES recall 不只是偶然提升
```

否则报告里写：

```txt
We attempted task-aware fine-tuning, but zero-shot MiniLM remained the most stable reranker. Therefore, we used fine-tuning as an analysis direction rather than the final default system.
```

这是合理的，不丢人。真正丢人的是明明实验输了还硬用。

---

# 6. 第四阶段：Classifier，不要直接粗暴拼 top-20

你现在已经生成了 classifier-ready JSONL，里面区分了 `final_evidence_candidates` 和 `classifier_evidence_context`，这是对的。 但 classifier 设计不能只是：

```txt
claim + top20 evidence concatenated -> 4-way label
```

这很可能被噪声污染。你应该至少做两个 classifier baseline。

## 6.1 Baseline 1：简单 concat classifier

输入：

```txt
[CLAIM] claim_text
[E1] evidence_1
[E2] evidence_2
...
[E20] evidence_20
[SEMANTIC] entities=..., years=..., numbers=...
```

输出：

```txt
SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED
```

优点：实现快。
缺点：噪声大，长文本截断严重，不知道哪个 evidence 起作用。

实验：

| Context           | Semantic Summary | Classifier          | Accuracy | Macro-F1 | REFUTES F1 |
| ----------------- | ---------------- | ------------------- | -------: | -------: | ---------: |
| top-3             | no               | transformer/sklearn |      new |      new |        new |
| top-10            | no               | transformer/sklearn |      new |      new |        new |
| top-20            | no               | transformer/sklearn |      new |      new |        new |
| top-20            | yes              | transformer/sklearn |      new |      new |        new |
| top-50 compressed | yes              | transformer/sklearn |      new |      new |        new |

## 6.2 Baseline 2：Evidence-wise verifier + aggregation

这是我更推荐的结构。不要让分类器一次读 20 条 evidence，而是先判断每条 evidence 对 claim 的关系：

[
f(q,e_j) \rightarrow
[
s_j^{support},
s_j^{refute},
s_j^{neutral}
]
]

然后聚合：

[
S_{support} = \max_j s_j^{support}
]

[
S_{refute} = \max_j s_j^{refute}
]

[
S_{neutral} = \text{mean/top-k aggregate}(s_j^{neutral})
]

最后 claim label：

```txt
if support high and refute low -> SUPPORTS
if refute high and support low -> REFUTES
if support high and refute high -> DISPUTED
if both low -> NOT_ENOUGH_INFO
```

这比 concat 更贴近 fact-checking 任务，因为 SUPPORTS/REFUTES 本质上是 claim-evidence relation，不是普通 document classification。

训练数据构造：

```txt
positive support/refute evidence:
  claim label = SUPPORTS or REFUTES
  gold evidence = positive relation evidence

neutral evidence:
  retrieved non-gold candidates
  random evidence
  topically related distractors
```

DISPUTED 比较麻烦。可以先把它当成 claim-level label，不强行给单条 evidence 标 disputed；聚合时如果 support/refute 两边都有高分，再判 disputed。这个假设不一定完美，但比直接拼接更可控。

## 6.3 Classifier input context 推荐

你应该比较：

```txt
MiniLM top-3
MiniLM top-10
MiniLM top-20
MiniLM top-50
hybrid candidate top-50 after rerank
```

我的默认选择：

```txt
final evidence output:
  hybrid candidates -> zero-shot MiniLM -> top-3

classifier context:
  hybrid candidates -> zero-shot MiniLM -> top-20 or top-50

classifier model:
  evidence-wise verifier + aggregation
```

先跑 top-20。top-50 只有在模型能处理或你做 evidence-wise aggregation 时才用；直接 concat top-50 基本会炸，因为上下文太长且噪声太多。

---

# 7. 第五阶段：专门处理 REFUTES

你文档里明确说 `REFUTES` recall 弱，这非常关键。 REFUTES 通常不是简单相似，而是需要识别：

```txt
negation mismatch
number mismatch
direction mismatch
comparison mismatch
causal mismatch
temporal mismatch
```

例子：

```txt
claim: emissions increased after policy X
evidence: emissions decreased after policy X
```

BM25/MiniLM 都会觉得很相似，因为实体和关键词高度重合，但真正 label 是 REFUTES。

所以你要给 REFUTES 单独做 error analysis：

| Error Type                        | Example                  | Retrieval issue | Classifier issue  | Fix               |
| --------------------------------- | ------------------------ | --------------- | ----------------- | ----------------- |
| missing refuting evidence         | gold not in candidates   | recall          | -                 | hybrid retrieval  |
| retrieved support-like distractor | topic similar            | rerank          | relation mismatch | semantic features |
| number mismatch missed            | same entity, diff number | rerank          | numeric reasoning | number features   |
| negation missed                   | "not", "no", "never"     | classifier      | negation          | negation features |

并且在 classifier 中加入显式特征：

```txt
claim_has_negation
evidence_has_negation
negation_xor
claim_numbers
evidence_numbers
number_exact_match
number_conflict
claim_comparison_direction
evidence_comparison_direction
direction_conflict
```

不要指望 MiniLM 自己稳定学会这些，尤其数据量不大时。

---

# 8. 实验执行顺序：不要同时开太多坑

下面是我建议你的实际执行顺序。

## Phase 0：冻结当前最好 baseline

先把当前结果固定成 baseline：

```txt
Baseline A:
BM25 top-50 -> zero-shot MiniLM -> final top-3

Baseline B:
BM25 top-50 -> zero-shot MiniLM -> classifier top-20 context
```

保存：

```txt
predictions
metrics
error analysis
runtime
config yaml/json
```

不要让后面实验把 baseline 搞乱。

## Phase 1：只做 candidate recall

跑：

```txt
BM25 top100/top200/top500
TFIDF top100/top200
char-TFIDF top100/top200
BM25+TFIDF union
BM25+charTFIDF union
BM25+dense union
BM25+TFIDF+dense union
```

只看：

```txt
candidate recall
hit-any
all-gold
class-specific recall
runtime
```

阶段目标：

```txt
candidate macro recall 从 0.4188 提到至少 0.55+
hit-any 明显提升
REFUTES candidate recall 明显提升
```

如果做不到 0.55，也至少要证明你尝试过多源候选。

## Phase 2：rerank 最好的 2-3 个 candidate pools

不要 rerank 所有组合，太浪费。选 Phase 1 recall 最好的 2-3 个：

```txt
CandidatePool_1 = BM25 top200
CandidatePool_2 = BM25 + TFIDF union
CandidatePool_3 = BM25 + TFIDF + dense union
```

然后：

```txt
-> zero-shot MiniLM rerank
-> output top1/top3/top5/top10
-> context top20/top50
```

目标：

```txt
final evidence F-score > 0.1642
or
classifier context recall/hit-any 明显提升
```

注意：final evidence F-score 不涨但 context recall 涨，也有价值，因为它可能提升 classification。

## Phase 3：训练 classifier

先跑两个版本：

```txt
Classifier 1:
claim + top20 concat -> 4-way label

Classifier 2:
claim-evidence pair verifier -> aggregation -> 4-way label
```

对比：

```txt
context top3
context top10
context top20
context top50 evidence-wise
```

重点看：

```txt
overall accuracy
macro-F1
REFUTES F1
DISPUTED F1
confusion matrix
```

## Phase 4：再考虑 reranker fine-tuning

只有当前三件事完成后才回头做：

```txt
candidate recall improved
zero-shot rerank on hybrid pool measured
classifier baseline measured
```

然后再做：

```txt
pairwise ranking loss
structured hard negatives
feature fusion reranker
```

否则就是乱调。

---

# 9. 最推荐的最终系统形态

我建议你的最终系统长这样：

```txt
Input claim
  |
  |-- Sparse retrieval:
  |     BM25 top200
  |     TFIDF top200
  |     char-TFIDF top100
  |     entity/year query BM25 top100
  |
  |-- Dense retrieval:
  |     MiniLM bi-encoder top200
  |
  |-- Candidate union + RRF
  |     keep top300 or top500
  |
  |-- Cross-encoder rerank:
  |     zero-shot ms-marco-MiniLM-L6-v2
  |
  |-- Branch A:
  |     final evidence output = top3
  |
  |-- Branch B:
  |     classifier context = top20/top50
  |     + semantic summary
  |     + evidence-wise verifier / classifier
  |
  |-- Output:
        evidence IDs + claim label
```

这就是一个标准的多阶段 retrieval/classification pipeline：cheap high-recall retrieval，expensive precise reranking，最后 task-specific classification。

---

# 10. 代码结构建议

你现在 repo 已经有模块化结构，这是好事。建议进一步拆成这些脚本：

```txt
experiments/retrieval/
  01_build_bm25_candidates.py
  02_build_tfidf_candidates.py
  03_build_dense_candidates.py
  04_merge_candidates_rrf.py
  05_eval_candidate_recall.py

experiments/rerank/
  01_rerank_with_minilm.py
  02_eval_reranked_outputs.py
  03_train_feature_fusion_reranker.py
  04_train_pairwise_reranker.py

experiments/classification/
  01_make_classifier_dataset.py
  02_train_concat_classifier.py
  03_train_pairwise_verifier.py
  04_eval_claim_classifier.py

experiments/analysis/
  01_error_by_label.py
  02_error_by_missing_gold.py
  03_refutes_error_analysis.py
  04_make_ablation_table.py
```

每个输出文件名都要带 config：

```txt
outputs/candidates/bm25_top200_dev.jsonl
outputs/candidates/hybrid_bm25_tfidf_dense_rrf_top500_dev.jsonl
outputs/rerank/hybrid_top500_minilm_dev.jsonl
outputs/classifier/dev_context_hybrid_minilm_top20_semantic.jsonl
```

不要用 `round06_final_v2_real_final.jsonl` 这种名字。会把自己搞死。

---

# 11. 报告写法：你应该怎样解释当前困难

报告里不要说：

```txt
Our model performs poorly because the task is hard.
```

要说：

```txt
The main bottleneck is first-stage candidate recall. BM25 top-100 macro recall is only 0.4188, which imposes an upper bound on any reranker operating only over BM25 candidates. Although zero-shot MiniLM improves final evidence F-score to 0.1642 at top-3, many false positives remain topic-relevant but not evidence-relevant. Therefore, we separate final evidence output from classifier evidence context: top-3 is used for compact final evidence prediction, while top-20/top-50 reranked candidates are used as wider classifier context.
```

这段逻辑是强的，因为它有实验支撑。你文档里的 problem framing 已经有这些事实，只需要系统化呈现。

---

# 12. 你现在最不该做的事

直接说：

1. **不要继续盲目 fine-tune MiniLM。** 当前 fine-tuning 没赢 zero-shot，说明你的训练目标/负样本/候选池至少有一个有问题。
2. **不要把 top-20 当 final evidence output。** 你自己已经看到 top-20 precision 很差，final F-score 会掉。
3. **不要只做 classifier，然后希望它自动修复 retrieval。** 如果 gold evidence 不在 context 里，classifier 学不到正确决策。
4. **不要把 semantic features 写成规则系统。** 它们应该是辅助信号，不是主模型。
5. **不要只报 final F-score。** 必须报 candidate recall、hit-any、all-gold、class-specific recall，尤其 REFUTES。
6. **不要用 dev set 反复调到过拟合后还说泛化好。** dev 可以做选择，但报告要诚实说明 test 没用。

---

# 13. 最小可行救火路线

如果时间很紧，你就按这个做：

```txt
Step 1:
Evaluate BM25 top200/top500 candidate recall.

Step 2:
Build BM25 top200 + TFIDF top200 union.
Evaluate candidate recall.

Step 3:
If possible, add dense top200.
Evaluate hybrid recall.

Step 4:
Use zero-shot MiniLM rerank best candidate pool.
Output final top3.

Step 5:
Create classifier context top20/top50 from same reranked list.

Step 6:
Train one simple classifier:
claim + top20 evidence + semantic summary -> 4 labels.

Step 7:
Make ablation table:
BM25
BM25+MiniLM
Hybrid+MiniLM
Hybrid+MiniLM+classifier context
```

这条路线最稳。它不追求花哨，但能最大概率救分。

---

# 14. 更强但稍复杂的路线

如果你还有时间，做这个：

```txt
Hybrid retrieval:
  BM25 + TFIDF + dense + entity/year query

Rerank:
  zero-shot MiniLM
  + logistic feature fusion

Classifier:
  evidence-wise verifier
  + aggregation rule/model

Fine-tuning:
  pairwise ranking loss
  + structured hard negatives
```

最终系统：

```txt
Final evidence:
  hybrid -> MiniLM/fusion -> top3

Claim label:
  hybrid -> MiniLM/fusion -> top20/top50
  -> evidence-wise verifier
  -> aggregate to SUPPORTS/REFUTES/NEI/DISPUTED
```

---

# 15. 你接下来第一天应该产出的东西

不要再写新模型。第一天只产出这张表：

| Candidate Source |       N | Macro Recall |  Hit-any | All-gold | SUPPORTS Recall | REFUTES Recall | NEI Recall | DISPUTED Recall |
| ---------------- | ------: | -----------: | -------: | -------: | --------------: | -------------: | ---------: | --------------: |
| BM25             |      50 |       0.3249 | existing |      new |             new |            new |        new |             new |
| BM25             |     100 |       0.4188 | existing |      new |             new |            new |        new |             new |
| BM25             |     200 |          new |      new |      new |             new |            new |        new |             new |
| BM25             |     500 |          new |      new |      new |             new |            new |        new |             new |
| BM25+TFIDF       |     200 |          new |      new |      new |             new |            new |        new |             new |
| BM25+TFIDF+dense | 300/500 |          new |      new |      new |             new |            new |        new |             new |

这张表出来之前，任何 reranker fine-tuning 都是赌博。

---

# 16. 最终判断

你的方向不是没希望，而是现在必须停止“模型崇拜”。这个任务的核心不是把 MiniLM 多训几轮，而是构建一个 **recall-aware, task-aware, classifier-aware** 的 pipeline。你已经发现了最关键的错误模式：topic relevance 不等于 evidence relevance；现在要把这个观察转化为系统设计，而不是继续在同一个 BM25 top-50 候选池里反复调参。

现在最强行动方案就是：

```txt
先扩大候选召回：
  BM25 top200/top500
  + TFIDF
  + dense retrieval
  + RRF union

再稳定 rerank：
  zero-shot MiniLM as default
  optional feature fusion

再做分类：
  top20/top50 context
  evidence-wise verifier preferred

最后才考虑：
  pairwise hard-negative reranker fine-tuning
```

按这个顺序做，你的实验会从“零散尝试”变成“有上限分析、有消融、有系统结构”的完整 assignment solution。

[1]: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html?utm_source=chatgpt.com "Retrieve & Re-Rank Pipeline"
