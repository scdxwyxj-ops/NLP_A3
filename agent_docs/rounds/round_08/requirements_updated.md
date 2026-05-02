你这轮审计是对的，而且比我上一轮建议更贴近真实约束。我要更新判断：**Round08 不应再围绕“加更多 retrieval 花样”展开，而应围绕“在扁平 evidence 数据结构下，如何最大化利用现有 top500 候选”展开。** 我之前建议里的 `title/body field-aware BM25`、`prev/current/next evidence expansion`、`document-level diversity` 基本要撤回，因为你已经确认 `data/evidence.json` 是扁平的 `evidence_id -> evidence text`，没有 title、document id、section、sentence index、previous/next sentence 等字段；连续 evidence id 也明显不是同一文档相邻句，gold evidence id 间距也不支持用 id 邻近恢复上下文。 

---

# 更新后的核心判断

Round07 后，问题已经不是：

```txt
BM25 / TF-IDF 太原始，所以继续救 candidate recall
```

而是：

```txt
candidate top500 中经常已经有 gold，
但 MiniLM / evidence selection 没有把它稳定推到 top20/top3。
```

所以 Round08 主线应该固定为：

```txt
Round07 sparse RRF baseline
-> error layer analysis
-> feature table construction
-> lightweight feature fusion reranker
-> classifier context selection
-> optional dense supplement
```

这和你审计后的优先级一致：error layer analysis、feature table、feature fusion reranker 应该排在 dense supplement 和 evidence-wise verifier 之前。

---

# 一、需要撤回或降级的建议

## 1. 撤回：真正的 field-aware BM25

我之前建议的：

```txt
BM25_title
BM25_body
BM25_section
BM25_document
```

在你当前数据下不可实现。因为 evidence 只有扁平文本，没有 title/body/section/document metadata。你最多能做退化版本：

```txt
BM25 over full evidence text
BM25 over extracted entity string
BM25 over extracted number/year string
```

但这不是真正的 field-aware retrieval，只是 query / feature ablation。不要把它写成主线，否则报告会显得不诚实。

---

## 2. 撤回：prev/current/next evidence expansion

这个也不能做。你已经审计出：

```txt
evidence-67730 / 67731 / 67732 / 67733 来自完全不同主题
```

并且 multi-evidence claim 的 gold evidence id 之间没有小间距关系，所以不能用 `evidence_id +/- 1` 恢复邻近句。

因此不要再做：

```txt
title + previous sentence + current sentence + next sentence
```

除非你找到额外 metadata。

---

## 3. 降级：claim decomposition

复杂 decomposition 暂时不该做。你没有 spaCy、dependency parser、entity linker，而且 query-boost ablation 已经没赢默认 pipeline。现在最多做低成本 ablation：

```txt
original claim
capitalized/domain entities query
number/year/percentage query
content-word query
```

但它不应进入 Round08 主线。你已有的 semantic extractor 可以抽取 entities、percentages、quantities、years、negation、comparison、causality、relation verbs，这些更适合作为 reranker features，而不是直接做复杂 query decomposition。

---

## 4. 降级：relation / comparison / contradiction features

第一版不要做：

```txt
subject/object consistency
comparison direction parsing
dependency relation matching
true contradiction detection
```

这些没有 parser 支撑，规则会很脆。保留浅层版本：

```txt
entity_overlap_count
entity_jaccard
number_overlap_count
year_overlap_count
percentage_overlap_count
claim_has_negation
evidence_has_negation
negation_presence_match
negation_xor
comparison_cue_overlap
causality_cue_overlap
relation_verb_overlap
```

这些可以进 feature fusion reranker。

---

# 二、Round08 的真正第一步：Error Layer Analysis

先不要急着训练新模型。第一件事是把错误分层算清楚：

```txt
gold not in top500
gold in top500 but not in MiniLM top50
gold in MiniLM top50 but not in top20
gold in top20 but not in top3
gold in top20/top50 but classifier label wrong
```

定义：

[
G_i = \text{gold evidence set for claim } i
]

[
C_i^{500} = \text{RRF candidate top500}
]

[
R_i^{50} = \text{MiniLM reranked top50}
]

[
R_i^{20} = \text{MiniLM reranked top20}
]

[
R_i^{3} = \text{final top3}
]

然后统计：

[
\text{missing@500}_i = |G_i - C_i^{500}|
]

[
\text{lost@50}_i = |(G_i \cap C_i^{500}) - R_i^{50}|
]

[
\text{lost@20}_i = |(G_i \cap R_i^{50}) - R_i^{20}|
]

[
\text{lost@3}_i = |(G_i \cap R_i^{20}) - R_i^{3}|
]

这个分析会决定后面做什么。如果大多数损失是 `top500 -> top50/top20`，就做 reranker；如果大多数损失是 `not in top500`，再救 dense/sparse recall。你目前的数据暗示前者更重要，但必须用表确认。

输出表建议：

| Layer                           | Count | Claim % | Gold evidence % | Main Fix                        |
| ------------------------------- | ----: | ------: | --------------: | ------------------------------- |
| gold not in top500              |   new |     new |             new | dense supplement / retrieval    |
| gold in top500 but not top50    |   new |     new |             new | rerank more candidates / fusion |
| gold in top50 but not top20     |   new |     new |             new | feature fusion                  |
| gold in top20 but not top3      |   new |     new |             new | final selection                 |
| gold in context but label wrong |   new |     new |             new | classifier aggregation          |

---

# 三、Feature Fusion Reranker 是现在最现实的主线

你的审计里最关键的一点是：当前 ranked top50 只保留了 MiniLM score/rank，RRF top500 只保留了 RRF score/rank；BM25 rank、char TF-IDF rank、source_count 等需要从候选池文件重新 join。

所以第一步不是训练模型，而是构造 feature table。

## 3.1 Feature table 结构

每一行：

```txt
claim_id
evidence_id
label_is_gold

minilm_rank
minilm_score
in_minilm_top50

rrf_rank
rrf_score

bm25_rank
bm25_score
in_bm25

char_tfidf_rank
char_tfidf_score
in_char_tfidf

source_count

entity_overlap_count
entity_jaccard
number_overlap_count
year_overlap_count
percentage_overlap_count
claim_has_negation
evidence_has_negation
negation_presence_match
negation_xor
comparison_cue_overlap
causality_cue_overlap
relation_verb_overlap
```

其中：

```txt
source_count = in_bm25 + in_char_tfidf
```

如果 dense 后面跑完，再加：

```txt
dense_rank
dense_score
in_dense
source_count_3way
```

---

## 3.2 一个重要坑：只用 MiniLM top50 会限制 fusion 上限

你现在的 `dev-rrf-bm25-char-minilm-ranked-top50.json` 只保存了 MiniLM top50。

这意味着如果你训练 fusion reranker 只重排 top50，它不能救：

```txt
gold in top500 but not in MiniLM top50
```

所以 feature fusion 有两个版本：

### 版本 A：Cheap Fusion，重排 MiniLM top50

优点：马上能做。
缺点：只能改善 top50 内排序，不能救 top50 之外的 gold。

```txt
RRF top500
-> MiniLM top50
-> feature fusion rerank top50
-> output top3/top20
```

这是第一版，可以快速验证 features 是否有用。

### 版本 B：Fuller Fusion，重新给更多候选打 MiniLM 分数

更推荐，但更慢：

```txt
RRF top500
-> MiniLM score top100 / top200 / top500
-> feature fusion rerank
-> output top3/top20
```

我建议先做：

```txt
MiniLM score top100
```

不要一上来 top500。top100 是性能和成本的折中。如果 top100 fusion 有提升，再扩到 top200。

---

## 3.3 模型选择

第一版不要用神经网络。用：

```txt
LogisticRegression(class_weight="balanced")
LinearSVC(class_weight="balanced")
RandomForest / GradientBoosting if already available
```

排序分数：

[
s(q,e)=P(y_{\text{gold}}=1\mid \phi(q,e))
]

按 claim group 内排序，输出 top-k。

注意：不要 row-level random split。必须：

```txt
train claims -> train
dev claims -> dev
```

否则同一个 claim 的候选泄漏会让结果虚高。

---

## 3.4 负样本比例

如果每个 claim top500 全用，负样本极多。可以先用：

```txt
positive:
  all gold candidates inside candidate pool

negative:
  top-ranked non-gold candidates
  MiniLM high-score non-gold candidates
  random non-gold candidates from top500
```

第一版建议：

```txt
每个 claim:
  all positives
  top 30 non-gold by MiniLM/RRF
  random 30 non-gold from remaining top500
```

然后用 `class_weight="balanced"`。

---

# 四、Dense retrieval 的位置：可以做，但不能阻塞主线

你的审计已经说明 dense retrieval 不是零成本：`sentence_transformers` 和 `faiss` 都没装，evidence 有 1,208,827 条，dev 154 条，train 1228 条。

所以 dense 的优先级应是：

```txt
主线之外的 supplement ablation
```

不要让 dense 卡住 Round08。

## 4.1 Dense 最小可行方案

```txt
install sentence-transformers
encode all evidence once
cache evidence embeddings
chunked exact search
save dense top500
RRF(BM25, char TF-IDF, dense)
```

重点是 **cache embeddings**。否则每次重跑都会浪费大量时间。

## 4.2 Dense 评估表

只看 candidate 层：

| Candidate Pool   | Macro Recall | Hit-any | All-gold | REFUTES Recall |
| ---------------- | -----------: | ------: | -------: | -------------: |
| BM25 top500      |       0.5861 |  0.8377 |   0.3247 |         0.5123 |
| char top500      |       0.6610 |  0.8896 |   0.4026 |         0.5556 |
| sparse RRF       |       0.6579 |  0.8896 |   0.4091 |         0.5926 |
| dense top500     |          new |     new |      new |            new |
| sparse+dense RRF |          new |     new |      new |            new |

如果 dense 不提升，就写成 ablation，不要继续纠缠。

---

# 五、Classifier 下一步也要降级成现实版本

我之前提 transformer concat classifier，现在要降级。原因很简单：

```txt
top20/top50 太长
噪声多
512 token 截断严重
Colab 训练稳定性不确定
```

所以现在 classifier 的实际顺序应该是：

```txt
1. 先用 fusion reranker 产生更可信 top20/top50
2. 重新跑现有 TF-IDF logreg baseline
3. 再考虑 evidence-wise verifier
4. 最后才考虑 transformer concat classifier
```

Evidence-wise verifier 仍然合理，但 label 要谨慎：

```txt
SUPPORTS claim + gold evidence -> support pair
REFUTES claim + gold evidence -> refute pair
retrieved non-gold evidence -> neutral pair
random evidence -> neutral pair
```

不要直接把 `DISPUTED` 或 `NOT_ENOUGH_INFO` 转成 pair-level positive。你审计里这一点是对的：`DISPUTED` 更像 claim-level aggregation 结果，不是一条 evidence 的固有标签。

---

# 六、更新后的 Round08 执行顺序

现在我建议按这个顺序做。

## Stage A：关闭 Round07，重定义 Round08

写清楚：

```txt
Round07:
  candidate recall rescue achieved

Round08:
  bottleneck shifts to reranking / evidence selection / classifier aggregation
```

不要再说“BM25/TF-IDF 太弱”作为主诊断。这个说法已经不精确。

---

## Stage B：Error Layer Analysis

脚本：

```txt
experiments/analysis/error_layer_analysis.py
```

输入：

```txt
data/dev-claims.json
outputs/round07/candidates/rrf_bm25_char_tfidf_top500.json
outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json
outputs/round07/dev-classifier-context-rrf-bm25-char-minilm-top20-semantic.jsonl
```

输出：

```txt
outputs/round08/error_layer_summary.csv
outputs/round08/error_layer_by_claim.csv
```

这一步必须先做。

---

## Stage C：Feature Table Construction

脚本：

```txt
experiments/rerank/build_feature_table.py
```

输入：

```txt
outputs/round07/candidates/bm25_top500.json
outputs/round07/candidates/tfidf_char_top500.json
outputs/round07/candidates/rrf_bm25_char_tfidf_top500.json
outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json
data/dev-claims.json
data/evidence.json
src/a3_factcheck/semantic/features.py
```

输出：

```txt
outputs/round08/dev_feature_table_top500.csv
```

如果有 train candidates，也做：

```txt
outputs/round08/train_feature_table_top500.csv
```

没有 train feature table，就不能严肃训练 fusion reranker，只能做 dev diagnostic。

---

## Stage D：Feature Fusion Reranker

第一版：

```txt
RRF top500
-> MiniLM top50
-> fusion rerank top50
```

对比：

| Method          | Candidate Scope | Output k | F-score | Macro Recall | Hit-any | REFUTES Recall |
| --------------- | --------------- | -------: | ------: | -----------: | ------: | -------------: |
| MiniLM only     | top50           |        3 |  0.1987 |       0.2331 |  0.4545 |         0.1790 |
| Fusion reranker | top50           |        3 |     new |          new |     new |            new |
| MiniLM only     | top50           |       20 |  0.1106 |       0.4540 |  0.7143 |         0.3951 |
| Fusion reranker | top50           |       20 |     new |          new |     new |            new |

第二版，如果时间够：

```txt
RRF top500
-> MiniLM score top100
-> fusion rerank top100
```

这更可能提升 top20/top3。

---

## Stage E：Dense Supplement Ablation

Dense 不要阻塞 Stage B-D。等它跑完，做：

```txt
dense top500
RRF(BM25,char,dense) top500
```

然后只判断是否保留。

---

## Stage F：Classifier Context Update

如果 fusion reranker 提升了 top20/top50 recall，就重新构造：

```txt
dev-classifier-context-fusion-top20.jsonl
dev-classifier-context-fusion-top50.jsonl
```

然后先跑现有 baseline：

```txt
TF-IDF over claim + evidence context
LogisticRegression(class_weight="balanced")
```

目标不是立刻做到很强，而是回答：

```txt
更好的 evidence context 是否让 classifier macro-F1 从 0.4242 往上走？
```

---

# 七、更新后的验收目标

Round08 不应验收“做了很多模型”，而应验收这四件事：

```txt
1. 错误分层表：
   明确 loss 主要发生在 top500->top50/top20/top3 的哪一层。

2. Feature table：
   成功 join RRF、BM25、char TF-IDF、MiniLM、semantic overlap features。

3. Fusion reranker：
   至少和 MiniLM-only 比较 top3/top20 evidence metrics。

4. Classifier context：
   给出 top20 vs top50 的证据上下文选择，并重新跑 classifier baseline。
```

Dense 是 bonus：

```txt
5. Dense supplement:
   如果跑完，就报告 sparse+dense RRF 是否提升 all-gold / REFUTES recall。
```

---

# 最终更新版建议

现在最现实、最强的 Round08 主线是：

```txt
不要继续幻想数据里不存在的 structure。
不要做 title/body field-aware。
不要做 prev/next evidence expansion。
不要一上来做复杂 relation parsing。
不要让 dense retrieval 阻塞主线。

固定 Round07:
  RRF(BM25 top500, char TF-IDF top500) -> MiniLM

先做:
  error layer analysis

再做:
  feature table construction

然后:
  lightweight feature fusion reranker

最后:
  用 fusion top20/top50 更新 classifier context
```

一句话：**你现在不是 retrieval 特征不够丰富的问题，而是 flat evidence 条件下的 reranking information integration 问题。** 先把已有信号 `RRF rank + BM25/char rank + MiniLM score + shallow semantic overlap` 融合好，比继续追求不存在的 document structure 更有价值。
