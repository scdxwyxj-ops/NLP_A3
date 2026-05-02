你的直觉对一半：**BM25/TF-IDF 确实还很原始，但现在最主要的低 hit-any 不是第一阶段特征不够，而是“候选池里有证据，但 MiniLM 没把它排进 top-k”**。看数字要分清两种 hit-any：`RRF(BM25,char) top500` 的 candidate hit-any 已经是 `0.8896`，这不低；真正低的是 rerank 后 final top-3 hit-any `0.4545`，top-20 hit-any `0.7143`。所以 Round08 的主瓶颈已经从 **candidate generation** 转移到 **reranking / evidence selection / classifier aggregation**。继续无脑加 sparse 特征会有收益，但不是最大收益。

---

## 1. 先判断：Round07 成功了吗？

成功，而且是实质性成功。

你原来的上限是：

```txt
BM25 top100 candidate macro recall = 0.4188
BM25 top50 -> MiniLM top3 evidence F = 0.1642
REFUTES recall = 0.0494
```

现在是：

```txt
RRF(BM25 top500, char TF-IDF top500) candidate macro recall = 0.6579
candidate hit-any = 0.8896
candidate all-gold = 0.4091
MiniLM top3 evidence F = 0.1987
MiniLM top3 REFUTES recall = 0.1790
```

这说明你之前的判断是正确的：**第一阶段 recall rescue 确实能提高最终 evidence F-score**。

但别被 `0.8896 hit-any` 骗了。hit-any 高只说明“至少一个 gold 在候选池里”，不说明所有 evidence 都在，也不说明排序对。你的 all-gold 只有 `0.4091`，top-3 hit-any 只有 `0.4545`，所以现在的问题是：

[
\text{candidate pool has some gold} ;;\not\Rightarrow;; \text{reranker selects correct evidence set}
]

这就是 Round08 的核心。

---

## 2. BM25 和 TF-IDF 是不是太原始？

是，但不要简单理解成“换成 dense 就解决”。更准确地说：

```txt
BM25 / TF-IDF 当前太原始的地方不是算法本身，
而是 query、document、field、feature 和 fusion 都太朴素。
```

BM25 和 TF-IDF 仍然是强 baseline。你现在 char TF-IDF top500 的 macro recall `0.6610` 甚至略高于 RRF 的 `0.6579`，这说明 lexical retrieval 还没榨干。真正的问题是你现在只用了“claim 原文 vs evidence 文本”的平面匹配，没有充分利用 fact-checking 里的结构信息：

```txt
entity
number
year
comparison direction
negation
relation verb
evidence title/page context
claim decomposition
class-aware retrieval
multi-hop evidence coverage
```

所以不要问“BM25 原不原始”，应该问：

```txt
BM25/TF-IDF 是否 field-aware？
是否 entity-aware？
是否 number-aware？
是否 claim-decomposition-aware？
是否 reranker-aware？
是否 evidence-set-aware？
```

答案：目前大概率都还不够。

---

## 3. 现在不要继续大幅细调 sparse retrieval，但可以做 4 个低成本增强

你说 recall 还不够，这个担心合理，尤其 all-gold 只有 `0.4091`。但是 Round08 的主线不应该是继续把 sparse retrieval 从 0.6579 磨到 0.68，而是做 **reranker 和 classifier**。不过下面 4 个 sparse 增强值得做，因为成本低、可解释、容易写进报告。

---

### 3.1 Field-aware BM25：标题 / 正文 / 实体分开打分

如果 evidence 有 title、page、sentence、section 或 document id，不要把它们直接拼成一个字符串。应该分别算分：

[
s(q,e) =
\alpha s_{\text{body}}(q,e)

* \beta s_{\text{title}}(q,e)
* \gamma s_{\text{entity}}(q,e)
* \delta s_{\text{number}}(q,e)
  ]

具体做法：

```txt
BM25_body
BM25_title
BM25_entity_string
BM25_number_year_string
char_TFIDF_body
char_TFIDF_title
```

然后 RRF：

```txt
RRF(
  BM25_body top500,
  BM25_title top300,
  BM25_entity top300,
  BM25_number_year top300,
  char_TFIDF_body top500
)
```

注意：你之前 query-boost top3 F-score 只有 `0.1927`，低于 `0.1987`。这不代表 entity/number 没用，只代表“直接 query expansion 后交给 MiniLM top3”没用。更正确的用法是 **field-aware candidate supplement** 或 **reranker feature**，不是把 query 粗暴变长。

---

### 3.2 Claim decomposition：把复杂 claim 拆成多个子查询

很多 claim 不是一个事实，而是复合事实：

```txt
A caused B in year Y
A increased by X percent after policy P
Person/organization claimed that ...
```

直接用整句检索会稀释关键词。你应该为每个 claim 生成多个 query：

```txt
q0 = original claim
q1 = entity-only query
q2 = entity + year + number query
q3 = entity + relation verb query
q4 = noun chunks / keyphrase query
q5 = without stopwords / without attribution words
```

然后每个 query 各自取 topN，最后 RRF。不要手动写复杂规则，先做简单版本：

```python
queries = [
    original_claim,
    " ".join(entities),
    " ".join(entities + years + numbers),
    " ".join(entities + relation_verbs),
    " ".join(key_nouns)
]
```

这比“把所有东西 append 到一个 query”更稳，因为 RRF 会保留不同 query 的检索路径。

---

### 3.3 Evidence expansion：给短 evidence 加局部上下文

如果 evidence 是 sentence-level，很多句子本身信息不完整。例如：

```txt
"That figure rose to 42% in 2018."
```

单看这句话没实体，BM25/TF-IDF 和 MiniLM 都会错。你应该构造 expanded evidence text：

```txt
evidence_for_retrieval =
  title + previous_sentence + current_sentence + next_sentence
```

但是 final output 仍然返回 current evidence id。这个技巧经常比换模型更有用，因为它修复的是输入信息不足。

实验表：

| Evidence Text                  | Candidate Recall | All-gold | Final F |
| ------------------------------ | ---------------: | -------: | ------: |
| sentence only                  |          current |  current | current |
| title + sentence               |              new |      new |     new |
| prev + sentence + next         |              new |      new |     new |
| title + prev + sentence + next |              new |      new |     new |

这一步我强烈建议做。它很可能提升 all-gold 和 REFUTES。

---

### 3.4 BM25 parameter sweep：别用默认参数迷信

BM25 不是只有一种。可以扫：

```txt
k1 = [0.8, 1.2, 1.5, 2.0]
b  = [0.2, 0.5, 0.75, 0.9]
```

短 evidence 场景下，`b` 过高可能惩罚长文本；expanded evidence 场景下，`b` 又可能需要重新调。这个实验成本低，但报告价值高。

---

## 4. Dense retrieval 现在慢，怎么处理？

Dense 慢有两种可能：

1. 你在用 cross-encoder 做 dense retrieval，这是错的；
2. 你在重复 encode evidence，没有缓存；
3. 你在 CPU 上暴力扫全库；
4. 你 batch 太小；
5. 你没有用 FAISS / matrix multiplication。

标准做法是 **bi-encoder retrieve，cross-encoder rerank**。Sentence Transformers 官方文档也是这个框架：先用 SentenceTransformer 计算 query/document embedding 做 semantic search，再用 CrossEncoder 对候选重排；CrossEncoder 更准，但因为每个 query-document pair 都要单独计算，所以通常只适合 rerank top-k，不适合扫全库。([SentenceTransformers][1])

你 dense 应该这样跑：

```txt
Step 1: encode all evidence once
Step 2: save evidence_embeddings.npy
Step 3: encode all dev claims once
Step 4: claim_embeddings @ evidence_embeddings.T
Step 5: topk
Step 6: save dense_top500_candidates.jsonl
```

如果 evidence 数量不是特别大，直接矩阵乘法就够。如果库大，用 FAISS；FAISS 是专门做大规模向量相似搜索和聚类的库，支持 Python wrapper 和 GPU，一些算法也能处理超过 RAM 的向量集合。([Faiss][2])

### Dense 最小实现建议

先别上复杂 ANN。先用 exact search：

```python
# pseudo
evidence_emb = model.encode(
    evidence_texts,
    batch_size=256,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

claim_emb = model.encode(
    claim_texts,
    batch_size=256,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

scores = claim_emb @ evidence_emb.T
topk = np.argpartition(-scores, kth=500, axis=1)[:, :500]
```

如果内存炸，再切 chunk：

```python
for evidence_chunk in chunks(evidence_emb, 50000):
    scores = claim_emb @ evidence_chunk.T
    keep local topk
merge global topk
```

如果还慢，上 FAISS：

```python
import faiss

index = faiss.IndexFlatIP(dim)
index.add(evidence_emb.astype("float32"))
scores, ids = index.search(claim_emb.astype("float32"), 500)
```

`IndexFlat` 是精确搜索，本质上会顺序比较所有向量，所以准确但大库会慢；FAISS wiki 也明确说 flat index 会把所有 indexed vectors 解码并和 query 比较。([GitHub][3]) 如果数据大，再考虑：

```txt
IndexHNSWFlat
IndexIVFFlat
```

但别一开始就上 IVF/PQ，因为 approximate search 会牺牲 recall，而你现在最怕的就是 recall 损失。

---

## 5. Dense retrieval 应该怎么并入主线？

不要直接替换当前 RRF。dense 应该作为 supplement：

```txt
candidate_pool =
  RRF(
    BM25 top500,
    char TF-IDF top500,
    dense top500
  )
```

然后比较四张表：

| Candidate Pool     | Macro Recall | Hit-any | All-gold | REFUTES Recall |
| ------------------ | -----------: | ------: | -------: | -------------: |
| BM25 top500        |       0.5861 |  0.8377 |   0.3247 |         0.5123 |
| char TF-IDF top500 |       0.6610 |  0.8896 |   0.4026 |         0.5556 |
| RRF sparse         |       0.6579 |  0.8896 |   0.4091 |         0.5926 |
| dense top500       |          new |     new |      new |            new |
| RRF sparse + dense |          new |     new |      new |            new |

你要注意一个可能结果：dense top500 可能 macro recall 不如 char TF-IDF，但 RRF 后 all-gold 或 REFUTES 变好。这就有价值。

---

## 6. 现在最该做的是：Feature Fusion Reranker

你的 top500 candidate hit-any 已经 `0.8896`，但是 MiniLM top3 hit-any 只有 `0.4545`。这说明：

```txt
gold evidence often exists in candidate pool,
but MiniLM does not reliably rank it into top3.
```

所以 Round08 第一主线应该是 **fusion reranker**，不是继续扩候选。

### 6.1 为什么 MiniLM 会错？

MS MARCO MiniLM 主要学的是 query-passage relevance，不是 fact verification。它容易把下面这种证据排高：

```txt
claim: X caused Y in 2015
candidate: X and Y are both mentioned, and 2015 appears
```

但真正 evidence 可能是：

```txt
candidate: Y happened before X, so X could not have caused Y
```

MiniLM 会偏向“语义相关”，不一定偏向“事实支持/反驳”。

所以你应该给每个 claim-evidence pair 加特征：

```txt
minilm_score
rrf_score
bm25_rank
char_tfidf_rank
entity_overlap
entity_missing_rate
number_overlap
year_overlap
percentage_overlap
negation_match
negation_xor
comparison_direction_match
relation_verb_overlap
evidence_position_features
candidate_source_count
```

然后训练：

```txt
LogisticRegression / LinearSVM / XGBoost / LightGBM
target = evidence_is_gold
```

如果不想加新依赖，就用 sklearn：

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
)
```

排序分数：

[
s(q,e)=
w_1s_{\text{MiniLM}}
+w_2s_{\text{RRF}}
+w_3f_{\text{entity}}
+w_4f_{\text{number}}
+w_5f_{\text{year}}
-w_6f_{\text{negation-conflict}}
+\cdots
]

然后 rerank top500，输出 top3。

这一步非常适合你，因为你已经有：

```txt
candidate pool
MiniLM scores
semantic features
gold evidence labels
```

不用训练 transformer，速度快，解释性强，报告好写。

---

## 7. 再做 Evidence Set Selection，而不是独立 top-k

你现在是：

```txt
score each evidence independently
take top3
```

这对 fact-checking 不够，因为 evidence 是集合。有些 top3 互相重复，可能都来自同一文档、同一角度，导致 all-gold 不上来。

应该加 diversity / coverage：

```txt
selected = []
while len(selected) < 3:
    choose candidate maximizing:
      rerank_score
      + entity_coverage_gain
      + number_coverage_gain
      + new_document_bonus
      - redundancy_penalty
```

形式上：

[
\text{score}(e)
===============

s_{\text{rerank}}(e)
+\lambda \text{coverage_gain}(e)
-\mu \max_{e' \in S}\text{sim}(e,e')
]

简单实现：

```python
final_score = (
    rerank_score
    + 0.1 * new_entity_gain
    + 0.1 * new_number_gain
    + 0.05 * new_doc_bonus
    - 0.1 * max_similarity_to_selected
)
```

这对 all-gold 可能有帮助，尤其多证据 claim。

---

## 8. Classifier 下一阶段：不要再停留在 TF-IDF logreg

你现在 classifier baseline：

```txt
claim + evidence context
TF-IDF
LogisticRegression balanced
Accuracy = 0.4610
Macro-F1 = 0.4242
```

这个 baseline 合格，但太弱。下一阶段建议两条线并行。

---

### 8.1 路线 A：Transformer concat classifier，最快可交付

输入：

```txt
[CLS] claim [SEP] evidence_1 [SEP] evidence_2 ... evidence_k
```

模型：

```txt
roberta-base
deberta-v3-base
microsoft/deberta-v3-small
```

如果资源紧，用 `deberta-v3-small` 或 `distilroberta-base`。

实验：

| Model          |          Context | Accuracy | Macro-F1 | REFUTES F1 |
| -------------- | ---------------: | -------: | -------: | ---------: |
| TF-IDF logreg  |            top50 |   0.4610 |   0.4242 |    current |
| RoBERTa concat |            top20 |      new |      new |        new |
| RoBERTa concat | top50 compressed |      new |      new |        new |

但 concat 有一个问题：top50 太长，会截断。你可以压缩 evidence：

```txt
每条 evidence 只保留前 40~60 tokens
按 MiniLM score 排序
总长度限制 512
```

---

### 8.2 路线 B：Evidence-wise verifier + aggregation，我更推荐

输入是单条 pair：

```txt
claim + one evidence
```

输出三类：

```txt
SUPPORT
REFUTE
NEUTRAL
```

然后对 top20/top50 evidence 聚合：

[
S_{\text{sup}} = \max_j p_{\text{sup}}(q,e_j)
]

[
S_{\text{ref}} = \max_j p_{\text{ref}}(q,e_j)
]

规则：

```txt
if S_sup high and S_ref low:
    SUPPORTS
elif S_ref high and S_sup low:
    REFUTES
elif S_sup high and S_ref high:
    DISPUTED
else:
    NOT_ENOUGH_INFO
```

这个结构比 concat 更适合你，因为你的 retrieval 已经能给 top20 高 recall context，接下来要判断每条 evidence 的立场，而不是让一个大 classifier 从一堆噪声里自己悟。

训练数据构造：

```txt
positive support:
  claim label SUPPORTS + gold evidence

positive refute:
  claim label REFUTES + gold evidence

neutral:
  non-gold retrieved evidence
  random evidence
  high MiniLM but non-gold evidence
```

DISPUTED 暂时不训练 pair-level disputed，因为一条 evidence 很难叫 disputed；DISPUTED 是 claim-level aggregation 结果。

---

## 9. Round08 的优先级排序

我建议你下一阶段按这个顺序做，不要乱开坑。

---

### Priority 1：Dense retrieval 跑完，但只当 supplement

输出：

```txt
dense_top500_dev.jsonl
rrf_bm25_char_dense_top500_dev.jsonl
```

看：

```txt
candidate macro recall
hit-any
all-gold
REFUTES recall
```

验收标准：

```txt
如果 sparse+dense all-gold 或 REFUTES recall 提升，就保留；
如果只 macro recall 小涨但 final F 不涨，可以报告为 ablation。
```

不要为了 dense 卡住主线。

---

### Priority 2：Feature Fusion Reranker

这是我认为收益最大的下一步。

对当前 top500 candidates，训练一个轻量 reranker：

```txt
input features:
  MiniLM score
  RRF score
  BM25 rank
  char TF-IDF rank
  entity overlap
  number/year overlap
  negation/comparison/relation features

target:
  evidence_id in gold_evidence
```

输出 top3/top5/top20，和 MiniLM 对比：

| Reranker                | top3 F | top3 hit-any | top3 REFUTES recall | top20 recall |
| ----------------------- | -----: | -----------: | ------------------: | -----------: |
| MiniLM only             | 0.1987 |       0.4545 |              0.1790 |       0.4540 |
| MiniLM + feature fusion |    new |          new |                 new |          new |

这一步比直接 fine-tune MiniLM 稳。

---

### Priority 3：Evidence-wise verifier

训练 claim-evidence pair classifier：

```txt
claim + evidence -> support/refute/neutral
```

然后用 top20/top50 聚合到 claim label。

这一步是 classification 的真正主线。

---

### Priority 4：Evidence expansion

如果你还有时间，做：

```txt
title + prev + current + next
```

然后重新跑：

```txt
BM25
char TF-IDF
RRF
MiniLM top3
classifier context
```

它可能同时提升 retrieval 和 reranking。

---

### Priority 5：BM25 field-aware / parameter sweep

这是锦上添花，不是主线。

---

## 10. 我会怎么安排 Round08 实验表

Round08 最少要交付 4 张表。

### 表 1：Dense supplement

| Pool             | Macro Recall | Hit-any | All-gold | REFUTES Recall |
| ---------------- | -----------: | ------: | -------: | -------------: |
| BM25 top500      |       0.5861 |  0.8377 |   0.3247 |         0.5123 |
| char top500      |       0.6610 |  0.8896 |   0.4026 |         0.5556 |
| sparse RRF       |       0.6579 |  0.8896 |   0.4091 |         0.5926 |
| dense top500     |          new |     new |      new |            new |
| sparse+dense RRF |          new |     new |      new |            new |

---

### 表 2：Reranker

| Candidate Pool   | Reranker                | top3 F | top3 Macro Recall | top3 Hit-any | REFUTES Recall |
| ---------------- | ----------------------- | -----: | ----------------: | -----------: | -------------: |
| sparse RRF       | MiniLM                  | 0.1987 |            0.2331 |       0.4545 |         0.1790 |
| sparse+dense RRF | MiniLM                  |    new |               new |          new |            new |
| sparse RRF       | MiniLM + feature fusion |    new |               new |          new |            new |
| sparse+dense RRF | MiniLM + feature fusion |    new |               new |          new |            new |

---

### 表 3：Classifier context

| Context         | Context k | Context Recall | Hit-any | Classifier           | Accuracy | Macro-F1 |
| --------------- | --------: | -------------: | ------: | -------------------- | -------: | -------: |
| MiniLM          |        20 |         0.4540 |  0.7143 | TF-IDF logreg        |  current |  current |
| MiniLM          |        50 |            new |     new | TF-IDF logreg        |   0.4610 |   0.4242 |
| Fusion reranker |        20 |            new |     new | transformer/verifier |      new |      new |
| Fusion reranker |        50 |            new |     new | transformer/verifier |      new |      new |

---

### 表 4：Error analysis

| Error Type                    | Count | Fix                                        |
| ----------------------------- | ----: | ------------------------------------------ |
| gold not in candidate top500  |   new | dense / evidence expansion                 |
| gold in top500 but not top20  |   new | fusion reranker                            |
| gold in top20 but label wrong |   new | verifier classifier                        |
| REFUTES confused as SUPPORTS  |   new | contradiction / number / negation features |
| all-gold missing              |   new | evidence set selection                     |

这张表非常重要。它会告诉你到底该继续救 retrieval、rerank，还是 classifier。

---

## 11. 你现在对“recall 不够”的判断要更精确

不要笼统说 recall 不够。要拆成三层：

### A. Candidate recall

```txt
RRF top500 macro recall = 0.6579
hit-any = 0.8896
all-gold = 0.4091
```

结论：候选池已经不错，但 all-gold 不够。

### B. Reranked context recall

```txt
MiniLM top20 macro recall = 0.4540
hit-any = 0.7143
```

结论：top20 可作为 classifier context，但仍然漏很多证据。

### C. Final output recall

```txt
MiniLM top3 macro recall = 0.2331
hit-any = 0.4545
```

结论：final evidence selection 很弱，主要是 reranker/evidence-set selection 问题。

所以真正要说的是：

```txt
First-stage hit-any is acceptable, but all-gold remains limited.
The larger bottleneck is that cross-encoder reranking does not preserve enough gold evidence in top-3/top-20.
```

这比“BM25/TF-IDF 特征不够丰富”更准确。

---

## 12. 最终建议：Round08 主线

我建议 Round08 这么定：

```txt
Fixed retrieval baseline:
  RRF(BM25 top500, char TF-IDF top500)

Add dense retrieval:
  dense top500
  RRF(BM25, char, dense)

Improve reranking:
  MiniLM score + sparse scores + semantic features
  train lightweight feature fusion reranker

Improve classification:
  evidence-wise verifier + aggregation
  compare with concat transformer classifier

Optional:
  evidence expansion with title/neighbor sentences
```

最重要的不是再把 candidate recall 从 `0.6579` 抠到 `0.67`，而是把：

```txt
top3 hit-any: 0.4545 -> 0.55+
top3 F: 0.1987 -> 0.22+
top20 hit-any: 0.7143 -> 0.78+
classifier macro-F1: 0.4242 -> 0.50+
```

作为 Round08 目标。

你现在已经不在“没救”的阶段了。Round07 证明 pipeline 是有效的；Round08 要做的是把它从 retrieval baseline 升级成 **task-aware fact verification system**。

[1]: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html?utm_source=chatgpt.com "Retrieve & Re-Rank Pipeline"
[2]: https://faiss.ai/index.html?utm_source=chatgpt.com "Welcome to Faiss Documentation — Faiss documentation"
[3]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes?utm_source=chatgpt.com "Faiss indexes · facebookresearch/faiss Wiki"
