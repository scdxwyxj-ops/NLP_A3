# Discussion: Round08 Requirements 第一轮回应

日期：2026-05-01

## 总体判断

Round08 requirements 的大方向是对的：Round07 已经完成 candidate recall rescue，
现在瓶颈不再是单纯“BM25/TF-IDF 太弱”，而是：

```txt
gold evidence 经常已经在 top500 candidate pool 里，
但 MiniLM 没有稳定把它排进 top3/top20。
```

所以 Round08 的主线应该从 candidate generation 转向：

```txt
reranking / evidence selection / classifier aggregation
```

不过 requirements 里有些建议目前还比较模糊，或者不适合作为第一优先级。
下面按“明确可执行、模糊待澄清、不适用或低优先级”分类。

## 已经明确、适合进入 Round08 主线的建议

### 1. 区分三层 recall

这是最重要的框架，应该保留。

当前不应该笼统说“recall 不够”，而应该拆成：

```txt
candidate recall:
  RRF(BM25,char) top500 macro recall = 0.6579
  hit-any = 0.8896
  all-gold = 0.4091

reranked context recall:
  MiniLM top20 macro recall = 0.4540
  hit-any = 0.7143

final output recall:
  MiniLM top3 macro recall = 0.2331
  hit-any = 0.4545
```

这个拆分能直接指导后续实验：

- 如果 gold 不在 top500：candidate generation 问题；
- 如果 gold 在 top500 但不在 top20/top3：reranker 问题；
- 如果 evidence context 里有 gold 但 label 错：classifier/aggregation 问题。

这个建议明确、适用，应该成为 Round08 的分析框架。

### 2. Dense retrieval 只作为 supplement

这个建议适用。

Dense retrieval 不应该替换当前 sparse RRF。正确实验是：

```txt
RRF(BM25 top500, char TF-IDF top500, dense top500)
```

然后只看它是否改善：

- macro recall
- hit-any
- all-gold
- REFUTES recall

如果 dense top500 本身不如 char TF-IDF，但 sparse+dense RRF 提高 all-gold
或 REFUTES recall，它仍然有价值。

注意：dense retrieval 目前还没有实际跑，所以不能在 Round07 表里填数。
Round08 可以把它作为独立 ablation。

### 3. Feature Fusion Reranker

这是我认为 Round08 最值得做的主线之一。

原因：

```txt
candidate hit-any = 0.8896
MiniLM top3 hit-any = 0.4545
```

说明很多 gold evidence 已经在候选池里，但 MiniLM 排序没有选出来。

Feature fusion reranker 可以用现有产物快速实现：

```txt
features:
  MiniLM score
  RRF score
  BM25 rank
  char TF-IDF rank
  source count
  entity/number/year overlap
  negation/comparison features if available

target:
  evidence_id in gold_evidence

model:
  sklearn LogisticRegression / LinearSVM
```

这比继续 fine-tune transformer 更稳、更快、更容易解释，也更适合报告。

### 4. Error analysis 四分表

这个建议非常适合 Round08，因为它能避免乱试模型。

建议把错误分成：

| Error Type | 说明 | 对应修复 |
|---|---|---|
| gold not in candidate top500 | 第一阶段没召回 | dense / evidence expansion |
| gold in top500 but not top20 | reranker 没排上去 | feature fusion reranker |
| gold in top20 but not top3 | final evidence selection 弱 | set selection / diversity |
| gold in context but label wrong | 分类器没利用证据 | verifier / aggregation |

这张表应该成为 Round08 的第一个验收输出之一。

### 5. Evidence-wise verifier + aggregation

这个方向适用，但要注意 label 构造。

比较稳的定义：

```txt
SUPPORTS claim + gold evidence -> support pair
REFUTES claim + gold evidence -> refute pair
retrieved non-gold evidence -> neutral pair
random evidence -> neutral pair
```

但是 `DISPUTED` 和 `NOT_ENOUGH_INFO` 不能直接变成 pair-level positive。
尤其是 `DISPUTED` 更像 claim-level aggregation 结果，不是一条 evidence 的关系标签。

所以这个方向可做，但需要清楚写明训练标签假设。

## 方向正确但目前比较模糊的建议

### 1. Field-aware BM25

建议本身合理，但当前是否可做取决于 evidence 数据结构。

requirements 里提到：

```txt
title / page / sentence / section / document id
```

但我们当前 `data/evidence.json` 主要是 evidence id -> text。是否有 title、
neighbor sentence、document metadata，还需要先审计数据格式。

如果没有这些字段，field-aware BM25 就不能直接做，只能退化成：

```txt
body text only
entity string
number/year string
```

所以它目前是“待数据审计后决定”，不应默认进入主线。

### 2. Claim decomposition

方向合理，但细节模糊。

requirements 里提到：

```txt
entity-only query
entity + year + number query
entity + relation verb query
noun chunks / keyphrase query
```

这里的问题是：

- entity 怎么抽？
- relation verb 怎么定义？
- noun chunks 是否需要 spaCy？
- query 太碎会不会引入更多 topic-level false positives？

我们已经做过简单 query-boost ablation，结果没有超过默认方案。
这说明“简单增加 query 内容”不一定有效。

如果做 claim decomposition，应该先限制为低成本版本：

```txt
original claim
numbers/years only
capitalized phrase query
content words query
```

并且只作为 candidate supplement，不作为主线替代。

### 3. Evidence expansion

这是潜在收益很大的建议，但目前模糊点在于：我们是否能从 evidence id 找到相邻句子。

requirements 里的设想是：

```txt
title + previous_sentence + current_sentence + next_sentence
```

如果 evidence ids 编码了文档和句子位置，或者原始数据里存在相邻信息，这个很值得做。
但如果 `evidence.json` 只有扁平 evidence text，就不能可靠恢复 prev/next。

因此它需要先做数据审计：

```txt
evidence id 是否包含 document id / sentence index？
是否存在 title 或 page context？
是否能按 id 排序恢复邻近句？
```

在审计前，不应承诺一定实现。

### 4. Comparison direction / relation verb / negation features

这些特征方向正确，但目前定义不够清楚。

例如：

```txt
comparison_direction_match
relation_verb_overlap
negation_xor
```

需要明确：

- 用什么规则抽取 comparison direction？
- 哪些词算 negation？
- relation verb 是所有动词，还是 claim 中的主谓关系？
- 错抽会不会增加噪声？

建议 Round08 先做最稳定的浅层版本：

```txt
number_overlap
year_overlap
percentage_overlap
negation_presence_match
capitalized_phrase_overlap
source_count
rank features
```

更复杂的 relation/comparison parsing 暂时作为后续增强，不作为第一版 fusion reranker 的依赖。

## 不适合第一优先级或暂时不适用的建议

### 1. 大幅继续调 sparse retrieval

不应作为 Round08 主线。

原因：

```txt
RRF(BM25,char) top500 hit-any = 0.8896
```

第一阶段已经能让多数 claim 至少看到一个 gold evidence。当前更大的损失发生在：

```txt
top500 -> MiniLM top3/top20
```

因此继续把 sparse recall 从 `0.6579` 慢慢磨到 `0.67`，收益可能小于改善 reranker。

Sparse retrieval 可以继续做 dense supplement / data audit / ablation，但不是主线。

### 2. 直接上复杂 FAISS ANN

目前不应第一步就上 `IndexIVFFlat`、`PQ` 或复杂 approximate search。

原因是当前最怕 recall 损失。Approximate search 可能更快，但会引入额外召回损失。

更合适的顺序：

```txt
1. exact dense search or chunked matrix multiplication
2. if memory/time fails, use FAISS IndexFlatIP
3. only if still太慢，再考虑 HNSW/IVF
```

### 3. RoBERTa/DeBERTa concat classifier 作为最先任务

这个方向可行，但不应排在 feature fusion reranker 前面。

原因：

- concat top20/top50 容易被 512 token 截断；
- evidence context 里噪声很多；
- 如果 reranker 没先改善，classifier 会吃到很多 topic-level false positives；
- Colab runtime 和训练稳定性还不确定。

因此 transformer concat classifier 可以作为后续对比，不建议作为 Round08 第一件事。

### 4. Evidence set selection 的复杂 coverage objective

方向正确，但第一版不宜做太复杂。

当前可以先做简单分析：

```txt
top3 是否重复来自相似文本？
top20 中是否有 gold 但被相似 false positive 压住？
```

如果确认有明显重复，再实现简单 diversity rerank。
不要一开始就设计复杂 set objective。

## 对 Round08 的建议执行顺序

我建议 Round08 不要同时开太多坑，按下面顺序：

### Stage A：Round07 关闭与 Round08 问题重定义

输出：

```txt
Round07 closed: retrieval rescue achieved
Round08 bottleneck: reranking / evidence selection / classifier aggregation
```

### Stage B：Error Layer Analysis

输出四类错误计数：

```txt
gold not in top500
gold in top500 but not top20
gold in top20 but not top3
gold in top20/top50 but classifier label wrong
```

这是后续所有实验的依据。

### Stage C：Dense Supplement Ablation

只做最小 dense retrieval：

```txt
dense top500
RRF(BM25,char,dense) top500
```

验收看：

- macro recall
- hit-any
- all-gold
- REFUTES recall

如果 dense 不提升，就记录为 ablation，不阻塞主线。

### Stage D：Feature Fusion Reranker

基于当前 top500 candidates 训练轻量 reranker：

```txt
MiniLM score + sparse ranks + overlap features -> evidence_is_gold
```

对比：

```txt
MiniLM only
MiniLM + feature fusion
```

主要目标：

- top3 F-score > `0.1987`
- top3 hit-any > `0.4545`
- top20 hit-any > `0.7143`
- REFUTES recall 提升

### Stage E：Classifier Aggregation

在更好的 top20/top50 context 上做：

```txt
evidence-wise verifier + aggregation
```

对比当前 baseline：

```txt
TF-IDF logreg accuracy = 0.4610
macro-F1 = 0.4242
```

## 当前 Round08 验收目标建议

不要把 Round08 验收目标设成“做很多模型”。

建议验收目标是：

```txt
1. 完成错误分层表，明确主要损失发生在哪一层。
2. 完成 dense supplement 表，确认 dense 是否值得保留。
3. 完成 feature fusion reranker 表，至少和 MiniLM only 对比。
4. 给出 classifier 下一步最可信输入：top20 or top50 context。
```

如果时间允许，再进入 evidence-wise verifier。

## 最终回应

Round08 requirements 里最适合立刻执行的是：

```txt
error layer analysis
dense as supplement
feature fusion reranker
evidence-wise verifier planning
```

比较模糊、需要先审计或降级实现的是：

```txt
field-aware BM25
claim decomposition
evidence expansion
relation/comparison parsing
```

不适合作为第一优先级的是：

```txt
继续大幅打磨 sparse retrieval
复杂 FAISS ANN
直接上 transformer concat classifier
复杂 evidence set objective
```

Round08 的主线应该是：

```txt
固定 Round07 retrieval default
-> 分析 top500/top20/top3 的损失层
-> 用 feature fusion reranker 改善 evidence selection
-> 再把更好的 top20/top50 交给 classifier aggregation
```

## 数据与实现审计结果

本节回应上面标记为“需要先审计”的建议。审计对象包括：

```txt
data/evidence.json
data/train-claims.json
data/dev-claims.json
outputs/round07/candidates/
outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json
outputs/round07/dev-classifier-context-rrf-bm25-char-minilm-top20-semantic.jsonl
src/a3_factcheck/semantic/features.py
```

### 1. Evidence 数据结构审计

`data/evidence.json` 是扁平结构：

```txt
evidence_id -> evidence text
```

示例：

```txt
evidence-0:
  John Bennet Lawes, English entrepreneur and agricultural scientist

evidence-67732:
  [citation needed] South Australia has the highest retail price for electricity in the country.
```

没有发现以下字段：

```txt
title
document id
section
page
sentence index
previous sentence
next sentence
source article
```

结论：

```txt
field-aware BM25 不能按 title/body/section/document fields 正常实现。
```

最多只能做退化版本：

```txt
body text BM25
entity string BM25
number/year string BM25
```

这类退化版本更像 feature/query ablation，不是真正的 field-aware retrieval。

### 2. Evidence expansion 审计

requirements 建议：

```txt
title + previous_sentence + current_sentence + next_sentence
```

但当前数据没有可靠的相邻句元数据。

我检查了 evidence id 是否可能按文档顺序排列。结果显示不可靠：

```txt
evidence-67730:
  Typically, each byte is from a range of 256 distinct values...

evidence-67731:
  The first initiative to use the name was Transition Town Totnes...

evidence-67732:
  [citation needed] South Australia has the highest retail price...

evidence-67733:
  The team plays in the 2nd Division of the Puerto Rico Soccer League.
```

连续 evidence id 明显来自不同主题，不是同一文档的相邻句。

Gold evidence id 之间也不接近：

```txt
multi-evidence claims: 1141
with any <=3 id gap: 0
gold evidence id spacing median: 187886
minimum spacing: 328
```

结论：

```txt
不能用 evidence id +/- 1 来恢复 previous/next sentence。
```

因此 `title + prev + current + next` 目前不适用，除非找到额外原始数据或 metadata。
Round08 不应承诺 evidence expansion。

### 3. Claim decomposition 审计

当前没有 spaCy、dependency parser 或 named entity linker 作为项目既有依赖。

已有 lightweight semantic extractor 位于：

```txt
src/a3_factcheck/semantic/features.py
```

它能抽取：

```txt
entities:
  domain terms + capitalized phrases

percentages:
  regex percentages

quantities:
  ppm / ppb / degrees / billion / million / tonnes / percent / per cent

years:
  18xx / 19xx / 20xx

negation cues:
  no / not / never / none / without / cannot ...

comparison cues:
  more than / less than / higher / lower / highest / lowest ...

causality cues:
  because / due to / caused by / lead to ...

relation verbs:
  produce / emit / reduce / increase / warm / cool / absorb / reflect ...
```

结论：

```txt
复杂 claim decomposition 暂时不适合做。
```

可做的降级版本是：

```txt
original claim
capitalized/domain entities query
number/year/percentage query
content-word query
```

但 Round07 的 query-boost ablation 已经显示，粗暴 query expansion 没有提升默认 pipeline。
因此 decomposition 只应作为低优先级 supplement，不应作为 Round08 主线。

### 4. Relation / comparison / negation feature 审计

现有 semantic extractor 已经有浅层规则，但不是句法级语义分析。

适合进入第一版 feature fusion reranker 的稳定特征：

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

不适合作为第一版依赖的复杂特征：

```txt
subject/object consistency
comparison direction parsing
dependency relation matching
true contradiction detection
```

原因是当前项目没有 parser 支撑，强行做会变成脆弱规则，误差可能很大。

### 5. Candidate / rerank 产物审计

当前有这些关键文件：

```txt
outputs/round07/candidates/rrf_bm25_char_tfidf_top500.json
outputs/round07/dev-rrf-bm25-char-minilm-ranked-top50.json
outputs/round07/dev-classifier-context-rrf-bm25-char-minilm-top20-semantic.jsonl
```

`rrf_bm25_char_tfidf_top500.json` 中每个 candidate 有：

```json
{
  "claim_id": "...",
  "evidence_id": "...",
  "rank": 1,
  "score": 0.0268
}
```

这里的 `score` 是 RRF score，不是 BM25/TF-IDF 原始分数。

`dev-rrf-bm25-char-minilm-ranked-top50.json` 中每个 candidate 有：

```json
{
  "evidence_id": "...",
  "score": 0.9987,
  "rank": 1
}
```

这里的 `score` 是 MiniLM reranker score。

注意：当前 ranked top50 文件没有保留：

```txt
bm25_rank
char_tfidf_rank
bm25_score
char_tfidf_score
source_count
```

但是这些信息可以从以下候选池重新 join：

```txt
outputs/round07/candidates/bm25_top500.json
outputs/round07/candidates/tfidf_char_top500.json
outputs/round07/candidates/rrf_bm25_char_tfidf_top500.json
```

结论：

```txt
feature fusion reranker 可做，但需要先构造一个 feature table。
```

建议 feature table 每一行是：

```txt
claim_id
evidence_id
label_is_gold
minilm_rank
minilm_score
rrf_rank
rrf_score
bm25_rank
bm25_score
char_tfidf_rank
char_tfidf_score
in_bm25
in_char_tfidf
source_count
semantic overlap features
```

### 6. Dense retrieval 可执行性审计

当前环境：

```txt
torch: installed
sentence_transformers: not installed
faiss: not installed
```

数据规模：

```txt
evidence count: 1,208,827
dev claims: 154
train claims: 1228
```

结论：

```txt
dense retrieval 可做，但不是零成本。
```

最小可行方案：

```txt
install sentence-transformers
use all-MiniLM-L6-v2 or multi-qa-MiniLM-L6-cos-v1
encode evidence once
cache embeddings
chunked exact search
save dense top500 candidates
RRF sparse+dense
```

如果不缓存 evidence embeddings，dense 会反复浪费时间。

### 7. 审计后的优先级修正

审计后，Round08 的优先级应调整为：

| Priority | 方向 | 审计后判断 |
|---:|---|---|
| 1 | Error layer analysis | 立即可做，依赖现有 top500/top50/top20/top3 文件 |
| 2 | Feature table construction | 立即可做，是 fusion reranker 前置步骤 |
| 3 | Feature fusion reranker | 可做，使用 sklearn 和浅层 semantic overlap features |
| 4 | Dense supplement | 可做，但需要安装依赖和缓存 embeddings |
| 5 | Evidence-wise verifier | 可规划，pair label 需要谨慎 |
| 6 | Claim decomposition | 只做低成本 ablation，不进主线 |
| 7 | Field-aware BM25 | 原始设想不适用，只能做退化版本 |
| 8 | Evidence expansion | 当前数据不支持，不做 |

### 8. 审计后的明确结论

可以直接做：

```txt
error layer analysis
feature table construction
feature fusion reranker
shallow semantic overlap features
dense supplement ablation if time/resources allow
```

需要降级做：

```txt
claim decomposition
field-aware BM25
relation/comparison features
```

不适合当前数据：

```txt
title/body/section field-aware retrieval
previous/current/next sentence evidence expansion
document-level diversity based on source document id
```

Round08 主线因此应更具体地改成：

```txt
1. 固定 Round07 sparse RRF + MiniLM baseline。
2. 生成 error layer analysis，确认 top500 -> top20/top3 的损失。
3. 从现有 candidate pools 和 MiniLM ranked output 构造 feature table。
4. 训练 lightweight feature fusion reranker。
5. 再考虑 dense supplement，而不是先做复杂 evidence expansion。
```
