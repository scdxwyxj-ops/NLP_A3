# Round04 Report B：Retrieval/ranking strategy plan

报告时间：`2026-04-29`

## 一、问题判断

第一步可以理解成 ranking task，但更准确地说是 evidence retrieval：

```txt
claim -> score every evidence passage -> rank evidence ids -> return top-k evidence ids
```

课程 `eval.py` 最终不直接评估排序指标，而是评估 top-k evidence set 和 gold evidence set 的 F-score。因此 ranking 是实现手段，真正要优化的是返回证据集合的 precision/recall/F-score。

## 二、为什么当前 TF-IDF 不够

当前 TF-IDF top-3 dev retrieval F-score 约 `0.0533`，说明简单词面相似不足。主要风险：
- claim 和 evidence 可能语义相关但词面不重合。
- evidence corpus 很大，泛主题相似文本会挤掉真正 gold evidence。
- claim 可能需要多条 evidence，固定 top-k 容易过多或过少。
- climate/science claim 常有数字、实体、术语和否定关系，普通 TF-IDF 不够稳。

## 三、推荐路线

### Step 1：强 lexical retriever

先做 BM25 或改进 TF-IDF，目标是快速超过当前 TF-IDF top-3。

优先尝试：
- BM25 over tokenized evidence。
- 保留数字、单位、专有名词，不做过度 stopword removal。
- 比较 top-k：`1, 3, 5, 10, 20, 50`。
- 输出 candidate pool，例如 top-50，用于后续 reranking/classifier。

### Step 2：轻量 query expansion

在不引入外部训练数据的前提下，用本地规则增强 query：
- claim 原文。
- lowercased 版本。
- 数字和实体 token 加权。
- 可选：从 claim 中保留关键 noun-like tokens，避免句子功能词干扰。

不要做复杂人工 if-then label 规则；这里只用于 retrieval query，不用于 classification。

### Step 3：reranker

在 lexical top-50 上做二阶段 reranking：
- Cross-encoder transformer：输入 `[claim] [SEP] [evidence]`，预测相关性。
- 训练正例：train gold evidence。
- 负例：BM25/TF-IDF top candidates 中非 gold evidence。
- 目标：把 gold evidence 从 top-50 推到 top-3/top-5。

这个 reranker 可以成为后续系统的 sequence modelling component，或者至少和 classifier 共享 transformer backbone 的设计理由。

### Step 4：cluster-based multicell-level selection

如果采用 cluster-based retrieval，不再使用 `context-level masks` 这个说法，统一改成 `multicell-level masks`。

这里的含义是：

```txt
claim -> lexical top-N candidates -> cluster candidates -> apply multicell-level masks -> select/rerank evidence set
```

`multicell-level` 强调模型处理的不是单个 evidence passage，也不是一个泛化的 context block，而是一组彼此相关的 evidence cells/candidates。每个 cluster 可以看成一个候选证据单元集合，mask 用来控制哪些 evidence cells 在当前 claim 下参与聚合、reranking 或最终输出。

这个表述比 `context-level masks` 更清楚，因为：
- evidence retrieval 的基本对象是 evidence id / evidence passage，而不是开放式上下文。
- 一个 claim 的判断可能依赖多个 evidence cells。
- cluster 内部可以有互补证据、重复证据和干扰证据，需要在 multicell 层面筛选。
- 后续 classifier 可以接收一个 masked evidence cluster，而不是只接收单条 evidence。

可执行方案：
- 先用 BM25/TF-IDF 取 top-50 或 top-100 candidates。
- 对 candidates 做 embedding 或 lexical similarity clustering。
- 每个 cluster 内保留若干代表 evidence cells。
- 用 `multicell-level mask` 标记哪些 cells 进入 aggregation/reranker。
- 最终从 cluster-level score 和 cell-level score 中选 top-k evidence ids。

训练监督可以来自 gold evidence：
- gold evidence 所在 cluster 是正 cluster。
- gold evidence cell 是 positive cell。
- 同 cluster 或高排名非 gold evidence 是 hard negative cell。

这样 cluster-based 部分可以自然连接 retrieval 和 classifier：retriever 不只是返回散乱 top-k，而是返回经过 multicell-level mask 过滤的 evidence group。

### Step 5：evidence set selection

不要只固定一个 top-k。需要 dev 上比较：
- fixed top-k。
- score threshold。
- top-k with max/min bounds，例如最少 1 条，最多 5 条。
- per-label 或 per-confidence selection 先不要做，避免和 classification 混在一起。

## 四、当前最实用的下一步

1. 实现 BM25 baseline，并生成 dev top-k 表。
2. 写一个 retrieval error analysis 脚本，抽样展示：
   - claim text
   - gold evidence ids/text
   - predicted top-k ids/text
   - overlap
3. 用分析结果判断失败主要来自：
   - 词面不匹配
   - 实体/数字没抓住
   - top-k 太小或太大
   - evidence passage 太泛
4. 若 BM25 仍弱，再做 reranker，而不是继续盲调 TF-IDF。
5. 如果做 cluster-based 版本，使用 `multicell-level masks` 组织 cluster 内 evidence cells，不再写 `context-level masks`。

## 五、给报告的叙事

可以把 retrieval 写成两阶段设计：
- Stage 1 retrieves a high-recall candidate pool using lexical matching.
- Stage 2 reranks or filters candidates using a sequence model that jointly encodes claim and evidence.
- For the cluster-based variant, candidate evidence is grouped into clusters and filtered with multicell-level masks before final evidence selection.

这样能解释为什么项目不是普通 text classification：如果 retrieval 找错，classifier 再强也只能基于错误证据判断。
