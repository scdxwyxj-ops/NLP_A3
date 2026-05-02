你现在不是“做了一堆乱实验”，而是已经形成了一条完整实验链：**先救 candidate recall，再定位 reranking 损失，再做 feature fusion，再做 calibrated blend，最后补 neural classifier / verifier 并确认最终系统应该解耦**。最重要的最终结论是：**不要用一个模型同时负责 evidence top3 和 claim label；最终系统应使用 Round09 blend 负责 evidence，DistilRoBERTa concat classifier 负责 label**。最终 dev 结果是 `Evidence F = 0.2105`，`Claim Accuracy = 0.5260`，`Harmonic Mean = 0.3007`，`Claim Macro-F1 = 0.4543`。

---

# 0. 先把任务拆清楚

这个 assignment 实际上有两个任务：

$$\text{Claim} \rightarrow \text{Evidence Retrieval}$$

$$\text{Claim} + \text{Evidence Context} \rightarrow \text{Claim Label}$$

也就是说，你同时要做：

```txt
1. 找 evidence：输出 top-k evidence ids。
2. 判 label：SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED。
```

早期你把它当成一个 retrieval pipeline；后面实验逐渐证明，**evidence selector 和 claim classifier 的最优输入不一样**。final evidence 需要小而准的 top3；claim classifier 反而需要一个更稳定的 context。最终报告也明确说，Round10 的核心问题是：最终系统应该是一个 single neural reranker，还是 evidence selection 与 label classification 分开优化；结果支持后者。

官方主指标里你要特别注意一个细节：

$$H = \frac{2 \cdot \text{EvidenceF} \cdot \text{Accuracy}}{\text{EvidenceF}+\text{Accuracy}}$$

不是 macro-F1 和 evidence F 的 harmonic。macro-F1 是我们内部看 label class imbalance 的辅助指标；官方 harmonic 看的是 evidence F 和 claim accuracy。你的最终报告里也写了这一点：official score 使用 evidence F、claim accuracy 和 harmonic mean，但内部仍看 macro-F1。

---

# 1. 初始阶段：TF-IDF / BM25 / MiniLM baseline

## 做了什么

最早你做了三个 retrieval baseline：

```txt
TF-IDF retrieval
BM25 retrieval
BM25 top50 -> zero-shot MiniLM cross-encoder reranking
```

BM25 比 TF-IDF 强，但仍然很弱。BM25 top-5 F-score 只有 `0.0772`，BM25 top-50 macro recall 是 `0.3249`，BM25 top-100 macro recall 是 `0.4188`。这意味着很多 gold evidence 在 reranker 之前就已经丢失，后面模型再强也救不回来。

之后你引入 `cross-encoder/ms-marco-MiniLM-L6-v2`：

```txt
claim
-> BM25 top50 candidates
-> MiniLM rerank
-> output top-k evidence
```

zero-shot MiniLM 明显超过 BM25：top-3 F-score 从 BM25 的 `0.0772` 量级提升到 `0.1642`。

## 还尝试了什么

你还做了 MiniLM fine-tuning：

```txt
positive = claim + gold evidence
negative = claim + BM25 non-gold evidence
loss = BCE
```

以及 task-aware hard negatives：

```txt
BM25 top100
-> zero-shot MiniLM rerank
-> high-ranked non-gold evidence as hard negatives
```

但 task-aware fine-tuned MiniLM 没有打败 zero-shot MiniLM。比如 top3 上 zero-shot MiniLM 是 `0.1642`，task-aware MiniLM 是 `0.1599`；top5 和 top20 也没有 overall 超过 zero-shot。

## 得到的结论

这一阶段的结论非常关键：

```txt
MiniLM 作为 zero-shot reranker 很强；
但 fine-tuning BCE 没有自然带来提升；
当前最大瓶颈可能不是 reranker 训练 epoch，而是 candidate recall 和 evidence relevance 的定义。
```

也就是说，MiniLM 找的是“语义相关 passage”，但这个任务要找的是“能支持/反驳/决定 claim 的 evidence”。你的错误分析也已经指出：false positives 往往不是随机错，而是 topic-relevant 但 evidence-irrelevant。

这直接导向 Round07：先别继续盲目训练 reranker，先把 candidate pool 召回救起来。

---

# 2. Round07：Candidate Recall Rescue

## 当时的问题

初始 BM25 top100 macro recall 只有 `0.4188`。这说明第一阶段候选召回太低。我们当时的判断是：

```txt
如果 gold evidence 不在 candidate pool 里，reranker 不可能选中它。
```

所以 Round07 的目标不是 final top3，而是：

```txt
先让 gold evidence 进入 top500 candidate pool。
```

## 做了什么

Round07 主要做 candidate generation：

```txt
BM25 top50 / top100 / top200 / top500
Char TF-IDF top500
RRF(BM25 top500, Char TF-IDF top500)
```

最终比较结果：

| Candidate Source        |   N | Macro Recall | Hit-any | All-gold |
| ----------------------- | --: | -----------: | ------: | -------: |
| BM25                    | 500 |       0.5861 |  0.8377 |   0.3247 |
| Char TF-IDF             | 500 |       0.6610 |  0.8896 |   0.4026 |
| RRF(BM25 + Char TF-IDF) | 500 |       0.6579 |  0.8896 |   0.4091 |

这个结果证明：BM25 单独不够；char TF-IDF 可以补 BM25 漏掉的 evidence；RRF 虽然 macro recall 略低于 char TF-IDF，但 all-gold 更高，所以更适合作为后续 reranker 输入。

然后你把这个 candidate pool 接到 MiniLM：

```txt
RRF(BM25 top500, Char TF-IDF top500)
-> MiniLM rerank
-> top3 evidence
```

结果：

| System                               | Evidence F | Macro Recall | REFUTES Recall | Hit-any |
| ------------------------------------ | ---------: | -----------: | -------------: | ------: |
| MiniLM zero-shot top3, old BM25 pool |     0.1642 |       0.1877 |         0.0494 |  0.3961 |
| RRF + MiniLM top3                    |     0.1987 |       0.2331 |         0.1790 |  0.4545 |

这说明 Round07 的 retrieval rescue 是有效的：candidate pool 变好以后，同一个 zero-shot MiniLM reranker 也能输出更好的 final evidence。

## Round07 的结论

```txt
Candidate recall rescue 成功。
下一步不应该继续无脑扩大 BM25/TF-IDF，而应该研究：
gold 已经在 top500 时，为什么没有进入 top20/top3？
```

这就是 Round08 的出发点。

---

# 3. Round08：Error Layer Analysis + Feature Fusion Reranker

## 当时的问题

Round07 后，RRF top500 的 hit-any 已经到 `0.8896`。也就是说，大多数 claim 至少有一个 gold evidence 已经进入 top500。但 final top3 hit-any 仍然只有 `0.4545` 左右。于是问题从：

```txt
candidate generation 不够
```

变成：

```txt
reranker / evidence selection 不够
```

这也是你 Round08 requirements 里最重要的重定义：现在瓶颈不再是单纯 BM25/TF-IDF 太弱，而是 gold evidence 经常已经在 top500 candidate pool 里，但 MiniLM 没有稳定排进 top3/top20。

## 做了什么

Round08 第一件事是 error layer analysis。你把 gold evidence 的丢失位置拆开：

| Layer                        | Count | Claim % | Gold Evidence % | Main Fix                               |
| ---------------------------- | ----: | ------: | --------------: | -------------------------------------- |
| gold not in top500           |   180 |  0.5909 |          0.3666 | candidate retrieval / dense supplement |
| gold in top500 but not top50 |    58 |  0.3052 |          0.1181 | score more candidates / fusion         |
| gold in top50 but not top20  |    52 |  0.2468 |          0.1059 | feature fusion                         |
| gold in top20 but not top3   |   108 |  0.4610 |          0.2200 | final evidence selection               |
| coverage top500              |   311 |  0.8896 |          0.6334 | diagnostic                             |
| coverage top20               |   201 |  0.7143 |          0.4094 | diagnostic                             |
| coverage top3                |    93 |  0.4545 |          0.1894 | diagnostic                             |

这张表非常重要。它证明：top500 已经不是完全没救，但从 top20 到 top3 仍然丢了大量 gold evidence。也就是说，Round08 不应该只做 dense，也不应该只做 classifier；rerank/final selection 是明确瓶颈。

然后你构造了 feature table。每个 claim-evidence pair 有：

```txt
MiniLM rank / score
RRF rank / score
BM25 rank / score
Char TF-IDF rank / score
source_count
entity / number / year / percentage overlap
negation / comparison / causality cue overlap
label_is_gold
```

这一步还有一个关键审计：`data/evidence.json` 是扁平结构，只有 `evidence_id -> evidence text`，没有 title、document id、section、page、previous/next sentence。因此 field-aware BM25 和 prev/next evidence expansion 不可作为主线。

## 模型实验

Round08 用这些 features 训练 lightweight fusion reranker：

```txt
MiniLM top50 candidates
-> feature table
-> LogisticRegression / GBDT
-> rerank top50
-> output top3/top20/top50
```

最好结果是 Fusion GBDT：

| Method      |  k | Evidence F | Macro Recall | Hit-any | REFUTES Recall |
| ----------- | -: | ---------: | -----------: | ------: | -------------: |
| MiniLM only |  3 |     0.1987 |       0.2331 |  0.4545 |         0.1790 |
| Fusion GBDT |  3 |     0.2011 |       0.2321 |  0.4545 |         0.0864 |
| MiniLM only | 20 |     0.1106 |       0.4540 |  0.7143 |         0.3951 |
| Fusion GBDT | 20 |     0.1153 |       0.4669 |  0.7403 |         0.4136 |

Round08 的提升很小，但不是没价值。它说明 `MiniLM score + sparse ranks + shallow semantic overlap` 有正向信号；同时也暴露出一个问题：GBDT 提高 overall F-score/context recall，但伤害 REFUTES top3 recall。

## Classifier context 实验

Round08 还用 Fusion GBDT top20/top50 重新构造 classifier context，并跑 TF-IDF LogisticRegression：

| Context              | Semantic Summary | Accuracy | Macro-F1 | Evidence F | Harmonic |
| -------------------- | ---------------- | -------: | -------: | ---------: | -------: |
| Round07 MiniLM top50 | no               |   0.4610 |   0.4242 |     0.1987 |   0.2778 |
| Fusion GBDT top20    | no               |   0.4481 |   0.3930 |     0.2011 |   0.2776 |
| Fusion GBDT top50    | no               |   0.4675 |   0.4376 |     0.2011 |   0.2812 |
| Fusion GBDT top50    | yes              |   0.4351 |   0.3895 |     0.2011 |   0.2751 |

结论：classifier 最喜欢的是 `Fusion GBDT top50 without semantic summary`。semantic summary 作为文本拼进去没有帮助，甚至降低结果。

## Round08 的结论

```txt
Feature fusion 有用，但只是 calibration，不是真正 verifier。
GBDT top3 overall 略升，但 REFUTES 被伤害。
Classifier context 最好用 Fusion GBDT top50 no-summary。
下一轮应该保留 GBDT 的 calibration，同时用 MiniLM 保住 REFUTES。
```

这直接导向 Round09：alpha blend + REFUTES calibration。

---

# 4. Round09：Top100 Scope + REFUTES Weighting + Alpha Blend

## 当时的问题

Round08 pure GBDT 有两个问题：

```txt
1. GBDT top3 F-score 略升，但 REFUTES recall 从 MiniLM 的 0.1790 掉到 0.0864。
2. Round08 cheap fusion 只重排 MiniLM top50，救不了 gold in top500 but not top50 的情况。
```

所以 Round09 的目标是：

```txt
扩大 MiniLM scoring scope 到 top100；
给 REFUTES positives 加权；
再用 alpha blend 合并 MiniLM 与 GBDT。
```

## Stage A：Top50 alpha blend

先检查 MiniLM-only 和 GBDT 是否互补：

|     alpha_minilm | top3 F-score | Macro Recall | REFUTES Recall |
| ---------------: | -----------: | -----------: | -------------: |
|   0.0, pure GBDT |       0.2011 |       0.2321 |         0.0864 |
|              0.5 |       0.2078 |       0.2416 |         0.1235 |
|              0.7 |       0.2104 |       0.2448 |         0.1235 |
| 1.0, pure MiniLM |       0.1987 |       0.2331 |         0.1790 |

这说明 blend 明显有用：它比 pure GBDT 和 pure MiniLM 都强。但 REFUTES 仍没有回到 MiniLM-only。

## Stage B：Gain/Loss Analysis

你对比 MiniLM vs Round08 Fusion GBDT top3：

| Label    | Bucket           | Count |
| -------- | ---------------- | ----: |
| REFUTES  | both_hit         |     4 |
| REFUTES  | both_miss        |    20 |
| REFUTES  | minilm_only      |     3 |
| REFUTES  | fusion_gbdt_only |     0 |
| SUPPORTS | fusion_gbdt_only |     3 |
| SUPPORTS | minilm_only      |     1 |

这解释了为什么 GBDT overall 小涨但 REFUTES 大跌：GBDT 对 SUPPORTS 有收益，但几乎没有新增 REFUTES hit，反而丢掉 MiniLM 找到的 REFUTES。

## Stage C：MiniLM Top100 Scope

Round08 只能重排 MiniLM top50。Round09 扩展到 top100：

```txt
RRF top500
-> MiniLM score top100
-> fusion rerank top100
```

但一个重要结论是：**MiniLM top100 scope 本身不会自动提高 top3/top20**，因为 top50 内排序保持一致：

| Method              | top3 F-score | top20 F-score |
| ------------------- | -----------: | ------------: |
| MiniLM top50 scope  |       0.1987 |        0.1106 |
| MiniLM top100 scope |       0.1987 |        0.1106 |

top100 的价值是给 fusion reranker 更多候选，而不是让 MiniLM 自己变强。

## Stage D：Top100 Fusion + REFUTES Weighting

你构造了 top100 feature table：

| Split |    Rows | Positive Rows |
| ----- | ------: | ------------: |
| train | 122,800 |         2,131 |
| dev   |  15,400 |           279 |

然后训练三个 GBDT：

| Model                  | top3 F-score | top20 F-score | Note            |
| ---------------------- | -----------: | ------------: | --------------- |
| GBDT top100 default    |       0.1975 |        0.1160 | top3 下降         |
| GBDT top100 REFUTES x2 |       0.2037 |        0.1154 | 最好的 pure fusion |
| GBDT top100 REFUTES x3 |       0.1922 |        0.1154 | 权重过强            |

结论：REFUTES x2 是合理校准；x3 太强，破坏 overall ranking。

## Stage E：Top100 REFUTES x2 Alpha Blend

最后你把 MiniLM top100 和 REFUTES x2 GBDT 做 alpha blend：

| alpha_minilm | top3 F-score | Macro Recall | REFUTES Recall | Harmonic |
| -----------: | -----------: | -----------: | -------------: | -------: |
|          0.0 |       0.2037 |       0.2393 |         0.1420 |   0.2788 |
|          0.3 |       0.2071 |       0.2417 |         0.1420 |   0.2819 |
|          0.4 |       0.2105 |       0.2446 |         0.1420 |   0.2851 |
|          0.5 |       0.2105 |       0.2446 |         0.1420 |   0.2851 |
|          0.6 |       0.2105 |       0.2446 |         0.1420 |   0.2851 |
|          0.8 |       0.2074 |       0.2444 |         0.1605 |   0.2822 |
|          1.0 |       0.1987 |       0.2331 |         0.1790 |   0.2741 |

推荐选择 `alpha_minilm = 0.4`，不是因为它唯一最好，而是因为 `0.4/0.5/0.6` 并列达到最高 top3 F-score `0.2105`，而 `0.4` 可以作为默认最低 MiniLM blend 权重。

## Stage G：Feature Importance

Top100 REFUTES x2 GBDT 的 feature importance：

| Feature                  | Importance |
| ------------------------ | ---------: |
| minilm_rank              |     0.5036 |
| minilm_score             |     0.2041 |
| rrf_score                |     0.1179 |
| char_tfidf_score         |     0.0419 |
| bm25_score               |     0.0362 |
| rrf_rank                 |     0.0362 |
| percentage_overlap_count |     0.0097 |
| entity_jaccard           |     0.0062 |

这个结果非常重要：GBDT 主要学的是 MiniLM 和 sparse rank calibration，而不是真正 semantic verifier。semantic overlap 有一点信号，但非常弱。

## Round09 的结论

```txt
当前最强 final evidence selector 是：
RRF(BM25 top500, Char TF-IDF top500)
-> MiniLM top100 scope
-> REFUTES x2 Fusion GBDT
-> alpha blend alpha=0.4
-> top3 evidence

Evidence F = 0.2105
Macro Recall = 0.2446
REFUTES Recall = 0.1420
```

但 Round09 也证明：继续调 GBDT 和 alpha 不能真正解决 REFUTES，因为 GBDT 不是 verifier。

这直接导向 Round10：补神经网络 baseline，验证是不是模型能力不够。

---

# 5. Round10：Neural Classifier + Neural Verifier

Round10 有两个不同目标：

```txt
1. 补齐神经网络 classifier baseline，避免项目像纯 retrieval engineering。
2. 测试 neural verifier 是否能替代 reranker，真正学习 SUPPORT / REFUTE / NEUTRAL。
```

## 5.1 DistilRoBERTa concat claim classifier

### 做了什么

模型：

```txt
model = distilroberta-base
input = claim + top10 retrieved evidence
output = 4-way claim label
max_length = 512
epochs = 5
batch_size = 8
lr = 2e-5
```

脚本：

```txt
experiments/neural/train_concat_transformer_classifier.py
```

输入是 classifier context JSONL，使用 retrieved evidence top10。

### 训练曲线

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro-F1 |
| ----: | ---------: | -------: | -----------: | -----------: |
|     1 |     1.2673 |   1.2907 |       0.4481 |       0.1658 |
|     2 |     1.2411 |   1.2442 |       0.4416 |       0.1532 |
|     3 |     1.1807 |   1.2973 |       0.3961 |       0.2321 |
|     4 |     1.0770 |   1.2019 |       0.4805 |       0.3425 |
|     5 |     0.8771 |   1.3339 |       0.5260 |       0.4543 |

结果：

| Model         | Context           | Accuracy | Macro-F1 | Evidence F-score | Harmonic |
| ------------- | ----------------- | -------: | -------: | ---------------: | -------: |
| TF-IDF logreg | Fusion GBDT top50 |   0.4675 |   0.4376 |           0.2011 |   0.2812 |
| DistilRoBERTa | Fusion GBDT top10 |   0.5260 |   0.4543 |           0.2011 |   0.2910 |

这个结果说明：“epoch 不够”对 classifier baseline 是成立的。DistilRoBERTa 到 5 epoch 才明显超过 TF-IDF logreg。

### 得到的结论

```txt
神经 concat classifier 是有效的；
它显著提高 claim accuracy；
但它不是 evidence selector，它只负责 label。
```

---

## 5.2 Neural claim-evidence verifier

### 做了什么

你构造了 pair-level verifier dataset：

```txt
SUPPORTS claim + gold evidence -> SUPPORT
REFUTES claim + gold evidence -> REFUTE
retrieved non-gold / random evidence -> NEUTRAL
DISPUTED gold evidence 暂不作为 pair-level positive
```

数据规模：

| Split |   Rows | SUPPORT | REFUTE | NEUTRAL |
| ----- | -----: | ------: | -----: | ------: |
| train | 14,537 |   1,343 |    914 |  12,280 |
| dev   |  1,825 |     171 |    114 |   1,540 |

训练设置：

```txt
model = distilroberta-base
epochs = 5
batch_size = 16
max_length = 256
lr = 2e-5
```



### Pair-level 结果

| Epoch | Train Loss | Dev Loss | Dev Accuracy | Dev Macro-F1 |
| ----: | ---------: | -------: | -----------: | -----------: |
|     1 |     0.4993 |   0.5031 |       0.8395 |       0.3244 |
|     2 |     0.3911 |   0.4737 |       0.8367 |       0.4091 |
|     3 |     0.2778 |   0.5973 |       0.8318 |       0.4635 |
|     4 |     0.2057 |   0.6133 |       0.8230 |       0.5027 |
|     5 |     0.1572 |   0.7176 |       0.8247 |       0.4404 |

这里非常清楚：verifier 在 epoch 4 达到最佳 pair-level macro-F1 `0.5027`；epoch 5 train loss 继续下降但 dev loss 变差，说明开始过拟合。

### Verifier 作为 reranker

你用：

```txt
score = max(p_support, p_refute)
```

对 dev top50 evidence 排序：

|  k | Evidence F-score | Claim Accuracy | Macro Recall | REFUTES Recall | Harmonic |
| -: | ---------------: | -------------: | -----------: | -------------: | -------: |
|  3 |           0.1029 |         0.4545 |       0.1048 |         0.0340 |   0.1678 |
| 20 |           0.1028 |         0.4481 |       0.4105 |         0.4074 |   0.1672 |
| 50 |           0.0611 |         0.4481 |       0.5475 |         0.5309 |   0.1076 |

结论很明确：

```txt
Verifier top3 很差，不能替代 Round09 evidence selector。
Verifier top20/top50 的 REFUTES recall 很高，说明它能发现 REFUTES 相关 evidence。
但它的 score 不适合直接控制 final top3。
```



### Hybrid verifier scoring

你又测试：

```txt
score = (1 - gamma) * Round09_blend_score + gamma * verifier_score
```

结果：

| gamma |  k | Evidence F-score | Macro Recall | REFUTES Recall | Harmonic |
| ----: | -: | ---------------: | -----------: | -------------: | -------: |
|   0.2 |  3 |           0.2038 |       0.2352 |         0.1235 |   0.2789 |
|   0.4 |  3 |           0.1987 |       0.2272 |         0.1173 |   0.2741 |
|   0.6 |  3 |           0.1788 |       0.1975 |         0.0679 |   0.2546 |
|   0.8 |  3 |           0.1312 |       0.1437 |         0.0340 |   0.2022 |
|   0.6 | 20 |           0.1206 |       0.4926 |         0.4568 |   0.1895 |

Hybrid 没有超过 Round09 top3 F-score `0.2105`。gamma 越大，top3 越差；但 top20 REFUTES recall 会提高。

## Round10 的结论

```txt
DistilRoBERTa concat classifier 有效，应该用于 claim label。
Neural verifier 学到了 pair-level stance signal，但不能直接替代 top3 reranker。
Verifier 更适合作为 REFUTES diagnostic / wide context supplement / aggregation feature。
```

这就回答了你之前的疑问：模型确实需要 neural baseline；但是 **single neural verifier 并没有自然解决 evidence ranking**。最终系统仍然应该解耦。

---

# 6. 最终实验是什么样的？

最终实验是 Round10 full experiment：把前几轮最优部件组合起来，并统一比较 evidence selection、classifier、final system。

## 实验资产

最终报告里记录了这些核心资产：

| 类型               | 路径                                                                                           |
| ---------------- | -------------------------------------------------------------------------------------------- |
| 统一分析脚本           | `experiments/analysis/build_round10_full_report_assets.py`                                   |
| 原始汇总输出           | `outputs/round10/full_experiment/`                                                           |
| 图表资产             | `assets/full_experiment/`                                                                    |
| evidence 对比表     | `assets/full_experiment/tables/evidence_system_comparison.csv`                               |
| classifier 对比表   | `assets/full_experiment/tables/classifier_system_comparison.csv`                             |
| final system 对比表 | `assets/full_experiment/tables/final_system_comparison.csv`                                  |
| final prediction | `assets/full_experiment/predictions/final_round09_evidence_distilroberta_fusion_labels.json` |
| summary          | `assets/full_experiment/summary.json`                                                        |

最终报告还给出了复现实验命令，包括训练 DistilRoBERTa classifier 和构建 full report assets。

## 最终 evidence selector

最终 evidence selector 是：

```txt
RRF(BM25 top500, Char TF-IDF top500)
-> MiniLM top100 scoring
-> Fusion GBDT with REFUTES positive weight x2
-> alpha blend, alpha_minilm = 0.4
-> output top3 evidence
```

结果：

```txt
Evidence F = 0.2105
Macro Recall = 0.2446
REFUTES Recall = 0.1420
Hit-any = 0.4610
```

这个是当前最强 evidence selector。它超过 TF-IDF、BM25、MiniLM zero-shot、RRF+MiniLM、Fusion GBDT、neural verifier 和 hybrid verifier。

## 最终 claim classifier

最终 claim classifier 是：

```txt
DistilRoBERTa concat classifier
input = claim + top10 retrieved evidence
output = SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED
```

有两个 classifier context 版本：

```txt
1. Fusion GBDT top10 context
2. Round09 blend top10 context
```

完整对比表里显示：

| System                                            | Evidence F | Accuracy | Macro-F1 | Harmonic | 结论                     |
| ------------------------------------------------- | ---------: | -------: | -------: | -------: | ---------------------- |
| Round08 TF-IDF logreg                             |     0.2011 |   0.4675 |   0.4376 |   0.2812 | strong sparse baseline |
| Round09 TF-IDF logreg                             |     0.2105 |   0.4481 |   0.4142 |   0.2865 | evidence 更好，但 label 较弱 |
| DistilRoBERTa fusion top10                        |     0.2011 |   0.5260 |   0.4543 |   0.2910 | 最高 label macro-F1      |
| DistilRoBERTa blend top10                         |     0.2105 |   0.5260 |   0.4045 |   0.3007 | 最高 official harmonic   |
| Final: Round09 evidence + fusion-label classifier |     0.2105 |   0.5260 |   0.4543 |   0.3007 | 推荐最终系统                 |
| Final: Round09 evidence + blend-label classifier  |     0.2105 |   0.5260 |   0.4045 |   0.3007 | H 相同，但 macro-F1 更低     |

因此最终推荐不是单纯 `DistilRoBERTa blend top10`，而是：

```txt
Evidence:
  Round09 blend alpha0.4 top3

Label:
  DistilRoBERTa classifier trained on Fusion GBDT top10 context
```

理由：official harmonic 同样是 `0.3007`，但 fusion-label classifier 的 macro-F1 更高：`0.4543` vs `0.4045`。

## 最终结果

最终 dev：

```txt
Evidence F = 0.2105
Claim Accuracy = 0.5260
Claim Macro-F1 = 0.4543
Harmonic Mean = 0.3007
```

这比 Round09 best harmonic `0.2865` 更高，也比 Round10 初版 neural classifier harmonic `0.2910` 更高。

---

# 7. 为什么最后要解耦？

因为每一轮都证明了同一件事：

```txt
最适合 evidence top3 的排序，不一定最适合 label prediction。
```

证据：

1. Round09 blend 是最强 evidence selector，Evidence F = `0.2105`。
2. Round08/Fusion-context 的 DistilRoBERTa classifier label macro-F1 更高。
3. Neural verifier 作为 reranker 很差，但作为 wide-context REFUTES diagnostic 有信号。
4. GBDT fusion 对 top20 context 有帮助，但 top3 REFUTES recall 会下降。
5. 官方 final harmonic 只需要 evidence F 和 accuracy；所以 evidence 和 label 可以分别用最强部件组合。

最终系统就是：

```txt
Candidate generation:
  RRF(BM25 top500, Char TF-IDF top500)

Evidence selector:
  MiniLM top100 + REFUTES x2 Fusion GBDT + alpha blend alpha=0.4
  -> final top3 evidence

Claim classifier:
  DistilRoBERTa concat classifier
  input = claim + top10 evidence context
  output = 4-way claim label

Final output:
  evidence ids from Round09 blend
  label from DistilRoBERTa classifier
```

这不是临时拼凑，而是实验支持的 decoupled design。

---

# 8. 每一轮如何推动下一轮

可以把整个项目写成下面这条链：

## Milestone 0：Lexical baseline 太弱

```txt
TF-IDF top3 F = 0.0533
BM25 top3 F = 0.0682
BM25 top100 macro recall = 0.4188
```

结论：

```txt
lexical retrieval 本身不够。
```

下一步：

```txt
用 cross-encoder rerank BM25 candidates。
```

## Milestone 1：MiniLM zero-shot 有效，但 candidate recall 仍是瓶颈

```txt
BM25 top50 -> MiniLM -> top3 F = 0.1642
```

结论：

```txt
neural rerank 有效；
但 BM25 candidate pool 丢了太多 gold。
```

下一步：

```txt
扩大 candidate generation：BM25 top500 + char TF-IDF + RRF。
```

## Milestone 2：Round07 candidate rescue 成功

```txt
RRF top500 candidate recall = 0.6579
hit-any = 0.8896
all-gold = 0.4091
RRF + MiniLM top3 F = 0.1987
```

结论：

```txt
candidate rescue 能显著提高 final evidence。
但 top500 -> top3 仍丢很多 gold。
```

下一步：

```txt
做 error layer analysis 和 feature fusion reranker。
```

## Milestone 3：Round08 feature fusion 有信号，但伤害 REFUTES

```txt
Fusion GBDT top3 F = 0.2011
MiniLM-only top3 F = 0.1987
但 REFUTES recall: 0.0864 vs MiniLM 0.1790
```

结论：

```txt
GBDT calibration 有用；
但不能替代 MiniLM，尤其不能解决 REFUTES。
```

下一步：

```txt
MiniLM + GBDT alpha blend；
REFUTES weighting；
扩 MiniLM scope 到 top100。
```

## Milestone 4：Round09 alpha blend 成为最强 evidence selector

```txt
Round09 blend alpha0.4:
Evidence F = 0.2105
Macro Recall = 0.2446
REFUTES Recall = 0.1420
```

结论：

```txt
alpha blend 是当前最强 final evidence selector；
但 REFUTES 仍未完全解决；
GBDT 主要是 rank calibration，不是真正 verifier。
```

下一步：

```txt
补 neural classifier / verifier，测试模型能力瓶颈。
```

## Milestone 5：Round10 neural classifier 有效，verifier 不适合直接 top3

```txt
DistilRoBERTa classifier:
Accuracy = 0.5260
Macro-F1 = 0.4543

Verifier top3:
Evidence F = 0.1029

Verifier top50:
REFUTES Recall = 0.5309
```

结论：

```txt
神经 classifier 提升 label；
verifier 学到 pair-level stance signal；
但 verifier score 不适合作为 final top3 ranking score。
```

下一步：

```txt
最终解耦：
Round09 blend 负责 evidence；
DistilRoBERTa 负责 label。
```

---

# 9. 最终系统的错误在哪里？

最终系统不是完美的。最终混淆矩阵显示：

| Gold \ Pred     | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED |
| --------------- | -------: | ------: | --------------: | -------: |
| SUPPORTS        |       47 |       9 |              11 |        1 |
| REFUTES         |        7 |      15 |               3 |        2 |
| NOT_ENOUGH_INFO |       20 |       3 |              16 |        2 |
| DISPUTED        |       10 |       2 |               3 |        3 |

最明显的问题是：

```txt
NOT_ENOUGH_INFO 和 DISPUTED 经常被预测成 SUPPORTS。
```

这说明 classifier 仍然容易被 retrieved evidence 的表面支持感误导，没有充分学会“证据不足”或“双向证据冲突”。

最终报告也指出当前上限仍来自 evidence selection：

```txt
top3 evidence F = 0.2105
hit-any = 0.4610
all-gold = 0.1104
```

这意味着多数 claim 还没有拿到完整 evidence set。

---

# 10. 你应该如何在报告里讲这个项目

我建议你不要按“Round07、Round08、Round09、Round10”机械罗列，而是按论文式故事讲：

## Stage 1：Sparse retrieval rescue

```txt
问题：
  BM25 candidate recall 太低。

方法：
  BM25 top500 + Char TF-IDF top500 + RRF。

结果：
  RRF top500 macro recall = 0.6579
  hit-any = 0.8896
  all-gold = 0.4091

结论：
  candidate rescue 成功，RRF candidate pool 进入后续 pipeline。
```

## Stage 2：Cross-encoder reranking + feature fusion

```txt
问题：
  gold 经常在 top500，但没有进 top3。

方法：
  MiniLM rerank；
  feature fusion GBDT；
  top100 scope；
  REFUTES weighting；
  alpha blend。

结果：
  MiniLM top3 F = 0.1642
  RRF + MiniLM top3 F = 0.1987
  Fusion GBDT top3 F = 0.2011
  Round09 blend top3 F = 0.2105

结论：
  Round09 blend 是最终 evidence selector。
```

## Stage 3：Neural classification and verification

```txt
问题：
  claim label 不能只靠 TF-IDF logreg；
  课程需要 neural baseline；
  需要验证 neural verifier 是否能替代 reranker。

方法：
  DistilRoBERTa concat classifier；
  DistilRoBERTa claim-evidence verifier；
  verifier hybrid scoring。

结果：
  DistilRoBERTa classifier accuracy = 0.5260
  macro-F1 = 0.4543
  verifier top3 evidence F = 0.1029
  hybrid top3 max = 0.2038 < Round09 0.2105

结论：
  neural classifier 有用；
  verifier 不能直接替代 final reranker；
  final system 应解耦。
```

## Final System

```txt
Evidence:
  Round09 blend alpha0.4 top3

Label:
  DistilRoBERTa concat classifier

Dev:
  Evidence F = 0.2105
  Accuracy = 0.5260
  Macro-F1 = 0.4543
  Harmonic = 0.3007
```

---

# 11. 哪些实验是“失败但有价值”的？

这部分你报告里一定要写，否则 marker 会觉得你只 cherry-pick。

## Fine-tuned MiniLM BCE

结果没有超过 zero-shot MiniLM。意义是：

```txt
BCE gold/non-gold objective 不够贴合 fact-checking ranking；
hard negative 不等于真正 stance-aware learning。
```

## Query boost / claim decomposition

简单 query expansion 没有超过默认 pipeline。意义是：

```txt
粗暴加 entity/year/number 会引入更多 topic-level false positives；
这些特征更适合作为 fusion features，而不是直接改 query。
```

## Feature Fusion GBDT

overall F 小涨，但 REFUTES recall 大跌。意义是：

```txt
GBDT 是 calibration，不是 verifier。
```

## Neural verifier

pair-level macro-F1 到 `0.5027`，说明它学到了 stance signal；但 top3 evidence F 只有 `0.1029`，说明：

```txt
stance probability 不能直接等价为 evidence ranking score。
```

它适合以后做 aggregation feature，而不是替代 reranker。

---

# 12. 最后一句判断

你的最终实验链是成立的。它不是“pipeline 到极限了然后随便加神经网络”，而是：

```txt
1. lexical retrieval 太弱；
2. candidate recall rescue 有效；
3. reranker selection 成为新瓶颈；
4. feature fusion 能校准但不是 verifier；
5. alpha blend 得到最强 evidence selector；
6. neural classifier 显著提升 label；
7. neural verifier 学到 stance，但不能直接替代 evidence selector；
8. 最终系统必须解耦。
```

最终推荐系统就是：

```txt
Round09 blend alpha0.4 evidence selector
+
DistilRoBERTa fusion-context label classifier
```

最终 dev：

```txt
Evidence F = 0.2105
Claim Accuracy = 0.5260
Claim Macro-F1 = 0.4543
Harmonic Mean = 0.3007
```

这条故事线已经可以写进 final report。不要再把它写成“我们试了很多方法”；要写成“每一轮根据上轮诊断定位一个新瓶颈，并用下轮实验验证假设”。
