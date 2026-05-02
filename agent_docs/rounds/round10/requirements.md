你这个判断是对的：**现在必须补神经网络 baseline**，否则报告会像一个 retrieval engineering project，而不是 NLP/deep learning project。但要直接说清楚：问题不只是“epoch 不够”，更大概率是 **训练目标错了**。你现在的 ranker 本质上在学：

[
\text{claim 与 evidence 是否相关}
]

但任务真正需要的是：

[
\text{evidence 是否能支持 / 反驳 / 解释 claim}
]

这两者不是一回事。MS MARCO MiniLM 这种 cross-encoder 很适合 rerank query-passage relevance；Sentence Transformers 文档也明确把 CrossEncoder 放在 retrieve-and-rerank 的第二阶段，并且它通常输出一个 relevance score，而不是事实立场判断。([SentenceTransformers][1]) 你的任务更接近 **claim-evidence verification / NLI / stance detection**，所以 Round10 不应该只是继续调 GBDT 或 alpha，而应该加一组神经模型来证明：到底是 pipeline 到极限了，还是 neural verifier/ranker 太弱。

---

# 1. 先纠正一个误区：不是“很多神经 baseline”，而是“三类必要 baseline”

你不需要乱跑十几个模型。真正该跑的是这三类：

```txt
A. Neural classifier baseline:
   claim + top-k evidence -> claim label

B. Neural pair verifier:
   claim + single evidence -> SUPPORT / REFUTE / NEUTRAL

C. Neural reranker:
   claim + candidate evidence -> evidence_is_gold / ranking score
```

其中最关键的是 B 和 C。A 是课程上好交代，B 是最贴近任务本质，C 是直接替换当前 MiniLM/GBDT ranker。

---

# 2. 你的“模型没找到本质”判断基本正确

现在的 pipeline 已经做了：

```txt
BM25 / char TF-IDF recall rescue
MiniLM rerank
GBDT feature fusion
REFUTES weighting
alpha blend
```

但从 Round09 的 feature importance 看，GBDT 主要还是依赖 `MiniLM rank/score`，浅层 semantic feature 没有真正学出“因果性 / 反驳 / 数字矛盾”。所以当前系统本质上仍是：

```txt
lexical relevance + semantic relevance + shallow calibration
```

而不是：

```txt
fact verification
```

所以你感觉“没有找到本质”是对的。尤其 REFUTES 很说明问题：MiniLM 能找到一些相关反驳证据，但 GBDT 会牺牲 REFUTES；说明模型没有可靠理解 contradiction，只是在做 relevance ranking。

---

# 3. epoch 不够有可能，但不是第一嫌疑

如果你 fine-tune 过 MiniLM reranker 但效果不好，原因可能有四类：

## 3.1 训练目标错

如果你用 BCE：

[
\mathcal{L}=-y\log\sigma(s)-(1-y)\log(1-\sigma(s))
]

它学的是 pair 是否 gold，不一定学 claim 内排序。更合适的是 pairwise ranking：

[
\mathcal{L}_{rank}=-\log\sigma(s(q,e^+)-s(q,e^-))
]

或者 margin ranking：

[
\mathcal{L}=\max(0,m-s(q,e^+)+s(q,e^-))
]

因为最终目标是：

```txt
same claim 下 gold evidence 排在 hard negative 前面
```

不是全局判断 pair positive/negative。

## 3.2 负样本错

你的 non-gold evidence 不一定是真负例。它可能是：

```txt
unannotated valid evidence
related but not sufficient evidence
partial evidence
same topic distractor
wrong relation evidence
```

如果全部当 0，模型会学乱。尤其 fact-checking 任务里，hard negative 需要分类：

```txt
random negative
topic hard negative
entity-overlap hard negative
number/year hard negative
MiniLM high-score hard negative
REFUTES-specific hard negative
```

## 3.3 label 太弱

如果你只告诉模型：

```txt
gold / non-gold
```

它不知道 gold 是 support 还是 refute，也不知道为什么 non-gold 不行。对这个任务，更强监督应该是：

```txt
claim + evidence -> SUPPORT / REFUTE / NEUTRAL
```

然后把 verifier score 转成 rank score。

## 3.4 epoch 可能不够，但要用曲线判断

你应该看：

```txt
train loss 是否还在下降
dev evidence F 是否早停
dev REFUTES recall 是否上升
positive/negative score gap 是否变大
```

如果 train loss 还很高、dev 也没 plateau，那 epoch 不够；如果 train loss 很低但 dev 不涨，是过拟合或目标错；如果 positive/negative score gap 变大但 top3 F 不涨，是 candidate/evaluation mismatch。

---

# 4. Round10 应该加的神经网络 baseline

我建议你做四个，按优先级排。

---

## Baseline 1：Transformer concat claim classifier

这是最容易向课程交代的神经 baseline。

输入：

```txt
[CLS] claim [SEP] evidence_1 [SEP] evidence_2 ... evidence_k
```

输出：

```txt
SUPPORTS / REFUTES / NEI / DISPUTED
```

模型：

```txt
distilroberta-base
roberta-base
microsoft/deberta-v3-small
```

DeBERTaV3 是合理选择，因为它是改进的 Transformer 预训练模型，论文中用 replaced token detection 改进预训练效率和 NLU 表现。([arXiv][2]) Hugging Face 的 `Trainer` 也能直接做 PyTorch 训练、混合精度等常规训练流程。([Hugging Face][3])

但是这个模型有明显上限：top50 太长，512 tokens 会截断。所以建议只跑：

```txt
top3
top5
top10
top20 compressed
```

不要直接 top50 全塞。

推荐实验：

| Model            |          Context | Max length | Accuracy | Macro-F1 |
| ---------------- | ---------------: | ---------: | -------: | -------: |
| TF-IDF logreg    |            top50 |          - |   0.4675 |   0.4376 |
| DistilRoBERTa    |            top10 |        512 |      new |      new |
| RoBERTa-base     |            top10 |        512 |      new |      new |
| DeBERTa-v3-small |            top10 |        512 |      new |      new |
| DeBERTa-v3-small | top20 compressed |        512 |      new |      new |

这个 baseline 的作用不是一定赢，而是证明你尝试了 neural classifier。

---

## Baseline 2：Claim-evidence verifier，这是最重要的

输入：

```txt
claim [SEP] single evidence
```

输出三类：

```txt
SUPPORT
REFUTE
NEUTRAL
```

这才是任务本质。

训练数据构造：

```txt
if claim label == SUPPORTS and evidence in gold:
    pair label = SUPPORT

if claim label == REFUTES and evidence in gold:
    pair label = REFUTE

retrieved non-gold evidence:
    pair label = NEUTRAL

random evidence:
    pair label = NEUTRAL
```

对于 `DISPUTED`，第一版不要强行做 pair-level label，因为 DISPUTED 可能需要 evidence set 才成立；可以先排除 disputed gold pairs，或者把 disputed 只用于 claim-level aggregation。

训练后，对 top20/top50 每条 evidence 计算：

[
p_j^{sup},p_j^{ref},p_j^{neu}
]

claim-level 聚合：

[
S_{sup}=\max_j p_j^{sup}
]

[
S_{ref}=\max_j p_j^{ref}
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

同时 evidence ranking score 可以定义成：

[
s_{evidence}(q,e)=1-p_{neu}(q,e)
]

或者：

[
s_{evidence}(q,e)=\max(p_{sup},p_{ref})
]

这比 MiniLM relevance score 更接近 evidence usefulness。

这是我最建议你做的神经模型。

---

## Baseline 3：Neural pairwise reranker

这个是直接回答你“rank 模型是不是弱”的问题。

用 cross-encoder：

```txt
claim + evidence -> scalar score
```

训练样本：

```txt
(q, e_positive, e_negative)
```

loss：

[
\mathcal{L}=-\log\sigma(s(q,e^+)-s(q,e^-))
]

负样本从当前 top100/top500 取：

```txt
MiniLM high-score non-gold
RRF high-rank non-gold
entity-overlap non-gold
number/year-overlap non-gold
random non-gold
```

模型：

```txt
cross-encoder/ms-marco-MiniLM-L6-v2 继续训练
roberta-base cross-encoder
deberta-v3-small cross-encoder
```

关键点：不要只跑 1 个 epoch。建议：

```txt
epochs = [1, 2, 3, 5]
learning_rate = [1e-5, 2e-5, 3e-5]
early stopping on dev top3 evidence F
```

但别做巨大 grid，先固定 LR=2e-5 跑 epoch curve。

记录：

```txt
train ranking loss
dev top3 evidence F
dev top20 recall
REFUTES recall
positive-negative score gap
```

如果 1->3 epoch 明显涨，说明之前确实没训够；如果 train loss 降但 dev 不涨，说明目标或数据构造有问题。

---

## Baseline 4：Verifier-reranker hybrid

这是最贴近最终系统的神经模型。

最终 evidence score：

[
s(q,e)=
\alpha s_{\text{MiniLM}}(q,e)
+\beta s_{\text{GBDT}}(q,e)
+\gamma (1-p_{\text{neutral}}(q,e))
+\delta \max(p_{\text{sup}},p_{\text{ref}})
]

尤其对 REFUTES：

[
s_{\text{refute}}(q,e)=p_{\text{ref}}(q,e)
]

这样 final evidence 不再只靠 relevance，而会引入 stance/verifier 信号。

实验：

| Reranker                       | top3 F | REFUTES Recall | top20 Recall |
| ------------------------------ | -----: | -------------: | -----------: |
| MiniLM-only                    | 0.1987 |         0.1790 |       0.4540 |
| GBDT blend                     | 0.2105 |         0.1420 |          new |
| Verifier score                 |    new |            new |          new |
| MiniLM + verifier blend        |    new |            new |          new |
| MiniLM + GBDT + verifier blend |    new |            new |          new |

这是 Round10 最有可能突破“模型没理解本质”的方向。

---

# 5. 为什么 neural verifier 可能比 reranker 更重要？

因为你的 claim 不是普通 query。普通 retrieval 关心：

```txt
evidence 是否谈到 claim 中的实体和主题
```

fact verification 关心：

```txt
evidence 是否给出足够事实关系
evidence 是否支持 claim 的方向
evidence 是否反驳 claim 的数字、时间、因果、比较关系
```

例如：

```txt
claim: A caused B after 2010.
evidence: B had already happened before A.
```

这个 evidence 和 claim lexical overlap 可能不高，但它是强 refutation。

而另一个：

```txt
evidence: A and B were both discussed in 2010.
```

overlap 很高，但不是证据。

所以 ranking 模型如果只学 relevance，永远差一点。你需要让模型学：

```txt
SUPPORT / REFUTE / NEUTRAL
```

这就是 neural verifier 的价值。

---

# 6. 训练集怎么做才不会炸？

你现在可以从已有数据构造三份训练集。

---

## 6.1 Claim classifier dataset

每个 claim 一条：

```json
{
  "text": "[CLAIM] ... [E1] ... [E2] ...",
  "label": "SUPPORTS"
}
```

context 用：

```txt
Round08 Fusion top10/top20
Round09 blend top10/top20
```

---

## 6.2 Pair verifier dataset

每个 claim 多条：

```json
{
  "claim": "...",
  "evidence": "...",
  "label": "SUPPORT" / "REFUTE" / "NEUTRAL"
}
```

构造：

```txt
positive:
  gold evidence for SUPPORTS/REFUTES

neutral:
  top-ranked non-gold from RRF/MiniLM
  random evidence
```

建议比例：

```txt
SUPPORT positive: all
REFUTE positive: all, maybe upsample x2
NEUTRAL: max 5-10 per claim
```

不要把每个 claim top500 全部作为 neutral，类别会爆炸失衡。

---

## 6.3 Pairwise ranking dataset

每条：

```json
{
  "claim": "...",
  "positive_evidence": "...",
  "negative_evidence": "..."
}
```

每个 positive 配：

```txt
1 random negative
2 MiniLM hard negatives
2 RRF hard negatives
1 entity/number hard negative
```

REFUTES positive 可以采样更多：

```txt
REFUTES positive pairs x2 or x3
```

---

# 7. epoch 到底怎么跑？

我建议不要盲目“多跑 epoch”。你要做一个学习曲线实验：

```txt
epochs: 1, 2, 3, 5
model: DeBERTa-v3-small or MiniLM cross-encoder
objective: pairwise reranking / verifier classification
```

每个 epoch 后评估：

```txt
dev top3 evidence F
dev top20 recall
REFUTES recall
claim macro-F1 if verifier aggregation
train loss
dev loss
```

判断逻辑：

```txt
train loss 降，dev F 涨：
  可以多训

train loss 降，dev F 不涨：
  过拟合 / 目标错 / 负样本错

train loss 不降：
  LR、batch、label 构造有问题

dev REFUTES 一直不上：
  verifier 没学到 contradiction，需要更强 pair label / upsample / NLI init
```

推荐初始超参：

```txt
model: microsoft/deberta-v3-small or roberta-base
max_length: 256 for pair verifier/reranker
batch_size: 16 or 32
lr: 2e-5
epochs: 3
warmup_ratio: 0.1
weight_decay: 0.01
fp16: yes if GPU supports
early stopping metric: dev evidence F or verifier macro-F1
```

---

# 8. 具体 Round10 实验路线

我建议 Round10 这样安排。

## Stage 1：Neural claim classifier baseline

先跑简单的：

```txt
DeBERTa-v3-small / RoBERTa-base
claim + top10 evidence
4-way classification
```

目标：建立课程需要的 neural baseline。

验收：

```txt
是否超过 TF-IDF logreg macro-F1 0.4376
```

就算没超过，也能说明 concat classifier 不适合噪声 evidence。

---

## Stage 2：Neural verifier

训练：

```txt
claim + evidence -> SUPPORT / REFUTE / NEUTRAL
```

评估两件事：

```txt
pair-level macro-F1
claim-level aggregation macro-F1
```

同时用 verifier score rerank evidence：

```txt
score = max(p_support, p_refute)
```

验收：

```txt
top3 evidence F 是否接近/超过 0.2105
REFUTES recall 是否超过 0.1420，最好接近 MiniLM 0.1790
```

---

## Stage 3：Pairwise neural reranker

训练：

```txt
claim, gold evidence, hard negative evidence
```

loss：

```txt
RankNet / margin ranking
```

评估：

```txt
top3 F
top20 recall
REFUTES recall
```

它直接检验“rerank 模型是不是没训够 / objective 不对”。

---

## Stage 4：Hybrid scoring

把 neural verifier/reranker 加回当前最强 pipeline：

```txt
score =
  alpha * MiniLM
  beta * GBDT
  gamma * verifier_evidence_score
```

扫少量权重：

```txt
gamma = [0.2, 0.4, 0.6]
alpha/beta 固定用 Round09 最优比例附近
```

不要大网格。

---

# 9. 最终报告应该怎么讲

你可以把故事改得更强：

```txt
Round07: Sparse retrieval rescue
  BM25/char TF-IDF/RRF improved candidate recall.

Round08: Feature fusion reranking
  GBDT showed that sparse + MiniLM + shallow semantic features help context selection.

Round09: Calibrated reranking
  REFUTES-weighted GBDT and alpha blending improved final evidence F-score.

Round10: Neural verification baselines
  We tested whether the remaining bottleneck is model capacity/objective.
  Transformer concat classifiers and claim-evidence verifiers were introduced.
```

这就非常符合 NLP 课程：有 classical retrieval，有 neural reranking，有 transformer classifier，有 error analysis。

---

# 10. 我的最终判断

你说得对：**必须补神经网络 baseline**。但不要只为了“神经网络”而跑模型。最应该补的是：

```txt
1. DeBERTa/RoBERTa concat claim classifier
2. DeBERTa/RoBERTa claim-evidence verifier
3. Cross-encoder pairwise neural reranker
4. MiniLM/GBDT/verifier hybrid ranker
```

其中最可能真正提升的是：

```txt
claim-evidence verifier + hybrid reranking
```

因为它把任务从“找相关 evidence”改成“判断 evidence 是否支持/反驳 claim”。如果这一轮仍然不涨，那你就可以非常有底气地说：当前限制不是 pipeline 没设计好，而是数据标注粒度、候选证据缺失、以及 pair-level stance supervision 不充分。

[1]: https://sbert.net/docs/cross_encoder/training_overview.html?utm_source=chatgpt.com "Training Overview — Sentence Transformers documentation"
[2]: https://arxiv.org/abs/2111.09543?utm_source=chatgpt.com "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"
[3]: https://huggingface.co/docs/transformers/en/main_classes/trainer?utm_source=chatgpt.com "Trainer"
