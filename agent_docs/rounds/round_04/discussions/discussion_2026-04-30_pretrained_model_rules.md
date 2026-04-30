# Discussion：作业是否允许预训练模型

讨论时间：`2026-04-30`

## 问题

作业要求里有没有说能不能用预训练模型？还是只能自己训练？

## 结论

可以用预训练模型，但有严格限制。

允许：
- 使用 **open-source pretrained models**。
- 使用 HuggingFace Transformers 做 model loading、training、inference。
- 使用 PyTorch / TensorFlow / Keras。
- 使用 open-source LLM，只要能在免费 Google Colab 跑。
- 对 open-source LLM 做 prompting、fine-tuning、LoRA、hybrid integration。

禁止：
- 使用 closed-source API 或 proprietary models，例如 OpenAI GPT、Claude、Gemini、Copilot。
- 使用不能在免费 Google Colab 跑的模型。
- 使用额外 external datasets 做训练或评测。
- 直接复制外部 GitHub/project solution。
- 只调用 off-the-shelf pretrained model 而没有清楚设计贡献和 justification。

## 对本项目的影响

我们不需要从零训练所有模型。合理方案是：

```txt
BM25 candidate retrieval
-> open-source pretrained transformer reranker/classifier
-> fine-tune on train-claims gold evidence / labels
-> evaluate on dev
```

推荐模型类型：
- `distilbert-base-uncased`
- `bert-base-uncased`
- `microsoft/deberta-v3-small`
- 其他能在 free Colab 运行的小型 open-source transformer

不推荐：
- 大型 LLM prompting 作为主系统。
- 需要付费 API 或外部推理服务的模型。
- 太大、免费 Colab 跑不动的模型。

## 和 retrieval/ranking 的关系

Retrieval 可以分两层：

1. 无监督 candidate generator：

```txt
BM25 top-50 / top-100
```

2. 有监督 reranker：

```txt
pretrained open-source transformer
fine-tuned on claim + evidence relevance pairs
```

训练数据来自课程提供的 train/dev labelled claims：

```txt
positive = claim + gold evidence
negative = claim + BM25 non-gold candidate
```

这样既符合规则，又能体现技术贡献：
- 我们不是直接套一个模型输出答案。
- 我们设计了 retrieval + hard negative + reranking pipeline。
- 训练和调参只用课程数据。

## 当前建议

下一步应该进入：

```txt
BM25 top-50 candidate pool
-> hard negative construction
-> supervised transformer reranker
```

这个方向符合作业规则，也比继续深挖无监督词向量更重要。

