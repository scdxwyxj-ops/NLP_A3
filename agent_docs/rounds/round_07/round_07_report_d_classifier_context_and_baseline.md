# Round07 报告 D：Classifier Context 与 Baseline

## 目的

Stage D/E 测试 Round07 改进后的 retrieval pipeline 是否能给 claim-label
classifier 提供更好的证据上下文。

## Retrieval 输入

当前选定的 retrieval pipeline：

```txt
RRF(BM25 top500, char TF-IDF top500)
-> zero-shot cross-encoder/ms-marco-MiniLM-L6-v2
```

Final evidence output 使用 top-3。Classifier context 使用 top-20 或 top-50。

## Classifier Context 文件

Dev：

```txt
outputs/round07/dev-classifier-context-rrf-bm25-char-minilm-top20-semantic.jsonl
outputs/round07/dev-classifier-context-rrf-bm25-char-minilm-top50-semantic.jsonl
```

Train：

```txt
outputs/round07_train/train-classifier-context-rrf-bm25-char-minilm-top20-semantic.jsonl
outputs/round07_train/train-classifier-context-rrf-bm25-char-minilm-top50-semantic.jsonl
```

每一行包含：

- claim id/text/label
- gold evidences
- final evidence candidates
- ranked classifier evidence context
- optional semantic features

## 快速 Classifier Baseline

脚本：

```txt
experiments/classification/train_concat_classifier.py
```

模型：

```txt
TF-IDF over claim + evidence context
LogisticRegression(class_weight="balanced")
```

结果：

| 方法 | Context k | Semantic Summary | Accuracy | Macro-F1 | Assignment Harmonic Mean |
|---|---:|---|---:|---:|---:|
| concat TF-IDF logreg | 20 | no | 0.4610 | 0.4162 | 0.2778 |
| concat TF-IDF logreg | 20 | yes | 0.4481 | 0.4013 | 0.2754 |
| concat TF-IDF logreg | 50 | no | 0.4610 | 0.4242 | 0.2778 |
| concat TF-IDF logreg | 50 | yes | 0.4545 | 0.4096 | 0.2766 |

## 解读

这个快速 classifier baseline 只是略高于 majority-like baseline。Round07 的 retrieval
pipeline 确实提高了 evidence F-score 和 context recall，但简单的 bag-of-words
concat classifier 不足以把证据上下文转化为更好的 label prediction。

Semantic summaries 没有帮助这个简单模型。这不代表这些特征没用，而是当前 TF-IDF
文本特征模型无法有效使用它们。

## 决策

保留 Round07 retrieval pipeline 作为当前默认：

```txt
RRF(BM25 top500, char TF-IDF top500) -> zero-shot MiniLM
```

分类器下一步应转向：

1. evidence-wise verifier + aggregation；或
2. 如果 Colab runtime 允许，使用 transformer classifier over top-k context。

不要继续在 simple concat TF-IDF/logistic regression 上投入太多时间；它只适合作为报告 baseline。

