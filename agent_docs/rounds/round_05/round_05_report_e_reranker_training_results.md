# Round05 Report E：MiniLM reranker training and evaluation results

报告时间：`2026-04-30`

## 一、本次实验目标

安装必要依赖后，实际测试 Round05 reranker 的效果。

已安装：

```txt
transformers
accelerate
```

本机环境：

```txt
torch 2.8.0+cu128
CUDA available
GPU: NVIDIA GeForce RTX 5070 Ti
```

## 二、实验设置

Candidate pool：

```txt
outputs/round05/dev-bm25-top50.json
```

训练数据：

```txt
outputs/round05/train-reranker-pairs-top50-neg5.jsonl
```

训练样本：
- total pairs：`10262`
- positives：`4122`
- hard negatives：`6140`

## 三、结果对比

### BM25 baseline

Round04 最佳：

```txt
BM25 top-5 F-score = 0.0772
```

### Zero-shot MS MARCO MiniLM cross-encoder

命令：

```bash
PYTHONPATH=src python experiments/rerank/rerank_with_cross_encoder.py \
  --candidate-pool outputs/round05/dev-bm25-top50.json \
  --model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output outputs/round05/dev-msmarco-minilm-zero-shot.json \
  --top-k-values 1,3,5,10 \
  --batch-size 64
```

结果：

```txt
top-1 F = 0.1299
top-3 F = 0.1642
top-5 F = 0.1578
top-10 F = 0.1218
```

当前最佳是 zero-shot top-3：`0.1642`。

### 错误的 2-class fine-tuning

第一版训练脚本把原始 single-logit reranking head 改成了 2-class classification head，导致 MS MARCO reranker head 被重新初始化。

结果：

```txt
top-1 F = 0.0216
top-3 F = 0.0318
top-5 F = 0.0334
top-10 F = 0.0372
```

结论：这个设置不可用，不能作为正式结果。

### Single-logit BCE fine-tuning

修正后保留 single-logit head，并使用 BCEWithLogits loss。

训练命令：

```bash
PYTHONPATH=src python experiments/rerank/train_cross_encoder.py \
  --model-name cross-encoder/ms-marco-MiniLM-L6-v2 \
  --train-pairs outputs/round05/train-reranker-pairs-top50-neg5.jsonl \
  --output-dir models/round05/minilm-reranker-bce \
  --epochs 1 \
  --batch-size 32 \
  --max-length 256 \
  --num-labels 1
```

训练概况：

```txt
runtime = 18.25s
train loss = 0.5754
```

dev 结果：

```txt
top-1 F = 0.0656
top-3 F = 0.1000
top-5 F = 0.1044
top-10 F = 0.0906
```

结论：single-logit fine-tuning 超过 BM25，但仍低于 zero-shot MS MARCO MiniLM。

## 四、当前判断

当前最强方法是：

```txt
BM25 top-50 candidates
-> zero-shot cross-encoder/ms-marco-MiniLM-L6-v2 reranker
-> top-3 evidence
```

它把 retrieval F-score 从 BM25 top-5 的 `0.0772` 提升到 `0.1642`。

监督 fine-tuning 方向仍然可行，但当前 hard-negative dataset 和训练设置还没有超过 zero-shot teacher。

## 五、下一步

优先级：

1. 固定 zero-shot MiniLM reranker 作为强 baseline。
2. 改 fine-tuning：
   - 更低 learning rate，例如 `5e-6`。
   - 尝试 `0.3` 到 `1.0` epoch。
   - 增加 dev pair evaluation 或 early stopping。
   - 改 hard negatives，不只取每个 claim 前 5 个。
3. 改 candidate generator recall：
   - BM25 + TF-IDF union。
   - top-100 candidates。
   - query expansion。
4. 再考虑 self-training / pseudo-label loop。
