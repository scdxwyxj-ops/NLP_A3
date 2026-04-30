# Round05 Report C：Reranker candidate dataset and API

报告时间：`2026-04-30`

## 一、本阶段目标

把 supervised reranker 做成后续 classifier 的预处理组件，而不是一次性实验脚本。

本阶段完成内容：
- BM25 top-50 candidate pool 构建。
- train hard-negative reranker dataset 构建。
- HuggingFace cross-encoder 训练入口。
- reranker inference API。
- dev reranking 评估入口。

## 二、工程结构

新增可复用代码：

```txt
src/a3_factcheck/rerank/
  api.py          # EvidenceReranker 用户 API
  candidates.py   # BM25 candidate pool 构建和 recall@k
  dataset.py      # positive / hard-negative pair 构造
```

新增实验入口：

```txt
experiments/rerank/
  build_hard_negative_dataset.py
  train_cross_encoder.py
  rerank_with_cross_encoder.py
```

新增配置：

```txt
configs/rerank/minilm_reranker.json
```

## 三、API 设计

`EvidenceReranker` 是后续 classifier 可以直接调用的接口：

```python
from a3_factcheck.rerank import EvidenceReranker

reranker = EvidenceReranker.from_pretrained("models/round05/minilm-reranker")
ranked = reranker.rerank_claim(
    claim_text=claim_text,
    evidence_by_id=evidence,
    candidate_ids=bm25_candidate_ids,
    top_k=5,
)
```

这个 API 的输入输出保持在 task 层面：
- 输入：claim text、evidence corpus、candidate ids。
- 输出：按相关性排序后的 evidence ids 和 scores。

因此 Round06 classifier 不需要知道 reranker 的训练细节，只需要消费 top-k evidence。

## 四、当前数据构造结果

命令：

```bash
PYTHONPATH=src python experiments/rerank/build_hard_negative_dataset.py \
  --output-dir outputs/round05 \
  --top-k 50 \
  --negatives-per-claim 5
```

结果：
- train claims：`1228`
- dev claims：`154`
- train pairs：`10262`
- positive pairs：`4122`
- hard negative pairs：`6140`
- train recall@50：`0.2889`
- dev recall@50：`0.3249`

## 五、重要观察

BM25 top-50 的 recall 仍然偏低。reranker 可以把候选集内的 gold evidence 往前排，但不能恢复没有被 BM25 召回的 evidence。

因此 Round05 第一阶段先验证 supervised reranking 是否提升 top-k precision；如果提升有限，下一步应改进 candidate generator，例如：
- BM25 + TF-IDF union。
- query expansion。
- top-k 提高后再 rerank。
- dense retrieval 作为第二候选源。

## 六、验证

已完成：

```bash
PYTHONPATH=src python -m compileall src experiments
PYTHONPATH=src python experiments/rerank/build_hard_negative_dataset.py \
  --output-dir outputs/round05 \
  --top-k 50 \
  --negatives-per-claim 5
```

待完成：
- 在安装 `torch` / `transformers` 的环境中训练 `cross-encoder/ms-marco-MiniLM-L6-v2`。
- 运行 `experiments/rerank/rerank_with_cross_encoder.py` 对 dev top-k 做正式对比。
