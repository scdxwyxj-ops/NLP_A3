# Round16 Colab Pipeline Tutorial for Group Presentation

这份 tutorial 用来给组员快速讲清楚我们目前的系统进展、最终 Colab notebook 做了什么、哪些结果可以汇报，以及哪些结果只能当作 dev diagnostic。

核心结论先讲：

```txt
最好的实验结果来自 Round16 脚本产物：
Evidence F = 0.267950
Top3 macro recall = 0.304221
Best saved-label full-system harmonic = 0.355033
Best dev label ensemble harmonic = 0.359365

当前单文件 Colab notebook 已经写好并能从 raw course data 跑完整 pipeline：
notebooks/round16_colab_full_pipeline.ipynb

但 notebook 的 raw-data local run 不等于最佳实验产物：
local notebook run Evidence F = 0.127149
local notebook run Accuracy = 0.493506
local notebook run Harmonic = 0.202202
```

所以汇报时要分两条线：

```txt
1. Research / best dev result: 展示 Round16 evidence selector + classifier 的最好已观察结果。
2. Colab handoff: 展示我们已经把完整 raw-data pipeline 整合进一个可运行 notebook。
```

不要说当前 notebook 已经复现了最好的 dev score。它的价值是可重跑、无本地 checkpoint 依赖、符合 final handoff 形态。

## 1. Assignment Task

这个项目是 fact-checking pipeline。输入是一个 claim，输出两样东西：

```txt
claim_label:
  SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED

evidences:
  top evidence IDs
```

官方主要分数不是单一 accuracy，而是 evidence retrieval 和 claim classification 的组合：

```txt
Harmonic = 2 * EvidenceF * Accuracy / (EvidenceF + Accuracy)
```

这就是为什么我们一直把系统拆成两部分：

```txt
Evidence selector: 负责找 top3 evidence
Claim classifier: 负责判断 claim label
```

实验结论也支持这个拆分：最适合 evidence top3 的 ranking 输入，不一定最适合 label classifier。

## 2. Progress Timeline

可以按这个顺序给组员讲：

| Stage | What Changed | Main Result |
| --- | --- | ---: |
| Early baseline | TF-IDF / BM25 retrieval | BM25 top5 F about `0.0772` |
| MiniLM rerank | BM25 top50 -> zero-shot MiniLM | top3 F `0.1642` |
| Round07 | BM25 + char TF-IDF RRF candidate rescue | RRF+MiniLM top3 F `0.1987` |
| Round08 | Feature fusion reranker | top3 F `0.2011`, harmonic `0.2812` |
| Round09 | REFUTES calibration + alpha blend | top3 F `0.2105`, harmonic `0.2865` |
| Round10 | DistilRoBERTa claim classifier | accuracy `0.5260`, harmonic `0.3007` |
| Round12 | Wide candidate generation | top500 macro recall `0.921645` |
| Round15 | TopN/top3 evidence selection | top3 F `0.216388`, top3 macro `0.254329` |
| Round16 | Binary evidence selector fusion | top3 F `0.267950`, top3 macro `0.304221` |

The main story is:

```txt
We first fixed candidate recall.
Then we diagnosed reranking loss.
Then we fused sparse, neural, and feature-based ranking signals.
Finally we separated evidence selection from claim classification.
```

## 3. Current Best Evidence Selector

Round16 的最佳 evidence output 是：

```txt
outputs/round16/top3_binary_selector_e3_k20_n80/best_binary_rrf_fusion_top3.json
```

它的结果是：

```txt
top3 macro recall = 0.304221
evidence F = 0.267950
hit_any = 0.597403
all_gold = 0.129870
```

方法是：

```txt
top4500 sparse gate
-> BGE / previous ranked signals
-> MiniLM binary evidence selector
-> RRF fusion with Round15 and Branch A fixed sources
-> top3 evidence
```

更具体地说，binary selector 是一个 claim-evidence cross-encoder：

```txt
model = cross-encoder/ms-marco-MiniLM-L6-v2
input = claim + evidence
label = whether this evidence is gold for this claim
```

重要 caveat：

```txt
binary selector model 是用 train claims 训练的；
但最终 RRF fusion weights 是 dev sweep 选出来的：
rrf_k = 20
weights = binary:2 | r15:0.5 | a_fixed:0.5
```

所以这可以作为 best dev result 汇报，但如果讲 final test-safe method，要说明 fusion rule 仍有 dev-selection risk。

## 4. Current Claim Classification Result

Round16 固定 top3 后，claim classification 主要有三种结果：

| Classifier Path | Accuracy | Evidence F | Harmonic | Use |
| --- | ---: | ---: | ---: | --- |
| Dev label ensemble | `0.545455` | `0.267950` | `0.359365` | best observed dev diagnostic |
| Round10 saved labels + Round16 top3 | `0.525974` | `0.267950` | `0.355033` | strongest immediately usable saved-label baseline |
| Saved DistilRoBERTa single model | `0.519481` | `0.267950` | `0.353542` | clean reproducible model path |

推荐汇报方式：

```txt
Best observed dev result:
  harmonic = 0.359365
  but it is a dev-output ensemble.

Clean reproducible single-model result:
  harmonic = 0.353542
  model = distilroberta-base
  saved model path = outputs/round16/claim_classifier/distilroberta_oldctx_newtop3_k10_e5_seed0/best_model/

Strongest saved-label baseline:
  harmonic = 0.355033
  path = outputs/round16/claim_classifier/round10_blend_labels_new_round16_top3.json
```

不要把 dev ensemble 说成 final deployable model，除非我们补齐相同 ensemble 的 test inference path。

## 5. Colab Notebook Structure

Colab notebook 在：

```txt
notebooks/round16_colab_full_pipeline.ipynb
```

它是一个 single-file raw-data pipeline，不读取本地 `outputs/round*` checkpoint。notebook 期望 `DATA_DIR` 里有：

```txt
train-claims.json
dev-claims.json 或 test-claims-unlabelled.json
evidence.json
```

notebook 主要 cell 结构：

```txt
0-2: intro / readme / dataset processing
3: dependency setup
4: runtime configuration
5-7: imports, JSON loading, metrics
9: query-restricted BM25
10: BGE-small dense rerank inside BM25 pool
11: MiniLM binary evidence selector
12: DistilRoBERTa claim classifier
14: final_predictions.json and run_summary.json
15: optional Google Drive copy
```

Pipeline 是：

```txt
raw claims + raw evidence
-> query-restricted BM25 candidate pool
-> BGE-small rerank inside sparse candidates
-> MiniLM binary evidence selector
-> final top3 evidence
-> DistilRoBERTa classifier with top-k evidence context
-> final_predictions.json
```

默认模型：

```txt
dense reranker = BAAI/bge-small-en-v1.5
binary selector = cross-encoder/ms-marco-MiniLM-L6-v2
claim classifier = distilroberta-base
```

默认关键参数：

```txt
BM25_TOP_K = 4500
DENSE_PREFILTER_K = 2000
DENSE_TOP_K = 200
BINARY_SCORE_TOP_K = 80
FINAL_EVIDENCE_TOP_K = 3
CLASSIFIER_CONTEXT_TOP_K = 10
```

## 6. How To Run In Colab

给组员的运行步骤：

1. 打开 `notebooks/round16_colab_full_pipeline.ipynb`。
2. 在 Colab runtime 里选择 GPU。
3. 上传或挂载课程数据，保证 `DATA_DIR` 能看到：

```txt
train-claims.json
dev-claims.json 或 test-claims-unlabelled.json
evidence.json
```

4. 如果用 test set，把 cell 4 里的 target path 改成：

```python
TARGET_CLAIMS_PATH = DATA_DIR / 'test-claims-unlabelled.json'
```

5. 从上到下运行所有 cell。
6. 最终输出在：

```txt
final_predictions.json
run_summary.json
```

如果需要保存到 Drive，把 cell 15 改成：

```python
COPY_TO_DRIVE = True
```

## 7. What To Say In The Presentation

建议用这个 narrative：

```txt
Our early BM25 and TF-IDF baselines had low evidence recall, so we first focused on candidate generation.
Char TF-IDF and sparse RRF improved recall because they captured lexical variants BM25 missed.
After candidate recall improved, the bottleneck moved to top3 evidence selection.
Feature fusion and calibrated blends improved top3 modestly, but the major Round16 gain came from adding a supervised binary claim-evidence selector and fusing it with previous rank priors.
For label prediction, we used a separate DistilRoBERTa classifier over evidence context, because the best evidence top3 is not necessarily the best classification context.
The current best dev result reaches evidence F 0.26795 and harmonic above 0.35.
We also prepared a single Colab notebook that rebuilds the full pipeline from raw course data without loading our local checkpoints.
```

## 8. What Not To Overclaim

这些点一定要讲谨慎：

```txt
The best Round16 evidence fusion uses dev-selected RRF weights.
The dev label ensemble is not yet a clean single-model test inference path.
The single-file Colab notebook is runnable and self-contained, but its local raw rerun did not reproduce the archived best dev result.
The final submission zip should not include course data or trained checkpoints unless the assignment instructions explicitly allow it.
```

如果组员问为什么 Colab notebook 的 raw run 分数低于脚本 best result，可以这样解释：

```txt
The notebook is a checkpoint-free handoff version.
It retrains lightweight models during the run and uses a simplified raw-data path.
The best dev result came from a broader experiment pipeline with saved ranked outputs, fixed contexts, and dev-selected fusion.
So the notebook is our reproducible integration baseline, while the archived scripts/results are our research-best evidence.
```

## 9. Files To Mention

最重要的文件：

```txt
notebooks/round16_colab_full_pipeline.ipynb
agent_docs/rounds/round_16/round_16_report_b_top3_recall_30_binary_selector.md
agent_docs/rounds/round_16/round_16_report_c_claim_classification_training.md
outputs/round16/top3_binary_selector_e3_k20_n80/summary.json
outputs/round16/claim_classifier/classification_summary.csv
outputs/round16_colab_full_pipeline_local/run_summary.json
```

如果只给组员一个入口，就给：

```txt
notebooks/round16_colab_full_pipeline.ipynb
docs/round16_colab_pipeline_tutorial.md
```

