# Round06 Report B：Stage plan and final comparison table schema

报告时间：`2026-04-30`

## 一、本报告目的

Round06 最终应该产出一个足够支撑下一步工作的对比实验表格，而不是一堆分散实验结果。

最终表格需要回答：

```txt
哪一种 evidence preprocessing 方案最适合进入最终 classifier？
```

因此 Round06 的指标不只看 retrieval F-score，还要看 classifier-readiness：
- top-k recall
- hit-any
- REFUTES recall
- FP/FN 类型
- classifier input context 是否足够
- semantic feature 是否有帮助

## 二、最终对比表 schema

建议最终表格列：

| column | meaning |
|---|---|
| `method_id` | 实验编号，例如 `R06-A1` |
| `candidate_source` | BM25 top-50、BM25 top-100、BM25+TFIDF union 等 |
| `reranker` | none、zero-shot MiniLM、BCE MiniLM、task-aware MiniLM |
| `negative_strategy` | none、BM25 non-gold、MiniLM-mined non-gold |
| `semantic_features` | none、entities+quantities、full semantic summary |
| `final_output_top_k` | 用于提交 evidence 的 k |
| `classifier_context_top_k` | 给 classifier 看的 evidence 数量 |
| `retrieval_f_score` | `eval.py` evidence F-score |
| `claim_accuracy_placeholder` | 当前 majority-label placeholder 或真实 classifier accuracy |
| `harmonic_mean` | `eval.py` harmonic mean |
| `precision` | evidence-level precision |
| `macro_recall` | claim-level gold evidence recall |
| `micro_recall` | evidence-level recall |
| `hit_any` | 至少命中一个 gold evidence 的 claim 比例 |
| `all_gold` | 完全命中 gold evidence set 的 claim 比例 |
| `refutes_recall` | REFUTES 子集 recall |
| `refutes_hit_any` | REFUTES 子集 hit-any |
| `topical_fp_pattern` | 是否仍明显存在 topic-relevant FP |
| `runtime_notes` | 训练/推理成本 |
| `decision` | keep / reject / needs classifier test |
| `interpretation` | 一句话结论 |

## 三、Stage A：Baseline consolidation and table scaffold

状态：`待开始`

目标：
- 把 Round05 结果整理成 Round06 表格的 baseline rows。
- 建立 `outputs/round06/comparison_table.csv` 或 equivalent JSON。

需要纳入的 baseline：
- BM25 top-5。
- BM25 top-50 / top-100 recall rows。
- zero-shot MiniLM top-3 / top-5 / top-20。
- Round05 BCE MiniLM top-5。

验收指标：
- 有表格初版。
- 每一行都有 metric source。
- 空缺值显式标记为 `pending` 或 `not_applicable`。

## 四、Stage B：MiniLM-mined task-aware hard negatives

状态：`待开始`

目标：
- 用 zero-shot MiniLM 找出高分 non-gold negatives。
- negatives 要更接近真实错误：topic-relevant but not evidence-relevant。

实验变量：
- candidate source：BM25 top-50 vs top-100。
- negatives per claim：`5 / 10 / 20`。
- positive handling：gold positives always injected。

输出：

```txt
outputs/round06/train-task-aware-negatives-*.jsonl
outputs/round06/task-aware-negative-stats-*.json
```

验收指标：
- 每个 negative 有 BM25 rank/score 和 MiniLM rank/score。
- examples 中能看到 topical FP 类型。
- 不使用 dev labels 训练，不使用 test labels。

## 五、Stage C：Task-aware reranker training comparison

状态：`待开始`

目标：
- 用 Stage B 数据重训 single-logit MiniLM reranker。
- 比较它是否超过 zero-shot MiniLM 或至少改善 classifier input recall。

实验变量：
- learning rate：`5e-6 / 1e-5`。
- epoch：`0.3 / 0.5 / 1.0`。
- negatives per claim：`5 / 10 / 20`。

主要指标：
- final output top-3/top-5 F-score。
- top-10/top-20 recall。
- REFUTES recall/hit-any。
- topic-relevant FP 是否减少。

验收指标：
- 至少完成一个 conservative variant。
- 不管结果好坏，都记录进 final table。
- 若 fine-tuning 低于 zero-shot，说明原因和下一步。

## 六、Stage D：Fine-grained semantic extraction prototype

状态：`待开始`

目标：
- 不做手写规则分类器。
- 只提取辅助语义信息，用于 analysis、hard negative mining 或 classifier input formatting。

候选字段：

```txt
entities
quantities
years
percentages
negation_cues
comparison_cues
causality_cues
relation_verbs
```

输出：

```txt
outputs/round06/semantic_features_sample.jsonl
outputs/round06/semantic_feature_examples.md
```

验收指标：
- 对 `claim-375` 这类例子能显示 claim 的关键数值和因果/否定 cue。
- 能解释为什么某些 FP 只是 topic-relevant。
- 如果噪声较大，标记为 analysis-only。

## 七、Stage E：Classifier-oriented evidence packaging

状态：`待开始`

目标：
- 为 Round07 classifier 准备数据，而不是直接开始复杂 classifier。

输出格式建议：

```json
{
  "claim_id": "...",
  "claim_text": "...",
  "claim_label": "...",
  "final_evidence_candidates": ["evidence-..."],
  "classifier_evidence_context": [
    {
      "evidence_id": "...",
      "text": "...",
      "reranker_score": 0.0,
      "semantic_features": {}
    }
  ]
}
```

需要对比：
- classifier context top-3。
- classifier context top-10。
- classifier context top-20。
- classifier context top-50。
- with vs without semantic feature summary。

验收指标：
- 有 classifier-ready train/dev JSONL。
- 每个版本可追溯到 retrieval method。
- 明确推荐 Round07 使用哪个输入版本。

## 八、Stage F：Final Round06 synthesis

状态：`待开始`

最终产物：

```txt
agent_docs/rounds/round_06/round_06_report_final_comparison.md
outputs/round06/comparison_table.csv
```

最终报告必须包含：
- 表格。
- 每个方法的优缺点。
- REFUTES 专项分析。
- topic relevance vs evidence relevance 分析。
- classifier input 推荐。
- 下一轮 Round07 classifier 的明确任务。

## 九、推荐执行顺序

```txt
Stage A
-> Stage B
-> Stage C
-> Stage D
-> Stage E
-> Stage F
```

但如果训练效果不稳定，可以先做：

```txt
Stage A
-> Stage D
-> Stage E
```

这样也能尽快进入 classifier baseline。
