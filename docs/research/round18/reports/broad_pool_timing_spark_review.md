# Broad Pool Timing Review (Spark)

Scope (this run): timing + metric readback only, no notebook execution, no dev-label-based tuning.

## 1) 可复跑命令与输入（现有 artifacts）

### Fixed sparse fusion gate（O-S7）

- `python round18/experiments/o_sparse/o_s7_plain_leaf_fusion/run_o_s7_plain_leaf_fusion.py`
- Inputs:
  - `--bm25-pool round18/outputs/o_sparse/o_s1_lexical_index_experiments/dev_full_bm25_dev_bm25_top500_candidates.json`
  - `--char-pool round18/outputs/o_sparse/o_s6_char_tfidf/dev_full_dev_o_s6_char_tfidf_tfidf_char_top500_candidates.json`
  - `--structured-pool round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposed_candidates.json`
  - `--prf-pool round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json`
  - `--candidate-k 500 --eval-k 100,500`

### Word+char broad pool（O-S9）

- `python round18/experiments/o_sparse/o_s9_union_gate/run_o_s9_union_gate.py`
- Inputs:
  - same 4 leaf pools as above
  - `--candidate-k 1000 --eval-k 3,10,64,100,500,1000`
  - `--bm25-top-k 500 --char-top-k 500 --structured-top-k 0 --prf-top-k 0 --policies rrf,round_robin,priority,union_upper --rrf-k 60`

### Four-signal broad pool（O-S9 four-source）

- `python round18/experiments/o_s9_union_gate/run_o_s9_union_gate.py`
- Inputs:
  - same 4 leaf pools as above
  - `--candidate-k 2000 --eval-k 3,10,64,100,500,1000,2000`
  - `--bm25-top-k 500 --char-top-k 500 --structured-top-k 500 --prf-top-k 500 --policies rrf,round_robin,priority,union_upper --rrf-k 60`

### Broad pool + cheap ranker（O-S10 wide hand feature）

- `round18/experiments/o_sparse/o_s10_wide_hand_feature/run_o_s10_wide_hand_feature.py`
- Inputs:
  - `--train-pool round18/outputs/o_sparse/o_s9_union_gate_bm25_char_train/train_full_train_o_s9_union_gate_bm25_char_round_robin_top1000_candidates.json`
  - `--dev-pool round18/outputs/o_sparse/o_s9_union_gate_bm25_char_dev/dev_full_dev_o_s9_union_gate_bm25_char_round_robin_top1000_candidates.json`
  - `--train-pool-limit 1000 --dev-pool-limit 1000 --candidate-k 500 --eval-k 3,10,64,100,500`

## 2) 运行时对比

### A) 基于现有 leaf pool（仅合并/压缩）

| Pipeline | wall time (s) | 候选规模（总/平均） | 备注 |
|---|---:|---|---|
| Fixed sparse fusion（O-S7） | `141.503` | `77,000` / `500` | 154 claims * 500 |
| Word+char broad（O-S9） | `20.084` | `126,385` / `820.68` | 每 claim 607~978 |
| Four-signal broad（O-S9） | `33.878` | `196,381` / `1275.20` | 每 claim 882~1697 |
| Broad + cheap ranker（O-S10） | `329.741` | `77,000` / `500` | 训练+rerank（dev/train pool 复用现有） |

### B) 从原始 leaf 重新生成（把“主要成本源”拆分）

#### 叶池重建（主干开销）

| Leaf stage | wall time (s) |
|---|---:|
| o_s1 BM25 | `53.714` |
| o_s6 char TF-IDF | `145.011` |
| o_s2 structured | `50.301` |
| o_s3 PRF | `143.595` |
| 合计（Leaf 前置） | `392.621` |

#### 由重建的 leaf 再合并出 broad 与 rerank

| Pipeline | wall time (s) |
|---|---:|
| Word+char broad（rebuild） | `8.498` |
| Four-signal broad（rebuild） | `13.375` |
| Broad + cheap ranker（rebuild） | `146.155` |

> 可比总时延（如果把 leaf 重建也计入）：
> - Fixed sparse: `~534.124`
> - Word+char broad: `~401.119`
> - Four-signal broad: `~405.996`
> - Broad + cheap ranker（基于现成 train pool）: `~538.776`

## 3) Recall@64 / Recall@500（快速读出）

| 方法 | recall@64 | recall@500 |
|---|---:|---:|
| Fixed sparse fusion | unavailable（该 run 仅 eval 100/500） | `0.657900` |
| Word+char broad | `0.412771` | `0.657900` |
| Four-signal broad | `0.425325` | `0.645996` |
| Broad + cheap ranker（现成 dev pool） | `0.487554` | `0.700541` |
| Broad + cheap ranker（dev pool重建后） | `0.492316` | `0.692208` |

## 4) 结论

1. Broad 候选池扩容确实带来额外 wall time，且放大不线性。与固定 500 候选相比，word+char 候选总量约 +64%（77,000→126,385），但对合并耗时影响不足 14%（20.08s vs 141.5s）；成本优势/劣势主要取决于下游排序与训练阶段是否吃满。  
2. 四信号扩容到约 1.28k/claim 是明显吞吐压力点（4-source broad 33.88s），候选总量约为 fixed 的 2.55x，但 wall time 仅约 1.7x（33.88s vs 20.08s/141.50s 取决于比较口径）。
3. Cheap ranker 总耗时仍被模型训练+rerank dominate（`~329.7s` 或重建链上 `146.155s`），其上游 broad 规模变化是次要项。
4. 若把“从原始叶池重建”算入，主要新增成本是 o_s1/o_s6/o_s2/o_s3 四段，约 `392.6s`，这个才是 broad candidate pool 显著增加时最容易被忽略的关键时延项。

## 5) 风险

- Broad+ranker 的重建结果只替换了 dev pool；train pool 仍使用现成 `train_full...`，因此“从零开始完整时间”会更高。  
- fixed/sparse/word+char/four-signal 的对比里，候选规模与 metric 采样口径不同（例如 fixed 的 recall@64 未产出），横向精度比较有限。  
- `wall_seconds` 来自各 run manifest 与阶段性 log，可能存在采样/预热差异；建议后续加 3~5 次重复运行取中位数。  
