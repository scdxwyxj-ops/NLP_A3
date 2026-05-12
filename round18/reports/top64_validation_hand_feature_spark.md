# Hand-Feature Top-64 Validation (Round18, Spark Pass)

## 1) Candidate files identified

- Fixed score-fusion gate (Round18 O-S4):  
  `round18/outputs/o_sparse/o_s4_fusion/dev_full_dev_o_s4_fusion_strict_rrf_prf_heavy_top500_candidates.json`
- Fixed score-fusion variant used by CE-S7 gate (Round18 O-S7):  
  `round18/outputs/o_sparse/o_s7_plain_leaf_fusion/dev_full_dev_o_s7_plain_leaf_fusion_strict_rrf_char_heavy_top500_candidates.json`
- Hand-feature reranker (Round18 O-S8):  
  `round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json`
- Word+character union pool (Round18 O-S9):  
  `round18/outputs/o_sparse/o_s9_union_gate_bm25_char_dev/dev_full_dev_o_s9_union_gate_bm25_char_rrf_top1000_candidates.json`
- Four-gate union pool (Round18 O-S9 wide):  
  `round18/outputs/o_sparse/o_s9_union_gate_four_source_wide_dev/dev_full_dev_o_s9_union_gate_four_source_wide_rrf_top2000_candidates.json`
- Union pool + cheap ranker (Round18 O-S10):  
  `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_top1000/dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_top1000_top500_candidates.json`
- MiniLM/CE semantic candidates:  
  `round18/outputs/o_dense/o_d3_cross_encoder/dev_full_dev_full_dev_top64_strict_top64_candidates.json`  
  `round18/outputs/o_dense/o_d3b_cross_encoder_s7/dev_full_dev_full_dev_top64_s7_strict_top64_candidates.json`

## 2) Recomputed macro recall@N on dev-claims

`N = [3,5,10,20,32,50,64,100,500]`, using evidence-id overlap recall per claim.

| Selector | 3 | 5 | 10 | 20 | 32 | 50 | 64 | 100 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed score-fusion (O-S4 strict RRF PRF heavy) | 0.112338 | 0.146970 | 0.207468 | 0.302165 | 0.355736 | 0.394913 | 0.414069 | 0.456710 | 0.619589 |
| Hand-feature reranker (O-S8) | 0.155519 | 0.201840 | 0.262446 | 0.358658 | 0.410498 | 0.471861 | **0.492749** | 0.532251 | 0.666667 |
| Word+char union (O-S9) | 0.108225 | 0.167424 | 0.223918 | 0.297186 | 0.340476 | 0.385498 | 0.412771 | 0.463745 | 0.657900 |
| Four-gate union (O-S9 wide) | 0.117641 | 0.153571 | 0.222078 | 0.308009 | 0.352489 | 0.408225 | 0.425325 | 0.483766 | 0.645996 |
| Union + cheap ranker (O-S10) | 0.146320 | 0.177489 | 0.255195 | 0.341775 | 0.388312 | 0.462662 | 0.487554 | 0.532684 | 0.700541 |
| MiniLM/CE strict top64 (O-D3) | 0.215909 | 0.263745 | 0.302922 | 0.360281 | 0.401407 | 0.410823 | 0.414069 | 0.414069 | 0.414069 |
| MiniLM/CE S7 strict top64 (O-D3b) | 0.219372 | 0.275433 | 0.334632 | 0.392208 | 0.426623 | 0.440476 | 0.440476 | 0.440476 | 0.440476 |
| Fixed score-fusion (O-S7 char-heavy) | 0.118831 | 0.158117 | 0.236905 | 0.308874 | 0.360390 | 0.408766 | 0.440476 | 0.488528 | 0.666667 |

## 3) Leakage check from manifests / configs

- Labels are explicitly tracked via `labels_used` in manifests.
- O-S4/O-S7/O-S9 gate runs: only `data/dev-claims.json` has `labels_used: true`; they do not consume training labels.
- O-S8 and O-S10: `data/train-claims.json` and `data/dev-claims.json` both have `labels_used: true`, consistent with training on train labels and scoring on dev; this is not train/dev leakage.
- CE runs O-D3/O-D3b: only `data/dev-claims.json` is labeled as used.
- No run manifest references `data/test-claims-unlabelled.json` or any non-dev source for scoring/evaluation.
- All target candidate files cover all 154 dev claims in recomputation (no missing claims).

## 4) Claim: is hand-feature best at top-64?

Yes, within the identified Round18 candidates, hand-feature reranker is highest at top-64 with `mR@64 = 0.492749`.

Next-best is Union + cheap ranker with `mR@64 = 0.487554`.

Margin: **0.005195** absolute (about **1.04%** relative improvement), i.e. a **small** gain.

Per-claim top-64 comparison (hand-feature vs union+cheap): better on 20 claims, worse on 18, tie on 116.

## 5) Files changed

- Added: `round18/reports/top64_validation_hand_feature_spark.md`
