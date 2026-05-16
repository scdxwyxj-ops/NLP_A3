# Top3 Score Fusion (CE + hand-feature/source) — Round18 Spark

## Inputs

- Claims: `data/dev-claims.json` (154 claims)
- Evidence corpus: `data/evidence.json`
- CE base pool (strict): `round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json`
- Hand-feature/source candidates:
  - `round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json`
  - `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_top1000/dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_top1000_top500_candidates.json`
  - `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_rrf_top1000/dev_full_dev_o_s10_wide_hand_feature_bm25_char_rrf_top1000_top500_candidates.json`
  - `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_union_upper_top1000/dev_full_dev_o_s10_wide_hand_feature_bm25_char_union_upper_top1000_top500_candidates.json`
  - `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_pos16/dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_pos16_top500_candidates.json`
  - `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_train500/dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_train500_top500_candidates.json`
- Sweep:
  - `ce_weight ∈ {0.0,0.25,0.5,0.75,1.0}` (hand weight = `1-ce_weight`)
  - `tie_break ∈ {min_source_rank, ce_rank_first, hand_rank_first}`
  - `candidate_k = 500`
- `k` values: `[1,3,5,10,64,500]`

## Best variant (dev-sweep winner)

- Selection status: **diagnostic-only**
  - Weights were tuned by dev recall grid sweep; this file is not strict-frozen.
- Best run: `ce=0.75`, `hand=0.25`, `tie_break=min_source_rank`
- Selection rule: maximize `macro recall@3`, then use `@10`, `@64`, and `@500` only as tie-breaks.
- Hand pool: `round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json`
- Best candidate file:
  - `round18/reports/top3_score_fusion_spark/top3_score_fusion_best_dev_full_dev_o_s8_hand_feature_ranker_top500_candidates_min_source_rank_ce0.75_top500_candidates.json`
- Dev macro recall (top500 candidates, 154 claims, all claims covered):

| mR@1 | mR@3 | mR@5 | mR@10 | mR@64 | mR@500 |
|---:|---:|---:|---:|---:|---:|
| 0.1176 | 0.2387 | 0.2957 | 0.3682 | 0.5607 | 0.6667 |

## Notes

- `min_source_rank`, `ce_rank_first`, and `hand_rank_first` produced identical metrics for the top CE-weighted hand-feature variant, so the top three tie-break variants are equivalent on the requested `k` points.
- The top-3 winner is not the same as the top-500/context winner: the best `R@3` setting keeps the original hand-feature pool, while the wider RRF pool has better `R@500`. This supports treating top-3 evidence submission and top-64 classifier context as separate operating points.
- This run reuses existing Round18 artifacts only; no external models/retraining or non-existing files were introduced.
- Output summary artifacts are in `round18/reports/top3_score_fusion_spark/` (CSV + JSON + best candidates).

## Status / leakage

- **Strict status**: **strict candidate sources + diagnostic selection.**  
  - CE base is a strict top500 diagnostic artifact.
  - Final fused variant is **diagnostic-only** (selected on dev).
- **Leakage risk**: **low** (no forbidden-token hits; only `data/dev-claims.json`/`data/evidence.json` plus existing strict/dev artifact pools used for pooling and scoring).

## Files changed

- `round18/experiments/top3_score_fusion_spark.py` (new)
- `round18/reports/top3_score_fusion_spark.md` (new)
- `round18/reports/top3_score_fusion_spark/top3_score_fusion_summary.json`
- `round18/reports/top3_score_fusion_spark/top3_score_fusion_summary.csv`
- `round18/reports/top3_score_fusion_spark/top3_score_fusion_best_metrics.json`
- `round18/reports/top3_score_fusion_spark/top3_score_fusion_best_dev_full_dev_o_s8_hand_feature_ranker_top500_candidates_min_source_rank_ce0.75_top500_candidates.json`
- `round18/reports/top3_score_fusion_spark/top3_score_fusion_report.json`
