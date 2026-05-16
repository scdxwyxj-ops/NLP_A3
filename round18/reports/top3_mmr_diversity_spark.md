# Top3/Top10 MMR Diversity Check (Spark)

## Setup

- Goal: test whether bounded, deterministic duplicate-control (MMR) on the O-A2c hand-feature aggregate improves top-k recall.
- Source pools (existing artifacts):
  - Sparse: `round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json`
  - CE: `round18/outputs/o_dense/o_d3b_cross_encoder_s7/dev_full_dev_full_dev_top64_s7_strict_top64_candidates.json`
- Scoring pool for final rerank: max 500 candidates/claim.
- Evaluation ks: 1, 3, 5, 10, 64 (plus 100, 500 in artifacts).
- MMR implementation: deterministic greedy re-rank of top positions only.
  - `score = λ * rank_score - (1-λ) * max_j Jaccard(tokens_i, tokens_j)`
  - Similarity from evidence text token overlap (alphanumeric unigram Jaccard), bounded by `top_k ∈ {3,10}`
  - λ fixed at `0.7`, no randomization.

## Command

`python round18/experiments/o_aggregate/o_a2c_plain_leaf_rank_s8/run_o_a2c_plain_leaf_rank_s8.py --eval-k 1,3,5,10,64,100,500 --output-dir round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr --run-id o_a2c_plain_leaf_rank_s8_top3_top10_mmr --manifest round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr/run_manifest.json --record-path round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr/run_record.json`

## Macro recall comparison (dev)

| Variant | mR@1 | mR@3 | mR@5 | mR@10 | mR@64 |
|---|---:|---:|---:|---:|---:|
| strict_rrf_sparse_ce (baseline) | 0.091775 | 0.187554 | 0.247403 | 0.327597 | 0.463528 |
| strict_rrf_sparse_ce_mmr_top3 | 0.093074 | 0.104113 | 0.198593 | 0.318831 | 0.463528 |
| strict_rrf_sparse_ce_mmr_top10 | 0.093074 | 0.104113 | 0.110173 | 0.123701 | 0.463528 |

## Delta vs baseline (mR)

| Variant | Δ@1 | Δ@3 | Δ@5 | Δ@10 | Δ@64 |
|---|---:|---:|---:|---:|---:|
| mmr_top3 | +0.001299 | -0.083442 | -0.048810 | -0.008766 | +0.000000 |
| mmr_top10 | +0.001299 | -0.083442 | -0.137229 | -0.203896 | +0.000000 |

## Verdict

For this hand-feature + top64 CE setup, deterministic MMR **hurts overall diversity-controlled ranking quality at top3/top5/top10**. The only visible gain is a tiny +0.0013 macro recall at top1, while top64 is unchanged. Top10-MMR is substantially worse beyond the first position. Therefore, **top3/top10 lexical-diversity control does not help this pipeline** (and likely hurts at user-visible cutoffs 3–10).

## Files produced/read

- `round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr/dev_full_dev_o_a2c_plain_leaf_rank_s8_top3_top10_mmr_comparison_table.json`
- `round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr/dev_full_dev_o_a2c_plain_leaf_rank_s8_top3_top10_mmr_comparison_table.csv`
- `round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr/dev_full_dev_o_a2c_plain_leaf_rank_s8_top3_top10_mmr_metrics.json`
- `round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr/*_metrics.json`
- `round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8_top3_top10_mmr/*_candidates.json`

## Code changes

- `round18/experiments/o_aggregate/o_a2c_plain_leaf_rank_s8/run_o_a2c_plain_leaf_rank_s8.py` (added strict variants: `strict_rrf_sparse_ce_mmr_top3`, `strict_rrf_sparse_ce_mmr_top10`)
