# Top-64 MiniLM/CE Aggregate Spark Report (Round18, existing artifacts only)

Date: 2026-05-09
Scope: Investigate MiniLM/semantic + hand-feature / multi-gate top-64 evidence reranking using only existing `round18` outputs.

## 1) Existing Dense / MiniLM / CE Artifacts

Round18 currently has three dense/lighter-semantic artifact families:

### O-D3 (`cross-encoder/ms-marco-MiniLM-L6-v2`)
- CE lane file: `round18/outputs/o_dense/o_d3_cross_encoder/dev_full_dev_full_dev_top64_strict_top64_metrics.json`
- Rerank source gate: `round18/outputs/o_sparse/o_s4_fusion/dev_full_dev_o_s4_fusion_strict_rrf_prf_heavy_top500_candidates.json`
- Command/model: `cross-encoder/ms-marco-MiniLM-L6-v2`, candidate-k=64, eval k=`1,3,5,10,64`
- recall@3: `0.2159090909090909`
- recall@64: `0.4140692640692641`

### O-D3b (`cross-encoder/ms-marco-MiniLM-L6-v2`)
- CE lane file: `round18/outputs/o_dense/o_d3b_cross_encoder_s7/dev_full_dev_full_dev_top64_s7_strict_top64_metrics.json`
- Rerank source gate: `round18/outputs/o_sparse/o_s7_plain_leaf_fusion/dev_full_dev_o_s7_plain_leaf_fusion_strict_rrf_char_heavy_top500_candidates.json`
- Command/model: `cross-encoder/ms-marco-MiniLM-L6-v2`, candidate-k=64, eval k=`1,3,5,10,64`
- recall@3: `0.21937229437229436`
- recall@64: `0.44047619047619047`

### O-D1 (`sentence-transformers/all-MiniLM-L6-v2`, fallback)
- Artifact: `round18/outputs/o_dense/o_d1_light_biencoder/dev_full_dev_o_d1_light_biencoder_fallback_tfidf_svd_metrics.json`
- Requested model: `sentence-transformers/all-MiniLM-L6-v2`
- Backend used: `sklearn_tfidf_svd` (fallback; missing `sentence_transformers`)
- Eval scope: `100,500` (no top-3/top-64 in this artifact)

## 2) Which Sparse Gate/Leaf Is Reranked

- O-D3: strict O-S4 score-fusion top500 (`strict_rrf_prf_heavy`).
- O-D3b: strict O-S7 plain-leaf fusion top500 (`strict_rrf_char_heavy`), i.e., multi-gate (BM25 + char TF-IDF + structured + PRF).
- O-A2_calibrated_rank:
  - sparse: O-S4 (`strict_rrf_prf_heavy`)
  - CE leaf: O-D3 top64
  - dense fallback leaf: O-D1 candidate lane
- O-A2b:
  - sparse: O-S7 (`strict_rrf_char_heavy`)
  - CE leaf: O-D3b top64
- O-A2c:
  - sparse: O-S8 hand-feature top500
  - CE leaf: O-D3b top64

## 3) Recall@3 and Recall@64 Snapshot (macro recall)

| Artifact / Variant | Sparse gate source | CE source | Recall@3 | Recall@64 |
|---|---|---|---:|---:|
| O-D3 | O-S4 (`strict_rrf_prf_heavy`) | O-D3 | `0.2159090909090909` | `0.4140692640692641` |
| O-D3b | O-S7 (`strict_rrf_char_heavy`) | O-D3b | `0.21937229437229436` | `0.44047619047619047` |
| O-A2b `strict_ce64_sparse_backfill` | O-S7 (`strict_rrf_char_heavy`) | O-D3b | `0.21937229437229436` | `0.44047619047619047` |
| O-A2b `strict_rrf_sparse_ce` | O-S7 (`strict_rrf_char_heavy`) | O-D3b | `0.19794372294372295` | `0.44047619047619047` |
| O-A2c `strict_ce64_sparse_backfill` | O-S8 (`hand-feature top500`) | O-D3b | `0.21937229437229436` | `0.44047619047619047` |
| O-A2c `strict_rrf_sparse_ce` | O-S8 (`hand-feature top500`) | O-D3b | `0.18755411255411256` | `0.46352813852813857` |
| O-A2_calibrated `strict_ce64_sparse_backfill` | O-S4 (`strict_rrf_prf_heavy`) | O-D3 | `0.2159090909090909` | N/A |
| O-A2_calibrated `strict_rrf_sparse_ce` | O-S4 (`strict_rrf_prf_heavy`) | O-D3 | `0.18625541125541126` | N/A |

## 4) Existing combinations of MiniLM/CE with hand-feature or multi-gate

- Yes, MiniLM/CE is already combined with multi-gate sparse source in O-A2b (O-S7 is multi-gate; CE leaf from O-D3b).
- Yes, MiniLM/CE is already combined with hand-feature sparse source in O-A2c (`strict_ce64_sparse_backfill` and `strict_rrf_sparse_ce` use O-S8 hand-feature as sparse gate).
- No existing artifact reranks the O-S8 top64 candidate pool with a MiniLM/CE model directly; O-D3b CE is still fed by O-S7 or O-S4 only.

## 5) Smallest next experiment to test the hypothesis

Single smallest change with controlled comparison:

1. Generate a new CE rerank over the hand-feature top500 pool:
   - reuse `round18/experiments/o_dense/o_d3b_cross_encoder_s7/run_o_d3b_cross_encoder_s7.py`
   - set `--sparse-pool` to `round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json`
   - keep `--candidate-k 64` and same `cross-encoder/ms-marco-MiniLM-L6-v2` settings.
2. Run one aggregate mode on that new CE top64 pool:
   - `round18/experiments/o_aggregate/o_a2c_plain_leaf_rank_s8/run_o_a2c_plain_leaf_rank_s8.py`
   - reuse `--sparse-pool` already at O-S8 and swap only `--ce-pool` to the new CE top64.

Compare `strict_ce64_sparse_backfill` against existing `O-A2c strict_rrf_sparse_ce` on macro recall@3 and recall@64. This directly answers whether MiniLM/CE adds top-64 gains when coupled to hand-feature source material, with minimal compute cost versus a full pipeline redesign.
