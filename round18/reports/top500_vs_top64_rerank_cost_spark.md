# Top500 vs Top64 Rerank Cost/Benefit (Round 18)

## 1) Candidate-pair counts and measured timing

- **Cross-encoder top64 runs are the only measured CE rerank experiments in `o_d3*`.**
  - `round18/outputs/o_dense/o_d3_cross_encoder/full_dev_top64_manifest.json`:
    - Command uses `--candidate-k 64` from `run_o_d3_cross_encoder.py` default.
    - Wall time: **20.018s**.
  - `round18/outputs/o_dense/o_d3_cross_encoder/dev_full_dev_full_dev_top64_strict_top64_data_flow_report.json`:
    - `counts.reranked_candidates = 9856` for 154 claims.
  - `round18/outputs/o_dense/o_d3b_cross_encoder_s7/full_dev_top64_s7_manifest.json`:
    - Command uses `--candidate-k 64`.
    - Wall time: **16.278s**.
  - `round18/outputs/o_dense/o_d3b_cross_encoder_s7/dev_full_dev_full_dev_top64_s7_strict_top64_data_flow_report.json`:
    - `counts.reranked_candidates = 9856` for 154 claims.

- **Top500 dense-like rerank attempt exists in O-D1, but it fell back (no SentenceTransformer model).**
  - `round18/outputs/o_dense/o_d1_light_biencoder/run_manifest.json` and `run_record.json`:
    - `--candidate-k 500 --max-candidates 500`.
    - Wall time: **433.818s**.
  - `round18/outputs/o_dense/o_d1_light_biencoder/dev_full_dev_o_d1_light_biencoder_fallback_tfidf_svd_metrics.json`:
    - `fallback_used = true`
    - `fallback_reason = "import failed: No module named 'sentence_transformers'"`
    - `candidate_top_k = 500`, 154 claims, `macro_recall_at_500 = 0.6195887446`.
  - `round18/outputs/o_dense/o_d1_light_biencoder/dev_full_dev_o_d1_light_biencoder_fallback_tfidf_svd_top500_candidates.json`:
    - 154 claims, **77,000** candidates (exactly 500 each), 44,842 unique sparse evidence IDs.

- **Aggregate top500 experiments use top500 fixed candidates, not extra CE inference per claim.**
  - `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/run_record.json`:
    - `--candidate-k 500`
    - CE source is top64 CE file from O-D3b (`.../full_dev_top64_s7_strict_top64_candidates.json`).
  - `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/run_manifest.json` runtime: **11.714s**.
  - `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/dev_full_dev_o_a2b_plain_leaf_rank_s7_strict_*_top500_candidates.json`:
    - 154 claims, **77,000** entries each (`candidate_top_k=500`).

## 2) Measured recall/F1 impacts (macro recall)

Top500 CE/aggregation variants were tested mainly as diagnostic/merging variants in `o_a2b_plain_leaf_rank_s7`:

| Variant | macro@1 | macro@3 | macro@5 | macro@10 | macro@64 | macro@100 | macro@500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_sparse_only | 0.07868 | 0.11883 | 0.15812 | 0.23690 | 0.44048 | 0.48853 | 0.66667 |
| strict_rrf_sparse_ce | 0.09091 | 0.19794 | 0.24957 | 0.29589 | 0.44048 | 0.48853 | 0.66667 |
| strict_ce64_sparse_backfill | 0.10195 | 0.21937 | 0.27543 | 0.33463 | 0.44048 | 0.48853 | 0.66667 |

Pattern is consistent with `o_a2c_plain_leaf_rank_s8` and `o_a2_calibrated_rank`: CE-based variants improve shallow recall, but all three `o_a2b`/`o_a2c`/`o_a2_calibrated` variants keep `@500` unchanged and do not show deep-tail gains beyond the rerank window.

## 3) Cost multiplier estimate

- Candidate pairs:  
  - Top64: `154 × 64 = 9856`  
  - Top500: `154 × 500 = 77000`
- Multiplier: **7.8125×** more candidates.
- Throughput extrapolation from CE top64 timing gives ~**127–156s** for top500 CE rerank on similar hardware (16.278s to 20.018s × 7.8125), excluding extra I/O and model-load effects.
- Observed top500 O-D1 fallback runtime (**433.8s**) is much higher, confirming top500 rerank is significantly slower when the sentence-transformer path is unavailable.

## 4) Verdict

- **Top500 rerank should not be the default** for production-like cross-encoder ranking under current evidence:
  - only top64 rerank is directly executed on CE today;
  - no direct top500 CE run is present in the current dense CE scripts (`o_d3*` default/recorded `--candidate-k=64`);
  - the only top500 CE-influenced runs are diagnostic merges that do not change deep recall (`@100/@500`) materially.
- Treat top500 CE/CE-backfill as a **diagnostic experiment** when validating candidate gating losses, not a routine default in Round 18.

## 5) Files reviewed

- `round18/outputs/o_dense/o_d3_cross_encoder/full_dev_top64_manifest.json`
- `round18/outputs/o_dense/o_d3_cross_encoder/dev_full_dev_full_dev_top64_strict_top64_data_flow_report.json`
- `round18/outputs/o_dense/o_d3_cross_encoder/dev_full_dev_full_dev_top64_strict_top64_metrics.json`
- `round18/outputs/o_dense/o_d3b_cross_encoder_s7/full_dev_top64_s7_manifest.json`
- `round18/outputs/o_dense/o_d3b_cross_encoder_s7/dev_full_dev_full_dev_top64_s7_strict_top64_data_flow_report.json`
- `round18/outputs/o_dense/o_d3b_cross_encoder_s7/dev_full_dev_full_dev_top64_s7_strict_top64_metrics.json`
- `round18/outputs/o_dense/o_d1_light_biencoder/run_manifest.json`
- `round18/outputs/o_dense/o_d1_light_biencoder/dev_full_dev_o_d1_light_biencoder_fallback_tfidf_svd_metrics.json`
- `round18/outputs/o_dense/o_d1_light_biencoder/dev_full_dev_o_d1_light_biencoder_fallback_tfidf_svd_data_flow_report.json`
- `round18/outputs/o_dense/o_d1_light_biencoder/dev_full_dev_o_d1_light_biencoder_fallback_tfidf_svd_top500_candidates.json`
- `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/run_record.json`
- `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/run_manifest.json`
- `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/dev_full_dev_o_a2b_plain_leaf_rank_s7_strict_sparse_only_metrics.json`
- `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/dev_full_dev_o_a2b_plain_leaf_rank_s7_strict_rrf_sparse_ce_metrics.json`
- `round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/dev_full_dev_o_a2b_plain_leaf_rank_s7_strict_ce64_sparse_backfill_metrics.json`
- `round18/reports/batch2_dense_wave1_review.md`
