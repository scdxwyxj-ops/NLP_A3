# Top500 CE-cutoff64 Direction Review (Spark)

## Executive conclusion

- **Do not promote Top500 CE-cutoff64 as the default strict candidate strategy yet.**
- The current artifacts show the CE signal is injected as a **top64 slice only** and then backfilled to 500; this improves shallow ranks (e.g., @3/@10) but does not raise `@500` in the fixed-pool variants.
- Treat CE-cutoff64 variants as **diagnostic ranking boosters**, not as the default top500 selection contract.

## Manifest and candidate-source audit

- **O-A2C / O-A2B / O-A2 calibrated** all use:
  - `--candidate-k 500` in aggregate output.
  - `--ce-pool` fixed to a top64 CE candidate file.
  - CE pool loaded with `_parse_pool(..., 64)` while sparse is loaded at `candidate_k`.
  - Evidence can be verified in:
    - [o_a2c_manifest](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/run_manifest.json)
    - [o_a2b_manifest](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/run_manifest.json)
    - [o_a2cal_manifest](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2_calibrated_rank/run_manifest.json)
- CE-backfill candidate files indeed contain a 64-item CE block followed by sparse backfill:
  - [o_a2c_ce_backfill_top500](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/dev_full_dev_o_a2c_plain_leaf_rank_s8_strict_ce64_sparse_backfill_top500_candidates.json)
  - [o_a2b_ce_backfill_top500](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/dev_full_dev_o_a2b_plain_leaf_rank_s7_strict_ce64_sparse_backfill_top500_candidates.json)
  - [o_a2cal_ce_backfill_top500](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2_calibrated_rank/dev_full_dev_o_a2_calibrated_rank_strict_ce64_sparse_backfill_top500_candidates.json)
    - all evidence files are built with 154 claims and `candidate_top_k=500`.
- Diagnostic variants using wider sparse pools behave consistently with the same mechanism:
  - [o_a2x_s8](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2x_s8_ce_s8_diag/run_manifest.json)
  - [o_a2x_s10](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2x_s10_ce_s10_diag/run_manifest.json)
- **Artifact inconsistency flag:** `o_a2x_s10_ce_s10_diag` command uses custom sparse/CE inputs in `command`/`data_flow_report`, but `input_files` in manifest still point to defaults due `build_input_records` hard-coding constants. That weakens provenance reproducibility even though strict checks pass.

## Leakage / contamination audit

- Strict contracts are enforced to dev split for scoring:
  - candidate and sparse sources are fixed artifacts under `round18/outputs/...`
  - labels are tracked in `input_files`; claims are `data/dev-claims.json` in all reviewed CE-cutoff64 aggregates.
- `strict_guard`/`validate_manifest` pass for key reviewed manifests:
  - [o_a2c](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/run_manifest.json)
  - [o_a2x_s10](/mnt/a/code/NLP/A3/round18/outputs/o_aggregate/o_a2x_s10_ce_s10_diag/run_manifest.json)
- No train/test split crossover is visible in the reviewed manifests; no `train-claims.json`/`test` scoring inputs for these CE-cutoff64 runs.
- The only clear governance issue is the **metadata drift** noted above (stale input-file paths).

## Metric interpretation

- `O-A2B` example:
  - sparse-only `macro@3=0.11883`, `macro@10=0.23690`, `macro@500=0.66667`, `micro@500=0.63951`
  - CE-backfill `macro@3=0.21937`, `macro@10=0.33463`, `macro@500=0.66667`, `micro@500=0.63951`
  - RRF sparse+CE `macro@10=0.29589`, `macro@500=0.66667`, `micro@500=0.63951`
- `O-A2C` and `O-A2 calibrated` show the same pattern: CE variants improve top3/top10, but preserve deep recall.
- `O-A2X_s10` objective tie at @500 leads to sparse-only being selected, even though CE-backfill improves @10, because `@500` and `@100` are unchanged by CE insertion.
- This is the expected consequence of **reranking inside a fixed 500 candidate universe**:
  - recall@500 is bounded by input sparse candidate coverage; CE reordering cannot add unseen evidence.

- Top500 CE-style rerank attempt (`O-D3x` top500_diag) is measured separately and is not using a dedicated CE-top500 source:
  - `sparse` only source metadata and `77,000` reranked candidates
  - runtime `44.845s` (`154 × 500` pairs)
  - candidate source does not come from a CE-top500 pool.

## Promotion status

- **Do not promote** any CE-cutoff64-top500 aggregate variant as global default yet.
- Keep CE-cutoff64 as a diagnostic layer for top3/top10-sensitive consumers only.
- Preserve strict defaults that maximize deep recall in the same candidate universe (for these runs, sparse-only wins in diagnostics where `@500` dominates).
- Fix provenance metadata bug before promoting/archiving this branch as a canonical strict line.

## Risk flags

1. **High:** `input_files` provenance mismatch in `o_a2x_s10_ce_s10_diag` manifest (stale default sparse/ce source paths).
2. **Medium:** diagnostic selection rule prioritizes `@500`/`@100`, so CE variants with clear top-k gains are not surfaced as "best" by default.
3. **Medium:** top500 CE intent is partially conflated with top500 dense-like rerank attempts; only explicit CE-cutoff64-backfill artifacts are present in most aggregate runs.
