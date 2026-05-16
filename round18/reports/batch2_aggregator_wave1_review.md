# Round18 Batch 2 Aggregator Wave 1 Review

Date: 2026-05-09 Australia/Sydney

## Scope

Reviewed `O-A2` leaf-flat fixed-rule aggregation over:

- sparse fusion top500;
- bounded cross-encoder top64;
- optional O-D1 fallback similarity leaf.

## Master Verification

```bash
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_aggregate/o_a2_calibrated_rank/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_aggregate/o_a2_calibrated_rank/run_manifest.json
```

Results:

- strict guard: passed.
- manifest validation: passed.

## Key Results

Sparse-only baseline:

- macro recall@3: `0.11233766233766235`
- macro recall@10: `0.20746753246753247`
- macro recall@100: `0.4567099567099567`
- macro recall@500: `0.6195887445887446`

CE top64 + sparse backfill:

- macro recall@3: `0.2159090909090909`
- macro recall@10: `0.3029220779220779`
- macro recall@100: `0.4567099567099567`
- macro recall@500: `0.6195887445887446`

RRF sparse + CE:

- macro recall@1: `0.10995670995670996`
- macro recall@3: `0.18625541125541126`
- macro recall@10: `0.2937229437229437`
- macro recall@100: `0.4567099567099567`
- macro recall@500: `0.6195887445887446`

MMR variants:

- did not improve top3/top10.
- keep as diagnostic design evidence, not recommended for the current strict path.

## Verdict

`APPROVED AS STRICT-CANDIDATE`

Recommended strict evidence top3 candidate:

```text
round18/outputs/o_aggregate/o_a2_calibrated_rank/dev_full_dev_o_a2_calibrated_rank_strict_ce64_sparse_backfill_top500_candidates.json
```

Reason:

- best top3 macro recall among fixed strict variants;
- preserves sparse top100/top500 recall by backfilling from sparse ranking;
- demonstrates dense-in-sparse improves early evidence ranking over sparse-only.

Secondary strict candidate:

```text
round18/outputs/o_aggregate/o_a2_calibrated_rank/dev_full_dev_o_a2_calibrated_rank_strict_rrf_sparse_ce_top500_candidates.json
```

Reason:

- best top1 and top10 macro recall;
- useful if a downstream classifier prefers top10 context quality.

## Hypothesis Status

H1 sparse fusion beats single sparse gate:

- supported by O-S4 strict fusion.

H2 dense scoring inside sparse candidates improves sparse-only ranking:

- supported for top3/top10 by O-D3 + O-A2.

H3 final leaf-flat aggregator beats individual leaves:

- partially supported.
- CE backfill beats sparse-only at top3/top10 and preserves top100/top500.
- Further train-only selection is still required before final strict freeze.

## Next Recommended Work

Start classifier/context work using:

- top3 evidence from `strict_ce64_sparse_backfill`;
- top10/top100 context comparison between `strict_ce64_sparse_backfill` and `strict_rrf_sparse_ce`.

Do not treat dev-selected diagnostic outputs as final.

