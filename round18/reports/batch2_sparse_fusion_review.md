# Round18 Batch 2 Sparse Fusion Review

Date: 2026-05-09 Australia/Sydney

## Scope

Reviewed `O-S4` sparse fusion over current-run Round18 sparse sources:

- O-S1 BM25 full dev pool.
- O-S2 structured sparse routing full dev pool.
- O-S3 feedback-expanded sparse full dev pool.

## Master Verification

```bash
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_sparse/o_s4_fusion/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_sparse/o_s4_fusion/run_manifest.json
```

Results:

- strict guard: passed.
- manifest validation: passed.

## Key Results

Baseline O-S1 BM25:

- macro recall@100: `0.41883116883116883`
- macro recall@500: `0.5861471861471862`
- hit-any@500: `0.8376623376623377`

Best fixed strict fusion:

- variant: `strict_rrf_prf_heavy`
- macro recall@100: `0.4567099567099567`
- macro recall@500: `0.6195887445887446`
- hit-any@100: `0.7272727272727273`
- hit-any@500: `0.8506493506493507`

Best diagnostic dev-selected fusion:

- variant: `diagnostic_rrf_k60_w1.0_0.8_1.3`
- macro recall@100: `0.4515151515151515`
- macro recall@500: `0.6225108225108225`
- status: `diagnostic-only`

## Verdict

`APPROVED AS STRICT-CANDIDATE`

Approved strict candidate:

```text
round18/outputs/o_sparse/o_s4_fusion/dev_full_dev_o_s4_fusion_strict_rrf_prf_heavy_top500_candidates.json
```

Do not use the diagnostic-best output as final strict input unless its weights are later selected by train-only CV or pre-registered by a non-dev rule.

## Interpretation

The fixed strict fusion improves over O-S1 BM25:

- +`0.03787878787878779` macro recall@100
- +`0.03344155844155841` macro recall@500

This satisfies the Round18 hypothesis that aggregated sparse sources can beat a single sparse gate, at least on frozen dev evaluation for fixed non-dev-selected weights.

## Next Gate

Dense-in-sparse may now start, but only inside:

```text
round18/outputs/o_sparse/o_s4_fusion/dev_full_dev_o_s4_fusion_strict_rrf_prf_heavy_top500_candidates.json
```

Dense workers must not score the full evidence corpus in the final path.

