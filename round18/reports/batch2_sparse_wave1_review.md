# Round18 Batch 2 Sparse Wave 1 Review

Date: 2026-05-09 Australia/Sydney

## Scope

Reviewed the first sparse optimization wave:

- `O-S1`: lexical BM25 / TF-IDF-capable sparse baseline.
- `O-S2`: structured query routing with entity-like, number, and date views.
- `O-S3`: lexical feedback expansion / RM3-lite prototype.

All worker edits stayed under `round18/experiments/o_sparse/...` and `round18/outputs/o_sparse/...`.

## Master Verification Commands

```bash
PYTHONPATH=src:. python -m round18.tools.strict_guard \
  round18/outputs/o_sparse/o_s1_lexical_index_experiments/run_manifest.json \
  round18/outputs/o_sparse/o_s2_structured/full_dev_manifest.json \
  round18/outputs/o_sparse/o_s3_prf/full_dev_manifest.json

PYTHONPATH=. pytest -q round18/tests
```

Results:

- strict guard: passed for reviewed full/dev manifests.
- tests: `4 passed`.

## O-S1: Lexical Baseline

Verdict: `APPROVED AS STRICT-CANDIDATE`

Primary files:

- `round18/experiments/o_sparse/o_s1_lexical_index_experiments/run_lexical_index_experiment.py`
- `round18/outputs/o_sparse/o_s1_lexical_index_experiments/dev_full_bm25_dev_bm25_top500_candidates.json`
- `round18/outputs/o_sparse/o_s1_lexical_index_experiments/dev_full_bm25_dev_metrics.json`
- `round18/outputs/o_sparse/o_s1_lexical_index_experiments/run_manifest.json`

Full dev metrics:

- macro recall@100: `0.41883116883116883`
- macro recall@500: `0.5861471861471862`
- hit-any@100: `0.6753246753246753`
- hit-any@500: `0.8376623376623377`
- wall time: about `56.41s`

Assessment:

- This is currently the best strict sparse candidate from this wave.
- It reads raw dev/evidence JSON only and writes Round18-isolated artifacts.
- It is a valid baseline for O-S4 fusion and later dense-in-sparse leaves.

## O-S2: Structured Sparse Routing

Verdict: `APPROVED AS STRICT-CANDIDATE PROTOTYPE`

Primary files:

- `round18/experiments/o_sparse/o_s2_structured/run_o_s2_structured.py`
- `round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposed_candidates.json`
- `round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposition_profile.json`
- `round18/outputs/o_sparse/o_s2_structured/dev_full_dev_metrics.json`
- `round18/outputs/o_sparse/o_s2_structured/full_dev_manifest.json`

Full dev metrics:

- macro recall@100: `0.37575757575757573`
- macro recall@500: `0.5454545454545454`
- hit-any@100: `0.6298701298701299`
- hit-any@500: `0.7922077922077922`

Assessment:

- It is strict and useful as a source of route features.
- As a standalone candidate source, it underperforms O-S1 BM25.
- Its output should feed O-S4 fusion/feature analysis, not replace O-S1.
- The inherited `[round13]` progress prefix from the reusable BM25 helper is a logging issue only; no Round13 artifact input was observed in the manifest.

## O-S3: PRF/RM3-Lite Expansion

Verdict: `APPROVED AS STRICT-CANDIDATE PROTOTYPE`

Primary files:

- `round18/experiments/o_sparse/o_s3_prf/run_o_s3_prf.py`
- `round18/outputs/o_sparse/o_s3_prf/candidate_pool_baseline_top500.json`
- `round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json`
- `round18/outputs/o_sparse/o_s3_prf/expanded_queries.json`
- `round18/outputs/o_sparse/o_s3_prf/full_dev_manifest.json`

Full dev metrics using O-S1 as current-run baseline input:

- baseline macro recall@100: `0.41883116883116883`
- PRF macro recall@100: `0.42673160173160174`
- baseline macro recall@500: `0.5861471861471862`
- PRF macro recall@500: `0.578030303030303`
- wall time: about `3m`

Assessment:

- The first O-S3 manifest failed hard guard because notes contained forbidden substrings in a negated sentence. The script was patched to remove those strings and rerun.
- Full dev PRF slightly improves recall@100 but slightly lowers recall@500.
- Treat O-S3 as a useful fusion leaf or top100-oriented variant, not a replacement for O-S1.

## Current Sparse Ranking

Strict sparse source ranking after Wave 1:

1. O-S1 BM25 full dev: best top500 recall and hit-any.
2. O-S3 PRF: useful top100 improvement, top500 regression.
3. O-S2 structured routing: useful route features, lower standalone recall.

## Next Recommended Work

Start `O-S4` sparse fusion:

- read O-S1 full candidate pool;
- read O-S2 full structured candidate pool;
- read O-S3 PRF candidate pool;
- implement RRF / weighted sparse fusion;
- select weights with train-only CV before any final strict selection;
- report dev only as frozen evaluation or mark diagnostic.

Do not start dense leaves until a sparse fusion candidate is frozen or at least one O-S4 strict candidate exists.

