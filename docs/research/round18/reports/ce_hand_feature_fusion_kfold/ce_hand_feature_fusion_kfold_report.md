# CE + Hand-Feature Fusion / Shallow-Ranker Tuning

- Run mode: `diagnostic-only`
- Tune mode: `fusion`
- Candidate budget: `top500`
- Eval ks: `[3, 10, 64, 500]`
- Selection policy: diagnostic-only (dev artifacts only)

## Inputs
- Train claims: `data/train-claims.json`
- Dev claims: `data/dev-claims.json`
- Evidence: `data/evidence.json`
- Train CE pool: `None`
- Train hand pool: `None`
- Dev CE pool: `round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json`
- Dev hand pool: `round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json`

## Command
```bash
round18/experiments/ce_hand_feature_fusion_kfold.py --smoke --smoke-claims 8 --output-dir round18/reports/ce_hand_feature_fusion_kfold
```

## Forbidden-token scan
- No forbidden tokens in scanned command/path fields.

## Status discipline
- Strict-vs-diagnostic marker: `diagnostic-only`
- Hyperparameters are selected from train-only k-fold only in strict mode.
- Dev-only selected hyperparameters are **not** promoted when strict mode is enabled.

## Diagnostic run only
- `--smoke` was used: run is bounded to `8` dev claims.
- Best diagnostic config: `fusion`
- CV macro@64: `0.695000`
- CV macro@3: `0.340000`
- CV macro@10: `0.485000`
- CV macro@500: `0.780000`

## Dev-only hyperparameter note
- This run cannot be used for strict submission selection; status is diagnostic-only.
- Use `--mode strict --train-ce-pool ... --train-hand-pool ...` once artifacts exist.

## Outputs
- Summary JSON: `round18/reports/ce_hand_feature_fusion_kfold/ce_hand_feature_fusion_kfold_summary.json`
- Summary CSV: `round18/reports/ce_hand_feature_fusion_kfold/ce_hand_feature_fusion_kfold_summary.csv`

## Top 10 candidates
| rank | tune_mode | status | ce_weight | hand_weight | model_c | tie_break | cv@64 | dev@64 |
|---:|---|---|---:|---:|---:|---|---:|---:|
| 1 | fusion | diagnostic-only | 0.75 | 0.25 |  | min_source_rank | 0.6950 | 0.6937 |
| 2 | fusion | diagnostic-only | 0.75 | 0.25 |  | ce_rank_first | 0.6950 | 0.0000 |
| 3 | fusion | diagnostic-only | 0.75 | 0.25 |  | hand_rank_first | 0.6950 | 0.0000 |
| 4 | fusion | diagnostic-only | 0.5 | 0.5 |  | min_source_rank | 0.6750 | 0.0000 |
| 5 | fusion | diagnostic-only | 0.5 | 0.5 |  | ce_rank_first | 0.6750 | 0.0000 |
| 6 | fusion | diagnostic-only | 0.5 | 0.5 |  | hand_rank_first | 0.6750 | 0.0000 |
| 7 | fusion | diagnostic-only | 1.0 | 0.0 |  | min_source_rank | 0.6750 | 0.0000 |
| 8 | fusion | diagnostic-only | 1.0 | 0.0 |  | ce_rank_first | 0.6750 | 0.0000 |
| 9 | fusion | diagnostic-only | 1.0 | 0.0 |  | hand_rank_first | 0.6750 | 0.0000 |
| 10 | fusion | diagnostic-only | 0.0 | 1.0 |  | min_source_rank | 0.5950 | 0.0000 |
