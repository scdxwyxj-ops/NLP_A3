# Top3 Fusion Grid Diagnostic

Status: diagnostic only. Weights are selected on dev metrics and must not be promoted without train-only selection.

Input candidates: `round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json`
Best candidate file: `round18/outputs/o_rerank/top3_fusion_grid/dev_top3_fusion_grid_diagnostic_best_candidates.json`

## Best by macro recall@3, then @10, then @64

| ce_w | hf_w | rank_w | mR@1 | mR@3 | mR@5 | mR@10 | mR@64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.00 | 0.50 | 0.1083 | 0.2302 | 0.2937 | 0.3643 | 0.5761 |
| 0.25 | 0.00 | 1.00 | 0.1083 | 0.2302 | 0.2988 | 0.3643 | 0.5761 |
| 0.50 | 0.00 | 1.00 | 0.1083 | 0.2302 | 0.2937 | 0.3643 | 0.5761 |
| 0.75 | 0.00 | 1.00 | 0.1083 | 0.2302 | 0.2937 | 0.3643 | 0.5761 |
| 0.50 | 0.00 | 0.25 | 0.1083 | 0.2289 | 0.2924 | 0.3643 | 0.5728 |
| 0.75 | 0.00 | 0.25 | 0.1062 | 0.2289 | 0.2924 | 0.3643 | 0.5728 |
| 1.00 | 0.00 | 0.50 | 0.1083 | 0.2289 | 0.2924 | 0.3643 | 0.5728 |
| 1.50 | 0.00 | 0.50 | 0.1062 | 0.2289 | 0.2924 | 0.3643 | 0.5728 |
| 2.00 | 0.00 | 1.00 | 0.1083 | 0.2289 | 0.2924 | 0.3643 | 0.5728 |
| 0.25 | 0.00 | 0.25 | 0.1083 | 0.2289 | 0.2924 | 0.3643 | 0.5715 |
