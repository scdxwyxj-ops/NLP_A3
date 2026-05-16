# n/N Residual Grid Visual Summary

- Input JSON: `/mnt/a/code/NLP/A3/round18/outputs/o_classifier/nN_residual_grid/nN_residual_grid_results.json`
- Evaluated grid cells: `19`
- Unique n values: `3, 5, 8, 10`
- Unique N values: `10, 15, 20, 32, 64`
- Figure: `round18/reports/figures/nN_residual_grid_c4_summary.png`

The left panel shows dev confirmation after choosing tail_alpha on the train holdout inside each (n, N) cell. Comparing cells by dev is still diagnostic unless the cell was pre-registered. The diagnostic panel, when present, shows dev-best comparison values and is explicitly non-promoted.

## Grid Table

| n | N | train-selected dev macro-F1 | selected tail_alpha | diagnostic dev-best macro-F1 | diagnostic status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 10 | 0.4422 | 1 | 0.4748 | non-promoted |
| 3 | 15 | 0.4748 | 0 | 0.4748 | non-promoted |
| 3 | 20 | 0.4726 | 1 | 0.4748 | non-promoted |
| 3 | 32 | 0.4590 | 0.1 | 0.4767 | non-promoted |
| 3 | 64 | 0.4670 | 0.05 | 0.4748 | non-promoted |
| 5 | 10 | 0.3957 | 1 | 0.5011 | non-promoted |
| 5 | 15 | 0.4219 | 1 | 0.5011 | non-promoted |
| 5 | 20 | 0.4416 | 1 | 0.5011 | non-promoted |
| 5 | 32 | 0.4430 | 1 | 0.5011 | non-promoted |
| 5 | 64 | 0.4393 | 1 | 0.5011 | non-promoted |
| 8 | 10 | 0.4133 | 1 | 0.4133 | non-promoted |
| 8 | 15 | 0.4040 | 1 | 0.4295 | non-promoted |
| 8 | 20 | 0.4217 | 1 | 0.4295 | non-promoted |
| 8 | 32 | 0.4270 | 0.25 | 0.4745 | non-promoted |
| 8 | 64 | 0.4259 | 1 | 0.4266 | non-promoted |
| 10 | 15 | 0.4516 | 0.25 | 0.4545 | non-promoted |
| 10 | 20 | 0.4634 | 1 | 0.4634 | non-promoted |
| 10 | 32 | 0.4326 | 0.1 | 0.4842 | non-promoted |
| 10 | 64 | 0.4256 | 1 | 0.4326 | non-promoted |

## Key Readout

- Best dev-confirmed cell after train-only alpha selection: `n=3`, `N=15`, macro-F1 `0.474811`.
- Best train-selected tail_alpha: `0`.
- This cell-level comparison uses dev labels; treat it as an audit result unless the cell was pre-registered.
- tail_alpha = 0 cells: `1`; tail_alpha > 0 cells: `18`.
- Best diagnostic cell: `n=5`, `N=10`, macro-F1 `0.501089`.
- Diagnostic dev-best is shown for audit context only; it is not the promoted choice.

Artifacts: `round18/reports/figures/nN_residual_grid_c4_summary.png`
Summary: `round18/reports/nN_residual_grid_c4_visual_summary.md`
