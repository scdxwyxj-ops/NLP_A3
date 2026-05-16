# n/N Residual Grid Visual Summary

- Input JSON: `/mnt/a/code/NLP/A3/round18/outputs/o_classifier/nN_residual_grid/nN_residual_grid_results.json`
- Evaluated grid cells: `19`
- Unique n values: `3, 5, 8, 10`
- Unique N values: `10, 15, 20, 32, 64`
- Figure: `round18/reports/figures/nN_residual_grid_summary.png`

The left panel shows dev confirmation after choosing tail_alpha on the train holdout inside each (n, N) cell. Comparing cells by dev is still diagnostic unless the cell was pre-registered. The diagnostic panel, when present, shows dev-best comparison values and is explicitly non-promoted.

## Grid Table

| n | N | train-selected dev macro-F1 | selected tail_alpha | diagnostic dev-best macro-F1 | diagnostic status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 10 | 0.4606 | 1 | 0.4606 | non-promoted |
| 3 | 15 | 0.4639 | 1 | 0.4639 | non-promoted |
| 3 | 20 | 0.4979 | 1 | 0.4979 | non-promoted |
| 3 | 32 | 0.4514 | 0 | 0.4845 | non-promoted |
| 3 | 64 | 0.4514 | 0 | 0.4738 | non-promoted |
| 5 | 10 | 0.4123 | 1 | 0.5147 | non-promoted |
| 5 | 15 | 0.4268 | 1 | 0.5147 | non-promoted |
| 5 | 20 | 0.4504 | 1 | 0.5151 | non-promoted |
| 5 | 32 | 0.4356 | 1 | 0.5147 | non-promoted |
| 5 | 64 | 0.4338 | 1 | 0.5157 | non-promoted |
| 8 | 10 | 0.4213 | 1 | 0.4292 | non-promoted |
| 8 | 15 | 0.4468 | 1 | 0.4468 | non-promoted |
| 8 | 20 | 0.4426 | 1 | 0.4426 | non-promoted |
| 8 | 32 | 0.4615 | 1 | 0.4615 | non-promoted |
| 8 | 64 | 0.4345 | 1 | 0.4345 | non-promoted |
| 10 | 15 | 0.4594 | 0.5 | 0.4594 | non-promoted |
| 10 | 20 | 0.4806 | 1 | 0.4806 | non-promoted |
| 10 | 32 | 0.4322 | 0 | 0.4854 | non-promoted |
| 10 | 64 | 0.4267 | 0.25 | 0.4374 | non-promoted |

## Key Readout

- Best dev-confirmed cell after train-only alpha selection: `n=3`, `N=20`, macro-F1 `0.497856`.
- Best train-selected tail_alpha: `1`.
- This cell-level comparison uses dev labels; treat it as an audit result unless the cell was pre-registered.
- tail_alpha = 0 cells: `3`; tail_alpha > 0 cells: `16`.
- Best diagnostic cell: `n=5`, `N=64`, macro-F1 `0.515727`.
- Diagnostic dev-best is shown for audit context only; it is not the promoted choice.

Artifacts: `round18/reports/figures/nN_residual_grid_summary.png`
Summary: `round18/reports/nN_residual_grid_visual_summary.md`
