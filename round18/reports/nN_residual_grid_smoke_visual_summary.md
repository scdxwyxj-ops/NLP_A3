# n/N Residual Grid Visual Summary

- Input JSON: `/mnt/a/code/NLP/A3/round18/outputs/o_classifier/nN_residual_grid/nN_residual_grid_results.json`
- Evaluated grid cells: `4`
- Unique n values: `3, 5`
- Unique N values: `10, 20`
- Figure: `round18/reports/figures/nN_residual_grid_smoke_summary.png`

The train-selected panel uses the train-only selection result. The diagnostic panel, when present, shows dev-best comparison values and is explicitly non-promoted.

## Grid Table

| n | N | train-selected dev macro-F1 | selected tail_alpha | diagnostic dev-best macro-F1 | diagnostic status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 10 | 0.4514 | 0 | 0.4514 | non-promoted |
| 3 | 20 | 0.4514 | 0 | 0.4514 | non-promoted |
| 5 | 10 | 0.5147 | 0 | 0.5147 | non-promoted |
| 5 | 20 | 0.5147 | 0 | 0.5151 | non-promoted |

## Key Readout

- Best train-selected cell: `n=5`, `N=10`, macro-F1 `0.514725`.
- Best train-selected tail_alpha: `0`.
- tail_alpha = 0 cells: `4`; tail_alpha > 0 cells: `0`.
- Best diagnostic cell: `n=5`, `N=20`, macro-F1 `0.515110`.
- Diagnostic dev-best is shown for audit context only; it is not the promoted choice.

Artifacts: `round18/reports/figures/nN_residual_grid_smoke_summary.png`
Summary: `round18/reports/nN_residual_grid_smoke_visual_summary.md`
