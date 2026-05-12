# n/N Residual Classifier Master Audit

## Goal

Test the architecture where the first `n` evidence items are the main context and ranks `n+1..N` are added as a residual branch. The intended question is whether deeper context can be used as an upgrade over the strong top-5 classifier without losing the top-5 signal.

## Implemented Experiment

- Main branch: claim + top-`n` evidence text + shallow factual features.
- Residual branch: ranks `n+1..N` evidence text + shallow factual features.
- Model: TF-IDF word 1-2 + shallow features, one-vs-rest logistic regression.
- Grid: `n={3,5,8,10}`, `N={10,15,20,32,64}`, `tail_alpha={0,0.05,0.1,0.25,0.5,1.0}`.
- Strict selector: train-holdout macro-F1, then accuracy, then lower top-class share.
- Dev is used only for confirmation and diagnostic analysis.

## Results

| run | strict global train-selected config | train macro-F1 | dev macro-F1 | best dev-confirmed cell after train alpha | dev-best diagnostic |
| --- | --- | ---: | ---: | --- | --- |
| `C=2.0` | `n=5,N=10,alpha=1.0` | 0.4034 | 0.4123 | `n=3,N=20,alpha=1.0`, 0.4979 | `n=5,N=64,alpha=0.1`, 0.5157 |
| `C=4.0` | `n=5,N=10,alpha=1.0` | 0.4093 | 0.3957 | `n=3,N=15,alpha=0.0`, 0.4748 | `n=5,N=10,alpha=0.0`, 0.5011 |

## Audit Readout

- The strict global train-holdout selector is not stable enough here. It chooses `alpha=1.0` in both C settings, but this transfers poorly to dev.
- The strongest diagnostic pattern is still top-5 dominated.
- Under `C=2.0`, the best diagnostic residual result is `n=5,N=64,alpha=0.1` with macro-F1 `0.5157`, but the same run's top-5-only setting (`n=5,alpha=0`) is already `0.5147`. The residual gain is only about `+0.0010`.
- Under `C=4.0`, the best diagnostic result is exactly `n=5,N=10,alpha=0.0`, meaning no residual contribution is selected.
- All strict selected dev predictions avoid class collapse by top-class share, but the macro-F1 is lower than the current top-5 story.

## Conclusion

Do not promote the n/N residual architecture as the final classifier yet. It is a useful diagnostic result: it shows that extra evidence ranks can be attached as a residual branch, but the actual gain over top-5-only is tiny and not stable across C. For the tutorial, the clean story is:

1. Top-5 context carries most of the classification signal.
2. Residual deeper context is a reasonable architecture to test.
3. In our strict audit, the residual branch did not produce a robust new winner.

## Artifacts

- `round18/experiments/o_classifier/nN_residual/run_nN_residual_grid.py`
- `round18/experiments/o_classifier/nN_residual/plot_nN_residual_results.py`
- `round18/reports/nN_residual_grid_report.md`
- `round18/reports/nN_residual_grid_c4_report.md`
- `round18/reports/figures/nN_residual_grid_summary.png`
- `round18/reports/figures/nN_residual_grid_c4_summary.png`
