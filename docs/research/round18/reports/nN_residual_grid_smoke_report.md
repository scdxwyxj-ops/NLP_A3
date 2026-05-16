# n/N Residual Classifier Grid

Architecture: head branch uses claim + top-n evidence; residual branch uses ranks n+1..N evidence.
Both branches use separate TF-IDF word 1-2 vectorizers and separate shallow feature scalers.

## References

- Top5 shallow dev-best: macro-F1 `0.495421`, accuracy `0.545455`, top class share `0.376623`.
- k20 promoted dev-best: macro-F1 `0.471928`, accuracy `0.519481`, top class share `0.435065`.

## Train-Holdout Selected Result

- selected config: `head3_tail10_a0p0`
- train-holdout macro-F1: `0.349945`
- dev macro-F1: `0.451373`
- dev accuracy: `0.493506`
- tail alpha: `0.0`
- head/tail dims: `50886` / `60000` word features, `47` / `47` side features

## Best (n, N) By Train-Selected Dev Confirmation

- best cell: `n=5`, `N=10`
- selected config: `head5_tail10_a0p0`
- selected tail alpha: `0.0`
- dev macro-F1: `0.514725`

## Dev Diagnostic Best

- best config: `head5_tail20_a0p1`
- dev macro-F1: `0.515110`
- dev accuracy: `0.558442`
- tail alpha: `0.1`

## Best (n, N) By Dev Diagnostic Upper Bound

- best diagnostic cell: `n=5`, `N=20`
- diagnostic config: `head5_tail20_a0p1`
- diagnostic tail alpha: `0.1`
- dev macro-F1: `0.515110`
- diagnostic values are non-promoted.

## Top Train-Holdout Configs

| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `head3_tail10_a0p0` | 3 | 10 | 0.00 | 0.349945 | 0.426829 | 0.418699 |
| 2 | `head3_tail20_a0p0` | 3 | 20 | 0.00 | 0.349945 | 0.426829 | 0.418699 |
| 3 | `head3_tail10_a0p1` | 3 | 10 | 0.10 | 0.324307 | 0.418699 | 0.426829 |
| 4 | `head3_tail20_a0p1` | 3 | 20 | 0.10 | 0.324307 | 0.418699 | 0.426829 |
| 5 | `head5_tail10_a0p0` | 5 | 10 | 0.00 | 0.317920 | 0.418699 | 0.414634 |
| 6 | `head5_tail20_a0p0` | 5 | 20 | 0.00 | 0.317920 | 0.418699 | 0.414634 |
| 7 | `head5_tail10_a0p1` | 5 | 10 | 0.10 | 0.314260 | 0.414634 | 0.422764 |
| 8 | `head5_tail20_a0p1` | 5 | 20 | 0.10 | 0.310000 | 0.406504 | 0.414634 |

## Top Dev Diagnostic Configs

| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `head5_tail20_a0p1` | 5 | 20 | 0.10 | 0.515110 | 0.558442 | 0.428571 |
| 2 | `head5_tail10_a0p0` | 5 | 10 | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 3 | `head5_tail20_a0p0` | 5 | 20 | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 4 | `head5_tail10_a0p1` | 5 | 10 | 0.10 | 0.506690 | 0.551948 | 0.428571 |
| 5 | `head3_tail10_a0p0` | 3 | 10 | 0.00 | 0.451373 | 0.493506 | 0.422078 |
| 6 | `head3_tail20_a0p0` | 3 | 20 | 0.00 | 0.451373 | 0.493506 | 0.422078 |
| 7 | `head3_tail10_a0p1` | 3 | 10 | 0.10 | 0.447613 | 0.487013 | 0.422078 |
| 8 | `head3_tail20_a0p1` | 3 | 20 | 0.10 | 0.447613 | 0.487013 | 0.422078 |

Artifacts: `round18/outputs/o_classifier/nN_residual_grid_smoke/nN_residual_grid_results.json`
