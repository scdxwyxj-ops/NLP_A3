# n/N Residual Classifier Grid

Architecture: head branch uses claim + top-n evidence; residual branch uses ranks n+1..N evidence.
Both branches use separate TF-IDF word 1-2 vectorizers and separate shallow feature scalers.

## References

- Top5 shallow dev-best: macro-F1 `0.495421`, accuracy `0.545455`, top class share `0.376623`.
- k20 promoted dev-best: macro-F1 `0.471928`, accuracy `0.519481`, top class share `0.435065`.

## Train-Holdout Selected Result

- selected config: `head5_tail10_a1p0`
- train-holdout macro-F1: `0.403400`
- dev macro-F1: `0.412322`
- dev accuracy: `0.506494`
- tail alpha: `1.0`
- head/tail dims: `60000` / `60000` word features, `47` / `47` side features

## Best (n, N) By Train-Selected Dev Confirmation

- best cell: `n=3`, `N=20`
- selected config: `head3_tail20_a1p0`
- selected tail alpha: `1.0`
- dev macro-F1: `0.497856`

## Dev Diagnostic Best

- best config: `head5_tail64_a0p1`
- dev macro-F1: `0.515727`
- dev accuracy: `0.558442`
- tail alpha: `0.1`

## Best (n, N) By Dev Diagnostic Upper Bound

- best diagnostic cell: `n=5`, `N=64`
- diagnostic config: `head5_tail64_a0p1`
- diagnostic tail alpha: `0.1`
- dev macro-F1: `0.515727`
- diagnostic values are non-promoted.

## Top Train-Holdout Configs

| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `head5_tail10_a1p0` | 5 | 10 | 1.00 | 0.403400 | 0.483740 | 0.475610 |
| 2 | `head3_tail20_a1p0` | 3 | 20 | 1.00 | 0.392750 | 0.475610 | 0.422764 |
| 3 | `head5_tail64_a1p0` | 5 | 64 | 1.00 | 0.379226 | 0.467480 | 0.422764 |
| 4 | `head5_tail20_a1p0` | 5 | 20 | 1.00 | 0.375987 | 0.443089 | 0.426829 |
| 5 | `head3_tail10_a1p0` | 3 | 10 | 1.00 | 0.369033 | 0.455285 | 0.459350 |
| 6 | `head10_tail20_a1p0` | 10 | 20 | 1.00 | 0.365253 | 0.451220 | 0.451220 |
| 7 | `head10_tail64_a0p25` | 10 | 64 | 0.25 | 0.364868 | 0.451220 | 0.422764 |
| 8 | `head10_tail64_a1p0` | 10 | 64 | 1.00 | 0.362906 | 0.467480 | 0.451220 |
| 9 | `head8_tail64_a1p0` | 8 | 64 | 1.00 | 0.362470 | 0.463415 | 0.430894 |
| 10 | `head8_tail10_a1p0` | 8 | 10 | 1.00 | 0.361269 | 0.475610 | 0.475610 |

## Top Dev Diagnostic Configs

| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `head5_tail64_a0p1` | 5 | 64 | 0.10 | 0.515727 | 0.558442 | 0.422078 |
| 2 | `head5_tail20_a0p05` | 5 | 20 | 0.05 | 0.515110 | 0.558442 | 0.428571 |
| 3 | `head5_tail20_a0p1` | 5 | 20 | 0.10 | 0.515110 | 0.558442 | 0.428571 |
| 4 | `head5_tail10_a0p0` | 5 | 10 | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 5 | `head5_tail15_a0p0` | 5 | 15 | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 6 | `head5_tail15_a0p05` | 5 | 15 | 0.05 | 0.514725 | 0.558442 | 0.422078 |
| 7 | `head5_tail20_a0p0` | 5 | 20 | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 8 | `head5_tail32_a0p0` | 5 | 32 | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 9 | `head5_tail64_a0p0` | 5 | 64 | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 10 | `head5_tail64_a0p05` | 5 | 64 | 0.05 | 0.514725 | 0.558442 | 0.422078 |

Artifacts: `round18/outputs/o_classifier/nN_residual_grid/nN_residual_grid_results.json`
