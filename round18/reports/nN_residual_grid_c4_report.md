# n/N Residual Classifier Grid

Architecture: head branch uses claim + top-n evidence; residual branch uses ranks n+1..N evidence.
Both branches use separate TF-IDF word 1-2 vectorizers and separate shallow feature scalers.

## References

- Top5 shallow dev-best: macro-F1 `0.495421`, accuracy `0.545455`, top class share `0.376623`.
- k20 promoted dev-best: macro-F1 `0.471928`, accuracy `0.519481`, top class share `0.435065`.

## Train-Holdout Selected Result

- selected config: `head5_tail10_a1p0`
- train-holdout macro-F1: `0.409267`
- dev macro-F1: `0.395676`
- dev accuracy: `0.480519`
- tail alpha: `1.0`
- head/tail dims: `60000` / `60000` word features, `47` / `47` side features

## Best (n, N) By Train-Selected Dev Confirmation

- best cell: `n=3`, `N=15`
- selected config: `head3_tail15_a0p0`
- selected tail alpha: `0.0`
- dev macro-F1: `0.474811`

## Dev Diagnostic Best

- best config: `head5_tail10_a0p0`
- dev macro-F1: `0.501089`
- dev accuracy: `0.545455`
- tail alpha: `0.0`

## Best (n, N) By Dev Diagnostic Upper Bound

- best diagnostic cell: `n=5`, `N=10`
- diagnostic config: `head5_tail10_a0p0`
- diagnostic tail alpha: `0.0`
- dev macro-F1: `0.501089`
- diagnostic values are non-promoted.

## Top Train-Holdout Configs

| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `head5_tail10_a1p0` | 5 | 10 | 1.00 | 0.409267 | 0.491870 | 0.500000 |
| 2 | `head3_tail20_a1p0` | 3 | 20 | 1.00 | 0.386209 | 0.471545 | 0.434959 |
| 3 | `head5_tail20_a1p0` | 5 | 20 | 1.00 | 0.383086 | 0.455285 | 0.434959 |
| 4 | `head10_tail64_a1p0` | 10 | 64 | 1.00 | 0.377265 | 0.475610 | 0.467480 |
| 5 | `head8_tail20_a1p0` | 8 | 20 | 1.00 | 0.372495 | 0.459350 | 0.434959 |
| 6 | `head5_tail10_a0p5` | 5 | 10 | 0.50 | 0.370873 | 0.451220 | 0.471545 |
| 7 | `head3_tail10_a1p0` | 3 | 10 | 1.00 | 0.366495 | 0.463415 | 0.479675 |
| 8 | `head8_tail15_a1p0` | 8 | 15 | 1.00 | 0.364452 | 0.475610 | 0.447154 |
| 9 | `head8_tail10_a1p0` | 8 | 10 | 1.00 | 0.363337 | 0.479675 | 0.471545 |
| 10 | `head10_tail20_a1p0` | 10 | 20 | 1.00 | 0.359756 | 0.443089 | 0.455285 |

## Top Dev Diagnostic Configs

| rank | config | n | N | tail alpha | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `head5_tail10_a0p0` | 5 | 10 | 0.00 | 0.501089 | 0.545455 | 0.467532 |
| 2 | `head5_tail10_a0p05` | 5 | 10 | 0.05 | 0.501089 | 0.545455 | 0.467532 |
| 3 | `head5_tail15_a0p0` | 5 | 15 | 0.00 | 0.501089 | 0.545455 | 0.467532 |
| 4 | `head5_tail15_a0p05` | 5 | 15 | 0.05 | 0.501089 | 0.545455 | 0.467532 |
| 5 | `head5_tail15_a0p1` | 5 | 15 | 0.10 | 0.501089 | 0.545455 | 0.467532 |
| 6 | `head5_tail20_a0p0` | 5 | 20 | 0.00 | 0.501089 | 0.545455 | 0.467532 |
| 7 | `head5_tail20_a0p05` | 5 | 20 | 0.05 | 0.501089 | 0.545455 | 0.467532 |
| 8 | `head5_tail32_a0p0` | 5 | 32 | 0.00 | 0.501089 | 0.545455 | 0.467532 |
| 9 | `head5_tail32_a0p05` | 5 | 32 | 0.05 | 0.501089 | 0.545455 | 0.467532 |
| 10 | `head5_tail32_a0p1` | 5 | 32 | 0.10 | 0.501089 | 0.545455 | 0.467532 |

Artifacts: `round18/outputs/o_classifier/nN_residual_grid_c4/nN_residual_grid_results.json`
