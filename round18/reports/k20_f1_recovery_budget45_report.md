# k20 F1 Recovery Audit

Goal: test whether a top20-context classifier can be tuned to exceed the top5 shallow dev macro-F1 without dev-selected hyperparameters.

## References

- Promoted k20 logistic: macro-F1 `0.471928`, accuracy `0.519481`.
- Top5 shallow dev-best: macro-F1 `0.495421`, accuracy `0.545455`.

## Train-Holdout Selected k20 Result

- selected config: `w1x2_mf30000_shallow_c1p0`
- train-holdout macro-F1: `0.391364`
- dev macro-F1: `0.456483`
- dev accuracy: `0.512987`
- top class share: `0.448052`
- collapse: `passed`

## Dev-Best Diagnostic k20 Result

- best dev config: `w1x2_mf30000_shallow_c2p0`
- dev macro-F1: `0.466094`
- dev accuracy: `0.519481`
- top class share: `0.435065`

## Top Dev Configs

| rank | config | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: |
| 1 | `w1x2_mf30000_shallow_c2p0` | 0.466094 | 0.519481 | 0.435065 |
| 2 | `w1x2_mf60000_c4p0` | 0.465511 | 0.519481 | 0.428571 |
| 3 | `w1x2_mf60000_c2p0` | 0.461031 | 0.512987 | 0.422078 |
| 4 | `w1x2_mf60000_char3x5_mf20000_shallow_c2p0` | 0.460655 | 0.512987 | 0.441558 |
| 5 | `w1x2_mf60000_c8p0` | 0.459917 | 0.512987 | 0.422078 |
| 6 | `w1x2_mf60000_char3x5_mf30000_shallow_c2p0` | 0.458435 | 0.512987 | 0.441558 |
| 7 | `w1x2_mf60000_char3x5_mf20000_shallow_c1p0` | 0.456991 | 0.506494 | 0.448052 |
| 8 | `w1x2_mf30000_shallow_c1p0` | 0.456483 | 0.512987 | 0.448052 |
| 9 | `w1x2_mf60000_char3x5_mf20000_shallow_c4p0` | 0.453500 | 0.506494 | 0.428571 |
| 10 | `w1x2_mf30000_c8p0` | 0.451131 | 0.500000 | 0.402597 |

Artifacts: `round18/outputs/o_classifier/k20_f1_recovery_budget45/k20_f1_recovery_results.json`
