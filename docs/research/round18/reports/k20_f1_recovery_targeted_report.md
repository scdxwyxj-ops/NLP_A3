# k20 F1 Recovery Audit

Goal: test whether a top20-context classifier can be tuned to exceed the top5 shallow dev macro-F1 without dev-selected hyperparameters.

## References

- Promoted k20 logistic: macro-F1 `0.471928`, accuracy `0.519481`.
- Top5 shallow dev-best: macro-F1 `0.495421`, accuracy `0.545455`.

## Train-Holdout Selected k20 Result

- selected config: `w1x2_mf60000_shallow_c1p0`
- train-holdout macro-F1: `0.388176`
- dev macro-F1: `0.447846`
- dev accuracy: `0.512987`
- top class share: `0.441558`
- collapse: `passed`

## Dev-Best Diagnostic k20 Result

- best dev config: `w1x2_mf60000_char3x5_mf30000_shallow_c2p0`
- dev macro-F1: `0.470732`
- dev accuracy: `0.525974`
- top class share: `0.441558`

## Top Dev Configs

| rank | config | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: |
| 1 | `w1x2_mf60000_char3x5_mf30000_shallow_c2p0` | 0.470732 | 0.525974 | 0.441558 |
| 2 | `w1x2_mf60000_c8p0` | 0.468461 | 0.519481 | 0.428571 |
| 3 | `w1x2_mf60000_char3x5_mf20000_shallow_c1p0` | 0.466962 | 0.519481 | 0.454545 |
| 4 | `w1x2_mf60000_char3x5_mf30000_shallow_c1p0` | 0.466962 | 0.519481 | 0.454545 |
| 5 | `w1x2_mf30000_shallow_c2p0` | 0.466466 | 0.519481 | 0.435065 |
| 6 | `w1x2_mf60000_c2p0` | 0.461973 | 0.512987 | 0.422078 |
| 7 | `w1x2_mf60000_char3x5_mf30000_shallow_c4p0` | 0.461065 | 0.506494 | 0.454545 |
| 8 | `w1x2_mf60000_char3x5_mf20000_shallow_c4p0` | 0.460271 | 0.506494 | 0.448052 |
| 9 | `w1x2_mf60000_c4p0` | 0.459583 | 0.512987 | 0.435065 |
| 10 | `w1x2_mf60000_char3x5_mf20000_shallow_c2p0` | 0.459349 | 0.512987 | 0.435065 |

Artifacts: `round18/outputs/o_classifier/k20_f1_recovery_targeted/k20_f1_recovery_results.json`
