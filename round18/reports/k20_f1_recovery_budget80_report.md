# k20 F1 Recovery Audit

Goal: test whether a top20-context classifier can be tuned to exceed the top5 shallow dev macro-F1 without dev-selected hyperparameters.

## References

- Promoted k20 logistic: macro-F1 `0.471928`, accuracy `0.519481`.
- Top5 shallow dev-best: macro-F1 `0.495421`, accuracy `0.545455`.

## Train-Holdout Selected k20 Result

- selected config: `w1x2_mf60000_shallow_c1p0`
- train-holdout macro-F1: `0.387640`
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
| 2 | `w1x2_mf60000_char3x5_mf20000_shallow_c1p0` | 0.466962 | 0.519481 | 0.454545 |
| 3 | `w1x2_mf60000_char3x5_mf30000_shallow_c1p0` | 0.466962 | 0.519481 | 0.454545 |
| 4 | `w1x2_mf60000_c2p0` | 0.466278 | 0.519481 | 0.428571 |
| 5 | `w1x2_mf60000_c4p0` | 0.464994 | 0.519481 | 0.428571 |
| 6 | `w1x2_mf60000_char3x5_mf20000_shallow_c2p0` | 0.464597 | 0.519481 | 0.441558 |
| 7 | `w1x2_mf60000_char3x5_mf30000_shallow_c4p0` | 0.460271 | 0.506494 | 0.448052 |
| 8 | `w1x2_mf60000_c8p0` | 0.456183 | 0.506494 | 0.428571 |
| 9 | `w1x2_mf30000_shallow_c1p0` | 0.455647 | 0.512987 | 0.448052 |
| 10 | `w1x2_mf30000_c4p0` | 0.453282 | 0.506494 | 0.428571 |

Artifacts: `round18/outputs/o_classifier/k20_f1_recovery_budget80/k20_f1_recovery_results.json`
