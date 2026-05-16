# k20 Rank-Residual Classifier Audit

Architecture: main branch is claim + top5 evidence; residual branch is ranks 6-20 scaled by tail_alpha.
When tail_alpha=0, the architecture degenerates to a top5-style classifier.

- Reference top5 shallow dev-best macro-F1: `0.495421`.

## Train-Holdout Selected Result

- selected config: `head5_mf60000_c4p0_tail1p0`
- train-holdout macro-F1: `0.383086`
- dev macro-F1: `0.441610`
- dev accuracy: `0.512987`
- tail_alpha: `1.0`

## Dev-Best Diagnostic Result

- best dev config: `head5_mf60000_c2p0_tail0p05`
- dev macro-F1: `0.515110`
- dev accuracy: `0.558442`
- tail_alpha: `0.05`

## Top Dev Configs

| rank | config | tail_alpha | macro-F1 | accuracy | top class share |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `head5_mf60000_c2p0_tail0p05` | 0.05 | 0.515110 | 0.558442 | 0.428571 |
| 2 | `head5_mf60000_c2p0_tail0p1` | 0.10 | 0.515110 | 0.558442 | 0.428571 |
| 3 | `head5_mf60000_c2p0_tail0p0` | 0.00 | 0.514725 | 0.558442 | 0.422078 |
| 4 | `head5_mf30000_c2p0_tail0p05` | 0.05 | 0.507091 | 0.558442 | 0.435065 |
| 5 | `head5_mf30000_c2p0_tail0p1` | 0.10 | 0.507091 | 0.558442 | 0.435065 |
| 6 | `head5_mf30000_c2p0_tail0p0` | 0.00 | 0.502394 | 0.551948 | 0.428571 |
| 7 | `head5_mf60000_c4p0_tail0p0` | 0.00 | 0.501089 | 0.545455 | 0.467532 |
| 8 | `head5_mf60000_c4p0_tail0p05` | 0.05 | 0.501089 | 0.545455 | 0.467532 |
| 9 | `head5_mf30000_c1p0_tail0p5` | 0.50 | 0.496216 | 0.571429 | 0.487013 |
| 10 | `head5_mf60000_c1p0_tail0p25` | 0.25 | 0.495257 | 0.551948 | 0.454545 |

Artifacts: `round18/outputs/o_classifier/k20_rank_residual/k20_rank_residual_results.json`
