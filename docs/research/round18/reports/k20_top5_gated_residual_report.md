# k20 Top5-Gated Residual Classifier Audit

Architecture: `final_logits = head_logits + sigmoid(top5_features @ w) * (tail_logits - head_logits)`.
Gate features are derived from top5 only, so the tail branch cannot decide by looking at its own text.

- Reference top5 shallow dev-best macro-F1: `0.495421`.
- Reference global residual dev-best macro-F1: `0.515110`.

## Train-Holdout Selected Result

- selected config: `gated_head5_mf60000_c2p0_l20p0_uncertainty_plus_shallow`
- train-holdout macro-F1: `0.406293`
- dev macro-F1: `0.467626`
- dev accuracy: `0.512987`
- gate mean/std: `0.5460` / `0.4974`

## Dev-Best Diagnostic Result

- best dev config: `gated_head5_mf30000_c2p0_l20p0_uncertainty`
- dev macro-F1: `0.500454`
- dev accuracy: `0.538961`
- gate mean/std: `0.5211` / `0.3834`

## Top Dev Configs

| rank | config | macro-F1 | accuracy | top class share | gate mean | gate std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gated_head5_mf30000_c2p0_l20p0_uncertainty` | 0.500454 | 0.538961 | 0.428571 | 0.5211 | 0.3834 |
| 2 | `gated_head5_mf60000_c2p0_l210p0_uncertainty` | 0.499452 | 0.551948 | 0.474026 | 0.3236 | 0.0000 |
| 3 | `gated_head5_mf60000_c2p0_l210p0_uncertainty_plus_shallow` | 0.499452 | 0.551948 | 0.474026 | 0.3236 | 0.0003 |
| 4 | `gated_head5_mf60000_c2p0_l20p1_uncertainty` | 0.495408 | 0.545455 | 0.467532 | 0.3270 | 0.0031 |
| 5 | `gated_head5_mf60000_c2p0_l20p1_uncertainty_plus_shallow` | 0.495408 | 0.545455 | 0.467532 | 0.3312 | 0.0220 |
| 6 | `gated_head5_mf60000_c2p0_l21p0_uncertainty` | 0.495408 | 0.545455 | 0.467532 | 0.3239 | 0.0003 |
| 7 | `gated_head5_mf60000_c2p0_l21p0_uncertainty_plus_shallow` | 0.495408 | 0.545455 | 0.467532 | 0.3244 | 0.0031 |
| 8 | `gated_head5_mf30000_c1p0_l20p1_uncertainty_plus_shallow` | 0.490420 | 0.545455 | 0.461039 | 0.3556 | 0.0287 |
| 9 | `gated_head5_mf30000_c2p0_l20p1_uncertainty` | 0.489603 | 0.538961 | 0.467532 | 0.3266 | 0.0036 |
| 10 | `gated_head5_mf30000_c2p0_l20p1_uncertainty_plus_shallow` | 0.489603 | 0.538961 | 0.467532 | 0.3315 | 0.0197 |

Artifacts: `round18/outputs/o_classifier/k20_top5_gated_residual/k20_top5_gated_residual_results.json`
