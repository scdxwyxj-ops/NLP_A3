# Score Distribution Audit

Method: paired bootstrap on the fixed dev set. This estimates sampling noise across dev claims, not a full train-randomness distribution.

## Point Scores And Bootstrap Intervals

| model | point macro-F1 | bootstrap mean | 95% interval | point accuracy |
| --- | ---: | ---: | ---: | ---: |
| Evidence-Aware Logistic | 0.471928 | 0.468291 | [0.379870, 0.557417] | 0.519481 |
| Top5 Shallow Candidate | 0.495421 | 0.492184 | [0.409174, 0.579978] | 0.545455 |
| Global Residual Diagnostic | 0.515110 | 0.511601 | [0.423725, 0.602302] | 0.558442 |
| Top5-Gated Residual Diagnostic | 0.500454 | 0.496703 | [0.406459, 0.587142] | 0.538961 |
| Top5-Gated Train-Selected | 0.467626 | 0.463699 | [0.370622, 0.551820] | 0.512987 |

## Paired Delta Versus Top5 Shallow Candidate

| model delta | mean delta | 95% interval | P(delta > 0) |
| --- | ---: | ---: | ---: |
| Evidence-Aware Logistic minus Top5 Shallow Candidate | -0.023893 | [-0.112834, 0.063577] | 0.294 |
| Global Residual Diagnostic minus Top5 Shallow Candidate | 0.019417 | [-0.024579, 0.067117] | 0.798 |
| Top5-Gated Residual Diagnostic minus Top5 Shallow Candidate | 0.004519 | [-0.071037, 0.079323] | 0.551 |
| Top5-Gated Train-Selected minus Top5 Shallow Candidate | -0.028485 | [-0.111213, 0.054283] | 0.252 |

## Interpretation

- If a paired-delta interval crosses zero, the apparent dev improvement is weak evidence and may be sampling noise.
- This audit does not rescue dev-selected methods; train-only selection stability remains required for promotion.

Figure: `round18/reports/figures/score_distribution_audit.png`
Results: `round18/outputs/o_classifier/score_distribution_audit/score_distribution_audit_results.json`
