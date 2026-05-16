# Round09 Acceptance Report

Date: 2026-05-01

## Summary

Round09 tested alpha blending, top100 MiniLM scoring scope, gain/loss analysis,
REFUTES-specific calibration, and classifier context impact.

Best final evidence result:

```txt
top100 REFUTES x2 Fusion GBDT + alpha blend
alpha_minilm = 0.4
top3 evidence F-score = 0.2105
```

Baseline comparison:

| Method | top3 F-score | REFUTES Recall |
|---|---:|---:|
| Round07 MiniLM-only | 0.1987 | 0.1790 |
| Round08 Fusion GBDT | 0.2011 | 0.0864 |
| Round09 blend alpha0.4 | 0.2105 | 0.1420 |

The blend improves overall evidence F-score but still does not fully recover
MiniLM's REFUTES recall.

## Classifier Context

| Context | Accuracy | Macro-F1 | Evidence F-score | Harmonic Mean |
|---|---:|---:|---:|---:|
| Round07 MiniLM top50 | 0.4610 | 0.4242 | 0.1987 | 0.2778 |
| Round08 Fusion GBDT top50 | 0.4675 | 0.4376 | 0.2011 | 0.2812 |
| Round09 blend top100 x2 alpha0.4 | 0.4481 | 0.4142 | 0.2105 | 0.2865 |

Recommendation:

- final evidence: use Round09 blend alpha0.4;
- classifier macro-F1: keep Round08 Fusion GBDT top50;
- assignment harmonic mean: Round09 blend alpha0.4 is currently best.

