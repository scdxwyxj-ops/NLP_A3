# Round10 Neural Baseline Report

Date: 2026-05-01

## Summary

Round10 added neural baselines.

Best neural result:

```txt
DistilRoBERTa concat classifier over Fusion GBDT top10
Accuracy = 0.5260
Macro-F1 = 0.4543
```

This beats the previous TF-IDF classifier baseline:

```txt
Accuracy = 0.4675
Macro-F1 = 0.4376
```

The neural verifier learned useful pair-level signal, but it is not a good final
top3 evidence selector yet.

## Comparison

| System | Evidence F | REFUTES Recall | Accuracy | Macro-F1 | Harmonic Mean |
|---|---:|---:|---:|---:|---:|
| Round09 blend alpha0.4 | 0.2105 | 0.1420 | 0.4481 | 0.4142 | 0.2865 |
| Round08 TF-IDF classifier baseline | 0.2011 | 0.0864 | 0.4675 | 0.4376 | 0.2812 |
| DistilRoBERTa concat classifier | 0.2011 | n/a | 0.5260 | 0.4543 | 0.2910 |
| DistilRoBERTa verifier top3 | 0.1029 | 0.0340 | 0.4545 | n/a | 0.1678 |
| Hybrid gamma0.2 top3 | 0.2038 | 0.1235 | 0.4416 | n/a | 0.2789 |
| Hybrid gamma0.6 top20 | 0.1206 | 0.4568 | 0.4416 | n/a | 0.1895 |

## Decision

Use separate components:

- final evidence: Round09 blend alpha0.4;
- claim label: DistilRoBERTa concat classifier;
- verifier: keep as diagnostic / REFUTES context supplement, not final top3 reranker.

