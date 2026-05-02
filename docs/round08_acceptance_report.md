# Round08 Acceptance Report

Date: 2026-05-01

## Summary

Round08 completed the requested acceptance outputs:

- error layer analysis;
- feature table construction;
- lightweight feature fusion reranker;
- updated classifier context baseline.

The best Round08 evidence result is:

```txt
Fusion GBDT top3 evidence F-score = 0.2011
Round07 MiniLM-only top3 evidence F-score = 0.1987
```

The best classifier-context result is:

```txt
Fusion GBDT top50 + TF-IDF logistic regression
Accuracy = 0.4675
Macro-F1 = 0.4376
Harmonic mean = 0.2812
```

## Main Tables

### Error Layer Analysis

| Layer | Count | Claim % | Gold Evidence % | Main Fix |
|---|---:|---:|---:|---|
| gold not in top500 | 180 | 0.5909 | 0.3666 | dense supplement / candidate retrieval |
| gold in top500 but not top50 | 58 | 0.3052 | 0.1181 | score more candidates / fusion reranker |
| gold in top50 but not top20 | 52 | 0.2468 | 0.1059 | feature fusion reranker |
| gold in top20 but not top3 | 108 | 0.4610 | 0.2200 | final evidence selection |
| coverage top500 | 311 | 0.8896 | 0.6334 | diagnostic |
| coverage top50 | 253 | 0.8052 | 0.5153 | diagnostic |
| coverage top20 | 201 | 0.7143 | 0.4094 | diagnostic |
| coverage top3 | 93 | 0.4545 | 0.1894 | diagnostic |

### Fusion Reranker

| Method | Output k | Evidence F-score | Macro Recall | Hit-any | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| MiniLM only | 3 | 0.1987 | 0.2331 | 0.4545 | 0.1790 |
| Fusion GBDT | 3 | 0.2011 | 0.2321 | 0.4545 | 0.0864 |
| MiniLM only | 20 | 0.1106 | 0.4540 | 0.7143 | 0.3951 |
| Fusion GBDT | 20 | 0.1153 | 0.4669 | 0.7403 | 0.4136 |

### Classifier Context

| Context | Semantic | Accuracy | Macro-F1 | Evidence F-score | Harmonic Mean |
|---|---|---:|---:|---:|---:|
| Round07 MiniLM top50 | no | 0.4610 | 0.4242 | 0.1987 | 0.2778 |
| Fusion GBDT top50 | no | 0.4675 | 0.4376 | 0.2011 | 0.2812 |
| Fusion GBDT top50 | yes | 0.4351 | 0.3895 | 0.2011 | 0.2751 |

## Decision

Use Fusion GBDT top50 as the next classifier-context candidate.

Do not claim that this solves REFUTES: top3 REFUTES recall dropped in this first fusion version.

Next work should score a wider MiniLM candidate scope, such as top100/top200, then rerun fusion.

