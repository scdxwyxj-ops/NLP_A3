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

The diagnostic should be read as two separate tables. The first table reports
cumulative coverage at each cutoff.

| Cutoff | Covered Gold Evidence | Gold Evidence Coverage | Claim Hit-any |
|---|---:|---:|---:|
| top500 | 311 / 491 | 0.6334 | 0.8896 |
| top50 | 253 / 491 | 0.5153 | 0.8052 |
| top20 | 201 / 491 | 0.4094 | 0.7143 |
| top3 | 93 / 491 | 0.1894 | 0.4545 |

The second table reports the loss between adjacent cutoffs.

| Loss Layer | Lost Gold Evidence | Lost Gold Evidence % | Affected Claims | Affected Claim % | Main Fix |
|---|---:|---:|---:|---:|---|
| not in top500 | 180 | 0.3666 | 91 | 0.5909 | dense supplement / candidate retrieval |
| top500 -> top50 | 58 | 0.1181 | 47 | 0.3052 | score more candidates / fusion reranker |
| top50 -> top20 | 52 | 0.1059 | 38 | 0.2468 | feature fusion reranker |
| top20 -> top3 | 108 | 0.2200 | 71 | 0.4610 | final evidence selection |

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
