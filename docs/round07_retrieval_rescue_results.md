# Round07 Retrieval Rescue Results

Date: 1 May 2026

## Summary

Round07 changed the retrieval strategy from "tune the reranker on BM25 top-50"
to "first improve candidate recall, then rerank a better candidate pool".

This helped immediately. The previous best dev evidence F-score was:

```txt
BM25 top-50 -> zero-shot MiniLM -> top-3
F-score = 0.1642
```

The current best Round07 result is:

```txt
RRF(BM25 top-500, char TF-IDF top-500)
-> zero-shot MiniLM
-> top-3 evidence

F-score = 0.1987
```

## Candidate Recall

| Candidate Source | Retained k | Macro Recall | Hit-any | All-gold | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| BM25 | 100 | 0.4188 | 0.6753 | 0.2013 | 0.2716 |
| BM25 | 500 | 0.5861 | 0.8377 | 0.3247 | 0.5123 |
| char TF-IDF | 500 | 0.6610 | 0.8896 | 0.4026 | 0.5556 |
| RRF(BM25, char TF-IDF) | 500 | 0.6579 | 0.8896 | 0.4091 | 0.5926 |

The main gain is that the correct evidence appears in the candidate pool more
often before reranking.

## Reranking Results

| Method | Output k | Evidence F-score | Macro Recall | REFUTES Recall | Hit-any |
|---|---:|---:|---:|---:|---:|
| old best: BM25 top50 + MiniLM | 3 | 0.1642 | 0.1877 | 0.0494 | 0.3961 |
| BM25 top500 + MiniLM | 3 | 0.1945 | 0.2286 | 0.1605 | 0.4416 |
| char TF-IDF top500 + MiniLM | 3 | 0.1920 | 0.2276 | 0.1605 | 0.4481 |
| RRF(BM25, char TF-IDF) top500 + MiniLM | 3 | 0.1987 | 0.2331 | 0.1790 | 0.4545 |

## Classifier Context

For final evidence output, top-3 is still better because it keeps precision
higher.

For the classifier, top-20 from the same reranked pool is more useful:

```txt
RRF(BM25, char TF-IDF) top500 + MiniLM top20
macro recall = 0.4540
hit-any      = 0.7143
```

This supports the two-branch design:

```txt
final evidence output:
  top-3

classifier context:
  top-20 or top-50
```

## Current Decision

Use this as the current default retrieval pipeline:

```txt
RRF(BM25 top500, char TF-IDF top500)
-> zero-shot cross-encoder/ms-marco-MiniLM-L6-v2
```

Then use:

- top-3 for final evidence output;
- top-20 first for classifier input;
- top-50 as a wider classifier-context ablation.

## Query-Boost Ablation

I also tested a salient-query BM25 supplement that repeats numeric expressions,
percentages, climate abbreviations, and entity-like fragments before BM25
scoring.

| Candidate Source | Retained k | Macro Recall | Hit-any | All-gold | REFUTES Recall |
|---|---:|---:|---:|---:|---:|
| RRF(BM25, char TF-IDF) | 500 | 0.6579 | 0.8896 | 0.4091 | 0.5926 |
| salient-boost BM25 | 500 | 0.5495 | 0.8117 | 0.2727 | 0.4012 |
| RRF(current best, salient-boost BM25) | 500 | 0.6419 | 0.8701 | 0.3701 | 0.6111 |

This is not an overall improvement. It slightly improves REFUTES candidate
recall in one variant, but it hurts macro recall and all-gold rate. After
MiniLM reranking, the query-boost supplement reached top-3 evidence F-score
`0.1927`, below the current default `0.1987`.

Decision: keep query-boost as a documented ablation, not the default pipeline.

## First Classifier Baseline

I also tested a fast classifier baseline using TF-IDF logistic regression over
the claim plus retrieved evidence context.

| Classifier | Context k | Semantic Summary | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|
| TF-IDF logistic regression | 20 | no | 0.4610 | 0.4162 |
| TF-IDF logistic regression | 20 | yes | 0.4481 | 0.4013 |
| TF-IDF logistic regression | 50 | no | 0.4610 | 0.4242 |
| TF-IDF logistic regression | 50 | yes | 0.4545 | 0.4096 |

This confirms that retrieval is improving, but the classifier still needs a
stronger design. The next classifier should probably be evidence-wise rather
than one large bag-of-words concatenation.
