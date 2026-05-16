# Reproduce First Meeting Top500 Recall

Date: 2026-05-09 Australia/Sydney

## Question

The first meeting report says the best top500 candidate recall was about `0.65`. Can we reproduce it?

Source report:

```text
group_meetings/first_meeting/first_meeting_report.pdf
group_meetings/first_meeting/first_meeting_report.md
```

The markdown source states:

```text
BM25 top500 macro recall:          0.5861
Char TF-IDF top500 macro recall:   0.6610
RRF top500 macro recall:           0.6579
```

So the `0.65` number is not BM25 alone. It refers to:

- `tfidf_char_top500`: `0.6610`
- `RRF(BM25 top500 + Char TF-IDF top500)`: `0.6579`

## Reproduction Command

```bash
PYTHONPATH=src:. python experiments/retrieval/evaluate_candidate_recall.py \
  --claims data/dev-claims.json \
  --evidence data/evidence.json \
  --output-dir round18/reports/reproduce_first_meeting_top500 \
  --top-k-values 500 \
  --pool-top-k 500 \
  --max-features 200000 \
  --char-max-features 300000 \
  --k1 1.5 \
  --b 0.75
```

This writes only under:

```text
round18/reports/reproduce_first_meeting_top500/
```

## Result

Reproduced exactly to the reported rounded values.

| method | macro recall@500 | micro recall@500 | hit-any@500 | all-gold@500 |
|---|---:|---:|---:|---:|
| `bm25_top500` | `0.5861471861471862` | `0.5560081466395111` | `0.8376623376623377` | `0.3246753246753247` |
| `tfidf_char_top500` | `0.6610389610389611` | `0.6334012219959266` | `0.8896103896103896` | `0.4025974025974026` |
| `rrf_bm25_char_tfidf_top500` | `0.6579004329004329` | `0.6334012219959266` | `0.8896103896103896` | `0.4090909090909091` |
| `rrf_bm25_word_char_tfidf_top500` | `0.6478354978354979` | `0.6252545824847251` | `0.8766233766233766` | `0.38311688311688313` |

## Interpretation

The first meeting `0.65` result is reproducible from raw course JSON using the Round07 script.

Important distinction:

- First meeting / Round07 used `BM25 + char TF-IDF` sparse RRF.
- Current Round18 sparse wave used `O-S1 BM25 + O-S2 structured routing + O-S3 feedback expansion`, not the original full char TF-IDF lane.
- That is why current Round18 `O-S4 strict_rrf_prf_heavy` top500 macro recall is `0.6195887445887446`, below the reproduced Round07 `0.6579004329004329`.

## Consequence For Round18

Round18 should add a strict char TF-IDF lane before freezing the sparse gate.

Recommended next task:

```text
O-S6 strict char TF-IDF lane
```

Inputs:

- `data/dev-claims.json`
- `data/train-claims.json`
- `data/evidence.json`

Output:

- `round18/outputs/o_sparse/o_s6_char_tfidf/`

Then rerun O-S4 fusion with:

- O-S1 BM25
- O-S6 char TF-IDF
- optional O-S2 structured routing
- optional O-S3 feedback expansion

Expected strict dev target:

- at least reproduce `rrf_bm25_char_tfidf_top500 = 0.6579004329004329`.

