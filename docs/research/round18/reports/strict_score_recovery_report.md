# Round18 Strict Score Recovery Report

## Executive Summary

This recovery wave implemented the strict plan as a new isolated Round18 line:

- `O-S9`: plain leaf union/wide gate over BM25, char, structured, and PRF pools.
- `O-S10`: train-only hand-feature compressor over explicit wide train/dev pools.
- `O-C3`: strict classifier context recovery with explicit train/dev pools, no hidden fallback, train-side selection, and collapse gates.

The strongest strict classifier result is:

| run | context | accuracy | macro-F1 | assignment_f | harmonic | gate |
|---|---:|---:|---:|---:|---:|---|
| old O-C2 logreg | top20 mixed-source | 0.4156 | 0.3586 | 0.0899 | 0.1478 | diagnostic-only |
| O-C3 logreg | top5 strict BM25+char O-S9 | 0.4545 | 0.3969 | 0.1093 | 0.1762 | pass |

The strongest strict evidence compression result is:

| run | macro@3 | macro@64 | macro@100 | macro@500 |
|---|---:|---:|---:|---:|
| old O-S8 hand-feature top500 | 0.1555 | 0.4927 | 0.5323 | 0.6667 |
| O-S10 BM25+char round-robin top1000 | 0.1463 | 0.4876 | 0.5327 | 0.7005 |

Interpretation: the new strict classifier is materially better and no longer collapses. The new wide hand-feature compression improves top500 substantially, but still does not beat the old O-S8 at top64/top3.

## Implemented Changes

- Added `round18/experiments/o_sparse/o_s9_union_gate/run_o_s9_union_gate.py`.
  - Supports train/dev split defaults, per-source caps, `rrf`, `round_robin`, `priority`, and `union_upper`.
  - Preserves `source_breakdown`, `source_count`, source ranks, and source scores.
  - Writes candidate pools, metrics, comparison tables, manifest, and record.

- Added `round18/experiments/o_sparse/o_s10_wide_hand_feature/run_o_s10_wide_hand_feature.py`.
  - Generalizes old O-S8 from a fixed top500 input to explicit wide train/dev pools.
  - Uses train labels only; dev labels are evaluation-only.

- Added `round18/experiments/o_classifier/o_c3_strict_context/run_o_c3_strict_context.py`.
  - Requires explicit train/dev pools from the same inferred family unless diagnostic is enabled.
  - Disables hidden fallback retrieval.
  - Uses train holdout or fixed pre-registered config selection in strict mode; dev selection forces diagnostic.
  - Reports prediction histogram, per-class recall, collapse gate, acceptance gate, assignment_f, and harmonic.

## Retrieval Findings

O-S9 confirmed the earlier hypothesis:

| gate | macro@64 | macro@100 | macro@500 | macro@1000/2000 |
|---|---:|---:|---:|---:|
| BM25+char RRF top1000 | 0.4128 | 0.4637 | 0.6579 | 0.7232 @1000 |
| four-source RRF top2000 | 0.4253 | 0.4838 | 0.6460 | 0.7445 @2000 |
| four-source round-robin top2000 | 0.4083 | 0.4442 | 0.6465 | 0.7445 @2000 |

The full four-source union has a real top2000 ceiling (`macro@2000=0.7445`), but simple top500 truncation does not beat the existing strict top500. This validates the plan: union should feed a compressor, not replace ranking directly.

O-S10 compression results:

| run | macro@3 | macro@64 | macro@100 | macro@500 | status |
|---|---:|---:|---:|---:|---|
| S7 baseline input | 0.1541 | 0.4800 | 0.5285 | 0.6667 | rejected for top64 |
| BM25+char RRF top1000 | 0.1424 | 0.4860 | 0.5221 | 0.6922 | top500 gain |
| BM25+char round-robin top1000 | 0.1463 | 0.4876 | 0.5327 | 0.7005 | best top500 |
| BM25+char union-upper top1000 | 0.1466 | 0.4622 | 0.5274 | 0.6918 | rejected |
| round-robin train500 | 0.1542 | 0.4720 | 0.5279 | 0.6832 | rejected |
| round-robin positive-weight16 | 0.1256 | 0.4518 | 0.5184 | 0.6843 | rejected |

Promotion recommendation for evidence: promote `O-S10 BM25+char round-robin top1000` only when the target is top500/context coverage. Keep old O-S8 for top64/top3-sensitive reranking until a better compressor is found.

## Classifier Findings

O-C3 strict top5 is the best final classifier run:

| context-k | token budget | accuracy | macro-F1 | assignment_f | harmonic | acceptance |
|---:|---:|---:|---:|---:|---:|---|
| 3 | full | 0.4026 | 0.3360 | 0.0942 | 0.1527 | failed macro-F1 |
| 5 | full | 0.4545 | 0.3969 | 0.1093 | 0.1762 | passed |
| 5 | 80 | 0.4545 | 0.3969 | 0.1093 | 0.1762 | passed |
| 5 | 120 | 0.4545 | 0.3969 | 0.1093 | 0.1762 | passed |
| 10 | full | 0.4481 | 0.3954 | 0.0931 | 0.1542 | passed |
| 20 | full | 0.4286 | 0.3586 | 0.0731 | 0.1249 | passed |
| 50 | full | 0.4481 | 0.3838 | 0.0430 | 0.0785 | passed |

The top5 logreg prediction distribution is balanced enough:

| label | predictions | recall |
|---|---:|---:|
| SUPPORTS | 62 | 0.5588 |
| REFUTES | 33 | 0.3333 |
| NOT_ENOUGH_INFO | 48 | 0.4634 |
| DISPUTED | 11 | 0.2222 |

The SVM branch remains rejected. Its selected strict run predicted `SUPPORTS=127`, `NOT_ENOUGH_INFO=26`, `REFUTES=1`, `DISPUTED=0`, with top class share `0.8247`.

## Strictness And Validation

Validated manifests:

- `round18/outputs/o_sparse/o_s9_union_gate_bm25_char_train/run_manifest.json`
- `round18/outputs/o_sparse/o_s9_union_gate_bm25_char_dev/run_manifest.json`
- `round18/outputs/o_sparse/o_s9_union_gate_four_source_wide_dev/run_manifest.json`
- `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_top1000/run_manifest.json`
- `round18/outputs/o_classifier/o_c3_context_k5_logreg_bm25_char_rrf_top1000/run_manifest.json`

Tests:

- `pytest -q round18/tests` -> `13 passed`.
- `python -m py_compile` passed for O-S9, O-S10, and O-C3.

Independent reviewer conclusion:

- No hard technical blocker was found for the promoted strict artifacts.
- `O-C3` top5 logreg and `O-S10` BM25+char round-robin top1000 are defensible strict promotions.
- `O-S9` diagnostic-best files must remain non-promoted; downstream scripts should not auto-consume `*_diagnostic_best_*`.

Strict caveats:

- O-C3 top5 uses the O-S9 BM25+char strict family for both train and dev, with no fallback.
- Four-source O-S9 is dev-only because train structured/PRF pools do not currently exist; it is evidence diagnostic/context research, not a classifier training source.
- Dev diagnostic best-policy files are written separately and must not be promoted.

## Promotion Recommendation

Promote for final classifier evaluation:

- `round18/outputs/o_classifier/o_c3_context_k5_logreg_bm25_char_rrf_top1000/o_c3_context_k5_logreg_bm25_char_rrf_top1000_tfidf_logreg_mf60000_ngram1x2_c4p0_dev_predictions.json`

Promote for evidence/context coverage research:

- `round18/outputs/o_sparse/o_s10_wide_hand_feature_bm25_char_round_robin_top1000/dev_full_dev_o_s10_wide_hand_feature_bm25_char_round_robin_top1000_top500_candidates.json`

Reject as final:

- Existing O-C2, because train/dev context sources differ.
- O-C3 SVM, because the collapse gate fails.
- Simple four-source top500 union truncations, because top500 does not beat current strict baseline.
