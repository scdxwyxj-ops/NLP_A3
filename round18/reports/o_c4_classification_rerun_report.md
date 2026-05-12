# O-C4 Classification Rerun Report

## Purpose

Rerun classification from the current Round18 retrieval/rerank logic instead of reusing old O-C3 outputs.

Old classifier results are kept only as risk context: previous SVM/logreg experiments showed class collapse, especially excessive `SUPPORTS` predictions. This rerun therefore enforces split-clean train/dev context, prediction histograms, per-class recall, and acceptance gates.

## Context Construction

Main context method:

- `CE + factual shallow features`
- Source script: `round18/experiments/o_rerank/run_ce_shallow_feature_kfold.py`
- Output directory: `round18/outputs/o_rerank/o_ce_factual_context_for_classifier/`
- Selection basis: train k-fold, primary metric `macro_recall@64`
- Selected variant: `ce_plus_factual`
- Context candidate metrics: `cv@64=0.515075`, `dev@64=0.567100`

Materialized strict train/dev candidate files:

- `round18/outputs/o_rerank/o_ce_factual_context_for_classifier/o_ce_factual_context_best_train_top500_candidates.json`
- `round18/outputs/o_rerank/o_ce_factual_context_for_classifier/o_ce_factual_context_best_dev_top500_candidates.json`

Alias paths were created only so O-C3/O-C4 strict family parsing sees the same `o_ce_factual_context` family for train and dev:

- `round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/train_full_train_o_ce_factual_context_top500_candidates.json`
- `round18/outputs/o_rerank/o_ce_factual_context_for_classifier_alias/dev_full_dev_o_ce_factual_context_top500_candidates.json`

## Runs

All classifier runs use:

- Script: `round18/experiments/o_classifier/o_c3_strict_context/run_o_c3_strict_context.py`
- Mode: `STRICT`
- Selection mode: `train_holdout`
- Model: TF-IDF + Logistic Regression unless otherwise noted
- Collapse threshold: `0.70`
- Minimum macro-F1 gate: `0.3585558388`

## Results

`assignment_f` and `harmonic` in this table are context-run diagnostics produced by the classifier script when it writes a complete prediction file. They are not the promoted final evidence score, because the final submission evidence uses the separate top3 ranker. Classification promotion is therefore based on claim accuracy, macro-F1, non-collapse, and strict train/dev context parity.

| run | context | model | accuracy | macro-F1 | assignment_f | harmonic | acceptance |
|---|---:|---|---:|---:|---:|---:|---|
| `o_c4_ce_only_context_k20_trainholdout` | CE-only k20 | logreg | 0.4740 | 0.4169 | 0.1106 | 0.1793 | pass |
| `o_c4_ce_factual_context_k5_trainholdout` | CE+factual k5 | logreg | 0.4351 | 0.4047 | 0.1896 | 0.2641 | pass |
| `o_c4_ce_factual_context_k20_trainholdout` | CE+factual k20 | logreg | 0.5195 | 0.4719 | 0.1142 | 0.1873 | pass |
| `o_c4_ce_factual_context_k64_trainholdout` | CE+factual k64 | logreg | 0.4286 | 0.3797 | 0.0500 | 0.0896 | pass |
| `o_c4_top3_fusion_context_k20_trainholdout` | top3-fusion k20 | logreg | 0.4545 | 0.4057 | 0.1189 | 0.1885 | pass |
| `o_c4_ce_factual_context_k5_trainholdout` | CE+factual k5 | linear SVM | 0.4221 | 0.2116 | 0.1896 | 0.2616 | fail |

## Main Promotion

Promote `o_c4_ce_factual_context_k20_trainholdout` as the current classification result.

It improves over the strict CE-only context at the same context depth:

- accuracy: `0.4740 -> 0.5195`
- macro-F1: `0.4169 -> 0.4719`
- all class recalls remain non-zero
- prediction distribution does not collapse

Prediction histogram for the promoted run:

| label | predictions | recall |
|---|---:|---:|
| SUPPORTS | 67 | 0.6029 |
| REFUTES | 29 | 0.4074 |
| NOT_ENOUGH_INFO | 43 | 0.5366 |
| DISPUTED | 15 | 0.3333 |

Confusion matrix:

```text
gold\pred,SUPPORTS,REFUTES,NOT_ENOUGH_INFO,DISPUTED
SUPPORTS,41,11,11,5
REFUTES,8,11,6,2
NOT_ENOUGH_INFO,13,4,22,2
DISPUTED,5,3,4,6
```

## Collapse Finding

The SVM branch still reproduces the old failure mode:

- prediction histogram: `SUPPORTS=131`, `REFUTES=3`, `NOT_ENOUGH_INFO=20`, `DISPUTED=0`
- top class share: `0.8506`
- acceptance failures:
  - `one_or_more_classes_have_zero_predictions`
  - `one_or_more_classes_have_zero_recall`
  - `max_class_share_above_threshold`
  - `macro_f1_below_baseline`

This confirms that the collapse gate is necessary and that SVM should not be promoted.

## Interpretation

Top64 evidence coverage is useful for retrieval analysis, but classification does not benefit from dumping all 64 evidence texts into the TF-IDF classifier. The best new strict classifier uses k20 context. The k64 context loses both classifier quality and assignment score, likely because extra evidence text adds noise.

Top3 fusion is a strong evidence-submission ranker, but it is not the best classifier context here. It is kept as a secondary comparison, not the promoted classification context. This keeps the system split cleanly: top3 ranking optimizes final evidence order, while k20 context optimizes claim classification.

## Promoted Files

- Metrics: `round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_metrics.json`
- Predictions: `round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_dev_predictions.json`
- Confusion matrix: `round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_confusion_matrix.csv`
- Manifest: `round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/run_manifest.json`
