# O-C5 Worker C: Fusion & Meta Learner
- baseline (A k5 TF-IDF + shallow macro-F1): `0.495421`
- strict alignment possible: `False`
- elapsed wall seconds: `32.22`

## A k5 selected config replay
- holdout macro-F1: `0.317170`
- dev macro-F1: `0.495421`
## claim_id alignment
- B holdout uses placeholder `val-*` rows: `True`
- B holdout placeholder mapping valid: `True`
- B holdout alignment: `failed_for_strict_claim_alignment`
- D holdout alignment missing count: `159`
- D dev alignment missing count: `0`
- B dev alignment missing count: `0`

## strict vs diagnostic
| mode | method | status | macro_f1 | accuracy | macro_recall | top_class_share | collapse | acceptance |
|---|---|---|---:|---:|---:|---:|---|---|
| strict | - | skipped | 0.000000 | 0.000000 | 0.000000 | 0.0000 | - | skipped |
| diagnostic | late_fusion | ok | 0.442963 | 0.500000 | 0.434511 | 0.4610 | n/a | n/a |

## collapse and per-class checks
Each result includes `prediction_histogram`, `per_class_recall`, and `confusion_matrix` in per-method metric file.

- Strict candidate achieved: no (not run)
