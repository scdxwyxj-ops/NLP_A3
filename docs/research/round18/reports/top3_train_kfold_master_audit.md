# Top3 Train-KFold Master Audit

## Scope

This audit checks whether the final evidence top-3 ranking should change after the top-64 context experiments were revised.

Top-64 and top-3 are separate operating points:

- Top-64 is the classifier context coverage target.
- Top-3 is the submitted evidence ranking target.

## Subagent Review

Reviewer Spark found that the old tutorial top-3 section mixed context-selection artifacts with top-3 submission artifacts. It also found that the strongest old top-3 blends were dev-selected diagnostics, so they should not be promoted as strict final evidence chains.

Validator Spark implemented a separate strict-kfold top-3 script with hand-feature and embedding components:

- Script: `round18/experiments/o_rerank/run_top3_strict_fusion_kfold.py`
- Main output: `round18/outputs/o_rerank/top3_strict_fusion_kfold/top3_strict_fusion_kfold_summary.json`

Its strict result selected an embedding-plus-prior variant, but the run recorded a CE train-pool coverage risk because the default CE train pool path was dev-only. I keep this result as a useful validator, not the promoted chain.

## Master Rerun

I implemented and ran a stricter same-family fusion experiment:

- Script: `round18/experiments/o_rerank/run_top3_train_kfold_fusion.py`
- Output: `round18/reports/top3_train_kfold_fusion_with_hand/top3_train_kfold_fusion_with_hand_summary.json`
- Selection basis: train-claim k-fold.
- Dev usage: confirmation only.
- Primary selection metric: evidence F@3.

Inputs were split-aligned:

- train sparse/CE/embedding/hand pools cover 1228 train claims and 0 dev claims.
- dev sparse/CE/embedding/hand pools cover 154 dev claims and 0 train claims.
- best dev candidate output covers 154 dev claims with 500 candidates each.
- forbidden input scan passed.

## Selected Top3 Fusion

Selected weights:

| component | weight |
|---|---:|
| CE score | 0.25 |
| embedding score | 0.50 |
| hand-feature score | 0.00 |
| source-rank prior | 0.00 |
| CE rank prior | 0.25 |
| embedding-rank prior | 0.00 |
| hand-rank prior | 0.00 |

Dev confirmation:

| method | dev evidence F@3 | dev macro R@3 | dev macro R@10 | dev macro R@64 |
|---|---:|---:|---:|---:|
| CE only | 0.1987 | 0.2331 | 0.3693 | 0.5637 |
| Embedding only | 0.1616 | 0.1883 | 0.3293 | 0.5649 |
| Hand score only | 0.1364 | 0.1555 | 0.2624 | 0.4927 |
| Train-selected CE+embedding fusion | 0.2047 | 0.2387 | 0.3959 | 0.5816 |

## Decision

Promote the train-selected CE+embedding fusion as the current top-3 evidence-ranking candidate. It improves the submitted evidence front over CE alone while preserving the same split discipline.

Do not claim that hand features improve top-3 in the final selected model. In this strict grid, hand-feature components were available but received zero weight. The right story is narrower:

- Hand/shallow features help top-64 context selection and help embedding in the top-64 analysis.
- For top-3 submission ranking, the strict train-selected gain comes from combining CE score, embedding score, and CE rank.

## Tutorial Update

Updated:

- `group_meetings/second_meeting /tutorial.ipynb`
- `round18/experiments/o_tutorial/build_second_meeting_tutorial.py`

Updated figures:

- `group_meetings/second_meeting /figures/round18_revised_top3_gate_cn.png`
- `group_meetings/second_meeting /figures/round18_revised_top3_submission_rn.png`
- `group_meetings/second_meeting /figures/round18_revised_top3_submission_bar.png`

The old dev-selected diagnostic blend is no longer shown as the top-3 main conclusion.
