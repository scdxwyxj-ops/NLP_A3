# Top64 CE Rerank Cutoff-64 Classifier/Context Check (Round18)

## Scope

Evaluate the context and classifier implications of using **top-500 sparse pools re-ranked by CE to top-64** (the `strict_ce64_sparse_backfill` pattern).

### Scripts inspected

- `round18/experiments/o_dense/o_d3b_cross_encoder_s7/run_o_d3b_cross_encoder_s7.py`
- `round18/experiments/o_classifier/o_c3_strict_context/run_o_c3_strict_context.py`

## Feasibility check

`run_o_c3_strict_context.py` enforces matching family IDs between `--train-pool` and `--dev-pool`.
For the CE-rerank candidate families in Round18, no matching train-side strict artifacts are present (all are dev-only), and CE rerank/aggregate scripts are hardcoded to `data/dev-claims.json`.

So **strict split-compatible O-C3 could not be run without adding train artifacts**.

I still ran a bounded strict-family comparison by setting both train and dev claims to the same dev split to validate script-level behavior and measure candidate-context effects within one family.

## Runs executed

```bash
python round18/experiments/o_classifier/o_c3_strict_context/run_o_c3_strict_context.py \
  --train-claims data/dev-claims.json --dev-claims data/dev-claims.json \
  --train-pool round18/outputs/o_aggregate/o_a2x_s8_ce_s8_diag/dev_full_dev_o_a2x_s8_ce_s8_diag_strict_sparse_only_top500_candidates.json \
  --dev-pool   round18/outputs/o_aggregate/o_a2x_s8_ce_s8_diag/dev_full_dev_o_a2x_s8_ce_s8_diag_strict_sparse_only_top500_candidates.json \
  --run-id top64_context_sparse_only_diagcheck \
  --manifest round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_manifest.json \
  --record round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_record.json

python round18/experiments/o_classifier/o_c3_strict_context/run_o_c3_strict_context.py \
  --train-claims data/dev-claims.json --dev-claims data/dev-claims.json \
  --train-pool round18/outputs/o_aggregate/o_a2x_s8_ce_s8_diag/dev_full_dev_o_a2x_s8_ce_s8_diag_strict_ce64_sparse_backfill_top500_candidates.json \
  --dev-pool   round18/outputs/o_aggregate/o_a2x_s8_ce_s8_diag/dev_full_dev_o_a2x_s8_ce_s8_diag_strict_ce64_sparse_backfill_top500_candidates.json \
  --run-id top64_context_ce64_backfill_diagcheck \
  --manifest round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_manifest.json \
  --record round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_record.json
```

Both runs used defaults (`train_context_k=20`, `dev_context_k=20`, `selection-mode=train_holdout`).

## Context/evidence recall@64

From corresponding candidate artifacts used above:

- `o_a2x_s8_ce_s8_diag_strict_sparse_only`:
  - `macro_recall@64 = 0.4927489`
  - `hit_any@64 = 0.7597403`
  - `macro_recall@10 = 0.2624459`
- `o_a2x_s8_ce_s8_diag_strict_ce64_sparse_backfill`:
  - `macro_recall@64 = 0.4927489`
  - `hit_any@64 = 0.7597403`
  - `macro_recall@10 = 0.3636364`

Implication: **top-64 evidence coverage is unchanged**, but CE reranking improves early-rank overlap (better top-10 recall).

### Top-64 set overlap check

- Per-claim top-64 evidence ID overlap between sparse-only and CE-backfill sets is **64/64 always**.
- Order changes materially: average top-20 overlap is ~**0.493** with average top-1 same only ~**0.208**.

## Classifier metrics

Metrics are read from:
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_record.json`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_record.json`

Best model by macro-F1 in each run:

| Candidate set | Selected model | accuracy | macro-F1 | harmonic (`harmonic_mean_F_A`) |
|---|---|---:|---:|---:|
| Sparse-only | `tfidf_logreg` | `0.9740` | `0.9733` | `0.1645` |
| CE64 backfill | `tfidf_logreg` | `0.9091` | `0.9059` | `0.1930` |

Alternative model in the CE64 run (`linear_svm`) improved from `acc=0.5584 / macro-F1=0.3129` (sparse-only) to `acc=0.7013 / macro-F1=0.5366` (CE64), showing the CE rerank changes context representation enough to affect model behavior.

## Operational implication

- Using CE top64 rerank at this stage does **not** increase `recall@64` of the evidence pool (coverage same set), but it does reshuffle to better-order evidence within that set.
- Classifier performance is therefore sensitive to context order/content rather than simply candidate count.
- The above numbers are a bounded proxy: they are not train/dev split-clean because no train-side CE candidate family artifacts exist.

## Blocker and exact next commands

To run a true split-clean strict-context eval (train 1228 / dev 154) with this CE family, unblock by generating matching train artifacts for the same family and then rerun O-C3 with those.

`run_o_c3_strict_context.py` command pattern:

```bash
python round18/experiments/o_classifier/o_c3_strict_context/run_o_c3_strict_context.py \
  --train-claims data/train-claims.json \
  --dev-claims data/dev-claims.json \
  --train-pool <train-family>_strict_ce64_sparse_backfill_top500_candidates.json \
  --dev-pool round18/outputs/o_aggregate/o_a2x_s8_ce_s8_diag/dev_full_dev_o_a2x_s8_ce_s8_diag_strict_ce64_sparse_backfill_top500_candidates.json \
  --train-context-k 20 --dev-context-k 20 \
  --manifest ...
  --record ...
```

Blocking script-level constraints:
- `o_a2c_plain_leaf_rank_s8` enforces `data/dev-claims.json` as claims input.
- `o_d3b_cross_encoder_s7` enforces `data/dev-claims.json` as claims input and current-run strict sparse-pool only.

Missing now: `<train-family>_...` path because corresponding train CE candidate artifacts are not present in `round18/outputs`.

## Files changed in this task

### New report
- `round18/reports/top64_context_classifier_spark.md`

### O-C3 strict context run artifacts (new)
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_manifest.json`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_record.json`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_dev_context_top20.jsonl`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_train_context_top20.jsonl`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_tfidf_logreg_*`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_sparse_only_linear_svm_*`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_manifest.json`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_record.json`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_dev_context_top20.jsonl`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_train_context_top20.jsonl`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_tfidf_logreg_*`
- `round18/outputs/o_classifier/o_c3_strict_context/top64_context_ce64_backfill_linear_svm_*`
