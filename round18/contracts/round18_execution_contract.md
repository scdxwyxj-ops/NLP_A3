# Round18 Execution Contract

## Allowed Inputs For Strict Mode

Strict-mode code may read:

- course claim JSON files:
  - `train-claims.json`
  - `dev-claims.json`
  - `test-claims-unlabelled.json` only in final prediction mode
- `evidence.json`
- source code and config files that are part of the current Round18 run
- open-source pretrained model weights downloaded by model name at runtime
- current-run artifacts under `round18/` with valid manifests

## Forbidden Inputs For Strict Mode

Strict-mode code must not read:

- `outputs/round*` historical ranked outputs
- old predictions
- old feature tables
- old teacher outputs
- old pseudo labels
- saved model checkpoints
- Google Drive artifacts without explicit provenance
- stale notebook cell outputs as data
- any test labels or manually inspected test-derived information

## Split Policy

- Train labels may be used for training and train-only cross-validation.
- Dev labels may be used for frozen final evaluation and error analysis.
- Dev labels must not be used for greedy feature/model/hyperparameter selection in strict results.
- Test claims must not be inspected for design decisions.
- Any dev-tuned result must be labeled `diagnostic-only`.

## Strict Mode Guard

Every strict run must fail fast if an input path or config references suspicious tokens such as:

```text
outputs/round
teacher
pseudo
checkpoint
best_model
label_source
drive.mount
old_prediction
cached_prediction
round15_selector_rank
claim_key_selector_rank
```

The guard is a contract requirement before final rebuild.

## Dynamic Milestone Rule

Spark agents may propose new methods beyond the initial plan. A new method can enter the strict candidate track only if it passes:

1. README compliance.
2. Allowed input check.
3. Free Colab feasibility estimate.
4. Reproducibility plan from raw JSON.
5. Train-only selection plan.
6. Clear output schema and manifest.

