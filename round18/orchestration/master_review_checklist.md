# Round18 Master Review Checklist

Use this checklist whenever a Spark worker returns a result.

## 1. Scope Check

- Did the worker edit only its assigned write scope?
- Did the worker avoid reverting unrelated user or agent changes?
- Are all generated outputs under `round18/outputs/<pool>/<agent>/`?
- Are scripts/configs under the assigned `round18/experiments/...` folder if implementation was required?

## 2. Input Check

Reject or mark diagnostic-only if strict output reads:

- `outputs/round*`
- `models/round*`
- old predictions
- old feature tables
- teacher outputs
- pseudo labels
- saved checkpoints
- stale notebook outputs
- Google Drive artifacts without provenance
- test labels or test-derived design feedback

## 3. Manifest Check

Every run must provide a `run_manifest.json` with:

- run id;
- mode;
- status;
- command;
- git commit;
- git status;
- input files with hashes;
- output files;
- random seed / CV seed when applicable;
- labels-used flags;
- forbidden-input scan;
- metrics;
- runtime;
- split-isolation summary;
- leakage and reproducibility risk.

Missing fields require `NEEDS INVESTIGATION`.

## 4. Split Check

- Train labels may be used for training and train-only CV.
- Dev labels may be used for frozen evaluation and error analysis.
- Dev labels must not be used for greedy final strict selection.
- Test claims must not influence design decisions.

If dev was used for hyperparameter search, label the result `diagnostic-only`.

## 5. Method Check

For sparse tasks:

- no dense/neural models if the lane claims sparse-only;
- no old sparse candidate artifacts;
- top-k candidate schema includes claim id, evidence id, rank, score, source/method.

For dense tasks:

- dense scoring must occur only inside sparse candidates;
- no full-corpus dense search in final path;
- model must be loaded by open-source runtime name, not local checkpoint.

For aggregator tasks:

- leaf-flat inputs only;
- no nested historical selector ranks;
- no dev-teacher distillation.

For classifier tasks:

- context must come from same frozen ranking;
- no old prediction/context/checkpoint dependency.

## 6. Reproducibility Check

- Can the command be rerun from repo root?
- Are seeds fixed where randomness exists?
- Is the output directory safe to delete and rebuild?
- Are dependency assumptions documented?
- Are runtime and memory constraints recorded?

## 7. Verdict

Use exactly one:

- `APPROVED AS STRICT-CANDIDATE`
- `APPROVED AS DIAGNOSTIC-ONLY`
- `REJECTED`
- `NEEDS INVESTIGATION`

Include minimal remediation for any non-approved result.

