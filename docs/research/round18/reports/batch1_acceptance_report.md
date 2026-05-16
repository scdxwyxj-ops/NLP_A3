# Round18 Batch 0/1 Acceptance Report

Date: 2026-05-09 Australia/Sydney

## Scope

Executed the first Round18 setup batch:

- isolated `round18/` workspace;
- execution contract;
- run manifest schema;
- sub-agent contract;
- dynamic milestones and Batch 2 task templates;
- strict-mode guard;
- raw JSON smoke run;
- repository-wide suspicious-token audit.

No existing `src/`, `experiments/`, `notebooks/`, `outputs/`, or `submissions/` files were edited by this batch.

## Files Added

- `round18/README.md`
- `round18/.gitignore`
- `round18/contracts/round18_execution_contract.md`
- `round18/manifests/run_manifest_schema.md`
- `round18/orchestration/spark_dynamic_plan.md`
- `round18/orchestration/dynamic_milestones_v2.md`
- `round18/subagents/subagent_contract.md`
- `round18/subagents/batch2_task_templates.yaml`
- `round18/tools/*.py`
- `round18/tests/test_tools.py`
- `round18/outputs/batch1_smoke/raw_json_summary.json`
- `round18/manifests/batch1_smoke/run_manifest.json`
- `round18/reports/batch1/static_audit.json`
- `round18/reports/batch1/static_audit_summary.json`

## Commands Run

```bash
PYTHONPATH=. python -m round18.tools.strict_guard --allow-empty
PYTHONPATH=. python -m round18.tools.raw_json_smoke --data-dir data --output-dir round18/outputs/batch1_smoke --manifest round18/manifests/batch1_smoke/run_manifest.json
PYTHONPATH=. python -m round18.tools.static_audit --root . --out round18/reports/batch1/static_audit.json
PYTHONPATH=. python -m round18.tools.summarize_static_audit --audit round18/reports/batch1/static_audit.json --out round18/reports/batch1/static_audit_summary.json
PYTHONPATH=. pytest -q round18/tests
PYTHONPATH=. python -m compileall -q round18/tools
PYTHONPATH=. python -m round18.tools.strict_guard round18/manifests/batch1_smoke/run_manifest.json
```

## Verification

- Strict guard empty scan: passed.
- Raw JSON smoke run: passed.
- Smoke manifest strict guard scan: passed.
- Round18 tool tests: `4 passed`.
- `round18/tools` compile check: passed.

Raw JSON summary:

- train claims: `1228`
- dev claims: `154`
- test-unlabelled claims: `153`
- evidence items: `1208827`

## Audit Findings

Repository-wide suspicious-token audit found:

- files with hits: `249`
- total hits: `1714`
- most common tokens:
  - `outputs/round`: `1275`
  - `checkpoint`: `223`
  - `teacher`: `66`
  - `pseudo`: `62`
  - `label_source`: `57`

Interpretation:

These hits are expected because the repository contains many Round15-Round17 scripts, reports, checkpoints, and diagnostic outputs. They are not automatically violations, but they confirm that strict Round18 runs need a hard guard and isolated write namespace.

High-risk historical areas:

- `outputs/round*`
- `models/round*`
- `experiments/rerank/round16_*`
- `experiments/rerank/round17_*`
- Round16 notebooks and submission artifacts

## Current Verdict

`APPROVED FOR BATCH 2 TASK ASSIGNMENT`

This approval is limited to assigning isolated Spark optimization tasks under `round18/outputs/...`.

It does not approve any final pipeline result yet. Final approval still requires:

- strict candidate manifests;
- train-only selection evidence;
- frozen strict rebuild from raw JSON;
- notebook execution path audit;
- clean rebuild and cache deletion tests.

