# Round18 Sub-Agent Contract

Every Round18 sub-agent must follow this contract.

## Default Rules

- Do not edit files outside the assigned write scope.
- Do not read forbidden inputs in strict mode.
- Do not inspect test claim contents for design decisions.
- Do not modify `eval.py`.
- Do not use historical `outputs/round*` artifacts for strict results.
- Do not use old predictions, teachers, pseudo labels, feature tables, or checkpoints for strict results.
- Do not call dense/neural methods sparse.
- Do not write into another sub-agent's namespace.

## Required Final Report

Each sub-agent must return:

1. patch or `no files changed`;
2. data-flow report;
3. files read;
4. files written;
5. commands run;
6. split-isolation explanation;
7. leakage-risk assessment;
8. reproducibility assessment;
9. exact metrics;
10. `run_manifest.json` path or reason none was produced;
11. final status recommendation: `strict-candidate`, `diagnostic-only`, or `rejected`.

## Write Scope Template

Each task assignment must specify:

```text
agent_name:
mode:
read scope:
write scope:
forbidden inputs:
allowed labels:
target metrics:
runtime budget:
expected manifest path:
exit criteria:
```

If any field is missing, the agent must ask the master for clarification before implementation.

