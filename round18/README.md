# Round18 Isolated Workspace

This directory is the isolated working area for Round18 orchestration.

Round18 goal:

```text
raw course JSON
-> strict sparse-only candidate gate
-> dense/neural scoring only inside sparse candidates
-> leaf-flat evidence aggregator
-> top3 evidence and topN classifier context
-> claim classifier
-> final predictions
```

## Isolation Rule

New Round18 planning, manifests, scratch outputs, exploratory outputs, and audit notes should live under `round18/` unless the master explicitly approves a reusable code patch elsewhere.

Default write roots:

- `round18/contracts/`
- `round18/orchestration/`
- `round18/subagents/`
- `round18/manifests/`
- `round18/scratch/`
- `round18/outputs/`
- `round18/reports/`

Existing project directories such as `experiments/`, `src/`, `notebooks/`, `outputs/`, and `submissions/` are not Round18 scratch space. They may be touched only after a scoped task contract names the exact files and the master approves the change.

## Status Labels

- `strict-candidate`: generated from allowed inputs under the strict contract and eligible for master review.
- `diagnostic-only`: useful evidence, but not eligible for final pipeline selection.
- `rejected`: violates the Round18 contract or lacks enough provenance.

Exploratory results are never final by default. A selected candidate must be rebuilt through the frozen strict path from raw course JSON.

