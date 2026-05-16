# Project Structure

```txt
assets/                 Shared static assets and versioned experiment configs
data/                   Local course data and generated outputs; ignored by Git
docs/                   Meeting notes, course spec mirror, and research archives
model/                  Local checkpoints; ignored by Git
src/a3_factcheck/       Reusable pipeline code
src/experiments/        Runnable experiment scripts and historical development code
src/tests/              Lightweight tests for reusable code
src/tools/              Helper scripts, including the course evaluation script
submissions/            Final code submission zip, report project, and exact source files
.agent/                 Local agent state and project memory; ignored by Git
```

## Code Boundaries

- `src/a3_factcheck/` should contain reusable modules: data loading, retrieval, classification, evaluation helpers, and prediction writing.
- `src/experiments/` should contain runnable scripts that combine modules, choose hyperparameters, write outputs, and print metrics.
- `data/outputs/` should be treated as generated artifacts. Keep important metrics in `docs/`.
- `data/` and `model/` should stay local and should not be included in the final submission zip unless course instructions change.
- `submissions/resource/` is the only source directory used to build `submissions/COMP90042_teamname_resource.zip`.
