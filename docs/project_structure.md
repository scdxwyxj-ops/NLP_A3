# Project Structure

```txt
agent_docs/             Agent memory, round reports, decisions, handoff notes
configs/                Versioned experiment configuration files
data/                   Local course data; do not include in final resource zip
docs/                   Human-facing project documentation
experiments/            Runnable experiment scripts and notebooks-in-progress logic
notebooks/              Course notebook template and final notebook work
outputs/                Generated predictions, metrics, and experiment artifacts
src/a3_factcheck/       Reusable pipeline code
tests/                  Lightweight tests for reusable code
eval.py                 Course evaluation script
README.md               Course/project specification mirror
```

## Code Boundaries

- `src/a3_factcheck/` should contain reusable modules: data loading, retrieval, classification, evaluation helpers, and prediction writing.
- `experiments/` should contain runnable scripts that combine modules, choose hyperparameters, write outputs, and print metrics.
- `outputs/` should be treated as generated artifacts. Keep important metrics in `agent_docs/` or `docs/`.
- `data/` should stay local and should not be included in the final submission zip unless course instructions change.

