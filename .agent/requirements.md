# Requirements

## Functional requirements

- Generate task-aware hard negatives from high-ranked non-gold evidence.
- Train or evaluate at least one task-aware reranker variant against Round05 baselines.
- Produce classifier-ready evidence packages with separate fields for:
  - final evidence output candidates
  - wider classifier input context
  - optional semantic feature summaries
- Produce a comparison table covering retrieval quality, recall, and error modes.
- Preserve the current assignment output format for `eval.py`.

## Non-functional requirements

- Keep all generated outputs under `outputs/`.
- Keep model checkpoints under `models/`.
- Do not commit course JSON data, generated predictions, checkpoints, or caches.
- Keep scripts runnable with `PYTHONPATH=src`.
- Prefer small, bounded experiments that can run on free Colab or the current local GPU.

## Engineering requirements

- Put reusable code under `src/a3_factcheck/`.
- Put experiment entry points under `experiments/`.
- Keep Round06 documentation under `agent_docs/rounds/round_06/`.
- Use structured JSON/CSV/JSONL outputs for every experiment result.
- Make the final comparison table reproducible from saved per-method result files.

## Testing requirements

- Run `PYTHONPATH=src pytest -q`.
- Run `PYTHONPATH=src python -m compileall src experiments`.
- For each experiment script, provide a minimal command in docs.
- For any generated prediction file, verify with `eval.py` when labels exist.

## Documentation requirements

- Record every stable result in `agent_docs/rounds/round_06/`.
- Maintain a table schema before running experiments.
- Include interpretation, not only raw metrics.
- Explicitly state whether a method is intended for final evidence output or classifier input.
- Record failure modes, especially topic-relevant false positives and REFUTES misses.

## Out of scope

- Manual inspection or use of test labels.
- Closed-source APIs or proprietary model services.
- Large hand-written if-then classification rules.
- Optimizing only retrieval F-score while ignoring classifier-readiness.

## Human decisions required

- Whether to prioritize final leaderboard retrieval output or classifier accuracy if they diverge.
- How much runtime is acceptable for classifier input packaging.
- Whether semantic features should be included in the final report as an implemented feature or as analysis only.
