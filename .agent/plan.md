# Plan

## Mission

Create a Round06 comparison framework that tells us which evidence preprocessing
strategy should feed the final classifier.

## Current state

- BM25 top-5 dev retrieval F-score: `0.0772`.
- Zero-shot `cross-encoder/ms-marco-MiniLM-L6-v2` top-3 dev retrieval F-score: `0.1642`.
- Single-logit BCE fine-tuned MiniLM top-5 dev retrieval F-score: `0.1044`.
- BM25 top-50 dev macro recall: `0.3249`.
- BM25 top-100 dev macro recall: `0.4188`.
- Main observed error: topic-relevant passages are often not fact-checking-relevant evidence.

## Target state

Round06 ends with:

- A reproducible experiment matrix.
- A final comparison table.
- Per-method TP/FP/FN/TN and label-level analysis.
- A recommendation for classifier evidence context.
- A handoff to classifier implementation.

## Constraints

- Use only train/dev labels for development and evaluation.
- Do not inspect test labels; test is unlabelled.
- Avoid closed-source APIs and proprietary models.
- Keep generated artifacts out of git.
- Avoid large brittle rule systems.

## Strategy

1. Establish a table schema before running more experiments.
2. Mine hard negatives from model-ranked non-gold evidence.
3. Fine-tune reranker variants conservatively.
4. Evaluate both final-output metrics and classifier-input metrics.
5. Prototype semantic extraction as auxiliary features, not prediction rules.
6. Use the table to decide what Round07 classifier should consume.

## Milestones

1. Stage A: table schema and baseline row consolidation.
2. Stage B: MiniLM-mined task-aware negatives.
3. Stage C: task-aware reranker comparison.
4. Stage D: semantic feature extraction prototype.
5. Stage E: classifier-oriented evidence packages.
6. Stage F: final Round06 analysis and recommendation.

## Verification strategy

- Unit tests and compile checks for code changes.
- `eval.py` for every prediction JSON.
- Recall/precision/TP/FP/FN/TN analysis for every method.
- Label-level analysis, especially REFUTES.
- Manual spot-check of representative examples in generated markdown.

## Risks

- Fine-tuning may overfit and underperform zero-shot MiniLM.
- Candidate recall may remain the bottleneck.
- Semantic extraction may add complexity without measurable gain.
- Classifier metrics may conflict with final evidence F-score.

## Non-goals

- Build the final classifier before evidence packaging is stable.
- Create a hand-written rule classifier.
- Submit or tune on test data.
