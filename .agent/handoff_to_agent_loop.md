# Handoff to Agent Loop

## Mission

Implement Round06 in bounded stages so the project ends with a detailed
comparison table and a recommendation for classifier evidence input.

## Constraints

- Do not edit `data/*.json`.
- Do not commit `outputs/`, `models/`, caches, or checkpoints.
- Do not inspect or infer test labels.
- Use open-source models only.
- Keep final evidence output and classifier input context as separate concepts.

## Current plan

1. Define and maintain the Round06 comparison table schema.
2. Mine task-aware hard negatives using zero-shot MiniLM-ranked non-gold evidence.
3. Fine-tune and evaluate single-logit reranker variants.
4. Prototype semantic feature extraction as auxiliary input.
5. Build classifier-ready evidence packages.
6. Write final Round06 analysis.

## Top 3 recommended first tasks

### Task 1

* Title: Mine task-aware hard negatives
* Objective: create train JSONL with gold positives and MiniLM-ranked non-gold hard negatives.
* Allowed paths: `src/a3_factcheck/rerank/`, `experiments/rerank/`, `configs/rerank/`, `agent_docs/rounds/round_06/`
* Forbidden paths: `data/*.json`, test label inspection, committed outputs
* Success criteria: JSONL rows include `claim_id`, `evidence_id`, text, label, source, BM25 rank/score, MiniLM rank/score.
* Verification command: `PYTHONPATH=src pytest -q`
* Risk level: medium
* Stop conditions: MiniLM inference unavailable or candidate pool missing.

### Task 2

* Title: Evaluate task-aware reranker variants
* Objective: train/evaluate conservative MiniLM BCE variants and fill comparison rows.
* Allowed paths: `experiments/rerank/`, `outputs/round06/`, `agent_docs/rounds/round_06/`
* Forbidden paths: test labels, closed-source APIs
* Success criteria: rows include F-score, precision, recall, hit-any, REFUTES recall, and interpretation.
* Verification command: `PYTHONPATH=src python -m compileall src experiments`
* Risk level: medium
* Stop conditions: if fine-tuning underperforms zero-shot, record underperformance and continue to analysis.

### Task 3

* Title: Build semantic feature prototype
* Objective: extract lightweight semantic cues for claim/evidence pairs without making rule-based predictions.
* Allowed paths: `src/a3_factcheck/semantic/`, `experiments/semantic/`, `agent_docs/rounds/round_06/`
* Forbidden paths: hand-written if-then classifier, external closed-source parsers
* Success criteria: feature JSONL and representative examples show entities, quantities, negation, comparison, and relation cues.
* Verification command: `PYTHONPATH=src pytest -q`
* Risk level: medium
* Stop conditions: if extraction is too noisy, mark as analysis-only.

## Tasks requiring human review

- Whether to prefer higher classifier recall over final evidence precision when outputs diverge.
- Whether semantic features should be implemented in final system or only discussed in report.
- Which classifier architecture to prioritize after Round06.

## Suggested command

PYTHONPATH=src pytest -q
