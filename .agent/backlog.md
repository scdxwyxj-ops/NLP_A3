# Backlog

Each item must be small, bounded, and verifiable.

## P0 - Define Round06 comparison table schema

* Type: documentation / analysis contract
* Objective: create the final table columns before implementing more experiments.
* Rationale: prevents optimizing isolated metrics and keeps work classifier-oriented.
* Evidence: Round05 showed final F-score and classifier recall needs diverge.
* Allowed paths: `agent_docs/rounds/round_06/`, `docs/`
* Forbidden paths: `data/*.json`, `models/`, `outputs/`
* Success criteria: table schema includes method, candidate source, reranker, negatives, semantic features, top-k output, classifier context top-k, F-score, recall, precision, hit-any, REFUTES recall, classifier status, and interpretation.
* Verification command: `git diff -- agent_docs/rounds/round_06 docs`
* Risk level: low
* Dependencies: Round05 metrics
* Stop conditions: if required metric cannot be computed from current outputs, mark it as planned rather than inventing a value.
* Expected output: Round06 report with table schema and stage map.
* Suitable for agent-loop-orchestrator: no

## P0 - Mine MiniLM task-aware hard negatives

* Type: implementation / data generation
* Objective: create JSONL training pairs where negatives are high-ranked zero-shot MiniLM non-gold candidates.
* Rationale: targets topic-relevant false positives observed in Round05.
* Evidence: `claim-375` and error analysis show topical FP dominates misses.
* Allowed paths: `src/a3_factcheck/rerank/`, `experiments/rerank/`, `configs/rerank/`, `agent_docs/rounds/round_06/`
* Forbidden paths: `data/*.json` edits, `test-claims-unlabelled.json` manual inspection
* Success criteria: output JSONL contains positives, mined negatives, source ranks/scores, and stats.
* Verification command: `PYTHONPATH=src pytest -q`
* Risk level: medium
* Dependencies: BM25 top-50/top-100 pool and zero-shot MiniLM reranker
* Stop conditions: if model inference fails or outputs are not reproducible.
* Expected output: `outputs/round06/*task-aware-negatives*.jsonl` plus stats.
* Suitable for agent-loop-orchestrator: yes

## P0 - Evaluate task-aware reranker variants

* Type: experiment
* Objective: compare zero-shot MiniLM, Round05 BCE, and task-aware BCE variants.
* Rationale: Round05 fine-tuning improved BM25 but did not beat zero-shot MiniLM.
* Evidence: Round05 Report E.
* Allowed paths: `experiments/rerank/`, `outputs/round06/`, `agent_docs/rounds/round_06/`
* Forbidden paths: test labels, closed-source APIs
* Success criteria: comparison rows for at least three methods with top-3/top-5 F-score and top-10/top-20 recall.
* Verification command: `PYTHONPATH=src python -m compileall src experiments`
* Risk level: medium
* Dependencies: task-aware negative dataset
* Stop conditions: if results underperform, record the result rather than discarding it.
* Expected output: model result rows and interpretation.
* Suitable for agent-loop-orchestrator: yes

## P1 - Prototype semantic feature extraction

* Type: implementation / analysis
* Objective: extract entities, quantities, negation, comparison, and relation cues from claim/evidence text.
* Rationale: user identified that subject/object and semantic constraints may help distinguish topical FP from useful evidence.
* Evidence: Round05 topic relevance vs evidence relevance discussion.
* Allowed paths: `src/a3_factcheck/semantic/`, `experiments/semantic/`, `agent_docs/rounds/round_06/`
* Forbidden paths: hand-written prediction rules for final label
* Success criteria: feature JSONL exists for sampled claim-evidence pairs and can be joined with reranker outputs.
* Verification command: `PYTHONPATH=src pytest -q`
* Risk level: medium
* Dependencies: selected retrieval outputs
* Stop conditions: if extraction is too noisy, keep it as analysis-only.
* Expected output: semantic feature files and examples.
* Suitable for agent-loop-orchestrator: yes

## P1 - Build classifier-oriented evidence packages

* Type: dataset preparation
* Objective: generate train/dev inputs for classifier with top-k evidence contexts and optional semantic summaries.
* Rationale: final goal is claim classification, not retrieval alone.
* Evidence: user explicitly wants final classification to be best.
* Allowed paths: `src/a3_factcheck/classification/`, `experiments/classification/`, `outputs/round06/`, `agent_docs/rounds/round_06/`
* Forbidden paths: test labels, large committed outputs
* Success criteria: packages for top-3/top-10/top-20/top-50 exist and are documented.
* Verification command: `PYTHONPATH=src python -m compileall src experiments`
* Risk level: medium
* Dependencies: stable reranker outputs
* Stop conditions: if evidence context becomes too long for target classifier, record truncation strategy.
* Expected output: classifier-ready JSONL and ablation plan.
* Suitable for agent-loop-orchestrator: yes

## P2 - Final Round06 analysis report

* Type: documentation / synthesis
* Objective: write the final comparison table and recommendation.
* Rationale: user needs a detailed analysis to decide next work.
* Evidence: all Round06 experiments.
* Allowed paths: `agent_docs/rounds/round_06/`, `docs/`
* Forbidden paths: source code unless fixing documentation commands
* Success criteria: final report names recommended classifier input configuration and next implementation task.
* Verification command: `git diff -- agent_docs/rounds/round_06 docs`
* Risk level: low
* Dependencies: Stage A-E outputs
* Stop conditions: if metrics conflict, include tradeoff rather than hiding it.
* Expected output: final Round06 report.
* Suitable for agent-loop-orchestrator: no
