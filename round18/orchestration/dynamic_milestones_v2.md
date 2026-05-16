# Round18 Dynamic Milestones V2

This plan upgrades Round18 from a fixed BM25/BGE/HistGB path to a dynamic exploration program.

The master may run multiple GPT-5.3 Codex Spark sub-agents for research and optimization, but exploratory outputs never become final automatically. Final pipeline results must be rebuilt through the strict path from raw course JSON.

## Repository Environment Plan

Round18 has an isolated workspace:

```text
round18/
  contracts/
  orchestration/
  manifests/
  subagents/
  scratch/
  outputs/
  reports/
```

Default rule:

- Research notes, manifests, scratch outputs, and exploration outputs stay under `round18/`.
- Existing directories are not scratch space:
  - `experiments/`
  - `src/`
  - `notebooks/`
  - `outputs/`
  - `submissions/`
- Reusable code may be added outside `round18/` only after a scoped task contract names the exact files and the master approves the write scope.
- No sub-agent may write into another sub-agent's namespace.

## Batch 0: Contract And Guard

Purpose: prevent contamination before optimization begins.

Milestones:

- `B0.1` Freeze allowed-input whitelist.
- `B0.2` Freeze forbidden-input blacklist.
- `B0.3` Freeze strict vs diagnostic mode.
- `B0.4` Freeze split policy.
- `B0.5` Freeze artifact schemas.
- `B0.6` Freeze `run_manifest.json` schema.
- `B0.7` Define strict-mode guard and suspicious-token scan.

Exit criteria:

- No implementation pool starts before this is approved.
- Strict mode must reject historical `outputs/round*`, old predictions, feature tables, teachers, checkpoints, stale notebook outputs, and test-derived design feedback.

## Batch 1: Research Scout Pools

Purpose: deliberately search beyond the known local plan before committing to implementation.

Research scouts do not implement final code. They produce method matrices, feasibility filters, risks, and proposed dynamic milestones.

### R-Sparse: Sparse Candidate Gate Research

Explore beyond BM25 and char TF-IDF:

- Lucene/Pyserini BM25 parameterization.
- BM25F or field-aware sparse scoring.
- analyzer variants: word, char, n-gram, edge n-gram.
- rule-light query rewriting.
- RM3 / PRF lexical expansion.
- entity, number, date, unit aware sparse routing.
- boolean/range/term structured matching.
- learned sparse methods such as SPLADE, uniCOIL, DeepImpact only as high-risk candidates.
- doc2query/docTTTTTquery as diagnostic unless reproducibility and cost are acceptable.

Scout deliverables:

- method list;
- expected benefit;
- implementation cost;
- leakage/compliance risk;
- Colab cost;
- strict/diagnostic/rejected classification;
- proposed optimization agents and milestones.

### R-Dense: Dense-In-Sparse Research

Explore reranking leaves beyond BGE-small and MiniLM-L6:

- bi-encoders:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `intfloat/e5-small-v2`
  - `intfloat/e5-base-v2`
  - `BAAI/bge-base-en-v1.5`
  - `sentence-transformers/all-mpnet-base-v2` as diagnostic-first
- cross-encoders:
  - `cross-encoder/ms-marco-MiniLM-L12-v2`
  - `cross-encoder/ms-marco-electra-base` diagnostic-first
  - TinyBERT variants if reliably available
- late interaction:
  - ColBERT-family candidate-only rerank as diagnostic-first
- score normalization:
  - train-fold quantile clipping;
  - rank normalization;
  - temperature scaling for logits;
  - sparse/dense prior blending.
- scheduling:
  - bi-encoder over sparse candidates;
  - cross-encoder on bounded topM;
  - adaptive topM by sparse score ambiguity.

Hard guard:

- no full-corpus dense retrieval in final path.
- dense/neural scoring must happen only inside sparse candidates.

### R-Aggregate-Classifier: Aggregator And Classifier Research

Explore beyond HistGB and DistilRoBERTa:

- leaf-calibrated RRF + isotonic calibration.
- MMR diversity-aware top3 rerank.
- pairwise preference learner / tiny RankNet.
- Bayesian leaf reliability fusion.
- linear/logistic calibration.
- tree-based learning.
- factual features:
  - numbers and units;
  - dates and years;
  - entity anchor agreement;
  - polarity, negation, hedge cues;
  - cross-evidence contradiction signals;
  - leaf consistency and rank variance.
- classifiers:
  - evidence MIL classifier with sentence embeddings and attention pooling.
  - per-evidence compact NLI logits + learned aggregation.
  - compact Transformer alternatives such as ALBERT/BERT/DeBERTa-small variants.
  - claim decomposition and context routing.
  - adaptive context width only if fixed by train-only CV; otherwise diagnostic.

## Batch 2: Optimization Spark Pools

Purpose: implement small, isolated experiments after research and contract approval.

Every optimization Spark must receive:

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

### O-Sparse Pool

Use multiple Spark agents with disjoint namespaces.

Initial lanes:

- `O-S1`: BM25/Pyserini/Lucene parameter sweep and raw JSON index reproducibility.
- `O-S2`: field-aware, query decomposition, entity/number/date sparse routing.
- `O-S3`: PRF/RM3-lite and rule-light query rewriting.
- `O-S4`: sparse fusion, RRF, strict-first layered candidate selection.
- `O-S5`: learned sparse feasibility only if approved as diagnostic-first.

Write roots:

```text
round18/outputs/o_sparse/o_s1_bm25/
round18/outputs/o_sparse/o_s2_structured/
round18/outputs/o_sparse/o_s3_prf/
round18/outputs/o_sparse/o_s4_fusion/
round18/outputs/o_sparse/o_s5_learned_sparse/
```

Promotion requirements:

- raw JSON reproducible;
- train-only selection;
- no dense/neural source if claiming sparse-only;
- candidate pool schema-compliant;
- manifest complete.

### O-Dense Pool

Initial lanes:

- `O-D1`: light bi-encoder comparison inside sparse candidates.
- `O-D2`: stronger bi-encoder diagnostic-first comparison.
- `O-D3`: bounded cross-encoder topM rerank.
- `O-D4`: score normalization and adaptive topM scheduling.
- `O-D5`: late-interaction feasibility diagnostic.

Write roots:

```text
round18/outputs/o_dense/o_d1_light_biencoder/
round18/outputs/o_dense/o_d2_strong_biencoder/
round18/outputs/o_dense/o_d3_cross_encoder/
round18/outputs/o_dense/o_d4_normalization/
round18/outputs/o_dense/o_d5_late_interaction/
```

Promotion requirements:

- only score sparse candidate evidence IDs;
- no full-corpus dense final path;
- no saved checkpoint dependency;
- model loaded by open-source runtime name;
- runtime and memory manifest complete.

### O-Aggregate Pool

Initial lanes:

- `O-A1`: leaf feature schema and factual features.
- `O-A2`: calibrated RRF, MMR, and Bayesian leaf fusion.
- `O-A3`: linear/logistic and tree-based learners.
- `O-A4`: pairwise/tiny RankNet feasibility.
- `O-A5`: top3/top100 multi-objective selection.

Write roots:

```text
round18/outputs/o_aggregate/o_a1_features/
round18/outputs/o_aggregate/o_a2_calibrated_rank/
round18/outputs/o_aggregate/o_a3_learners/
round18/outputs/o_aggregate/o_a4_pairwise/
round18/outputs/o_aggregate/o_a5_multiobjective/
```

Promotion requirements:

- inputs come only from current-run sparse/dense leaves;
- train-only group folds;
- dev not used for feature/model selection;
- output is leaf-flat, not nested historical selectors.

### O-Classifier Pool

Initial lanes:

- `O-C1`: fixed context width train-CV selection.
- `O-C2`: claim-only and TF-IDF/logreg baselines.
- `O-C3`: compact Transformer classifier.
- `O-C4`: evidence MIL classifier.
- `O-C5`: compact per-evidence NLI aggregation.

Write roots:

```text
round18/outputs/o_classifier/o_c1_context_width/
round18/outputs/o_classifier/o_c2_baselines/
round18/outputs/o_classifier/o_c3_transformer/
round18/outputs/o_classifier/o_c4_mil/
round18/outputs/o_classifier/o_c5_nli/
```

Promotion requirements:

- context comes from same frozen aggregator ranking;
- train-only selection;
- no old context/prediction/checkpoint dependency;
- labels and evidence outputs share the same final pipeline provenance.

## Batch 3: Master Review And Candidate Freeze

Purpose: convert exploration into a small number of frozen candidates.

Milestones:

- `B3.1` Collect manifests and reports.
- `B3.2` Classify each result as `strict-candidate`, `diagnostic-only`, or `rejected`.
- `B3.3` Audit suspicious IO and split policy.
- `B3.4` Compare strict candidates by fixed metrics:
  - top3 evidence F;
  - top3 macro recall;
  - top100 context recall/hit-any;
  - classifier accuracy and macro-F1 when applicable;
  - runtime and peak memory;
  - Colab feasibility;
  - train-CV variance.
- `B3.5` Freeze final config.

Rule:

- dev-tuned winners are diagnostic-only.
- if the highest dev score is diagnostic-only, report it honestly but do not use it as final strict config.

## Batch 4: Frozen Strict Rebuild

Purpose: rebuild the selected candidate from raw JSON.

Milestones:

- `B4.1` Delete declared Round18 strict outputs.
- `B4.2` Run frozen sparse gate.
- `B4.3` Run frozen dense leaves inside sparse candidates.
- `B4.4` Run frozen leaf-flat aggregator.
- `B4.5` Generate top3 and topN classifier context.
- `B4.6` Run frozen classifier.
- `B4.7` Generate final dev metrics and manifest.

Rule:

- Spark exploration artifacts may be used as design evidence, not as final inputs.
- If any cache is used, it must be reproducible from raw JSON and safe to delete/rebuild.

## Batch 5: Final Notebook, Audit, And Packaging

Milestones:

- `B5.1` Notebook executes the frozen pipeline from raw JSON.
- `B5.2` Narrative distinguishes sparse-only gate, dense-in-sparse leaves, leaf-flat aggregator, and classifier.
- `B5.3` Resource zip excludes data, checkpoints, hidden caches, and old artifacts.
- `B5.4` Clean rebuild passes.
- `B5.5` Cache deletion test passes.
- `B5.6` Suspicious IO grep passes.
- `B5.7` `eval.py` integrity check passes.
- `B5.8` Stale notebook output check passes.
- `B5.9` Master emits final verdict:
  - `APPROVED`
  - `REJECTED`
  - `NEEDS INVESTIGATION`

