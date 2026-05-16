# Round18 Dynamic Spark Orchestration Plan

## Master Role

The master agent owns:

- contract and schema approval;
- task assignment;
- write-scope approval;
- review of manifests and diffs;
- strict/diagnostic/rejected classification;
- final frozen config approval;
- final verdict.

The master does not silently accept exploratory artifacts as final pipeline inputs.

## Spark Exploration Policy

Spark agents are allowed to explore methods beyond the initial hand-designed plan. They may propose new retrieval, ranking, aggregation, or classification ideas if they remain open-source, Colab-feasible, and reproducible from raw course JSON.

Exploration output must stay under the assigned `round18/outputs/<pool>/<agent_name>/` namespace.

## Research Scout Pools

### Pool R-Sparse

Purpose: discover sparse-only candidate gate methods beyond BM25 and char TF-IDF.

Possible directions:

- lexical retrieval variants;
- query decomposition;
- numeric, date, unit, and entity-aware sparse matching;
- PRF/RM3-lite query expansion;
- sparse rank fusion;
- learned sparse methods only if clearly labeled and README/Colab feasible.

Dynamic milestones:

1. Literature/method scan.
2. Strict feasibility filter.
3. Minimal raw-JSON prototype proposal.
4. Expected metrics and failure modes.
5. Recommendation: strict-candidate, diagnostic-only, or reject.

### Pool R-Dense

Purpose: discover dense or neural reranking leaves that operate only inside sparse candidates.

Possible directions:

- open-source bi-encoders;
- cross-encoders;
- late interaction if Colab-feasible;
- score normalization;
- topM scheduling;
- runtime/memory guardrails.

Dynamic milestones:

1. Candidate model list with source and license/compliance notes.
2. Cost model for sparse topN/topM.
3. Scoring schema proposal.
4. Strict-mode guard plan proving no full-corpus dense final path.
5. Recommendation.

### Pool R-Aggregate

Purpose: discover leaf-flat aggregation and classifier strategies.

Possible directions:

- linear calibration;
- tree-based rank aggregation;
- pairwise/listwise ranking if feasible;
- multi-objective top3/top100 optimization;
- factual features;
- lightweight NLI/classifier options;
- context construction strategies.

Dynamic milestones:

1. Candidate feature/learner matrix.
2. Train-only CV selection design.
3. Top3/top100 metric plan.
4. Classifier context plan.
5. Recommendation.

## Optimization Spark Pools

Optimization agents may implement small experiments only after the relevant contract and write scope are approved. They must use `round18/` as their default output root and must not write shared files.

### Pool O-Sparse

Suggested initial agents:

- `O-S1`: BM25 and lexical parameter exploration.
- `O-S2`: query decomposition and entity/number/date sparse variants.
- `O-S3`: PRF/RM3-lite sparse expansion.
- `O-S4`: sparse fusion and train-CV selection.

### Pool O-Dense

Suggested initial agents:

- `O-D1`: BGE-small or better open-source bi-encoder inside sparse pool.
- `O-D2`: bounded cross-encoder inside sparse topM.
- `O-D3`: dense cost-benefit and score normalization.

### Pool O-Aggregate

Suggested initial agents:

- `O-A1`: feature family ablation.
- `O-A2`: learner family comparison.
- `O-A3`: top3/top100 multi-objective ranking.

### Pool O-Classifier

Suggested initial agents:

- `O-C1`: context width selection using train-only CV.
- `O-C2`: classifier baseline and Transformer comparison.

## Non-Delegable Gates

These are serial master gates:

- contract approval;
- split policy approval;
- schema approval;
- final candidate classification;
- frozen config approval;
- clean rebuild acceptance;
- final notebook/package verdict.

