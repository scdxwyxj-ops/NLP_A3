# O-C5 Worker D: Embedding + Shallow Classifier
## Scope
- context family: **strict**
- selection: **train_holdout**
- baseline macro-F1: **0.4719**
- strict vs diagnostic: **strict-candidate**
- embedding backend fallback: **transformers:sentence-transformers/all-MiniLM-L6-v2**

## Contract
- Train/dev hyperparameter tuning is forbidden.
- Train split holdout / CV only for model selection.
- Dev is used only for final scoring and output payloads.
- collapse gate: all 4 classes must appear, all per-class recall > 0, top-class share <= 0.70.

- Best macro-F1: **0.4430**
- Exceeds baseline 0.4719: `False`

| run_id | context_k | feature_mode | selected_algo | macro_f1 | macro_recall | top_class_share | acceptance | collapse |
|---|---:|---|---|---:|---:|---:|---|---|
| worker_d_embedding_classifier_k20 | 20 | embedding_only | mlp | 0.4430 | 0.4345 | 0.4610 | failed | passed |
| worker_d_embedding_classifier_k20 | 20 | shallow_only | mlp | 0.3213 | 0.3494 | 0.6558 | failed | passed |
| worker_d_embedding_classifier_k20 | 20 | embedding_plus_shallow | mlp | 0.3276 | 0.3279 | 0.4286 | failed | passed |

## Best Variant
- run_id: `worker_d_embedding_classifier_k20`
- context_k: `20`
- feature_mode: `embedding_only`
- selected_algo: `mlp`
- selected_config: `{"hidden_units": 64, "alpha": 0.0001}`
- macro_F1: `0.4430`
- strict runs count: `3` (strict mode enforced)
- diagnostic outputs: holdout/OoF and dev proba are written for fusion handoff.
