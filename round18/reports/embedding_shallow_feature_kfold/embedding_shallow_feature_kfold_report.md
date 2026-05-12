# Embedding + Shallow Feature K-Fold Reranker

Goal: test whether explicit shallow matching features complement MiniLM embedding inner-product scores for top64 context selection.

## Strict Policy
- Variant selection uses train-claim k-fold CV only.
- Dev is used once for confirmation after the selected variant is fixed.
- Primary metric: `macro_recall@64`.

## Inputs
- Train embedding pool: `round18/outputs/o_dense/o_d1x_embedding_bm25_char_train_top500/train_full_train_o_d1x_embedding_bm25_char_train_top500_strict_top500_candidates.json`
- Dev embedding pool: `round18/outputs/o_dense/o_d1x_embedding_bm25_char_dev_top500/dev_full_dev_o_d1x_embedding_bm25_char_dev_top500_strict_top500_candidates.json`
- Candidate budget: `top500`
- K folds: `5`

## Selected Variant
- Variant: `embedding_plus_shallow`
- CV macro@64: `0.509026`
- Dev macro@64: `0.578355`
- Embedding-only dev macro@64: `0.564935`
- Dev delta vs embedding-only: `+0.013420`

## Leakage Scan
- No forbidden path markers found.

## Variant Table
| variant | cv@3 | cv@10 | cv@64 | dev@3 | dev@10 | dev@64 |
|---|---:|---:|---:|---:|---:|---:|
| embedding_plus_shallow | 0.1720 | 0.3079 | 0.5090 | 0.1981 | 0.3982 | 0.5784 |
| embedding_plus_factual | 0.1692 | 0.2967 | 0.5039 | 0.2016 | 0.3825 | 0.5804 |
| embedding_only | 0.1496 | 0.2709 | 0.4818 | 0.1883 | 0.3293 | 0.5649 |
| shallow_only | 0.1304 | 0.2298 | 0.4218 | 0.1491 | 0.2772 | 0.4903 |

## Runtime
- Wall seconds: `173.200`
