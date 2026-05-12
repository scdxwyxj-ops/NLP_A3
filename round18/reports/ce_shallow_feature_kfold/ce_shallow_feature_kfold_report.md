# CE + Shallow Feature K-Fold Reranker

Goal: test whether explicit shallow matching features complement the CE/MiniLM score for top64 context selection.

## Strict Policy
- Variant selection uses train-claim k-fold CV only.
- Dev is used once for confirmation after the selected variant is fixed.
- Primary metric: `macro_recall@64`.

## Inputs
- Train CE pool: `round18/outputs/o_dense/o_d3x_bm25_char_train_top500/train_full_train_o_d3x_bm25_char_train_top500_strict_top500_candidates.json`
- Dev CE pool: `round18/outputs/o_dense/o_d3x_bm25_char_dev_top500/dev_full_dev_o_d3x_bm25_char_dev_top500_strict_top500_candidates.json`
- Candidate budget: `top500`
- K folds: `5`

## Selected Variant
- Variant: `ce_plus_factual`
- CV macro@64: `0.515075`
- Dev macro@64: `0.567100`
- CE-only dev macro@64: `0.563745`
- Dev delta vs CE-only: `+0.003355`

## Leakage Scan
- No forbidden path markers found.

## Variant Table
| variant | cv@3 | cv@10 | cv@64 | dev@3 | dev@10 | dev@64 |
|---|---:|---:|---:|---:|---:|---:|
| ce_plus_factual | 0.1796 | 0.3156 | 0.5151 | 0.2242 | 0.3670 | 0.5671 |
| ce_plus_shallow | 0.1769 | 0.3132 | 0.5139 | 0.2145 | 0.3814 | 0.5672 |
| ce_only | 0.1837 | 0.3196 | 0.5103 | 0.2331 | 0.3693 | 0.5637 |
| shallow_only | 0.1281 | 0.2216 | 0.4058 | 0.1358 | 0.2652 | 0.4669 |

## Runtime
- Wall seconds: `157.376`
