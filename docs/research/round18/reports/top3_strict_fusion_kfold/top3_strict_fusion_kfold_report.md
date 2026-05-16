# Top3 Strict Fusion K-Fold Reranker

## Policy
- Mode: `strict-candidate`
- Variants tested: `ce_only`, `embedding_only`, `ce_plus_shallow`, `embedding_plus_shallow`.
- Evaluation targets include R@1, R@3, R@5, R@10 and extended cuts R@64, R@500.
- Candidate budget: `top500`.
- Eval cuts: `[1, 3, 5, 10, 64, 500]`.

## Inputs
- Train claims: `data/train-claims.json`
- Dev claims: `data/dev-claims.json`
- Train CE pool: `round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json`
- Dev CE pool: `round18/outputs/o_dense/o_d3x_cross_encoder_s8_top500_diag/dev_full_dev_full_dev_top500_s8_diag_strict_top500_candidates.json`
- Train embedding pool: `round18/outputs/o_dense/o_d1x_embedding_bm25_char_train_top500/train_full_train_o_d1x_embedding_bm25_char_train_top500_strict_top500_candidates.json`
- Dev embedding pool: `round18/outputs/o_dense/o_d1x_embedding_bm25_char_dev_top500/dev_full_dev_o_d1x_embedding_bm25_char_dev_top500_strict_top500_candidates.json`
- Train hand pool: `round18/outputs/o_sparse/o_s8_hand_feature_ranker/train_full_o_s8_hand_feature_ranker_rrf_bm25_char_top500_pool.json`
- Dev hand pool: `round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json`

## Selected Variant
- Variant: `embedding_plus_shallow`
- CV@1: `0.077088`
- CV@3: `0.155828`
- CV@5: `0.205257`
- CV@10: `0.286814`
- Dev@1: `0.100866`
- Dev@3: `0.217100`
- Dev@5: `0.264394`
- Dev@10: `0.353463`

## Top Records by Selection Score
| variant | ce_w | emb_w | hand_w | prior_w | cv@1 | cv@3 | cv@5 | cv@10 | dev@1 | dev@3 | dev@5 | dev@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| embedding_plus_shallow | 0.000 | 1.000 | 0.000 | 0.250 | 0.0771 | 0.1558 | 0.2053 | 0.2868 | 0.1009 | 0.2171 | 0.2644 | 0.3535 |
| embedding_only | 0.000 | 0.250 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_only | 0.000 | 0.500 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_only | 0.000 | 0.750 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_only | 0.000 | 1.000 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_plus_shallow | 0.000 | 0.250 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_plus_shallow | 0.000 | 0.500 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_plus_shallow | 0.000 | 0.750 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_plus_shallow | 0.000 | 1.000 | 0.000 | 0.000 | 0.0769 | 0.1496 | 0.2017 | 0.2709 | 0.0880 | 0.1883 | 0.2517 | 0.3293 |
| embedding_only | 0.000 | 1.000 | 0.000 | 0.250 | 0.0761 | 0.1583 | 0.2069 | 0.2870 | 0.0994 | 0.2169 | 0.2686 | 0.3565 |
| embedding_only | 0.000 | 0.750 | 0.000 | 0.250 | 0.0744 | 0.1556 | 0.2044 | 0.2807 | 0.1015 | 0.2063 | 0.2649 | 0.3475 |
| embedding_plus_shallow | 0.000 | 0.750 | 0.000 | 0.250 | 0.0739 | 0.1527 | 0.2025 | 0.2804 | 0.1067 | 0.2126 | 0.2701 | 0.3529 |
| embedding_plus_shallow | 0.000 | 1.000 | 0.250 | 0.000 | 0.0732 | 0.1597 | 0.2107 | 0.2887 | 0.0989 | 0.2132 | 0.2656 | 0.3724 |
| embedding_plus_shallow | 0.000 | 1.000 | 0.250 | 0.250 | 0.0732 | 0.1517 | 0.1983 | 0.2689 | 0.1045 | 0.2294 | 0.2819 | 0.3558 |
| embedding_plus_shallow | 0.000 | 0.750 | 0.250 | 0.000 | 0.0724 | 0.1571 | 0.2071 | 0.2840 | 0.1037 | 0.2123 | 0.2728 | 0.3732 |

## Forbidden-token scan
- Passed

## Risk notes
- Strict mode selected: `True`.
- CE-only and CE+shallow depend on train CE pool coverage; if train CE claims are sparse, fusion scores can drift toward prior-only behavior.
- Source-rank prior is additional unsupervised signal (`1/(60+min_source_rank)` normalized per claim).
- Diagnostic mode only reports smoke/dev-subset metrics and should not be promoted as final selection.

## Runtime
- Wall seconds: `616.611`
