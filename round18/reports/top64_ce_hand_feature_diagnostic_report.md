# Top64 CE/Hand-Feature Diagnostic

## Question
We checked whether the surprising result is real: pure hand-feature ranking appears best for the top64 candidate set. We also checked whether MiniLM/CE plus hand-feature or multi-gate aggregation should be preferred, and why top500 CE reranking is not the default.

## Findings

### 1. Hand-feature is the best top64 selector, but the margin is small
Recomputed macro recall on dev candidates:

| method | mR@3 | mR@10 | mR@64 | mR@100 | mR@500 |
|---|---:|---:|---:|---:|---:|
| Fixed score-fusion gate | 0.1188 | 0.2369 | 0.4405 | 0.4885 | 0.6667 |
| Hand-feature reranker | 0.1555 | 0.2624 | 0.4927 | 0.5323 | 0.6667 |
| Union pool + cheap ranker | 0.1463 | 0.2552 | 0.4876 | 0.5327 | 0.7005 |

Spark validator independently found the same result: hand-feature mR@64=0.492749, union+cheap mR@64=0.487554, margin 0.005195. This is real but not a large separation.

### 2. MiniLM/CE should be used as an inside-top64 reranker, not as the top64 selector
Existing Round18 CE artifacts reranked score-fusion top64, not hand-feature top64. That made the old tutorial comparison incomplete.

New diagnostic runs:

| method | CE input set | mR@1 | mR@3 | mR@5 | mR@10 | mR@64 | time |
|---|---|---:|---:|---:|---:|---:|---:|
| CE on score-fusion top64 | score-fusion top64 | 0.1019 | 0.2194 | 0.2754 | 0.3346 | 0.4405 | 16.278s |
| CE on hand-feature top64 | hand-feature top64 | 0.1075 | 0.2195 | 0.2821 | 0.3636 | 0.4927 | 16.125s |
| CE on union+cheap top64 | union+cheap top64 | 0.1083 | 0.2173 | 0.2800 | 0.3652 | 0.4876 | 13.492s |

Interpretation: CE does not change mR@64 when it reranks a fixed 64-candidate set. It improves the ordering inside that set, especially mR@10. Therefore the clean pipeline is: choose the best top64 candidate set with hand-feature, then use CE to order that set.

### 3. MiniLM/CE + multi-gate is competitive but not strictly better
Aggregate/backfill diagnostics:

| method | mR@3 | mR@10 | mR@64 | mR@100 | mR@500 |
|---|---:|---:|---:|---:|---:|
| S8 hand-feature sparse only | 0.1555 | 0.2624 | 0.4927 | 0.5323 | 0.6667 |
| S8 hand-feature + CE backfill | 0.2195 | 0.3636 | 0.4927 | 0.5323 | 0.6667 |
| S10 union+cheap sparse only | 0.1463 | 0.2552 | 0.4876 | 0.5327 | 0.7005 |
| S10 union+cheap + CE backfill | 0.2173 | 0.3652 | 0.4876 | 0.5327 | 0.7005 |

For top3/top10 ranking, S8+CE and S10+CE are essentially tied. For top64 selection, S8 is slightly better. For top500 context coverage, S10 is higher but much more expensive. That makes S8+CE the cleaner default for top64 evidence ranking; S10+CE is a high-cost diagnostic or context-heavy option.

### 4. Why not rerank top500 by CE by default
Top64 CE scores 154 x 64 = 9,856 claim-evidence pairs. Top500 CE would score 154 x 500 = 77,000 pairs, a 7.8125x increase.

Existing measured runtime:
- CE top64: 16.1-16.3s in the new diagnostic runs.
- Linear extrapolated CE top500: about 126s.
- Existing top500 dense run using fallback TF-IDF/SVD: 433.818s, and it underperformed on recall.

More importantly, reranking top500 cannot increase recall@500 for a fixed input pool. It only changes order. Since our downstream evidence uses top3/top10/top64, the practical default is to run CE only after selecting a bounded top64 set.

## Recommended tutorial statement
Use this language:

- Hand-feature reranking is not the final semantic reranker. It is the best cheap top64 selector among the tested strict candidates.
- MiniLM/CE is useful after the top64 set is fixed, because it improves the front of the ranking: mR@10 rises from 0.2624 to 0.3636 on the hand-feature top64 set.
- We do not run CE over top500 by default because it is at least 7.8x more candidate pairs and cannot improve recall@500 for the same pool.
