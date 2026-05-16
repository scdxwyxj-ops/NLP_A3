# Round18 Batch 2 Hand-Feature Leaf Review

Date: 2026-05-09 Australia/Sydney

## Scope

Reviewed the old Branch-A hand-feature result and reproduced the idea as a current Round18 leaf:

- old source inspected: `experiments/rerank/round16_branch_a_requirement_selector.py`
- old output inspected: `outputs/round16/branch_a_requirement/summary.csv`
- new strict leaf: `O-S8 hand-feature ranker`
- new aggregate: `O-A2c plain leaf rank over O-S8 + O-D3b`

## Old Result Audit

The old `ranker_claim_key` result had:

- top100 macro recall: `0.4722943722943722`
- top3 macro recall: `0.14805194805194805`
- note: `train-only LightGBM LambdaRank over lexical Claim-Key MaxSim features`

The useful part appears non-leaky:

- model fit used `data/train-claims.json` labels.
- feature importances for old output show `top100_rank`, `top3_rank`, `minilm_rank`, `bge_rank`, and `gbdt_rank` all at `0.0`.
- effective features were rank/score, lexical overlap, claim-key coverage, char overlap, entity, number, year, logic cues, and length/shape.

The old artifact itself is not promoted because it consumed old Round14/Round16 candidate files. The method is reusable; the artifact is not.

## Round18 Reimplementation

New train-side artifacts were generated from current Round18 raw-data leaves:

- train BM25:
  `round18/outputs/o_sparse/o_s1_lexical_index_experiments/train_train_full_bm25_bm25_top500_candidates.json`
- train char TF-IDF:
  `round18/outputs/o_sparse/o_s6_char_tfidf/train_full_train_train_full_o_s6_char_tfidf_tfidf_char_top500_candidates.json`

O-S8 then:

- builds a fixed train BM25+char RRF pool.
- trains LightGBM LambdaRank on train labels only.
- reranks the strict O-S7 dev top500 pool.
- does not read old `outputs/round14`, `outputs/round15`, `outputs/round16`, or `outputs/round17` artifacts.

## Verification

```bash
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_sparse/o_s1_lexical_index_experiments/train_full_bm25_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_sparse/o_s1_lexical_index_experiments/train_full_bm25_manifest.json
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_sparse/o_s6_char_tfidf/train_full_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_sparse/o_s6_char_tfidf/train_full_manifest.json
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_sparse/o_s8_hand_feature_ranker/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_sparse/o_s8_hand_feature_ranker/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/run_manifest.json
```

Results:

- strict guard: passed.
- manifest validation: passed.

## Key Results

O-S7 fixed strict sparse baseline:

- macro recall@64: `0.44047619047619047`
- macro recall@100: `0.48852813852813853`
- macro recall@500: `0.6666666666666667`

O-S8 hand-feature reranker over the same O-S7 top500 set:

- macro recall@10: `0.2624458874458874`
- macro recall@64: `0.49274891774891777`
- macro recall@100: `0.5322510822510823`
- macro recall@500: `0.6666666666666667`
- hit-any@100: `0.7857142857142857`

O-A2c aggregate over O-S8 + O-D3b:

- `strict_ce64_sparse_backfill` macro recall@10: `0.3346320346320346`
- `strict_ce64_sparse_backfill` macro recall@100: `0.5356060606060606`
- `strict_ce64_sparse_backfill` macro recall@500: `0.6666666666666667`
- `strict_rrf_sparse_ce` macro recall@100: `0.5356060606060606`
- `strict_sparse_only` macro recall@100: `0.5322510822510823`

## Feature Signal

Top O-S8 feature importances:

- `source_score`: `731`
- `claim_key_soft_maxsim`: `682`
- `source_rank`: `645`
- `char3_jaccard`: `595`
- `char4_jaccard`: `552`
- `claim_key_weighted_coverage`: `512`
- `claim_key_exact_coverage`: `502`
- `length_ratio`: `458`

Old rank-signal features from prior pipelines remain unused:

- `top100_rank`: `0`
- `top3_rank`: `0`
- `minilm_rank`: `0`
- `bge_rank`: `0`
- `gbdt_rank`: `0`

## Verdict

`APPROVED AS STRICT-CANDIDATE`

Recommended sparse ordering leaf:

```text
round18/outputs/o_sparse/o_s8_hand_feature_ranker/dev_full_dev_o_s8_hand_feature_ranker_top500_candidates.json
```

Recommended aggregate pool:

```text
round18/outputs/o_aggregate/o_a2c_plain_leaf_rank_s8/dev_full_dev_o_a2c_plain_leaf_rank_s8_strict_ce64_sparse_backfill_top500_candidates.json
```

## Interpretation

The hand-feature model should be added to the final plan as a plain leaf. It does not improve top500 recall when applied only inside the existing O-S7 top500 pool, because the candidate set is unchanged. It does improve the ordering substantially:

- O-S7 macro@100: `0.48852813852813853`
- O-S8 macro@100: `0.5322510822510823`
- O-A2c CE backfill macro@100: `0.5356060606060606`

So it is immediately useful for top100/top64 compression and for final top-k evidence selection. For true top500 selection, the next experiment should apply the same O-S8 ranker to a wider strict pool, such as O-S7 top1000/top2000, then cut back to top500.
