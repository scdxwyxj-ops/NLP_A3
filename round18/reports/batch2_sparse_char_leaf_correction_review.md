# Round18 Batch 2 Char Leaf Correction Review

Date: 2026-05-09 Australia/Sydney

## Scope

Reviewed the correction requested after reproducing the first-meeting top500 recall result:

- O-S6: add char TF-IDF as a first-class sparse leaf.
- O-S7: plain sparse leaf fusion over BM25, char TF-IDF, structured sparse, and PRF leaves.
- O-D3b: bounded cross-encoder rerank over the fixed O-S7 strict sparse pool.
- O-A2b: plain aggregate rank over O-S7 sparse leaf and O-D3b CE leaf.

The final promoted path avoids the old O-S4 nested sparse fusion and the old O-A2 aggregate.

## Verification

```bash
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_sparse/o_s6_char_tfidf/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_sparse/o_s6_char_tfidf/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_sparse/o_s7_plain_leaf_fusion/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_sparse/o_s7_plain_leaf_fusion/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_dense/o_d3b_cross_encoder_s7/full_dev_top64_s7_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_dense/o_d3b_cross_encoder_s7/full_dev_top64_s7_manifest.json
PYTHONPATH=src:. python -m round18.tools.strict_guard round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/run_manifest.json
PYTHONPATH=src:. python -m round18.tools.validate_manifest round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/run_manifest.json
PYTHONPATH=. pytest -q round18/tests
```

Results:

- strict guard: passed for O-S6, O-S7, O-D3b, and O-A2b.
- manifest validation: passed for O-S6, O-S7, O-D3b, and O-A2b.
- tests: `4 passed`.

## Reviewer Follow-Up

An independent reviewer agent checked the corrected chain and found one guardrail gap: O-S7 enforced BM25 and char pool defaults but did not enforce the structured and PRF pool defaults. The final strict run had used the correct current-run paths, but a manual override could have pointed those leaves elsewhere.

Fix applied:

- `round18/experiments/o_sparse/o_s7_plain_leaf_fusion/run_o_s7_plain_leaf_fusion.py` now enforces O-S2 structured and O-S3 PRF default current-run paths.
- O-S7, O-D3b, and O-A2b were rerun after the fix.
- Strict guard, manifest validation, and tests passed after rerun.

## Key Results

First-meeting reproduction anchor:

- char TF-IDF top500 macro recall: `0.6610389610389611`
- BM25 top500 macro recall: `0.5861471861471862`
- BM25+char RRF top500 macro recall: `0.6579004329004329`

O-S6 char TF-IDF leaf:

- macro recall@100: `0.4100649350649351`
- macro recall@500: `0.6610389610389611`
- micro recall@500: `0.6334012219959266`
- hit-any@500: `0.8896103896103896`

O-S7 best fixed strict sparse leaf fusion:

- variant: `strict_rrf_char_heavy`
- macro recall@100: `0.48852813852813853`
- macro recall@500: `0.6666666666666667`
- micro recall@500: `0.639511201629328`
- hit-any@500: `0.8896103896103896`

O-S7 diagnostic-only winner:

- variant: `diagnostic_rrf_bm250.8_char1.3_structured0.8_prf0.8_k80`
- macro recall@100: `0.49047619047619045`
- macro recall@500: `0.6771645021645022`
- status: `diagnostic-only`

O-D3b cross-encoder over O-S7 top64:

- macro recall@1: `0.10194805194805194`
- macro recall@3: `0.21937229437229436`
- macro recall@10: `0.3346320346320346`
- macro recall@64: `0.44047619047619047`
- reranker: `cross_encoder`

O-A2b plain aggregate:

- `strict_sparse_only` macro recall@500: `0.6666666666666667`
- `strict_ce64_sparse_backfill` macro recall@10: `0.3346320346320346`
- `strict_ce64_sparse_backfill` macro recall@500: `0.6666666666666667`
- `strict_rrf_sparse_ce` macro recall@10: `0.2958874458874459`
- `strict_rrf_sparse_ce` macro recall@500: `0.6666666666666667`

## Verdict

`APPROVED AS STRICT-CANDIDATE`

Promoted sparse pool:

```text
round18/outputs/o_sparse/o_s7_plain_leaf_fusion/dev_full_dev_o_s7_plain_leaf_fusion_strict_rrf_char_heavy_top500_candidates.json
```

Promoted aggregate pool for downstream evidence selection:

```text
round18/outputs/o_aggregate/o_a2b_plain_leaf_rank_s7/dev_full_dev_o_a2b_plain_leaf_rank_s7_strict_ce64_sparse_backfill_top500_candidates.json
```

The aggregate backfill variant is preferred when downstream top-k is small because it preserves O-S7 top500 coverage while using CE order inside the top64 band. The sparse-only variant remains the cleanest retrieval baseline and ties at recall@500.

## Interpretation

The first-meeting `~0.65` top500 recall is reproducible and was caused by the char TF-IDF leaf, not by the earlier BM25-only baseline. Adding char TF-IDF as a plain leaf fixes the main Round18 sparse gap:

- O-S1 BM25 macro@500: `0.5861471861471862`
- O-S4 old strict sparse fusion macro@500: `0.6195887445887446`
- O-S6 char TF-IDF macro@500: `0.6610389610389611`
- O-S7 fixed strict sparse fusion macro@500: `0.6666666666666667`

The corrected O-S7/O-A2b path is intentionally plain:

```text
BM25 leaf
char TF-IDF leaf
structured sparse leaf
PRF leaf
  -> fixed sparse RRF leaf fusion
  -> bounded CE leaf over top64
  -> CE64 + sparse backfill aggregate
```

No diagnostic-best output is promoted.

## Next Gate

Use O-A2b `strict_ce64_sparse_backfill` as the next strict candidate pool for claim verification or evidence selection experiments. Keep `strict_sparse_only` as the retrieval coverage baseline and use diagnostic O-S7 only for analysis, not promotion.
