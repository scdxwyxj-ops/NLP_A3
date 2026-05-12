# Round18 Batch 2 Dense Wave 1 Review

Date: 2026-05-09 Australia/Sydney

## Scope

Reviewed first dense-in-sparse wave:

- `O-D1`: light bi-encoder lane with deterministic fallback.
- `O-D3`: bounded cross-encoder lane inside sparse top64.

Both lanes read the approved strict sparse candidate pool:

```text
round18/outputs/o_sparse/o_s4_fusion/dev_full_dev_o_s4_fusion_strict_rrf_prf_heavy_top500_candidates.json
```

## Master Verification

```bash
PYTHONPATH=src:. python -m round18.tools.strict_guard \
  round18/outputs/o_dense/o_d1_light_biencoder/run_manifest.json \
  round18/outputs/o_dense/o_d3_cross_encoder/full_dev_top64_manifest.json

PYTHONPATH=src:. python -m round18.tools.validate_manifest \
  round18/outputs/o_dense/o_d1_light_biencoder/run_manifest.json \
  round18/outputs/o_dense/o_d3_cross_encoder/full_dev_top64_manifest.json
```

Results:

- strict guard: passed.
- manifest validation: passed.

## O-D1: Light Bi-Encoder Lane

Verdict: `APPROVED AS STRICT FALLBACK BASELINE`

Important caveat:

- `sentence-transformers` was not installed.
- The run used deterministic fallback: `sklearn_tfidf_svd`.
- Therefore this is not evidence for semantic bi-encoder quality.

Full dev metrics:

- macro recall@100: `0.3496753246753247`
- macro recall@500: `0.6195887445887446`
- hit-any@100: `0.6298701298701299`
- hit-any@500: `0.8506493506493507`
- runtime: about `433.818s`

Assessment:

- Strict scope is correct: it reranks only sparse candidate IDs.
- Top500 recall is unchanged from the input sparse pool, as expected for reranking inside fixed top500.
- Top100 recall drops substantially relative to sparse fusion, so this fallback should not drive final ranking.
- Use it as a baseline/diagnostic leaf, not as the main dense semantic result.

## O-D3: Bounded Cross-Encoder Lane

Verdict: `APPROVED AS STRICT-CANDIDATE`

Full dev top64 metrics:

- macro recall@1: `0.09805194805194804`
- macro recall@3: `0.2159090909090909`
- macro recall@5: `0.2637445887445887`
- macro recall@10: `0.3029220779220779`
- macro recall@64: `0.4140692640692641`
- hit-any@3: `0.42207792207792205`
- hit-any@10: `0.525974025974026`
- reranked pairs: `9856`
- model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- device: `cuda`

Assessment:

- This is the first true dense/neural strict leaf in Round18.
- It scores only sparse top64 candidates per claim.
- It does not perform full-corpus dense retrieval.
- It is suitable for the next aggregator stage as a top3/top10 precision leaf.

## Dense-In-Sparse Hypothesis Status

Current evidence:

- O-D3 proves that a real cross-encoder can run inside the sparse gate and produce top3/top10 ranked outputs.
- It does not yet prove the final dense-in-sparse system beats sparse-only, because cross-encoder top64 must be aggregated with the sparse top500 source instead of replacing the whole sparse order.

Next step:

Start leaf-flat aggregation using:

- sparse fusion top500;
- O-D3 cross-encoder top64 scores/ranks;
- optional O-D1 fallback scores/ranks as a weak lexical-similarity leaf.

