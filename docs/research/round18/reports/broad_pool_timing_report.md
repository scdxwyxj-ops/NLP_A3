# Broad Pool Timing Report

Date: 2026-05-09 Australia/Sydney

## Question

Does the broad candidate pool materially increase runtime, and how should we balance recall against cost?

## Setup

The timing runs reuse existing Round18 leaf candidate pools and write only under:

```text
round18/outputs/timing/broad_pool/
```

This measures the cost after sparse leaf pools already exist. It does not include recomputing BM25, char TF-IDF, structured, or PRF leaves from raw evidence.

## Results

| method | what it does | external wall time | script wall time | peak RSS | avg candidates/claim | macro@64 | macro@500 | macro@1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Word+char broad pool | merge BM25 top500 + char top500, keep up to 1000 | 22.00s | 19.81s | 0.78GB | 820.7 | 0.4128 | 0.6579 | 0.7232 |
| Four-signal broad pool | merge BM25/char/structured/PRF top500, keep up to 2000 | 40.35s | 35.22s | 1.10GB | 1275.2 | 0.4253 | 0.6460 | 0.7146 |
| Broad pool + cheap ranker | train hand-feature ranker on train pool, compress BM25+char broad dev pool to top500 | 383.85s | 339.42s | 12.02GB | 500.0 | 0.4876 | 0.7005 | n/a |

Reference non-timing artifacts:

| method | script wall time | macro@64 | macro@500 |
|---|---:|---:|---:|
| Hand-feature reranker over fixed sparse pool | 83.18s | 0.4927 | 0.6667 |
| Previous broad pool + cheap ranker run | 329.74s | 0.4876 | 0.7005 |

## Interpretation

The broad pool merge itself is not the main bottleneck. Going from a two-source broad pool to a four-source broad pool roughly doubles merge time, but both remain under a minute on full dev.

The expensive step is compression with the hand-feature ranker. The current broad-pool cheap-ranker run takes about 6 minutes wall time and peaks around 12GB RSS, because it builds feature rows for a 1000-candidate train pool, trains LightGBM LambdaRank, and scores a 1000-candidate dev pool.

The recall tradeoff is also asymmetric:

- Broad pools improve the reservoir at large N, for example word+char broad pool reaches macro@1000 = 0.7232.
- Simple broad-pool ordering does not improve top64 or top500 by itself.
- Compression improves top500 substantially: macro@500 goes from 0.6667 fixed sparse / 0.6579 word+char broad ordering to 0.7005 after cheap ranking.
- For top64, the current broad-pool cheap ranker is slightly below the fixed-pool hand-feature reranker: 0.4876 vs 0.4927.

## Recommendation

Use broad pools only when the target is top500/context coverage or when a downstream compressor is available.

For top64-sensitive evidence selection, the current best strict gate is still the hand-feature reranker over the fixed sparse pool. The next useful experiment is not a larger raw broad pool by itself; it is a cheaper/better compressor:

```text
broad pool size: 500 / 1000 / 1500
compress to: top500 / top100 / top64
measure: runtime, peak memory, macro@64, macro@500
```

The practical operating point should likely be:

```text
BM25 + char broad pool top1000
-> train-only hand-feature compressor
-> top500 for classifier context
-> separate top64 selection for semantic rerank
```

Avoid four-source broad pool unless it gives a measurable downstream gain after compression. In the current uncompressed RRF result, it is slower and does not improve top500 over the simpler word+char broad pool.

## Commands

Word+char broad pool:

```bash
/usr/bin/time -f 'WALL_SECONDS %e\nMAX_RSS_KB %M' \
python round18/experiments/o_sparse/o_s9_union_gate/run_o_s9_union_gate.py \
  --claims data/dev-claims.json \
  --evidence data/evidence.json \
  --bm25-pool round18/outputs/o_sparse/o_s1_lexical_index_experiments/dev_full_bm25_dev_bm25_top500_candidates.json \
  --char-pool round18/outputs/o_sparse/o_s6_char_tfidf/dev_full_dev_o_s6_char_tfidf_tfidf_char_top500_candidates.json \
  --structured-pool round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposed_candidates.json \
  --prf-pool round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json \
  --candidate-k 1000 \
  --eval-k 3,10,64,100,500,1000 \
  --bm25-top-k 500 \
  --char-top-k 500 \
  --structured-top-k 0 \
  --prf-top-k 0 \
  --policies rrf,round_robin,priority,union_upper \
  --rrf-k 60 \
  --run-id timing_o_s9_bm25_char \
  --output-dir round18/outputs/timing/broad_pool/o_s9_bm25_char_dev \
  --manifest round18/outputs/timing/broad_pool/o_s9_bm25_char_dev/run_manifest.json \
  --record-path round18/outputs/timing/broad_pool/o_s9_bm25_char_dev/run_record.json \
  --stage o_s9_union_gate
```

Four-signal broad pool:

```bash
/usr/bin/time -f 'WALL_SECONDS %e\nMAX_RSS_KB %M' \
python round18/experiments/o_sparse/o_s9_union_gate/run_o_s9_union_gate.py \
  --claims data/dev-claims.json \
  --evidence data/evidence.json \
  --bm25-pool round18/outputs/o_sparse/o_s1_lexical_index_experiments/dev_full_bm25_dev_bm25_top500_candidates.json \
  --char-pool round18/outputs/o_sparse/o_s6_char_tfidf/dev_full_dev_o_s6_char_tfidf_tfidf_char_top500_candidates.json \
  --structured-pool round18/outputs/o_sparse/o_s2_structured/dev_full_dev_decomposed_candidates.json \
  --prf-pool round18/outputs/o_sparse/o_s3_prf/candidate_pool_prf_top500.json \
  --candidate-k 2000 \
  --eval-k 3,10,64,100,500,1000,2000 \
  --bm25-top-k 500 \
  --char-top-k 500 \
  --structured-top-k 500 \
  --prf-top-k 500 \
  --policies rrf,round_robin,priority,union_upper \
  --rrf-k 60 \
  --run-id timing_o_s9_four_source_wide \
  --output-dir round18/outputs/timing/broad_pool/o_s9_four_source_wide_dev \
  --manifest round18/outputs/timing/broad_pool/o_s9_four_source_wide_dev/run_manifest.json \
  --record-path round18/outputs/timing/broad_pool/o_s9_four_source_wide_dev/run_record.json \
  --stage o_s9_union_gate
```

Broad pool + cheap ranker:

```bash
/usr/bin/time -f 'WALL_SECONDS %e\nMAX_RSS_KB %M' \
python round18/experiments/o_sparse/o_s10_wide_hand_feature/run_o_s10_wide_hand_feature.py \
  --train-pool round18/outputs/o_sparse/o_s9_union_gate_bm25_char_train/train_full_train_o_s9_union_gate_bm25_char_round_robin_top1000_candidates.json \
  --dev-pool round18/outputs/o_sparse/o_s9_union_gate_bm25_char_dev/dev_full_dev_o_s9_union_gate_bm25_char_round_robin_top1000_candidates.json \
  --train-pool-limit 1000 \
  --dev-pool-limit 1000 \
  --candidate-k 500 \
  --eval-k 3,10,64,100,500 \
  --output-dir round18/outputs/timing/broad_pool/o_s10_bm25_char_round_robin_top1000 \
  --run-id timing_o_s10_bm25_char_round_robin_top1000 \
  --manifest round18/outputs/timing/broad_pool/o_s10_bm25_char_round_robin_top1000/run_manifest.json \
  --record-path round18/outputs/timing/broad_pool/o_s10_bm25_char_round_robin_top1000/run_record.json
```
