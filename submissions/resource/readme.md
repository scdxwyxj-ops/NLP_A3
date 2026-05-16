# COMP90042 Project Colab Upload Bundle

This folder contains a self-contained Colab notebook.
The canonical submission zip built from these files is stored at
`../COMP90042_teamname_resource.zip`. Report sources are kept separately under
`../../docs/report/latex/` so they are not mixed into the Colab runtime
directory.

## Files Read

The notebook reads only course JSON files from this folder, or from common Colab upload locations such as the current working directory, `/content`, `/`, or `/root` when the same files are uploaded there:

- `train-claims.json`
- `dev-claims.json` or `test-claims-unlabelled.json`
- `evidence.json`

It does not read repository-level `docs/`, `data/`, `model/`, `src/`, saved ranked artifacts, or any path outside the selected notebook data folder. It does download/load the declared Hugging Face models at runtime.

## Files Written

Generated outputs, and optional debug intermediates when enabled, are written under:

- `outputs/self_generated/`

By default, the notebook only saves:

- `final_predictions.json`
- `run_summary.json`

Set `A3_SAVE_ARTIFACTS=1` to also save these generated intermediate files for debugging:

- `train_sparse_fusion_top500.json`
- `target_sparse_fusion_top500.json`
- `train_top64_embedding_hand.json`
- `target_top64_embedding_hand.json`
- `target_top3_ce_embedding_source_fusion.json`
- `predicted_labels.json`
- `candidate_metrics.json`
- `top64_metrics.json`
- `top3_metrics.json`
- `classifier_metrics.json`
- `final_predictions.json`
- `run_summary.json`

## Pipeline

- Candidate: sklearn/scipy sparse-matrix BM25 gates for word, character n-gram, structured cue, and PRF/query expansion, fused with fixed char-heavy RRF weights:
  `{"bm25": 0.75, "char": 2.0, "structured": 0.25, "prf": 0.5}`.
- Top64: MiniLM embedding inner product plus shallow factual hand features.
- Top3: top500-level CE, embedding, embedding-rank, and source-rank fusion with CE scored on a bounded prefilter.
- Classifier: enhanced Context20 TF-IDF One-vs-Rest Logistic Regression with fixed `C=0.125`, `class_weight="balanced"`, and word 1-2 grams. The context keeps top-ranked evidence denser than tail evidence, repeats top5 evidence once, and adds compact rank/overlap/number/negation cue tokens before each evidence line.

Optional runtime control:

- Top3 CE scoring uses a bounded candidate prefilter inside the notebook (`TOP3_CE_PREFILTER_K`, default 256).

For final prediction on test, set:

```bash
A3_TARGET_FILE=test-claims-unlabelled.json
```
