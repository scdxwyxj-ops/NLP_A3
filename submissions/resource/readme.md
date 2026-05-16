# COMP90042 Project Colab Upload Bundle

This folder contains a self-contained Colab notebook.
The canonical submission zip built from these files is stored at
`../COMP90042_teamname_resource.zip`. Report sources are kept separately under
`../report/latex/` so they are not mixed into the Colab runtime
directory.

## Files Read

Recommended Google Drive layout for Colab:

```text
MyDrive/NLP_A3/
  data/
    train-claims.json
    dev-claims.json
    test-claims-unlabelled.json
    evidence.json
  src/
    GroupID_COMP90042_Project_2026.ipynb
    requirements.txt
    readme.md
  outputs/
    self_generated/
```

The notebook reads only course JSON files from `A3_DATA_DIR` when set,
otherwise from `MyDrive/NLP_A3/data`, a local repo `data/` directory, or common
Colab upload locations such as the current working directory, `/content`, and
`/content/colab_notebooks`:

- `train-claims.json`
- `dev-claims.json` or `test-claims-unlabelled.json`
- `evidence.json`

Apart from optional local course JSONs in `data/`, it does not read saved
ranked artifacts, checkpoints, or generated outputs. It does download/load the
declared Hugging Face models at runtime.

## Files Written

Generated outputs, and optional debug intermediates when enabled, are written
under:

- `outputs/self_generated/`

In the recommended Drive layout this resolves to
`MyDrive/NLP_A3/outputs/self_generated/`. In the local repo layout it remains
under the selected local data directory unless `A3_OUTPUT_DIR` is set.

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
- Stage visualizations are enabled by default. Set `A3_VISUALIZE=0` to disable
  matplotlib plots while keeping all prediction and metric outputs.
- On labelled targets, the notebook reports the three course metrics from the
  official README: Evidence Retrieval F-score `F`, Claim Classification
  Accuracy `A`, and Harmonic Mean of `F` and `A`.

For final prediction on test, set:

```bash
A3_TARGET_FILE=test-claims-unlabelled.json
```

The unlabelled test file is only needed for optional leaderboard/test
prediction. It has no gold labels, so the notebook writes
`final_predictions.json` and skips accuracy/F-score evaluation for that target.
