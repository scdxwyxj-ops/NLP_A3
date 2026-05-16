# COMP90042 A3 Resource Submission

This folder contains the tutorial-aligned final prediction driver.

## Files

- `GroupID_COMP90042_Project_2026.ipynb`: final submission notebook.

## Expected Inputs

The notebook is aligned with `group_meetings/second_meeting /tutorial.ipynb`. It expects:

- `data/dev-claims.json` or `A3_TARGET_CLAIMS_PATH`
- `data/evidence.json` or `A3_EVIDENCE_PATH`
- selected Round18 artifacts under `round18/outputs/` and `round18/reports/`

For an unlabelled test run, provide the matching test artifacts through:

```bash
A3_TARGET_CLAIMS_PATH=/path/to/test-claims-unlabelled.json
A3_TOP3_POOL_PATH=/path/to/test_top3_ce_embedding_fusion.json
A3_CLASSIFIER_PREDICTIONS_PATH=/path/to/test_context20_logreg_predictions.json
A3_OUTPUT_DIR=/path/to/outputs
```

## Aligned Method

The notebook now follows the selected tutorial pipeline exactly:

1. Candidate: BM25 word, character n-gram, structured cue, and query-expansion/PRF sparse gates fused with fixed RRF.
2. Top64 context: MiniLM embedding inner product plus shallow factual hand features.
3. Top3 evidence: train-selected CE plus embedding fusion.
4. Classifier: Context20 TF-IDF Logistic Regression with `max_features=60000`, word `1-2` grams, and `C=4.0`.

It no longer uses the previous BM25-only candidate gate, BGE-small final top3 fallback, archived Round16 shortcut, or DistilRoBERTa classifier path.

## Verified Dev Metrics

Using the aligned dev artifacts, the selected classifier has:

```txt
claim accuracy          = 0.5194805194805194
classification macro-F1 = 0.4719276094276095
```

The selected top3 CE+embedding fusion has:

```txt
top3 macro recall       = 0.23874458874458873
top3 evidence F         = 0.204700061842919
```
