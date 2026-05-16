# O-C5 Classification Comparison Master Review

## Scope

This review audits the O-C5 classification comparison requested after the O-C4 classifier was found to be too shallow. The master agent did not run the experiments directly; separate workers produced the experimental artifacts, and this file records the quality-control decision.

Current O-C4 reference:

- context: CE + factual shallow context, k20
- classifier: TF-IDF + Logistic Regression
- macro-F1: `0.4719276094`
- accuracy: `0.5194805195`
- status: strict candidate, collapse gate passed

## Strictness Correction

After reviewing Worker A selection traces, there is an important correction:

- Each Worker A run is strict internally: train/dev context family matches, and each model config is selected by train holdout.
- However, selecting `k=5` from the full `k={5,10,20,32,64}` sweep because it has the best dev macro-F1 would be cross-k dev tuning.
- The train-holdout trace does not select `k=5` across depths. Among TF-IDF + shallow runs, train-holdout macro-F1 is highest at `k=64`, while its dev macro-F1 is only `0.3828417210`.
- Worker F independently audited this point from Worker A selection traces. Its train-only winner is `tfidf_plus_shallow @ k=64`, train-holdout macro-F1 `0.4089419183`, dev macro-F1 `0.3828417210`, below the O-C4 baseline.

Therefore `k5 TF-IDF + shallow side features` is a strong same-dev positive result, but it cannot be promoted by selecting it from the same dev sweep.

## Best Same-Dev Candidate

- method: `k5 TF-IDF + shallow side features`
- artifact: `round18/outputs/o_classifier/o_c5_comparison/worker_a_tfidf_side/worker_a_tfidf_side_k5/tfidf_logreg_plus_side_mf30000_ngram1x2_c1p0_sf47_metrics.json`
- macro-F1: `0.4954212454`
- accuracy: `0.5454545455`
- macro recall: `0.5024609437`
- prediction histogram: `SUPPORTS=58`, `REFUTES=25`, `NOT_ENOUGH_INFO=50`, `DISPUTED=21`
- top class share: `0.3766233766`
- collapse gate: passed

This improves over O-C4 by `+0.0234936360` macro-F1 on dev, but final robustness testing does not support promotion.

## Final Candidate Robustness

Worker G fixed the best candidate and compared it against an O-C4-like baseline inside train only:

- candidate: `k=5 TF-IDF + shallow`, max features `30000`, ngram `1-2`, `C=1.0`
- baseline: `k=20 TF-IDF-only`, max features `60000`, ngram `1-2`, `C=4.0`
- repeated train holdout, 5 seeds: mean delta macro-F1 `-0.010880`
- 5-fold train CV: mean delta macro-F1 `-0.006603`

The same dev split still shows `+0.023494` macro-F1 for the candidate, but that is confirmation on the same dev split that exposed the candidate. The train-only robustness result is the promotion decision.

## Strict Results

| branch | method | strict status | accuracy | macro-F1 | macro recall | top class share | decision |
|---|---|---|---:|---:|---:|---:|---|
| O-C4 | k20 TF-IDF LogReg | strict | 0.5195 | 0.4719 | 0.4701 | 0.4351 | keep promoted |
| Worker A/G | k5 TF-IDF + shallow side features | same-dev positive, train-robustness negative | 0.5455 | 0.4954 | 0.5025 | 0.3766 | do not promote |
| Worker A | k32 TF-IDF + shallow side features | strict | 0.5130 | 0.4757 | 0.4790 | 0.4221 | secondary positive |
| Worker B | k20 DistilRoBERTa tokenized classifier | strict | 0.5390 | 0.4178 | 0.4362 | 0.6104 | reject for macro-F1 |
| Worker D | k20 MiniLM embedding-only classifier | strict | 0.5000 | 0.4430 | 0.4345 | 0.4610 | reject for macro-F1 |
| Worker F | train-only cross-k/mode selection winner: k64 TF-IDF + shallow | strict selection audit | 0.4221 | 0.3828 | n/a | 0.3442 | reject; confirms no O-C5 strict promotion |
| Worker G | fixed k5 shallow train-only robustness | strict robustness audit | n/a | delta -0.0109 / -0.0066 | n/a | n/a | reject promotion |

## Diagnostic Results

Worker C attempted late fusion over Worker A, B, and D outputs. It correctly refused strict promotion because the holdout probability files were not claim-id aligned:

- B holdout used `val-*` placeholder IDs, not A claim IDs.
- D holdout was missing 159 A holdout claim IDs.

The only fusion result is therefore diagnostic-only:

- method: dev-only late fusion
- macro-F1: `0.4429629723`
- accuracy: `0.5000`
- best diagnostic weight: A `0.0`, B `0.0`, D `1.0`
- decision: not promoted

## Negative Findings

- A full transformer tokenizer/classifier was tested on the current strict context. It reached higher accuracy than O-C4 but lower macro-F1, mainly because `DISPUTED` remained weak: only one `DISPUTED` prediction and recall `0.0556`.
- MiniLM embedding features were real transformer embeddings, not fallback, but did not beat TF-IDF + shallow side features.
- Shallow side features are valuable when attached to TF-IDF classification, but shallow-only and embedding+shallow variants were not competitive in Worker D.
- Fusion cannot be promoted until all participating models emit out-of-fold probabilities for the same train claim IDs.

## Quality-Control Decision

Do not promote a new final strict classifier. Keep O-C4 as the current strict promoted classifier. Record `k5 TF-IDF + shallow side features` as a non-promoted finding: it improves the same dev split but does not improve train-only repeated holdout or CV on average.

Next strict upgrade should use one shared train-only selection protocol for context depth, side features, transformer models, and fusion before any dev confirmation.

For fusion, rerun A/B/D under one shared split protocol that writes aligned out-of-fold probabilities for every train claim. Only then can late fusion or meta learning be evaluated as strict rather than diagnostic.
