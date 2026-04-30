# Current Experiment Report

Date: 30 April 2026

This report summarises the work completed so far on the evidence retrieval and
reranking part of the COMP90042 Assignment 3 fact-checking system. The main goal
of this stage was not to finalise the whole system, but to build a reliable
evidence preprocessing pipeline for the later claim classification model.

## 1. Task Framing

The assignment has two connected subtasks:

1. Retrieve relevant evidence passages for each claim.
2. Classify the claim into `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, or
   `DISPUTED`.

The experiments below focus mainly on the first part. This is important because
the classifier can only make a good label decision if the retrieved evidence
contains enough useful information.

## 2. Repository Structure

I reorganised the code into a more reusable structure:

```txt
src/a3_factcheck/
  retrieval/      # TF-IDF and BM25 retrieval
  rerank/         # cross-encoder reranking utilities
  semantic/       # lightweight semantic feature extraction
  metrics.py      # shared metric helpers

experiments/
  retrieval/      # retrieval baseline scripts
  rerank/         # reranker training and inference scripts
  analysis/       # error analysis and comparison table scripts
  semantic/       # semantic feature extraction scripts
  classification/ # classifier-ready evidence packaging
```

The cleaned GitHub version excludes raw data, generated outputs, model
checkpoints, and local-only planning notes.

## 3. Baseline Retrieval Experiments

### TF-IDF

The first baseline used TF-IDF lexical matching between each claim and the
evidence corpus.

Best observed dev retrieval result:

```txt
TF-IDF top-3 F-score = 0.0533
```

This was useful as a sanity check for the data loading, prediction JSON format,
and `eval.py` workflow, but the retrieval quality was weak.

### BM25

The next baseline used BM25-style lexical scoring. BM25 performed better than
TF-IDF and became the candidate generator for later reranking.

Best observed dev retrieval result:

```txt
BM25 top-5 F-score = 0.0772
```

Candidate recall was still limited:

```txt
BM25 top-50 macro recall  = 0.3249
BM25 top-100 macro recall = 0.4188
```

This means that even with top-100 candidates, many gold evidence passages are
still not retrieved. A reranker cannot recover evidence that is missing from the
candidate pool.

## 4. MiniLM Cross-Encoder Reranking

I tested `cross-encoder/ms-marco-MiniLM-L6-v2` as a reranker on top of BM25
candidates.

The pipeline was:

```txt
claim
-> BM25 top-50 evidence candidates
-> MiniLM cross-encoder scores each claim-evidence pair
-> rerank candidates by score
-> output top-k evidence
```

The zero-shot MiniLM reranker gave the strongest retrieval result so far:

```txt
zero-shot MiniLM top-1 F-score  = 0.1299
zero-shot MiniLM top-3 F-score  = 0.1642
zero-shot MiniLM top-5 F-score  = 0.1578
zero-shot MiniLM top-10 F-score = 0.1218
```

Current best final evidence setting:

```txt
BM25 top-50 candidates
-> zero-shot MiniLM reranker
-> final top-3 evidence
```

This improved the dev evidence retrieval F-score from `0.0772` to `0.1642`.

## 5. Fine-Tuning Attempts

### Initial BCE Fine-Tuning

I built a supervised training set:

```txt
positive = claim + gold evidence
negative = claim + BM25 non-gold candidate
```

The first useful fine-tuning setup used a single-logit BCE loss, preserving the
reranker-style output head.

Result:

```txt
MiniLM BCE top-5 F-score = 0.1044
```

This beat BM25, but it was still worse than the zero-shot MiniLM reranker.

### Task-Aware Hard Negatives

The next experiment mined harder negatives using MiniLM itself:

```txt
BM25 top-100 candidates
-> zero-shot MiniLM reranking
-> choose high-ranked non-gold evidence as negatives
```

Generated training data:

```txt
claims = 1228
total pairs = 16402
positive pairs = 4122
negative pairs = 12280
mean negative MiniLM rank = 6.15
mean negative BM25 rank = 35.33
```

This means the negatives are not random. They are passages the current model
would rank highly, but which are not gold evidence.

Training setup:

```txt
model = cross-encoder/ms-marco-MiniLM-L6-v2
loss = single-logit BCE
learning rate = 5e-6
epochs = 0.5
batch size = 32
```

Dev comparison:

| Method | Top-k | F-score | Precision | Macro Recall | Hit-any | REFUTES Recall |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot MiniLM | 3 | 0.1642 | 0.1688 | 0.1877 | 0.3961 | 0.0494 |
| Task-aware MiniLM | 3 | 0.1599 | 0.1667 | 0.1790 | 0.3766 | 0.0494 |
| Zero-shot MiniLM | 5 | 0.1578 | 0.1325 | 0.2332 | 0.4545 | 0.0679 |
| Task-aware MiniLM | 5 | 0.1526 | 0.1273 | 0.2256 | 0.4351 | 0.0864 |
| Zero-shot MiniLM | 20 | 0.0777 | 0.0461 | 0.3035 | 0.5519 | 0.1327 |
| Task-aware MiniLM | 20 | 0.0766 | 0.0455 | 0.3000 | 0.5519 | 0.1327 |

The task-aware fine-tuned model came close to the zero-shot model but did not
beat it overall. However, at top-5 it improved `REFUTES` recall from `0.0679` to
`0.0864`, which may be worth further investigation.

## 6. Error Analysis

I generated TP/FP/FN/TN error analysis files for BM25, zero-shot MiniLM, and
fine-tuned MiniLM variants.

One important observation is that many false positives are not random. They are
often topically related to the claim but not useful as fact-checking evidence.

For example, a claim about Australian emissions and the effect of emissions
reduction may retrieve passages about Australia, CO2, or electricity emissions.
These passages are topic-relevant, but they may not address the key reasoning
needed for the label decision.

This suggests a distinction:

```txt
topic relevance:
  The passage is about a similar topic.

evidence relevance:
  The passage helps support, refute, or decide the claim label.
```

The current zero-shot MiniLM reranker is strong at topic relevance, but it is not
fully task-aware for fact verification.

## 7. Recall vs Final Evidence F-score

The best final evidence F-score comes from outputting only a few high-confidence
passages. However, the classifier may need a wider evidence context.

Zero-shot MiniLM recall on BM25 top-50 candidates:

```txt
top-3 macro recall  = 0.1877
top-5 macro recall  = 0.2332
top-10 macro recall = 0.2758
top-20 macro recall = 0.3035
top-50 macro recall = 0.3249
```

This leads to the current recommendation:

```txt
final evidence output:
  use zero-shot MiniLM top-3

classifier input:
  use a wider MiniLM-ranked context, such as top-20
```

The final submission should not output too many passages because precision will
drop. But the classifier can still read more candidate evidence internally.

## 8. Semantic Feature Prototype

I also implemented a lightweight semantic feature extractor. It does not make
predictions by itself and should not be treated as a hand-written rule system.

Current extracted fields:

```txt
entities
percentages
quantities
years
negation cues
comparison cues
causality cues
relation verbs
```

The intended use is to help the classifier or the analysis stage notice whether
the retrieved evidence covers important parts of the claim, such as numbers,
negation, causal wording, or comparison terms.

## 9. Classifier-Ready Evidence Package

I generated classifier-ready JSONL files:

```txt
outputs/round06/train-classifier-context-minilm-top20-semantic.jsonl
outputs/round06/dev-classifier-context-minilm-top20-semantic.jsonl
```

Each row contains:

```txt
claim_id
claim_text
claim_label
gold_evidences
final_evidence_candidates
classifier_evidence_context
optional semantic features
```

This separates two different needs:

```txt
final_evidence_candidates:
  small top-k evidence list for evaluator output

classifier_evidence_context:
  wider evidence context for label prediction
```

## 10. Current Recommendation

The most stable evidence preprocessing setup right now is:

```txt
BM25 top-50
-> zero-shot MiniLM reranker
-> top-3 evidence for final evidence output
-> top-20 evidence context for classifier input
```

The task-aware fine-tuning experiment is promising but not yet better than the
zero-shot reranker. It should remain in the comparison table, but I would not
use it as the default final evidence method yet.

## 11. Suggested Next Steps

1. Build a first claim classifier using the MiniLM top-20 evidence context.
2. Compare classifier input sizes: top-3, top-10, top-20, and possibly top-50.
3. Compare with and without semantic feature summaries.
4. Analyse classification errors separately for `REFUTES`, since retrieval for
   that class is currently weak.
5. If retrieval remains the bottleneck, improve candidate recall with BM25
   top-100, BM25+TF-IDF union, or another candidate source.

## 12. Important Caveats

These are the main points we should not overclaim:

- The zero-shot MiniLM reranker is currently the strongest retrieval method, but
  it is still not enough for a full fact-checking system by itself.
- Fine-tuning has not yet improved over the zero-shot reranker overall.
- The semantic features are currently auxiliary analysis/input features, not a
  proven performance improvement.
- The current classifier-ready JSONL is prepared for the next stage, but no
  final classifier result is included in this report yet.
- The test set has not been used for evaluation.
