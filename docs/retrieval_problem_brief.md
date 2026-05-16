# Retrieval Problem Brief

Date: 1 May 2026

This document describes the current state of our evidence retrieval experiments,
why the current scores are low, and how we should describe the problem when
asking for external help.

## Short Summary

We are working on the evidence retrieval part of a fact-checking pipeline. For
each claim, the system must retrieve the correct evidence passages from a large
evidence corpus. The retrieved evidence will then be used by a downstream claim
classifier.

The current retrieval results are still weak:

- final evidence F-score is low;
- top-k recall is low;
- many false positives are topic-relevant distractors rather than useful
  fact-checking evidence;
- the downstream classifier is likely to fail if the correct evidence is not in
  the retrieved context.

At this stage, improving recall is more important than only optimizing final
top-k F-score. The classifier can tolerate some extra evidence context, but it
cannot recover from missing gold evidence.

## Task Definition

The full assignment has two connected tasks:

1. Evidence retrieval:
   retrieve relevant evidence IDs for each claim.
2. Claim classification:
   predict one of `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, or `DISPUTED`.

Our current work focuses on evidence retrieval and reranking. The retrieval
module is also a preprocessing step for the final classifier.

The input is:

```txt
claim_id
claim_text
large evidence corpus
```

The desired output is:

```txt
claim_id -> ranked evidence IDs
```

For final evidence evaluation, we need a small high-precision list. For
classification, we probably need a wider high-recall evidence context.

## Current Methods Tested

### 1. TF-IDF

TF-IDF was used as the first lexical baseline.

Observed result:

```txt
TF-IDF top-3 F-score = 0.0533
```

This confirmed that the data loading, prediction format, and evaluation flow
were working, but retrieval quality was poor.

### 2. BM25

BM25 improved over TF-IDF and became the main candidate generator.

Observed result:

```txt
BM25 top-5 F-score = 0.0772
```

Candidate pool recall is still a serious bottleneck:

```txt
BM25 top-50 macro recall  = 0.3249
BM25 top-100 macro recall = 0.4188
```

This means many gold evidence passages are missing even before reranking.

### 3. Zero-Shot MiniLM Cross-Encoder Reranking

We used `cross-encoder/ms-marco-MiniLM-L6-v2` to rerank BM25 candidates.

Pipeline:

```txt
claim
-> BM25 top-50 evidence candidates
-> MiniLM cross-encoder reranking
-> output top-k evidence
```

Observed dev results:

```txt
zero-shot MiniLM top-1 F-score  = 0.1299
zero-shot MiniLM top-3 F-score  = 0.1642
zero-shot MiniLM top-5 F-score  = 0.1578
zero-shot MiniLM top-10 F-score = 0.1218
```

Current best final evidence setting:

```txt
BM25 top-50 -> zero-shot MiniLM -> top-3 evidence
```

This is better than BM25, but still not strong enough.

### 4. Fine-Tuned MiniLM With BCE

We trained a reranker using positive claim-gold evidence pairs and negative
claim-non-gold evidence pairs.

Observed result:

```txt
MiniLM BCE top-5 F-score = 0.1044
```

This was better than BM25, but worse than the zero-shot MiniLM reranker.

### 5. Task-Aware Hard-Negative Fine-Tuning

We mined hard negatives from high-ranked MiniLM non-gold candidates.

Training data:

```txt
claims = 1228
total pairs = 16402
positive pairs = 4122
negative pairs = 12280
mean negative MiniLM rank = 6.15
mean negative BM25 rank = 35.33
```

Training setup:

```txt
model = cross-encoder/ms-marco-MiniLM-L6-v2
loss = single-logit BCE
learning rate = 5e-6
epochs = 0.5
batch size = 32
```

Comparison:

| Method | Top-k | F-score | Precision | Macro Recall | Hit-any |
|---|---:|---:|---:|---:|---:|
| Zero-shot MiniLM | 3 | 0.1642 | 0.1688 | 0.1877 | 0.3961 |
| Task-aware MiniLM | 3 | 0.1599 | 0.1667 | 0.1790 | 0.3766 |
| Zero-shot MiniLM | 5 | 0.1578 | 0.1325 | 0.2332 | 0.4545 |
| Task-aware MiniLM | 5 | 0.1526 | 0.1273 | 0.2256 | 0.4351 |
| Zero-shot MiniLM | 20 | 0.0777 | 0.0461 | 0.3035 | 0.5519 |
| Task-aware MiniLM | 20 | 0.0766 | 0.0455 | 0.3000 | 0.5519 |

The task-aware model did not beat the zero-shot model overall. It may still be
useful for analysis because it slightly improved some class-specific recall
patterns, but it is not the current default method.

## Why The Problem Is Difficult

### 1. Candidate Recall Is Low

The first-stage retriever is missing many gold evidence passages.

Even with BM25 top-100:

```txt
macro recall = 0.4188
```

This is a hard ceiling for any reranker that only sees BM25 candidates. If the
gold evidence is not in the candidate pool, the reranker cannot recover it.

### 2. Topic Relevance Is Not Evidence Relevance

Many false positives are not random. They are often about the same topic as the
claim, but they do not support, refute, or decide the claim.

For example, a claim about Australian carbon emissions can retrieve passages
about:

- Australia;
- CO2;
- global emissions;
- percentages;
- electricity emissions.

These passages look related, but may not contain the exact evidence needed for
the label decision.

This distinction is central:

```txt
topic relevance:
  The passage is about a similar topic.

evidence relevance:
  The passage helps support, refute, or decide the claim label.
```

Current rerankers are still closer to topic relevance than true fact-checking
evidence relevance.

### 3. Final F-score And Classifier Recall Pull In Different Directions

For final evidence output, top-3 currently gives the best F-score because it
keeps precision higher.

For classifier input, top-3 is probably too narrow:

```txt
zero-shot MiniLM top-3 macro recall  = 0.1877
zero-shot MiniLM top-5 macro recall  = 0.2332
zero-shot MiniLM top-20 macro recall = 0.3035
BM25 top-100 macro recall            = 0.4188
```

The classifier needs enough evidence context to make the label decision. This
means the best final evidence output is not necessarily the best classifier
input.

### 4. `REFUTES` Appears Especially Weak

The current retrieval setup has weak `REFUTES` recall. This matters because the
classifier must distinguish between support, refutation, insufficient evidence,
and disputed claims. If refuting evidence is missing, the classifier may learn
or predict the wrong label.

### 5. Fine-Tuning Has Not Yet Improved The Reranker

The hard-negative strategy is reasonable, but the first fine-tuning run did not
beat the zero-shot MiniLM baseline. Possible reasons:

- too few epochs or insufficient hyperparameter search;
- negative sampling may still be noisy;
- some gold labels may be incomplete or sparse;
- reranker training objective may not match evidence-set evaluation;
- BM25 candidate recall may be the larger bottleneck.

## Current Working Hypothesis

The most important bottleneck is recall, not only top-k F-score.

For the next stage, we should separate two goals:

```txt
Goal A: final evidence output
  Need small, high-precision evidence list.

Goal B: classifier evidence context
  Need wider, high-recall evidence context.
```

The system should probably use a two-stage or multi-stage pipeline:

```txt
high-recall candidate generation
-> stronger reranking / filtering
-> compact final evidence output
-> wider classifier context
```

## Candidate Directions For Next Experiments

### Direction 1: Improve Candidate Generation Recall

Try candidate generators beyond a single BM25 ranking:

- BM25 top-100 or top-200;
- BM25 + TF-IDF union;
- BM25 with alternative tokenization or query expansion;
- claim keyword/entity expansion;
- dense retrieval candidates if allowed by the assignment rules;
- hybrid sparse+dense retrieval.

Primary metric:

```txt
candidate pool macro recall
hit-any rate
all-gold rate
```

### Direction 2: Rerank Larger Candidate Pools

Instead of reranking only BM25 top-50, test:

```txt
BM25 top-100 -> MiniLM reranker
BM25 top-200 -> MiniLM reranker
hybrid top-N union -> MiniLM reranker
```

The tradeoff is runtime. However, if this is only for offline experiment and
final Colab execution remains acceptable, it may be worth it.

### Direction 3: Better Hard Negatives

The next supervised reranker should use more carefully designed negatives:

- high-ranked non-gold negatives from the zero-shot reranker;
- negatives that share entities/numbers with the claim;
- negatives from the same broad topic but wrong relation;
- class-aware negatives, especially for `REFUTES`;
- avoid treating possible unlabeled-but-valid evidence as too many hard
  negatives if labels are incomplete.

### Direction 4: Add Fine-Grained Semantic Signals

We have started extracting lightweight semantic features:

- entities;
- percentages;
- quantities;
- years;
- negation cues;
- comparison cues;
- causality cues;
- relation verbs.

These should not replace the neural model. They may help as:

- analysis features;
- reranker auxiliary features;
- classifier input summaries;
- filters for obvious entity/number mismatches.

### Direction 5: Build The Classifier With Wider Evidence Context

Because final evidence F-score is low, the classifier should initially receive a
wider context:

```txt
MiniLM top-20 evidence context
or BM25/MiniLM top-50 evidence context
```

Then compare:

- top-3 context;
- top-10 context;
- top-20 context;
- top-50 context;
- with and without semantic summaries.

## What We Need Help With

If asking for external help, the question should not be vague like:

```txt
How do we improve retrieval?
```

A better question is:

```txt
We are doing evidence retrieval for claim verification. Our current BM25
candidate recall is low, and a MiniLM cross-encoder reranker improves top-k
F-score but still retrieves many topically related distractors. The best final
evidence F-score is about 0.164 on dev, while BM25 top-100 macro recall is only
about 0.419. We need a practical pipeline that improves recall enough for a
downstream classifier, while still allowing a compact final evidence output.

What candidate generation and reranking strategy would you try next under a
Colab-friendly assignment setting? In particular, how should we handle hard
negatives, larger candidate pools, and topic-relevant but evidence-irrelevant
distractors?
```

## External Help Prompt

Use this prompt when asking another person, tutor, or model for advice:

```txt
We are working on a fact-checking assignment with two subtasks: evidence
retrieval and claim classification. For each claim, we need to retrieve evidence
passages from a large corpus, then classify the claim as SUPPORTS, REFUTES,
NOT_ENOUGH_INFO, or DISPUTED.

Current retrieval results are weak:
- TF-IDF top-3 F-score: 0.0533
- BM25 top-5 F-score: 0.0772
- BM25 top-50 macro recall: 0.3249
- BM25 top-100 macro recall: 0.4188
- zero-shot cross-encoder/ms-marco-MiniLM-L6-v2 reranking BM25 top-50:
  top-3 F-score 0.1642, top-3 macro recall 0.1877
- MiniLM top-20 gives higher recall, around 0.3035, but poor final precision
  and low final F-score.

The main error pattern is that many false positives are topic-relevant
distractors. They share entities, keywords, numbers, or climate-related terms
with the claim, but they do not actually help support/refute/decide the claim.
For example, a claim about Australian carbon emissions retrieves passages about
Australia, CO2, and percentages, but misses the exact evidence needed for the
fact-checking decision.

We have tried:
1. TF-IDF baseline.
2. BM25 baseline.
3. BM25 top-50 + zero-shot MiniLM cross-encoder reranking.
4. MiniLM BCE fine-tuning with BM25 negatives.
5. Task-aware hard negatives mined from high-ranked MiniLM non-gold candidates.

The fine-tuned reranker has not beaten the zero-shot MiniLM baseline yet.

Our current hypothesis is that recall is the bottleneck. For final evidence
output, a small top-k list is needed, but for the downstream classifier we may
need a wider high-recall evidence context.

Question:
What should we try next to improve retrieval recall and downstream
classification performance in a Colab-friendly pipeline? Would you recommend a
hybrid sparse+dense candidate generator, larger BM25 candidate pools, query
expansion, stronger hard-negative training, semantic/entity/number matching
features, or a different two-stage retrieval/reranking design?

Please focus on practical experiments we can implement and compare quickly.
```

## Suggested Next Local Experiments

Priority order:

1. Measure candidate recall for larger candidate pools:
   `BM25 top-100`, `top-200`, and possibly `top-500`.
2. Build candidate unions:
   `BM25 + TF-IDF`, and possibly alternative BM25 tokenization.
3. Rerank larger pools with MiniLM and compare top-3/top-5/top-10/top-20.
4. Build classifier inputs from wider contexts and test whether label accuracy
   improves even when final evidence F-score stays low.
5. Revisit hard-negative training only after candidate recall improves.

## Success Criteria For The Next Stage

The next stage should produce a comparison table with:

- candidate source;
- candidate pool size;
- reranker;
- final output top-k;
- classifier context top-k;
- evidence F-score;
- precision;
- macro recall;
- hit-any rate;
- all-gold rate;
- class-specific recall, especially `REFUTES`;
- runtime notes;
- decision for final output vs classifier input.

The immediate target is not just a higher final evidence F-score. We also need a
retrieval setup that gives the classifier enough correct evidence to improve
final claim classification.
