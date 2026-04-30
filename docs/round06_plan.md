# Round06 Plan

Round06 is designed to produce a decision-ready comparison table for the next
classifier stage.

The main question is:

```txt
Which evidence preprocessing strategy should feed the final claim classifier?
```

## Final Table

The final Round06 table should compare:

- BM25 baseline rows.
- Zero-shot MiniLM reranker rows.
- Task-aware hard-negative fine-tuned reranker rows.
- Candidate pool variants such as BM25 top-50 vs top-100.
- Classifier context variants such as top-3, top-10, top-20, and top-50.
- Optional semantic feature variants.

Core columns:

```txt
method_id
candidate_source
reranker
negative_strategy
semantic_features
final_output_top_k
classifier_context_top_k
retrieval_f_score
precision
macro_recall
micro_recall
hit_any
all_gold
refutes_recall
refutes_hit_any
decision
interpretation
```

## Stage Order

1. Consolidate Round05 baselines into the table.
2. Mine MiniLM-ranked non-gold hard negatives.
3. Train and evaluate task-aware reranker variants.
4. Prototype fine-grained semantic extraction.
5. Build classifier-oriented evidence packages.
6. Write the final Round06 comparison report.

## Expected Decision

Round06 should recommend:

- which method produces final evidence output;
- which wider top-k evidence context should feed the classifier;
- whether semantic features are worth carrying forward;
- what the first Round07 classifier experiment should be.
