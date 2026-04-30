# Mission

Plan and execute Round06 so it produces a rigorous comparison table and detailed
analysis for the next project step: improving final claim classification through
better evidence preprocessing.

Round06 should compare:

- Round05 baselines: BM25 and zero-shot MiniLM reranking.
- Task-aware hard-negative fine-tuned rerankers.
- Candidate pool variants such as BM25 top-50 vs top-100.
- Classifier input packaging variants such as top-3/top-10/top-20/top-50.
- Optional fine-grained semantic feature summaries.

The final output must be decision-ready: a table with retrieval, recall,
error-analysis, and classifier-oriented metrics, plus written analysis explaining
which configuration should feed Round07 classifier work.
