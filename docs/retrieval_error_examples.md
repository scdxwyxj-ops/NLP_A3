# Retrieval Error Examples

This note gives a few examples of why the evidence retrieval task is difficult.
The current baselines often find passages that are topically related to the
claim, but still not useful enough as gold fact-checking evidence.

## Current Baseline Scores

| Method | Setting | Evidence F-score |
|---|---|---:|
| TF-IDF | top-3 | 0.0533 |
| BM25 | top-5 | 0.0772 |
| MiniLM cross-encoder | BM25 top-50 -> rerank top-3 | 0.1642 |
| Fine-tuned MiniLM BCE | top-5 | 0.1044 |
| Task-aware fine-tuned MiniLM | top-3 | 0.1599 |

The strongest current retrieval method is still the zero-shot MiniLM
cross-encoder reranker. The absolute score is not high, which suggests that
retrieval remains a major bottleneck before claim classification.

## Example 1: Topical Match But Wrong Evidence

Claim:

> when 3 per cent of total annual global emissions of carbon dioxide are from
> humans and Australia produces 1.3 per cent of this 3 per cent, then no amount
> of emissions reduction here will have any effect on global climate.

The model often retrieves passages about Australia, carbon dioxide, global
emissions, or percentages. These are topically similar, but they do not
necessarily address the exact reasoning needed by the claim.

Observed issue:

- false positives contain related keywords such as `Australia`, `carbon
  dioxide`, `emissions`, and percentages
- missed gold evidence includes broader evidence about human CO2 emissions,
  anthropogenic emissions, and country-level contribution

This shows the difference between topic relevance and evidence relevance.

## Example 2: Similar Surface Words Are Not Enough

Claim:

> CO2 limits won't cool the planet.

The lexical baselines can retrieve passages containing `CO2`, `cool`, or
`planet`, but several of these passages are not actually useful evidence for
the claim. Some are only loosely related to cooling mechanisms, while gold
evidence may involve atmospheric energy balance, global warming definitions, or
CO2 exposure limits.

Observed issue:

- keyword overlap is high
- semantic usefulness is still weak
- the model needs to understand the relation between `CO2 limits`, cooling, and
  the claim's intended argument

## Example 3: Rerankers Still Prefer Topic-Relevant Distractors

Claim:

> The claim that 97 percent of scientists believe humans are causing climate
> change has been debunked by the "head" of the United Nations
> Intergovernmental Panel on Climate Change.

The model can retrieve passages about the IPCC, the United Nations, or climate
science, but the gold evidence is more specific: it needs evidence about the
97% consensus claim and whether scientists agree that warming is human-caused.

Observed issue:

- many false positives are about IPCC-related entities
- missed gold evidence is closer to the actual proposition being checked
- this suggests the reranker needs more task-aware negatives, not only generic
  semantic similarity

## Implication For Next Experiments

The next stage should test stronger pipelines rather than relying on one
retriever:

1. Use a high-recall candidate generator first.
2. Use a stronger reranker or a task-aware reranker second.
3. Pass a wider but ranked evidence context into the final classifier.
4. Compare whether hard negatives and semantic feature summaries help the
   classifier choose the correct label.

The goal is not only to retrieve text about the same topic. The model needs to
retrieve evidence that helps support, refute, or decide the claim.
