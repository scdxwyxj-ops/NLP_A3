# Unknowns

## Blocking questions

None for planning. Implementation can begin with task-aware hard negative mining.

## Non-blocking questions

- Will task-aware fine-tuning beat zero-shot MiniLM, or only improve classifier input quality?
- Should semantic extraction be done with lightweight regex/NLP features or a small open-source sequence model?
- What top-k context will optimize downstream classifier accuracy: top-10, top-20, or top-50?
- How much evidence context can the final classifier handle in Colab?

## Assumptions made

- Round06 should optimize for final classification usefulness, not only retrieval F-score.
- Generated models and outputs remain local and ignored by git.
- Zero-shot MiniLM remains the strongest fixed baseline until task-aware fine-tuning proves otherwise.
- REFUTES retrieval weakness deserves explicit reporting.

## Human decisions required

- Decide after Round06 whether to move to classifier implementation or continue retrieval recall work.
