# Confusion Matrix Figures

Each PNG is one evidence-level binary confusion matrix for a retrieval/reranking method.

| Method | TP | FP | FN | TN | Precision | Recall | Image |
|---|---:|---:|---:|---:|---:|---:|---|
| bm25_top5 | 50 | 720 | 441 | 6828 | 0.065 | 0.102 | [bm25_top5_confusion_matrix.png](bm25_top5_confusion_matrix.png) |
| minilm_zero_top3 | 78 | 384 | 413 | 7164 | 0.169 | 0.159 | [minilm_zero_top3_confusion_matrix.png](minilm_zero_top3_confusion_matrix.png) |
| minilm_zero_top5 | 102 | 668 | 389 | 6880 | 0.132 | 0.208 | [minilm_zero_top5_confusion_matrix.png](minilm_zero_top5_confusion_matrix.png) |
| minilm_zero_top20 | 142 | 2938 | 349 | 4610 | 0.046 | 0.289 | [minilm_zero_top20_confusion_matrix.png](minilm_zero_top20_confusion_matrix.png) |
| minilm_bce_top5 | 67 | 703 | 424 | 6845 | 0.087 | 0.136 | [minilm_bce_top5_confusion_matrix.png](minilm_bce_top5_confusion_matrix.png) |
| task_aware_neg10_lr5e6_ep05_top3 | 77 | 385 | 414 | 7163 | 0.167 | 0.157 | [task_aware_neg10_lr5e6_ep05_top3_confusion_matrix.png](task_aware_neg10_lr5e6_ep05_top3_confusion_matrix.png) |
| task_aware_neg10_lr5e6_ep05_top5 | 98 | 672 | 393 | 6876 | 0.127 | 0.200 | [task_aware_neg10_lr5e6_ep05_top5_confusion_matrix.png](task_aware_neg10_lr5e6_ep05_top5_confusion_matrix.png) |
| task_aware_neg10_lr5e6_ep05_top20 | 140 | 2940 | 351 | 4608 | 0.045 | 0.285 | [task_aware_neg10_lr5e6_ep05_top20_confusion_matrix.png](task_aware_neg10_lr5e6_ep05_top20_confusion_matrix.png) |
