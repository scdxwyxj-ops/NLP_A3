# Retrieval Figure Data Audit (third_meeting)

目标：把 second_meeting 图表结构映射到当前 `Group_131_COMP90042_Project_2026.ipynb` 的提交版逻辑（candidate/top500 sparse fusion、top64 embedding+shallow、top3 CE 预过滤融合、Enhanced Context20），并只变更方法名、参数、数值与注释，不改 tutorial 文件。

## 现有可用数据

- `colab_notebooks/outputs/self_generated/run_summary.json`（存在）：有 `candidate / top64 / top3` 的单点指标（仅 @500/@64/@3）。
- `colab_notebooks/outputs/self_generated/final_predictions.json`（存在）：仅最终提交结果。
- `group_meetings/third_meeting /tutorial.ipynb`：图像路径和章节结构与 second_meeting 对齐，但未改内容。

## 缺失的中间 ranked artifacts（`SAVE_ARTIFACTS=1` 需要补跑）

- `colab_notebooks/outputs/self_generated/train_sparse_fusion_top500.json`
- `colab_notebooks/outputs/self_generated/target_sparse_fusion_top500.json`
- `colab_notebooks/outputs/self_generated/candidate_metrics.json`
- `colab_notebooks/outputs/self_generated/train_top64_embedding_hand.json`
- `colab_notebooks/outputs/self_generated/target_top64_embedding_hand.json`
- `colab_notebooks/outputs/self_generated/top64_metrics.json`
- `colab_notebooks/outputs/self_generated/target_top3_ce_embedding_source_fusion.json`
- `colab_notebooks/outputs/self_generated/top3_metrics.json`

> 不重跑时只能使用 run_summary 的单点数；不满足完整 R-N 曲线重建。

## 逐图审计（重点：candidate / score fusion / top64 / top3）

| 图文件 | 对应段落 | 动作 | 当前是否已有 | 需要的数据 | 建议数据源 | 图标题/方法名替换 |
|---|---|---|---|---|---|---|
| `round18_revised_single_sparse_rn.png` | `# 4. Sparse Gates And Two Fusion Routes` | 保留（方法名更新） | 仅 `run_summary` 有单点 (@500) | 每个单源叶门控的 ranked list（bm25/char/structured/PRF） | rerun 后的 `target_sparse_fusion_top500.json`（若包含源级排序）+（若有）原始叶门控文件 | 标题不变；方法名统一为 `BM25 word/Character n-gram/Structured cue/Query expansion` |
| `round18_revised_sparse_fusion_weights_and_rn.png` | `## 4.1 Score-Based Fusion` | 保留 | 仅 `run_summary` 有单点 | `candidate` 阶段的 ranked list（至少 1,3,5,10,64,100,500...） | `target_sparse_fusion_top500.json`（重跑后） | 标题保持；改成 `Sparse RRF Score Fusion（BM25 0.75 / char 2.0 / structured 0.25 / PRF 0.5，k=500, rrf_k=500）` |
| `round18_revised_union_strategy_rn.png` | `## 4.2 Candidate-Union Fusion` | 删除 | 不符合当前提版本逻辑 | 旧版联合候选池数据 | 无（当前提交版本不使用独立 union 流） | 该图段落整体移除 |
| `round18_revised_union_cost_benefit.png` | `# 5. Why The Final Top-500 Uses Score Fusion` | 删除/重写 | 不符合当前提版本逻辑 | 旧版 union 成本/召回数据 | 无（仅保留可重算 top500 分支） | 建议改为“score fusion 成本-收益（无 union 对照）”或者移除该段 |
| `round18_revised_top64_selector_three_way_bar.png` | `# 7. Top-64 Context Selection` | 保留（改为当前方法） | 仅 `run_summary` 有 @64 单点 | `target_top64_embedding_hand.json` + `top64_metrics.json`（含完整 ranked） | `target_top64_embedding_hand.json`（重跑） | 标题改为 `Top-64 selector: MiniLM embedding inner-product + shallow feature fusion`；权重文字标注 `0.65*embed + 0.35*hand` |
| `round18_revised_shallow_complement_two_panel_rn.png` | `# 8. Shallow Features As A Complement` | 删除/重写 | 缺失可复现实验数据 | 旧 CE/embedding-only 候选对照曲线 | 无（当前提交无独立 CE-only / embedding-only 比较产物） | 当前版本建议移除该图（或替换为单一 top64 方法说明） |
| `round18_revised_top3_gate_cn.png` | `# 9. Top-3 Evidence For Submission` | 保留（改为“最终 CE+融合”） | 仅 `run_summary` 有 @3 单点 | top3 候选 ranked（需 `target_top3_ce_embedding_source_fusion.json`） | `target_top3_ce_embedding_source_fusion.json`（重跑） | 标题改为 `Top-3 final evidence: CE-prefiltered multi-signal fusion` |
| `round18_revised_top3_submission_rn.png` | `# 9. Top-3 Evidence For Submission` | 保留（改为当前方法） | 仅 `run_summary` 有 @3 单点 | top3 ranked（需 full ranked 列表） | `target_top3_ce_embedding_source_fusion.json`（重跑） | 标题改为 `Fused top-3 ranker: CE prefilter + embedding + source signal`；方法名改为 `CE/embedding/source-rank fusion (ce_score 0.25, embedding_score 0.25, ce_rank 0.00, embedding_rank 0.25, source_rank 0.25)` |
| `round18_revised_top3_submission_bar.png` | `# 9. Top-3 Evidence For Submission` | 保留（改为当前方法） | 仅 `run_summary` 有 @3 单点 | top3 @3 评估条目 | `top3_metrics.json`（重跑） | 标题改为 `Final top-3 evidence ranking`；柱状值标签标注 `CE-prefiltered multi-signal fusion` |

## 方法参数（当前提版需落图注）

- **Candidate / sparse top500**：`weights={"bm25":0.75,"char":2.0,"structured":0.25,"prf":0.5}, CANDIDATE_TOP_K=500, SPARSE_RRF_K=500`
- **Top-64**：`top64_k=64`, `top64_score = 0.65 * embedding_norm + 0.35 * hand_prob`, hand model=`HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, class_weight="balanced")`
- **Top-3**：`ce_prefilter_k=min(256,500)=256`, `weights={"ce_score":0.25,"embedding_score":0.25,"ce_rank":0.0,"embedding_rank":0.25,"source_rank":0.25}`, CE model=`cross-encoder/ms-marco-MiniLM-L6-v2`
- **Classifier（本问任务不改）**：`Enhanced Context20 TF-IDF One-vs-Rest Logistic Regression`

## 当前可直接抽出的替换数值（来自 run_summary）

见 `group_meetings/third_meeting /retrieval_figure_metrics.json`：
- candidate@500: `macro_recall=0.6754329004329005`, `hit_any=0.9025974025974026`, `evidence_f=0.008273851035835564`
- top64@64: `macro_recall=0.5783549783549783`, `hit_any=0.8636363636363636`, `evidence_f=0.05173643362054943`
- top3@3: `macro_recall=0.23549783549783548`, `hit_any=0.4675324675324675`, `evidence_f=0.2021026592455164`
