# 缺失 ablation 输入清单（用于复刻 second tutorial）

当前 artifacts 仅有：
- `target_sparse_fusion_top500.json`（Sparse RRF 融合后的最终候选池）
- `target_top64_embedding_hand.json`（Top-64 embedding+shallow 融合后的最终候选池）
- `target_top3_ce_embedding_source_fusion.json`（Top-3 CE+embedding+source 融合后的最终候选池）

下面这些曲线需要单独的 ranked pool 才能复现；现在缺失，不能由现有 artifacts 可靠推导：

| 缺失线 | 说明 | 应在 notebook 的保存点 |
|---|---|---|
| BM25 word | 单源词项 BM25 检索排位曲线 | Sparse 候选阶段：对每个 claim 直接保存 BM25 排序列表（如 `target_sparse_bm25_top500.json`） |
| Character n-gram | 单源字符 n-gram 检索排位曲线 | Sparse 候选阶段：对每个 claim 直接保存 char n-gram 排序列表（如 `target_sparse_char_top500.json`） |
| Structured | 单源 structured cue 检索排位曲线 | Sparse 候选阶段：对每个 claim 直接保存 structured 排序列表（如 `target_sparse_structured_top500.json`） |
| PRF | Query expansion/PRF 检索排位曲线 | Sparse 候选阶段：对每个 claim 直接保存 PRF 扩展检索排序列表（如 `target_sparse_prf_top500.json`） |
| Embedding-only | 顶层 Top-64 阶段只用 embedding 分排序 | Top64 选择器阶段：从同一候选母池另存 embedding 评分排名（如 `target_top64_embedding_only_top500.json`） |
| Shallow-only | 顶层 Top-64 阶段只用 shallow hand 特征排序 | Top64 选择器阶段：从同一候选母池另存 shallow 评分排名（如 `target_top64_shallow_only_top500.json`） |
| CE-only | Top-3 阶段只用 CE score 排序 | Top3 CE 预筛阶段：保存 CE 单路排序（如 `target_top3_ce_only_top500.json`） |
| Source-rank-only | Top-3 阶段只按 source rank 排序 | Top3 生成阶段：保存 source-rank 单路排序（如 `target_top3_source_rank_only_top500.json`） |

建议这些文件均保存为与现有输出同格式：`claim_id -> [{evidence_id, score, rank, ...}]`，以便直接复用统一的 `recall_at`/曲线重算逻辑。
