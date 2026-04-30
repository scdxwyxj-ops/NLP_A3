# Round04 Report C：Repository structure cleanup

报告时间：`2026-04-30`

## 一、目标

把项目从单个实验脚本整理成更清晰的工程结构：
- `experiments/` 专门放可运行实验脚本。
- `src/a3_factcheck/` 放可复用 pipeline 代码。
- `docs/` 放面向队友和报告的稳定项目文档。
- `configs/` 放实验配置。
- `tests/` 预留轻量测试。

## 二、当前结构

```txt
agent_docs/             Agent memory, round reports, decisions, handoff notes
configs/                Versioned experiment configuration files
data/                   Local course data; do not include in final resource zip
docs/                   Human-facing project documentation
experiments/            Runnable experiment scripts
notebooks/              Course notebook template and final notebook work
outputs/                Generated predictions, metrics, and experiment artifacts
src/a3_factcheck/       Reusable pipeline code
tests/                  Lightweight tests for reusable code
eval.py                 Course evaluation script
README.md               Course/project specification mirror
```

## 三、已完成改动

- 新增 reusable package：
  - `src/a3_factcheck/data.py`
  - `src/a3_factcheck/evaluation.py`
  - `src/a3_factcheck/retrieval/tfidf.py`
- 新增实验入口：
  - `experiments/retrieval/tfidf_baseline.py`
- 保留旧入口兼容：
  - `src/tfidf_retrieval_baseline.py`
- 新增项目文档：
  - `docs/README.md`
  - `docs/project_structure.md`
  - `docs/experiments.md`
- 新增配置：
  - `configs/retrieval/tfidf_baseline.json`
- 新增工程元数据：
  - `pyproject.toml`
  - `.gitignore`
  - `tests/.gitkeep`

## 四、运行方式

推荐新命令：

```bash
PYTHONPATH=src python experiments/retrieval/tfidf_baseline.py \
  --top-k-values 1,3,5,10,20 \
  --output outputs/round04/dev-tfidf.json
```

旧命令仍可用：

```bash
python src/tfidf_retrieval_baseline.py --help
```

## 五、验证

已验证：
- Python compile 通过。
- 新实验入口 `--help` 可运行。
- 旧 wrapper `--help` 可运行。

未重新完整运行 TF-IDF baseline，因为完整运行会重新向量化 1,208,827 条 evidence；旧结果仍保留在 `outputs/round04/`。

## 六、后续建议

- 新增 BM25 时放在：
  - reusable code：`src/a3_factcheck/retrieval/bm25.py`
  - experiment entry：`experiments/retrieval/bm25_baseline.py`
  - config：`configs/retrieval/bm25_baseline.json`
- 如果队友给 Drive access，先审计已有结构，再决定是否把当前结构同步到 GitHub/Drive。
