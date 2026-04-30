# Round03 索引：数据资产、template、baseline 可运行化

报告时间：`2026-04-29`

## 一、本轮总目标
- 收集课程提供的 Assignment 3 数据文件、notebook template 和 `eval.py`。
- 确认本地目录中每个必需资产的位置、用途和提交边界。
- 跑通课程 baseline evaluation，建立后续 retrieval/classification 实验的最低可复现起点。

## 二、本轮当前状态
- 状态：`进行中`
- 当前 stage：`Stage C：跑通 baseline evaluation`
- 当前结论：课程数据、`eval.py` 和 notebook template 已下载到本地；baseline evaluation 已跑通。
- 当前阻塞：无技术阻塞；下一步是进入 Round04 retrieval baseline，或先把 notebook template 改成小组工作副本名称。

## 三、Stage 划分
### Stage A：收集课程文件
报告：

```txt
agent_docs/rounds/round_03/round_03_report_a_data_assets_audit.md
```

状态：`已达标`

验收指标：
- 项目目录或约定路径中存在 `train-claims.json`、`dev-claims.json`、`test-claims-unlabelled.json`、`evidence.json`、`dev-claims-baseline.json`、`eval.py`。
- 记录数据文件来源和存放位置。
- 明确数据文件不应放入最终 zip，除非课程明确要求。

### Stage B：建立 notebook template 工作副本
状态：`已达标`

验收指标：
- 课程指定 `.ipynb` template 已复制或保存为小组工作版本。
- notebook 可以在 Colab 打开。
- notebook 中保留课程要求的核心结构。

### Stage C：跑通 baseline evaluation
状态：`已达标`

验收指标：
- `python eval.py --predictions dev-claims-baseline.json --groundtruth dev-claims.json` 能跑通。
- 记录 baseline 的 F-score、accuracy、harmonic mean。
- 明确 dev prediction JSON 的 required schema。

## 四、下一步
1. 开启 Round04：实现 TF-IDF/BM25 retrieval baseline，记录 dev retrieval F-score。
2. 在 notebook template 中加入数据读取、baseline evaluation 和后续 retrieval/classification pipeline 代码。
3. 与队友确认任务分工和是否使用 GitHub/Drive 共享代码与实验日志。
