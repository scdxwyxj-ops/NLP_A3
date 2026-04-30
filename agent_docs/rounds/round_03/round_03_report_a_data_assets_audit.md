# Round03 Report A：课程文件资产审计

报告时间：`2026-04-29`

## 一、审计目的

本报告接手 Round02 的长期规划，检查 Round03 Stage A 是否已经满足：课程数据、notebook template 和 `eval.py` 是否已在本地项目中可用。

## 二、已检查位置

工作目录：

```txt
/mnt/a/OneDrive/文档/2026/NLP/A3
```

已检查：
- 项目根目录。
- `agent_docs/` 目录。
- 三层以内子目录。
- 所有 `.json`、`.py`、`.ipynb`、`.tex`、`.zip`、`.pdf` 文件。

## 三、当前文件状态

已存在：
- `README.md`
- `agent_docs/`

已下载：
- `data/train-claims.json`
- `data/dev-claims.json`
- `data/test-claims-unlabelled.json`
- `data/evidence.json`
- `data/dev-claims-baseline.json`
- `eval.py`
- `notebooks/GroupID_COMP90042_Project_2026.ipynb`

保留说明：
- `data/evidence.md` 是课程 GitHub 中的下载说明文件，包含 `evidence.json` 的 Google Drive/Canvas 来源。

## 四、课程要求摘要

根据 `README.md`，本项目需要：
- 使用 labelled train/dev claims 开发和调参。
- 使用 `evidence.json` 作为 evidence corpus。
- 每条 claim 至少输出一个 evidence id。
- 输出四分类 label：`SUPPORTS`、`REFUTES`、`NOT_ENOUGH_INFO`、`DISPUTED`。
- 使用 `eval.py` 计算 Evidence Retrieval F-score、Claim Classification Accuracy 和两者的 Harmonic Mean。
- 所有最终实验代码必须放入课程指定 `.ipynb` template。
- 最终系统必须包含 RNN、LSTM、GRU 或 Transformer 等 sequence modelling component。
- 不得使用闭源 API、proprietary models、外部训练/评测数据，或手工修改 predictions。

## 五、Stage A 判定

状态：`已达标`

原因：
- 必需数据文件、`eval.py` 和 notebook template 已放置到本地项目目录。
- `eval.py` 已通过 Python bytecode 编译检查。
- baseline evaluation 已成功运行。

## 六、资产位置

当前文件位置：

```txt
data/train-claims.json
data/dev-claims.json
data/test-claims-unlabelled.json
data/evidence.json
data/dev-claims-baseline.json
eval.py
notebooks/GroupID_COMP90042_Project_2026.ipynb
```

轻量验证结果：

```txt
data/train-claims.json: 1228 records
data/dev-claims.json: 154 records
data/dev-claims-baseline.json: 154 records
data/evidence.json: 1208827 records
data/test-claims-unlabelled.json: 153 records
notebooks/GroupID_COMP90042_Project_2026.ipynb: 9 cells
```

注意：对 `test-claims-unlabelled.json` 只做 JSON 解析和样本数统计，不检查具体内容。

## 七、Baseline evaluation

运行命令：

```bash
python eval.py --predictions data/dev-claims-baseline.json --groundtruth data/dev-claims.json
```

输出：

```txt
Evidence Retrieval F-score (F)    = 0.3377705627705628
Claim Classification Accuracy (A) = 0.35064935064935066
Harmonic Mean of F and A          = 0.3440894901357093
```

## 八、下一步

Round03 已满足关闭条件。下一步建议开启 Round04：
- 建立数据读取和统计脚本。
- 实现 TF-IDF 或 BM25 lexical retrieval baseline。
- 在 dev set 上比较 top-k retrieval F-score，为后续 classifier 提供 evidence candidates。
