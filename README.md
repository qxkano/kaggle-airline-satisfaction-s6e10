# 航空旅客满意度的数据管理、预测与可视化分析

本项目以 Kaggle Playground Series S6E10 合成数据为例，把官方 CSV 转换为可查询的旅客记录表，训练单棵决策树并逐条生成预测；最后分别分析已观测的满意率、拟合与留出误差、无标签记录的预测分布。

[阅读课程论文](文档/课程论文.md) · [实验结果与核对记录](文档/实验结果报告.md) · [论文各节与证据对照](文档/实验结论与论文写作依据.md)

## 复现课程实验

1. 从 [比赛数据页](https://www.kaggle.com/competitions/playground-series-s6e10/data) 下载官方压缩包，在本项目下新建 `data/raw/`，将 `train.csv` 和 `test.csv` 解压到其中。
2. 使用 Python 3.12 创建并激活虚拟环境。在项目根目录运行 `python -m pip install -r requirements-course.txt`。
3. 依次运行 `python src/course_pipeline.py` 和 `python src/course_tree.py`。Windows PowerShell 也可执行 `./run.ps1`。如果 Python 位于其他目录，使用 `./run.ps1 -Python '解释器完整路径'`。
4. 打开 `results/course/` 查看质量记录、拟合与留出指标、预测结果和分组图。数据库在 `data/processed/airline.sqlite`，可用 SQLite 客户端查询。再次运行会从原始 CSV 重建数据库和汇总结果。重建时需要同时容纳新旧两个数据库文件。

## 从记录到图表

| 环节 | 处理 | 产物 |
|---|---|---|
| 采集与预处理 | 每批读取 50,000 行，校验字段、评分范围和标签，保留到达延误的缺失状态 | [文件指纹与行数核对](results/course/quality.json) |
| 存储与管理 | 使用 id 主键和 split 来源字段，核对入库行数、缺失数和已知标签数 | 本地 SQLite 的 responses 表 |
| 分析方法与工具 | 用 SQL 按舱位汇总，再按出行目的下钻；用 Pandas 接收查询结果 | [舱位](results/course/cabin.csv)、[舱位与出行目的](results/course/cabin_travel.csv)、[到达延误](results/course/arrival.csv) |
| 训练与预测 | 决策树拟合、留出评价、对无标签记录输出概率 | [拟合与留出指标](results/course/fit_holdout.csv)、[预测分组](results/course/prediction_cabin.csv) |
| 可视化分析 | 对照观测数据、训练效果与待预测记录的输出 | [原始分层](results/course/satisfaction_groups.png)、[拟合与留出](results/course/fit_holdout.png)、[预测分布](results/course/prediction_profile.png) |

## 课程决策树预测

从清洗后的 SQLite 读取数据，按满意标签分层留出 20% 评估。一次运行会生成 [评估指标](results/course/tree_metrics.json)、[树的上层规则](results/course/tree_rules.png)、[ROC 与混淆矩阵](results/course/tree_evaluation.png)，并在本地保存 `models/course_tree.joblib` 及 `results/course/tree_predictions.csv`。后者按 id 输出官方测试集中每一行的满意概率。[Word 论文](文档/课程论文.docx) · [PDF 论文](文档/课程论文.pdf)
