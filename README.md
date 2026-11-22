# 航空旅客满意度的数据管理、建模预测与可视化分析

本项目基于 Kaggle Playground Series S6E10 合成数据集，围绕数据科学课程的核心流程，完成了从原始 CSV 校验清洗、SQLite 关系型入库、单棵决策树分类建模与批量预测，到多维度可视化分析与结果核对的完整实验。


[查看课程论文（Word）](文档/数据科学课程论文.docx)

## 实验复现步骤

1. 前往 Kaggle Playground Series S6E10 比赛数据页下载官方压缩包，在项目根目录下新建 `data/raw/` 文件夹，并将 `train.csv` 与 `test.csv` 解压至该目录。
2. 建议使用 Python 3.12 创建并激活独立虚拟环境，在项目根目录执行以下命令安装依赖：
   ```bash
   python -m pip install -r requirements-course.txt
   ```
3. 依次运行数据处理与建模脚本：
   ```bash
   python src/course_pipeline.py
   python src/course_tree.py
   ```
   在 Windows PowerShell 环境下也可直接执行 `./run.ps1` 一键运行；如需指定其他路径的 Python 解释器，可执行 `./run.ps1 -Python '解释器完整路径'`。
4. 运行完成后，可在 `results/course/` 目录查看数据质量校验记录、拟合与留出集评估指标、无标签测试集预测结果及全部统计图表；清洗后的明细数据库保存在 `data/processed/airline.sqlite`，支持直接使用 SQLite 客户端进行 SQL 查询。重新运行流水线时会先写入临时库并通过完整性核验后再替换旧库，因此磁盘需预留足够空间以同时容纳新旧两个数据库文件。

## 数据处理与分析流程

| 课程环节 | 核心处理逻辑 | 主要输出产物 |
| :--- | :--- | :--- |
| 数据采集与预处理 | 按每批 50,000 行流式读取原始 CSV，校验字段名、服务评分范围（0—5）与非负数值约束，保留到达延误缺失状态 | `quality.json`（SHA-256 文件指纹与行数核对） |
| 数据存储与管理 | 以 `id` 为主键、`split` 区分来源写入 SQLite，强制约束标签取值并建立 `(split, "Class", "Type of Travel")` 组合索引 | `data/processed/airline.sqlite`（`responses` 表） |
| 探索性分析与统计 | 通过 SQL 按舱位、出行目的与到达延误区间分组聚合，交由 Pandas 整理制表与绘图 | `cabin.csv`、`cabin_travel.csv`、`arrival.csv` |
| 模型训练与预测 | 基于信息熵准则训练最大深度为 6 的决策树，在 20% 分层留出集上评估并对无标签测试集逐条评分 | `fit_holdout.csv`、`tree_metrics.json`、`tree_predictions.csv` |
| 可视化与结果核对 | 对照原始观测分组差异、模型拟合与留出表现、树深边际贡献及待预测样本概率分布 | 论文图 1—5 及 `tree_rules.png` |

## 决策树建模与输出说明

建模脚本从清洗后的 SQLite `responses` 表中提取 699,635 条有标签记录，按满意度标签分层留出 20%（139,927 条）作为评估集，其余 80%（559,708 条）用于拟合。模型采用信息熵分裂准则，限定最大深度为 6、叶节点最小样本数为 200，最终生成含 58 个叶节点的决策树，留出集 ROC-AUC 为 0.9444，准确率为 90.17%。

单次运行除输出评估指标、树的上层分裂规则（`tree_rules.png`）、ROC 曲线与混淆矩阵（`tree_evaluation.png`）外，还会将预处理器与模型打包保存至 `models/course_tree.joblib`，并在 `results/course/tree_predictions.csv` 中按 `id` 输出官方测试集全部 299,844 条记录的预测满意概率。
