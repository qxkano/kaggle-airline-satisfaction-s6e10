# 航空旅客满意度的数据管理、建模预测与可视化分析

这个数据科学课程作业用 Kaggle 的航空旅客满意度合成数据做实验。先校验并清洗原始 CSV，再把记录写入 SQLite 数据库。用有满意度标签的记录训练一棵决策树，给没有标签的记录预测满意概率。

[查看课程论文（Word）](文档/数据科学课程论文.docx)

## 怎样运行

1. 从 Kaggle Playground Series S6E10 比赛数据页下载官方压缩包，把 `train.csv` 和 `test.csv` 解压到项目的 `data/raw/`。
2. 建议使用 Python 3.12。在项目根目录安装依赖：

   ```powershell
   python -m pip install -r requirements-course.txt
   ```

3. 在 PowerShell 运行：

   ```powershell
   .\run.ps1
   ```

   如果要指定 Python，使用 `.\run.ps1 -Python '解释器完整路径'`。也可以依次运行 `python src/course_pipeline.py` 和 `python src/course_tree.py`。

4. 清洗后的数据库在 `data/processed/airline.sqlite`，图表和预测结果在 `results/course/`。重跑时，程序会校验新库后再替换旧库，因此需要能同时存放两个数据库文件的磁盘空间。

## 数据和模型

程序每次读取 50,000 行，检查列名、服务评分和标签。到达延误缺失时，数据库保留空值，训练时用拟合集的中位数填补，并单独标记原来是否缺失。数据库用 `id` 唯一标识每条记录，用 `split` 区分数据来源。输入文件指纹和入库行数见[数据核对记录](results/course/quality.json)。

有标签的 699,635 条记录按满意度分层划分。559,708 条用于拟合，139,927 条留到最后评估。决策树最大深度为 6，每个叶节点至少有 200 条拟合记录。留出集的 ROC-AUC（衡量满意旅客被排在前面的能力）为 **0.9444**，准确率为 **90.17%**。预处理器与模型保存在本地的 `models/course_tree.joblib`。

官方测试集的 299,844 条记录没有满意度标签。程序按 `id` 输出[逐条预测概率](results/course/tree_predictions.csv)，用来描述模型的预测分布。

## 论文图表对应的实验文件

| 论文图 | 对应的仓库图 | 核对用数据 |
|---|---|---|
| 图 1 按舱位、出行目的和到达延误分组的实际满意率 | [分组图](results/course/satisfaction_groups.png) | [舱位](results/course/cabin.csv)、[舱位与出行目的](results/course/cabin_travel.csv)、[延误](results/course/arrival.csv) |
| 图 2 拟合集和留出集的指标对比 | [指标对比图](results/course/fit_holdout.png) | [拟合与留出指标](results/course/fit_holdout.csv) |
| 图 3 留出集混淆矩阵 | [留出集评估图](results/course/tree_evaluation.png) | [评估指标及混淆矩阵](results/course/tree_metrics.json) |
| 图 4 决策树深度与排序能力 | [深度曲线](results/course/tree_depth_auc.png) | [各深度 AUC](results/course/tree_depth_auc.csv) |
| 图 5 预测概率分布与分舱位汇总 | [预测分布图](results/course/prediction_profile.png) | [概率区间统计](results/course/prediction_bins.csv)、[分舱位统计（表 3）](results/course/prediction_cabin.csv) |

论文中的图按正文版式排版。仓库保留脚本输出的对应图，其中留出集评估图同时画了 ROC 曲线和混淆矩阵。论文表 1 和表 2 对应数据核对记录及数据处理脚本。
