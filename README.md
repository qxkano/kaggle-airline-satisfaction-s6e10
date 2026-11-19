# 航空旅客满意度预测课程大作业

官方比赛：https://www.kaggle.com/competitions/playground-series-s6e10

## 文件入口

- `EXPERIMENT_PLAN.md`：在正式训练前确定的实验方案。
- `文档/实验结果报告.md`：实验完成后由实际指标生成的报告。
- `results/cv_summary.csv`：九种模型配置的开发集三折交叉验证结果。
- `results/holdout_metrics.csv`：冻结方案后统一计算的独立留出结果。
- `results/stability_metrics.csv`：固定最佳单模型的三个种子结果。
- `results/selection_frozen.json`：根据开发集确定的方案和权重。
- `results/deterministic_verification.json`：从原始标签与保存预测重新计算的核验。
- `results/figures/`：类别分布、ROC、混淆矩阵、SHAP图。
- `submission.csv`：全量有标签数据训练后的最终竞赛预测。
- `models/`：含预处理的可加载模型；只加载自己信任的joblib文件。
- `logs/experiment_run.log`：实际运行日志。

## 运行

环境：`E:/codex-download/environments/datascience-s6e10`，精确依赖版本见`requirements-lock.txt`。在PowerShell项目目录执行 `& ./run.ps1`，或者直接运行其中的Python命令。run.ps1仅为本次进程设置E盘缓存与线程数，并在训练前执行环境小样本检查。

主脚本会复用已有配置预测缓存。run.ps1先检查已保存的原始文件及代码、方案、依赖清单哈希，发生变化时拒绝复用。**已有缓存仅适用于相同数据、代码、划分及参数**；如果修改上述内容，请先把现有results、models、logs及submission.csv移动到另一个备份目录，建立新的空输出目录，再运行，不可混合旧预测。完整复现从空输出目录开始。直接运行主脚本会绕过入口哈希检查，应优先使用run.ps1。

原始数据位于`data/raw`，包含官方zip及三个CSV。不能将样例提交的常数概率当作测试标签。不得据无标签测试文件计算测试AUC。

## 结果解释边界

开发集OOF用于配置与融合权重选择，有模型选择偏差；独立留出集提供最终本地评价。holdout_metrics.csv对应三个CV模型平均（单个模型只训练约2/3开发集）；stability_metrics.csv对应完整开发集重训。二者训练规模不同，必须分别说明。三种子只反映固定划分下算法随机性。

本任务数据为Playground合成数据，SHAP只解释预测关联，无法证明业务因果。此项目未使用别人的OOF或预测，最终分数来自自行训练。生成submission.csv不代表已上传Kaggle或取得线上分数。
