import os
os.environ.setdefault('MPLCONFIGDIR','E:/codex-download/tmp/matplotlib-s6e10')
import importlib.util,json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
R=Path('E:/codex-download/项目/数据科学课程大作业-S6E10')
spec=importlib.util.spec_from_file_location('experiment',R/'src/run_experiments.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
completion=json.loads((R/'results/completion.json').read_text(encoding='utf-8'))
cfgs=json.loads((R/'results/configurations.json').read_text(encoding='utf-8'))
test=pd.read_csv(R/'data/raw/test.csv',nrows=1000);p=np.zeros(len(test));checks=[]
for item in completion['final_refits']:
    cfg=next(c for c in cfgs if c['name']==item['name']);model=joblib.load(R/'models'/f"{item['name']}_full.joblib")
    pp=model.predict_proba(module.feature_frame(test,cfg['fe']))[:,1];p+=item['weight']*pp
    saved=np.load(R/'results/predictions'/f"{item['name']}_competition_test.npz")['p'][:len(test)]
    error=float(np.max(np.abs(pp-saved)));assert error<1e-10
    checks.append({'model':item['name'],'rows':len(test),'max_abs_prediction_difference':error})
sub=pd.read_csv(R/'submission.csv',nrows=len(test));error=float(np.max(np.abs(p-sub.satisfaction)))
assert error<1e-10
(R/'results/model_reload_verification.json').write_text(json.dumps({'status':'PASS','checks':checks,'blend_max_abs_difference':error},indent=2),encoding='utf-8')
cv=pd.read_csv(R/'results/cv_summary.csv').set_index('model');hm=pd.read_csv(R/'results/holdout_metrics.csv').set_index('model')
stable=pd.read_csv(R/'results/stability_metrics.csv');shap=pd.read_csv(R/'results/shap_importance.csv')
cm=np.asarray(json.loads((R/'results/confusion_matrix.json').read_text(encoding='utf-8')))
fig,ax=plt.subplots(figsize=(5.5,4.5));im=ax.imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted satisfaction',ylabel='Actual satisfaction',title='Frozen blend: threshold 0.5')
fig.colorbar(im);fig.tight_layout();fig.savefig(R/'results/figures/confusion_matrix.png',dpi=180);plt.close(fig)
pred=pd.read_csv(R/'results/predictions/holdout_predictions.csv')
fig,ax=plt.subplots(figsize=(7.5,5.5))
for n,label,color,style in [('LogisticRegression','Logistic regression','#0072B2','-'),('RandomForest','Random forest','#E69F00','--'),('LightGBM_700_L63','LightGBM tuned','#009E73','-.'),('CatBoost_tuned','CatBoost tuned','#CC79A7',':'),('Selected_LGB_Cat_blend','Frozen blend','#D55E00','-')]:
    fpr,tpr,_=roc_curve(pred.y_true,pred[n]);ax.plot(fpr,tpr,label=f'{label}: {hm.loc[n,"auc"]:.6f}',color=color,linestyle=style,lw=1.5)
ax.plot([0,1],[0,1],'k--',lw=.8);ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Independent holdout ROC (CV-model averaging)');ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(R/'results/figures/roc_curves.png',dpi=180);fig.savefig(R/'results/figures/roc_curves.svg');plt.close(fig)
sel=completion['selected'];best=sel['best_single'];b=sel['blend'];h=hm.loc['Selected_LGB_Cat_blend']
lines=['# 实验结论与论文写作依据','',
       '本文件只总结已完成实验，供后续课程论文使用。全部指标均为本地评价，未取得Kaggle线上评分。','',
       '## 已完成的工作','',
       '- 官方压缩包含699635条有标签训练记录、299844条无标签竞赛测试记录及提交样例；实际建模21个特征，排除id。',
       '- 292个到达延误缺失值；基础逻辑回归/随机森林/LightGBM在折内填补，CatBoost使用原生数值缺失处理。',
       '- 559708条开发数据与139927条独立留出数据；开发集共享三折划分，九配置共27次折内训练。',
       '- 已完成有限参数比较、两组匹配的特征工程消融、五个融合权重比较、最佳单模型三种子稳定性、SHAP和错误案例分析。',
       '- 最终融合两组件已在全部699635条标签数据上重训，生成299844行submission.csv。','',
       '## 主要发现','',
       f"1. 树模型优于线性基线：逻辑回归OOF AUC={cv.loc['LogisticRegression','oof_auc']:.6f}，随机森林={cv.loc['RandomForest','oof_auc']:.6f}，基础LightGBM={cv.loc['LightGBM_base','oof_auc']:.6f}，基础CatBoost={cv.loc['CatBoost_base','oof_auc']:.6f}。",
       f"2. 本次有限配置中最佳单模型是{best}，OOF AUC={cv.loc[best,'oof_auc']:.6f}，较基础LightGBM增加{cv.loc[best,'oof_auc']-cv.loc['LightGBM_base','oof_auc']:.6f}。不能据此称其在所有参数配置下均优于其他算法。",
       f"3. 构造特征未带来提升：31叶对照的OOF AUC差为{cv.loc['LightGBM_FE_L31','oof_auc']-cv.loc['LightGBM_700_L31','oof_auc']:.6f}；63叶对照的差为{cv.loc['LightGBM_FE_L63','oof_auc']-cv.loc['LightGBM_700_L63','oof_auc']:.6f}。这些结果支持本次特征组合未改善当前配置，而非所有特征工程均无效。",
       f"4. 开发集选择的融合为LightGBM权重{b['lgb_weight']:.2f}、CatBoost权重{b['cat_weight']:.2f}；OOF AUC={b['oof_auc']:.6f}，比最佳单模型增加{b['oof_auc']-sel['best_single_oof_auc']:.6f}。",
       f"5. 冻结融合方案的独立留出AUC={h.auc:.6f}，Accuracy={h.accuracy:.6f}，F1={h.f1:.6f}，Log Loss={h.log_loss:.6f}。相对同口径最佳单模型的AUC增加{h.auc-hm.loc[best,'auc']:.6f}，属于较小的点估计差异，尚未检验统计显著性。",
       f"6. 最佳单模型在完整开发集上用42、2026、3407三种子重训，留出AUC均值={stable.auc.mean():.6f}，样本标准差={stable.auc.std(ddof=1):.6f}。只能说明固定划分下的算法随机性较小，不能据此称对不同数据划分或其他数据集都稳健。",
       '7. SHAP排名前三的特征为：'+ '、'.join(shap.feature.head(3))+ '。它们是模型预测的重要依据；SHAP不是因果效应，也不能直接推导航空公司服务改进的收益。','',
       '## 必须在论文中交代的评价口径','',
       '模型选择和融合权重仅依据开发集OOF；该开发分数存在模型选择乐观偏差。独立留出评分在selection_frozen.json保存后统一计算，没有据留出评分调整配置。',
       '描述性EDA在划分前计算了完整有标签数据的分布和分组满意率，因此不能写“从未查看任何留出标签”。正确表述是“留出集模型评分不参与方案选择”。构造特征为预先列出的行内计算，不从留出标签估计参数。',
       'holdout_metrics.csv对应三个CV模型平均后再做融合；单个CV模型训练约373139条记录。stability_metrics.csv对应559708条完整开发集训练；最终submission的组件对应699635条全量训练。三个口径不得混写。',
       '原定复现入口run.ps1先检查数据与代码哈希。直接运行主脚本会绕过该入口保护，修改数据/代码/参数后必须使用全新输出目录。',
       '数据为Playground合成数据，研究仅覆盖一个数据集和有限配置；不能直接宣称已验证真实航空业务效果，也不能把线上未知分数写为0.960010。','',
       '## 可用于论文的图表与证据','',
       '- 字段与数据质量：results/field_summary.csv、data_profile.json。',
       '- 模型比较：results/cv_summary.csv、fold_metrics.csv、holdout_metrics.csv。',
       '- 消融与融合：两组FE对照、results/blend_search.csv与selection_frozen.json。',
       '- 稳定性与解释：results/stability_metrics.csv、shap_importance.csv、error_cases.csv。',
       '- 图：results/figures中的类别分布、ROC、混淆矩阵、SHAP；图件已检查文字及布局。',
       '- 可复现性：原始文件SHA256、splits.npz、requirements-lock.txt、src及models。',
       '- 核验：results/deterministic_verification.json、model_reload_verification.json及单独审查报告。']
(R/'文档/实验结论与论文写作依据.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
report=R/'文档/实验结果报告.md'
text=report.read_text(encoding='utf-8')
text=text.replace('留出标签只在selection_frozen.json写出后评价，未据此修改方案。','留出集的模型评分只在selection_frozen.json写出后计算，未据此修改方案。划分前的描述性EDA已读取完整训练数据标签并输出分组满意率；因此不能表述为“从未查看留出标签”。')
report.write_text(text,encoding='utf-8')
print('MODEL_RELOAD_PASS',json.dumps(checks));print('INTERPRETATION_SAVED')
