"""S6E10 experiments. Run from any directory; data and artifacts stay on E:."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('MPLCONFIGDIR','E:/codex-download/tmp/matplotlib-s6e10')
import argparse, copy, hashlib, json, platform, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score, precision_score,
                            recall_score, f1_score, roc_curve, confusion_matrix)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

ROOT=Path('E:/codex-download/项目/数据科学课程大作业-S6E10')
RAW=ROOT/'data/raw'; OUT=ROOT/'results'; MOD=ROOT/'models'
CAT=['Gender','Customer Type','Type of Travel','Class']
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
         'Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment',
         'On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def metric(y,p):
    q=np.asarray(p)>=.5
    return {'auc':roc_auc_score(y,p),'log_loss':log_loss(y,p),'accuracy':accuracy_score(y,q),
            'precision':precision_score(y,q,zero_division=0),'recall':recall_score(y,q,zero_division=0),
            'f1':f1_score(y,q,zero_division=0)}
def feature_frame(raw,fe=False):
    x=raw.drop(columns=['id','satisfaction'],errors='ignore').copy()
    for c in CAT:x[c]=x[c].fillna('__MISSING__').astype(str)
    if fe:
        r=x[RATINGS]
        for key,val in [('mean',r.mean(axis=1)),('std',r.std(axis=1)),('min',r.min(axis=1)),
                        ('max',r.max(axis=1)),('zero_count',(r==0).sum(axis=1)),('low_count',(r<=2).sum(axis=1))]:
            x['rating_'+key]=val
        d=x['Departure Delay in Minutes']; a=x['Arrival Delay in Minutes']
        x['delay_sum']=d+a; x['arrival_minus_departure']=a-d
        x['any_delay']=((d>0)|(a>0)).astype('int8')
        x['log_flight_distance']=np.log1p(x['Flight Distance'].clip(lower=0))
    return x
def configs():
    return [dict(name='LogisticRegression',kind='lr',fe=False,params={'C':1.,'max_iter':600}),
            dict(name='RandomForest',kind='rf',fe=False,params={'n_estimators':120,'max_depth':18,'min_samples_leaf':5,'max_features':'sqrt'}),
            dict(name='LightGBM_base',kind='lgb',fe=False,params={'n_estimators':300,'num_leaves':31,'learning_rate':.05,'reg_lambda':1.}),
            dict(name='CatBoost_base',kind='cat',fe=False,params={'iterations':600,'depth':6,'learning_rate':.07,'l2_leaf_reg':3.}),
            dict(name='LightGBM_700_L31',kind='lgb',fe=False,params={'n_estimators':700,'num_leaves':31,'learning_rate':.05,'reg_lambda':5.,'min_child_samples':50}),
            dict(name='LightGBM_700_L63',kind='lgb',fe=False,params={'n_estimators':700,'num_leaves':63,'learning_rate':.05,'reg_lambda':5.,'min_child_samples':50}),
            dict(name='CatBoost_tuned',kind='cat',fe=False,params={'iterations':1000,'depth':7,'learning_rate':.06,'l2_leaf_reg':5.}),
            dict(name='LightGBM_FE_L31',kind='lgb',fe=True,params={'n_estimators':700,'num_leaves':31,'learning_rate':.05,'reg_lambda':5.,'min_child_samples':50}),
            dict(name='LightGBM_FE_L63',kind='lgb',fe=True,params={'n_estimators':700,'num_leaves':63,'learning_rate':.05,'reg_lambda':5.,'min_child_samples':50})]
def new_model(cfg,x,seed,gpu):
    num=[c for c in x if c not in CAT]
    if cfg['kind']=='cat':
        return CatBoostClassifier(**cfg['params'],loss_function='Logloss',random_seed=seed,
                thread_count=8,verbose=False,allow_writing_files=False,task_type='GPU' if gpu else 'CPU',
                **({'devices':'0','gpu_ram_part':.5} if gpu else {}),cat_features=CAT,nan_mode='Min')
    if cfg['kind']=='lgb':
        enc=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),
              ('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CAT)],verbose_feature_names_out=False)
        est=LGBMClassifier(**cfg['params'],random_state=seed,n_jobs=8,verbosity=-1,
              deterministic=True,force_col_wise=True)
    else:
        enc=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),
              ('scale',StandardScaler())]) if cfg['kind']=='lr' else SimpleImputer(strategy='median'),num),
              ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],verbose_feature_names_out=False)
        est=LogisticRegression(**cfg['params'],random_state=seed) if cfg['kind']=='lr' else RandomForestClassifier(**cfg['params'],random_state=seed,n_jobs=8)
    return Pipeline([('preprocess',enc),('model',est)])
def fit_model(model,cfg,x,y):
    if cfg['kind']=='lgb':
        return model.fit(x,y,model__categorical_feature=list(range(len(x.columns)-len(CAT),len(x.columns))))
    return model.fit(x,y)
def profile(train,test):
    d={'train_shape':train.shape,'test_shape':test.shape,'positive_count':int(train.satisfaction.sum()),
       'positive_ratio':float(train.satisfaction.mean()),'missing_train':train.isna().sum().to_dict(),
       'missing_test':test.isna().sum().to_dict(),'duplicate_features':int(train.drop(columns=['id','satisfaction']).duplicated().sum()),
       'source':'Official competition zip, downloaded via authenticated Kaggle browser by user',
       'source_url':'https://www.kaggle.com/competitions/playground-series-s6e10/data',
       'sha256':{p.name:sha(p) for p in RAW.iterdir() if p.is_file()},'test_labels_available':False}
    assert d['duplicate_features']==0,'Duplicate features require a grouped split protocol.'
    assert train.id.is_unique and test.id.is_unique and not set(train.id)&set(test.id)
    assert set(train.satisfaction.unique())=={False,True}
    assert list(train.drop(columns='satisfaction').columns)==list(test.columns)
    save_json(OUT/'data_profile.json',d)
    fields=pd.DataFrame({'feature':train.columns,'dtype':[str(t) for t in train.dtypes],
                          'missing':[int(train[c].isna().sum()) for c in train],
                          'unique':[int(train[c].nunique()) for c in train]})
    fields.to_csv(OUT/'field_summary.csv',index=False)
    train.describe(include='all').to_csv(OUT/'descriptive_statistics.csv')
    cat_rows=[]
    for c in CAT:
        for name,group in train.groupby(c):cat_rows.append({'feature':c,'value':name,'count':len(group),'satisfaction_rate':group.satisfaction.mean()})
    pd.DataFrame(cat_rows).to_csv(OUT/'category_satisfaction.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4)); train.satisfaction.value_counts().sort_index().plot.bar(ax=ax,color=['#7570b3','#1b9e77'])
    ax.set(xlabel='Satisfaction',ylabel='Number of rows',title='Training label distribution');fig.tight_layout();fig.savefig(OUT/'figures/label_distribution.png',dpi=180);plt.close(fig)
    return d
def explain(cfg,model,x,y):
    """Use first frozen dev-only seed model; explanations are associations."""
    rng=np.random.default_rng(42); idx=rng.choice(len(x),min(1200,len(x)),replace=False); sample=x.iloc[idx]
    if cfg['kind']=='lgb':
        z=model.named_steps['preprocess'].transform(sample)
        contrib=model.named_steps['model'].booster_.predict(z,pred_contrib=True)[:,:-1]
        names=model.named_steps['preprocess'].get_feature_names_out()
    elif cfg['kind']=='cat':
        contrib=model.get_feature_importance(Pool(sample,cat_features=CAT),type='ShapValues')[:,:-1];names=sample.columns
    else:return
    importance=pd.DataFrame({'feature':names,'mean_abs_shap':np.abs(contrib).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    importance.to_csv(OUT/'shap_importance.csv',index=False)
    np.savez_compressed(OUT/'shap_values.npz',values=contrib,feature_names=np.asarray(names,dtype=str),holdout_sample_indices=idx)
    top=importance.head(15).iloc[::-1];fig,ax=plt.subplots(figsize=(9,6));ax.barh(top.feature,top.mean_abs_shap,color='#1b9e77')
    ax.set(xlabel='Mean absolute SHAP value (raw log-odds)',title='Frozen single model: prediction associations');fig.tight_layout();fig.savefig(OUT/'figures/shap_importance.png',dpi=180);fig.savefig(OUT/'figures/shap_importance.svg');plt.close(fig)

def main():
    for d in [OUT,MOD,OUT/'figures',OUT/'predictions']:d.mkdir(parents=True,exist_ok=True)
    start=time.time(); train=pd.read_csv(RAW/'train.csv'); test=pd.read_csv(RAW/'test.csv')
    profile_data=profile(train,test);y=train.satisfaction.astype('int8').to_numpy()
    dev,hold=train_test_split(np.arange(len(train)),test_size=.2,stratify=y,random_state=2026)
    fold_id=np.full(len(train),-1,dtype=int);kf=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    folds=list(kf.split(dev,y[dev]))
    for k,(_,va) in enumerate(folds):fold_id[dev[va]]=k
    np.savez_compressed(OUT/'splits.npz',dev=dev,holdout=hold,fold_id=fold_id)
    assert len(np.intersect1d(dev,hold))==0 and len(dev)+len(hold)==len(train)
    gpu=json.loads((OUT/'environment_witness.json').read_text())['catboost_gpu']['ok']
    cfgs=configs();save_json(OUT/'configurations.json',cfgs)
    oofs={};hp={};rows=[];cv=[]
    for cfg in cfgs:
        name=cfg['name'];cache=OUT/'predictions'/f'{name}.npz'
        if cache.exists():
            z=np.load(cache);oofs[name]=z['oof'];hp[name]=z['holdout']
            rr=json.loads((OUT/f'{name}_folds.json').read_text());rows+=rr
            print('RESUME',name,flush=True)
        else:
            x=feature_frame(train,cfg['fe']);oof=np.empty(len(dev));ph=np.zeros(len(hold));rr=[]
            for fold,(tr,va) in enumerate(folds):
                t=time.time();model=new_model(cfg,x,42,gpu)
                print('START',name,'fold',fold+1,flush=True)
                fit_model(model,cfg,x.iloc[dev[tr]],y[dev[tr]])
                oof[va]=model.predict_proba(x.iloc[dev[va]])[:,1]
                ph+=model.predict_proba(x.iloc[hold])[:,1]/3
                r={'model':name,'fold':fold+1,'train_rows':len(tr),'validation_rows':len(va),'seconds':time.time()-t,**metric(y[dev[va]],oof[va])}
                rr.append(r);rows.append(r);print('DONE',json.dumps(r),flush=True)
            oofs[name]=oof;hp[name]=ph;np.savez_compressed(cache,oof=oof,holdout=ph)
            save_json(OUT/f'{name}_folds.json',rr)
        cv.append({'model':name,'oof_auc':roc_auc_score(y[dev],oofs[name]),
                   'fold_auc_mean':np.mean([r['auc'] for r in rr]),'fold_auc_std':np.std([r['auc'] for r in rr],ddof=1),
                   'total_cv_seconds':sum(r['seconds'] for r in rr),**{'oof_'+k:v for k,v in metric(y[dev],oofs[name]).items() if k!='auc'}})
        pd.DataFrame(cv).to_csv(OUT/'cv_summary.csv',index=False)
        pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False)
    best=max(cv,key=lambda r:r['oof_auc'])['model']
    lgb=max([c['name'] for c in cfgs if c['kind']=='lgb'],key=lambda n:roc_auc_score(y[dev],oofs[n]))
    cat=max([c['name'] for c in cfgs if c['kind']=='cat'],key=lambda n:roc_auc_score(y[dev],oofs[n]))
    blend=[]
    for w in [0.,.25,.5,.75,1.]:
        blend.append({'lgb_weight':w,'cat_weight':1-w,'oof_auc':roc_auc_score(y[dev],w*oofs[lgb]+(1-w)*oofs[cat])})
    b=max(blend,key=lambda r:r['oof_auc']);single_auc=roc_auc_score(y[dev],oofs[best])
    use_blend=b['oof_auc']>single_auc+1e-12
    selected={'best_single':best,'best_single_oof_auc':single_auc,'lgb':lgb,'cat':cat,'blend':b,
              'use_blend':use_blend,'selection_basis':'development OOF only; holdout metrics not computed before freeze',
              'threshold':.5,'stability_seeds':[42,2026,3407],'submission_seed':42,
              'holdout_prediction_protocol':'equal average of three CV models trained on two thirds of development data',
              'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save_json(OUT/'selection_frozen.json',selected);pd.DataFrame(blend).to_csv(OUT/'blend_search.csv',index=False)
    # First evaluation of holdout labels happens only after recipe freeze above.
    hrows=[{'model':n,**metric(y[hold],p)} for n,p in hp.items()]
    pblend=b['lgb_weight']*hp[lgb]+b['cat_weight']*hp[cat]
    hrows.append({'model':'Selected_LGB_Cat_blend',**metric(y[hold],pblend)})
    pd.DataFrame(hrows).to_csv(OUT/'holdout_metrics.csv',index=False)
    ph=pblend if use_blend else hp[best]
    predictions=pd.DataFrame({'id':train.iloc[hold].id.to_numpy(),'y_true':y[hold],**hp,'Selected_LGB_Cat_blend':pblend})
    predictions.to_csv(OUT/'predictions/holdout_predictions.csv',index=False)
    devpred=pd.DataFrame({'id':train.iloc[dev].id.to_numpy(),'y_true':y[dev],**oofs})
    devpred.to_csv(OUT/'predictions/oof_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5))
    for n in ['LogisticRegression','RandomForest','LightGBM_base','CatBoost_base',best]:
        fpr,tpr,_=roc_curve(y[hold],hp[n]);ax.plot(fpr,tpr,label=f'{n}: {roc_auc_score(y[hold],hp[n]):.4f}')
    ax.plot([0,1],[0,1],'k--',lw=.8);ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Independent holdout ROC (CV-model averaging)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'figures/roc_curves.png',dpi=180);plt.close(fig)
    cm=confusion_matrix(y[hold],ph>=.5);save_json(OUT/'confusion_matrix.json',cm.tolist())
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Actual',title='Frozen selected recipe, threshold 0.5');fig.colorbar(im);fig.tight_layout();fig.savefig(OUT/'figures/confusion_matrix.png',dpi=180);plt.close(fig)
    cases=train.iloc[hold].copy();cases['predicted_probability']=ph;cases['absolute_error']=np.abs(y[hold]-ph)
    cases.sort_values('absolute_error',ascending=False).head(100).to_csv(OUT/'error_cases.csv',index=False)
    # Stability: identical frozen single configuration, fixed split, three predefined seeds.
    cfg=next(c for c in cfgs if c['name']==best);x=feature_frame(train,cfg['fe']);stability=[]
    for seed in [42,2026,3407]:
        cache=OUT/'predictions'/f'stability_{seed}.npz'
        if cache.exists():
            p=np.load(cache)['p'];r=json.loads((OUT/f'stability_{seed}.json').read_text())
        else:
            t=time.time();print('STABILITY',best,seed,flush=True)
            model=new_model(cfg,x,seed,gpu);fit_model(model,cfg,x.iloc[dev],y[dev]);p=model.predict_proba(x.iloc[hold])[:,1]
            r={'seed':seed,'model':best,'seconds':time.time()-t,**metric(y[hold],p)}
            np.savez_compressed(cache,p=p);save_json(OUT/f'stability_{seed}.json',r)
            if seed==42:
                joblib.dump(model,MOD/'frozen_single_dev_seed42.joblib',compress=3)
                explain(cfg,model,x.iloc[hold],y[hold])
        stability.append(r);print('STABILITY_DONE',json.dumps(r),flush=True)
    pd.DataFrame(stability).to_csv(OUT/'stability_metrics.csv',index=False)
    # Final all-labelled refit: predetermined recipe, seed42, never select using holdout.
    components=[(lgb,b['lgb_weight']),(cat,b['cat_weight'])] if use_blend else [(best,1.)]
    ptest=np.zeros(len(test));trained=[]
    for name,weight in components:
        if weight==0:continue
        cfg=next(c for c in cfgs if c['name']==name);x=feature_frame(train,cfg['fe']);xt=feature_frame(test,cfg['fe'])
        print('FINAL_REFIT',name,'weight',weight,flush=True);t=time.time()
        model=new_model(cfg,x,42,gpu);fit_model(model,cfg,x,y)
        p=model.predict_proba(xt)[:,1];ptest+=weight*p
        joblib.dump(model,MOD/f'{name}_full.joblib',compress=3)
        np.savez_compressed(OUT/'predictions'/f'{name}_competition_test.npz',p=p)
        trained.append({'name':name,'weight':weight,'training_rows':len(train),'seconds':time.time()-t})
    submission=pd.DataFrame({'id':test.id,'satisfaction':ptest});template=pd.read_csv(RAW/'sample_submission.csv')
    assert list(submission)==list(template) and np.array_equal(submission.id,template.id)
    assert len(submission)==len(test) and np.isfinite(ptest).all() and ((ptest>=0)&(ptest<=1)).all()
    submission.to_csv(ROOT/'submission.csv',index=False)
    save_json(OUT/'completion.json',{'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
       'elapsed_seconds':time.time()-start,'holdout_rows':len(hold),'development_rows':len(dev),
       'competition_test_rows':len(test),'gpu_catboost':gpu,'selected':selected,'final_refits':trained,
       'submission_sha256':sha(ROOT/'submission.csv'),'kaggle_submission_performed':False,
       'selected_recipe_holdout_metrics':metric(y[hold],ph)})
    write_report(profile_data,cv,hrows,stability,selected)
    print('EXPERIMENTS_COMPLETE',json.dumps(json.loads((OUT/'completion.json').read_text()),ensure_ascii=False),flush=True)
def write_report(profile_data,cv,hrows,stability,selected):
    lines=['# 航空旅客满意度预测：实验结果报告','',
       '全部数字由本地训练和官方训练标签计算；尚未上传Kaggle，未取得线上分数。','',
       f"训练样本：{profile_data['train_shape'][0]}；竞赛测试样本：{profile_data['test_shape'][0]}；正类比例：{profile_data['positive_ratio']:.6f}。",
       '数据来自Playground合成数据任务，不能直接据此声称真实航空业务效果。','',
       '## 数据与协议','',
       '20%独立留出集；80%开发集上共享三折分层交叉验证。全部预处理仅在折内训练部分拟合，id不参与建模；无外部数据。固定迭代次数，不在评分折早停。',
       '独立留出集各配置预测来自三个CV模型等权平均（每个模型只使用约2/3开发集）；稳定性表则在完整开发集重训，两者训练规模不同。',
       'OOF用于有限配置与融合权重选择，属于开发结果，存在选择乐观偏差。留出标签只在selection_frozen.json写出后评价，未据此修改方案。','',
       '## 模型开发结果','', '| 配置 | OOF AUC | 折AUC均值±标准差 | CV训练与预测秒数 |','|---|---:|---:|---:|']
    for r in cv:lines.append(f"| {r['model']} | {r['oof_auc']:.6f} | {r['fold_auc_mean']:.6f} ± {r['fold_auc_std']:.6f} | {r['total_cv_seconds']:.1f} |")
    lines+=['','## 独立留出结果（冻结后统一评估）','','| 配置 | AUC | Log Loss | Accuracy | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in hrows:lines.append('| '+r['model']+' | '+' | '.join(f"{r[k]:.6f}" for k in ['auc','log_loss','accuracy','precision','recall','f1'])+' |')
    lines+=['','## 冻结方案','',f"最佳单模型：{selected['best_single']}；是否选择融合：{selected['use_blend']}。",
       f"融合比较：{selected['lgb']} 与 {selected['cat']}；开发集选择的LightGBM权重：{selected['blend']['lgb_weight']}。",'',
       '## 固定单模型的随机种子稳定性（完整开发集训练）','', '| 种子 | 独立留出AUC | F1 |','|---|---:|---:|']
    for r in stability:lines.append(f"| {r['seed']} | {r['auc']:.6f} | {r['f1']:.6f} |")
    vals=[r['auc'] for r in stability];lines+=['',f"三种子AUC均值：{np.mean(vals):.6f}；样本标准差：{np.std(vals,ddof=1):.6f}。此处只评估固定划分下的算法随机性，不代表数据划分不确定性。",'',
       '## 特征工程与限制','',
       'LightGBM_700_L31 / LightGBM_FE_L31、LightGBM_700_L63 / LightGBM_FE_L63为对应的特征消融对照。负收益如实保留。',
       'SHAP图来自固定种子42、仅用开发集训练的最佳单模型，1200个留出样本；SHAP表示预测关联，不是因果效应。',
       '各算法调参预算有限且不同，不声称算法全局最优；本实验只覆盖一个数据集。GPU CatBoost存在数值非确定性，固定种子也不保证逐位一致。',
       '## 交付文件','',
       '- submission.csv：全量有标签数据重训后的竞赛测试预测；按官方样例校验列名、id、行数及概率范围。',
       '- results/：逐折指标、OOF和留出预测、划分、数据哈希、模型选择冻结记录、SHAP及图表。',
       '- models/：可加载的完整模型与预处理器。',
       '- EXPERIMENT_PLAN.md、requirements-lock.txt、src/：运行前方案、依赖及可复现实验代码。']
    (ROOT/'文档/实验结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
