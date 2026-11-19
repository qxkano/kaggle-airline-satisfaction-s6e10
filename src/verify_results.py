"""Recompute reported scores from saved predictions and original labels."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score,precision_score,recall_score,f1_score
ROOT=Path('E:/codex-download/项目/数据科学课程大作业-S6E10');R=ROOT/'results'
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def metrics(y,p):
    q=p>=.5
    return dict(auc=roc_auc_score(y,p),log_loss=log_loss(y,p),accuracy=accuracy_score(y,q),
        precision=precision_score(y,q,zero_division=0),recall=recall_score(y,q,zero_division=0),f1=f1_score(y,q,zero_division=0))
def main():
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'pass':bool(condition)})
        assert condition,name
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    profile=json.loads((R/'data_profile.json').read_text(encoding='utf-8'))
    for f,h in profile['sha256'].items():check('raw SHA256 '+f,digest(ROOT/'data/raw'/f)==h)
    s=np.load(R/'splits.npz');dev=s['dev'];hold=s['holdout'];fid=s['fold_id'];y=train.satisfaction.astype(int).to_numpy()
    check('split disjoint',len(np.intersect1d(dev,hold))==0)
    check('split covers every labelled row',np.array_equal(np.sort(np.r_[dev,hold]),np.arange(len(train))))
    check('holdout excluded from CV',(fid[hold]==-1).all())
    check('all dev rows have one CV fold',set(fid[dev])=={0,1,2})
    cv=pd.read_csv(R/'cv_summary.csv');folds=pd.read_csv(R/'fold_metrics.csv');hm=pd.read_csv(R/'holdout_metrics.csv')
    for row in cv.to_dict('records'):
        n=row['model'];z=np.load(R/'predictions'/f'{n}.npz');p=z['oof'];h=z['holdout']
        check(n+' finite probability',np.isfinite(p).all() and np.isfinite(h).all() and ((p>=0)&(p<=1)).all() and ((h>=0)&(h<=1)).all())
        check(n+' OOF AUC',np.isclose(roc_auc_score(y[dev],p),row['oof_auc'],atol=1e-12,rtol=0))
        for fr in folds[folds.model==n].to_dict('records'):
            mask=fid[dev]==fr['fold']-1
            check(n+f" fold {fr['fold']} AUC",np.isclose(roc_auc_score(y[dev][mask],p[mask]),fr['auc'],atol=1e-12,rtol=0))
        rr=hm[hm.model==n].iloc[0]
        for key,v in metrics(y[hold],h).items():check(n+' holdout '+key,np.isclose(v,rr[key],atol=1e-12,rtol=0))
    sel=json.loads((R/'selection_frozen.json').read_text(encoding='utf-8'));best=cv.loc[cv.oof_auc.idxmax(),'model']
    check('single selected by development OOF only',sel['best_single']==best)
    lgb=np.load(R/'predictions'/f"{sel['lgb']}.npz");cat=np.load(R/'predictions'/f"{sel['cat']}.npz")
    b=sel['blend'];blend=b['lgb_weight']*lgb['oof']+b['cat_weight']*cat['oof']
    check('blend development score',np.isclose(roc_auc_score(y[dev],blend),b['oof_auc'],atol=1e-12,rtol=0))
    ph=b['lgb_weight']*lgb['holdout']+b['cat_weight']*cat['holdout']
    rr=hm[hm.model=='Selected_LGB_Cat_blend'].iloc[0]
    for key,v in metrics(y[hold],ph).items():check('blend holdout '+key,np.isclose(v,rr[key],atol=1e-12,rtol=0))
    for rr in pd.read_csv(R/'stability_metrics.csv').to_dict('records'):
        p=np.load(R/'predictions'/f"stability_{rr['seed']}.npz")['p']
        for key,v in metrics(y[hold],p).items():check(f"seed {rr['seed']} "+key,np.isclose(v,rr[key],atol=1e-12,rtol=0))
    sub=pd.read_csv(ROOT/'submission.csv');sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    check('submission columns',list(sub)==['id','satisfaction'])
    check('submission ids and order',np.array_equal(sub.id,sample.id) and np.array_equal(sub.id,test.id))
    p=sub.satisfaction.to_numpy();check('submission finite probability',np.isfinite(p).all() and ((p>=0)&(p<=1)).all())
    comp=json.loads((R/'completion.json').read_text(encoding='utf-8'));expected=np.zeros(len(sub))
    for item in comp['final_refits']:expected+=item['weight']*np.load(R/'predictions'/f"{item['name']}_competition_test.npz")['p']
    check('submission equals frozen recipe predictions',np.allclose(p,expected,atol=1e-12,rtol=0))
    check('submission SHA256',digest(ROOT/'submission.csv')==comp['submission_sha256'])
    out={'status':'PASS','checks_passed':len(checks),'checks':checks,'scope':'Deterministic metric, label, split, hash and submission checks; semantic audit separate'}
    (R/'deterministic_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('VERIFICATION_PASS',len(checks))
if __name__=='__main__':main()
