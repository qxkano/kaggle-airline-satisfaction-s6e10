import json, os, sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np
import pandas as pd
import sklearn, lightgbm, catboost
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
ROOT=Path('E:/codex-download/项目/数据科学课程大作业-S6E10')
X,y=make_classification(n_samples=500,n_features=10,random_state=42)
out={'python':sys.version,'versions':{m.__name__:m.__version__ for m in [np,pd,sklearn,lightgbm,catboost]}}
for name,model in [('lightgbm_cpu',LGBMClassifier(n_estimators=10,n_jobs=8,verbosity=-1,random_state=42)),('catboost_gpu',CatBoostClassifier(iterations=10,depth=4,task_type='GPU',devices='0',random_seed=42,verbose=False,allow_writing_files=False))]:
    try:
        model.fit(X,y)
        pred=model.predict_proba(X)[:,1]
        assert np.isfinite(pred).all() and len(pred)==500
        out[name]={'ok':True,'shape':list(pred.shape),'training_smoke_auc':roc_auc_score(y,pred)}
    except Exception as e:
        out[name]={'ok':False,'error':repr(e)}
(ROOT/'results/environment_witness.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('WITNESS',json.dumps(out))
