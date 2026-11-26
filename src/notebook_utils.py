"""Shared reproducible notebook setup; does not modify production training."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve
from imblearn.pipeline import Pipeline
from .features import features,EXCLUDE
from .preprocessing import build
from .models.imbalance import sampler
from .evaluation.metrics import scores
from .train import split

def load_cohort(root,quick=True):
    raw=pd.read_csv(root/'data/raw/diabetic_data.csv',dtype=str,keep_default_na=False).drop_duplicates(); raw=raw[~raw.discharge_disposition_id.isin(EXCLUDE)].copy()
    if quick:
        ids=raw.patient_nbr.drop_duplicates().sample(min(3000,raw.patient_nbr.nunique()),random_state=42); raw=raw[raw.patient_nbr.isin(ids)]
    raw=raw.reset_index(drop=True); y=raw.readmitted.eq('<30').astype(int); g=raw.patient_nbr; x=features(raw)
    dev,test=split(y,g); tr,va=split(y.iloc[dev],g.iloc[dev],4,43); train=dev[tr]; valid=dev[va]; ca,th=split(y.iloc[valid],g.iloc[valid],2,44)
    parts=dict(train=train,calibration=valid[ca],threshold=valid[th],test=test)
    assert sum(len(set(g.iloc[i])) for i in parts.values())==g.nunique(); assert sum(map(len,parts.values()))==len(raw)
    return raw,x,y,g,parts

def split_table(y,g,parts):
    return pd.DataFrame([dict(part=k,encounters=len(v),patients=g.iloc[v].nunique(),positive_rate=y.iloc[v].mean()) for k,v in parts.items()])

def make_model(X,est,kind='none',quick=True):
    ord_,enc,cat=build(X); est=clone(est)
    if quick and 'n_estimators' in est.get_params(): est.set_params(n_estimators=20)
    steps=[('ordinal',ord_)]
    if kind=='SMOTENC': steps.append(('mixed_resample',sampler(kind,cat)))
    steps.append(('encode',enc))
    if kind=='ADASYN': steps.append(('resample',sampler(kind,cat)))
    return Pipeline(steps+[('model',est)])

def folds(X,Y,G): return StratifiedGroupKFold(3,shuffle=True,random_state=45).split(X,Y,G)

def group_cv(est,X,Y,G):
    p=np.zeros(len(Y)); rows=[]
    for t,v in folds(X,Y,G):
        assert not set(G.iloc[t]) & set(G.iloc[v]); m=clone(est).fit(X.iloc[t],Y.iloc[t]); p[v]=m.predict_proba(X.iloc[v])[:,1]; rows.append((average_precision_score(Y.iloc[v],p[v]),roc_auc_score(Y.iloc[v],p[v])))
    return p,np.array(rows)

def plot_curves(y,preds):
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for k,p in preds.items():
        f,t,_=roc_curve(y,p); pr,re,_=precision_recall_curve(y,p); ax[0].plot(f,t,label=k); ax[1].plot(re,pr,label=k)
    ax[0].plot([0,1],[0,1],'--'); ax[1].axhline(np.mean(y),linestyle='--'); ax[0].set(xlabel='False positive rate',ylabel='Recall'); ax[1].set(xlabel='Recall',ylabel='Precision'); ax[0].legend(); ax[1].legend(); fig.tight_layout(); plt.show()
