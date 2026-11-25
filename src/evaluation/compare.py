import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

def compare(y,g,preds,winner,out,n=300):
    rng=np.random.default_rng(42); groups,inv=np.unique(g,return_inverse=True); order=np.argsort(inv); blocks=np.split(order,np.flatnonzero(np.diff(inv[order]))+1); names=list(preds); draws=[]
    for _ in range(n):
        idx=np.concatenate([blocks[i] for i in rng.integers(len(groups),size=len(groups))]); yy=y[idx]
        if len(np.unique(yy))<2: continue
        draws.append([[average_precision_score(yy,preds[k][idx]),roc_auc_score(yy,preds[k][idx])] for k in names])
    arr=np.asarray(draws); rows=[]
    for i,k in enumerate(names):
        for j,m in enumerate(['pr_auc','roc_auc']):
            lo,hi=np.quantile(arr[:,i,j],[.025,.975]); d=arr[:,names.index(winner),j]-arr[:,i,j]; dl,dh=np.quantile(d,[.025,.975]); rows.append(dict(model=k,metric=m,lower=lo,upper=hi,winner_minus_model_lower=dl,winner_minus_model_upper=dh))
    pd.DataFrame(rows).to_csv(out/'patient_bootstrap.csv',index=False)
