import argparse,json,time,os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedGroupKFold,RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from .features import features,EXCLUDE
from .preprocessing import build
from .models.baselines import models as baseline
from .models.boosting import models as boosting
from .models.imbalance import sampler,weights
from .evaluation.metrics import scores
from .evaluation.calibration import calibrate,apply,threshold
from .eda import run as eda

def split(y,g,n=5,seed=42):
    return next(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--quick',action='store_true',help='Smoke run on a patient subset; not final results'); p.add_argument('--data',default='data/raw/diabetic_data.csv'); a=p.parse_args()
    out=Path('reports/quick' if a.quick else 'reports/metrics'); out.mkdir(parents=True,exist_ok=True); modeldir=Path('models/quick' if a.quick else 'models'); modeldir.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.data,dtype=str,keep_default_na=False).drop_duplicates(); raw=raw[~raw.discharge_disposition_id.isin(EXCLUDE)].copy()
    if a.quick:
        ids=raw.patient_nbr.drop_duplicates().sample(n=min(3000,raw.patient_nbr.nunique()),random_state=42); raw=raw[raw.patient_nbr.isin(ids)]
    raw=raw.reset_index(drop=True); y=raw.readmitted.eq('<30').astype(int); g=raw.patient_nbr; x=features(raw)
    dev,test=split(y,g); tr0,val0=split(y.iloc[dev],g.iloc[dev],n=4,seed=43); train=dev[tr0]; valid=dev[val0]; ca,th=split(y.iloc[valid],g.iloc[valid],n=2,seed=44); cal=valid[ca]; tune=valid[th]
    parts=[train,cal,tune,test]; assert sum(map(len,parts))==len(raw); assert sum(len(set(g.iloc[i])) for i in parts)==g.nunique()
    (out/'cohort.json').write_text(json.dumps({'quick':a.quick,'encounters':len(raw),'patients':g.nunique(),'positive_rate':y.mean(),'splits':{k:len(v) for k,v in zip(['train','calibration','threshold','test'],parts)}},indent=2))
    eda(x.iloc[train],y.iloc[train],out/'eda')
    ord_,enc,cat=build(x.iloc[train]); candidates={**baseline(),**boosting()}; rows=[]; fitted={}; validation={}
    def fit(name,est,kind='none',search=False):
        start=time.time(); steps=[('ordinal',clone(ord_))]
        if kind=='SMOTENC': steps.append(('mixed_resample',sampler(kind,cat)))
        steps.append(('encode',clone(enc)))
        if kind=='ADASYN': steps.append(('resample',sampler(kind,cat)))
        steps.append(('model',clone(est))); pipe=Pipeline(steps)
        kw={'model__sample_weight':weights(y.iloc[train])} if kind=='weighted' else {}
        if search:
            cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=45)
            rs=RandomizedSearchCV(pipe,{'model__max_depth':[2,3,4,5,6],'model__n_estimators':[60,100,150,220],'model__learning_rate':[.03,.06,.1,.2],'model__reg_lambda':[1,3,8]},n_iter=2 if a.quick else 10,scoring='average_precision',cv=cv,random_state=42,n_jobs=1)
            rs.fit(x.iloc[train],y.iloc[train],groups=g.iloc[train]); pipe=rs.best_estimator_; pd.DataFrame(rs.cv_results_).to_csv(out/'tuning.csv',index=False)
        else: pipe.fit(x.iloc[train],y.iloc[train],**kw)
        cp=pipe.predict_proba(x.iloc[cal])[:,1]; cm=calibrate(cp,y.iloc[cal]); vp=apply(cm,pipe.predict_proba(x.iloc[tune])[:,1]); t=threshold(y.iloc[tune],vp)
        validation[name]=scores(y.iloc[tune],vp,t)['pr_auc']; fitted[name]=(pipe,cm,t)
        print(name,'validation AP',round(validation[name],4),'seconds',round(time.time()-start,1),flush=True)
    for name,est in candidates.items():
        if a.quick and 'n_estimators' in est.get_params(): est.set_params(n_estimators=12)
        fit(name,est)
    for kind in ['weighted','SMOTENC','ADASYN']:
        fit('XGBoost_'+kind,candidates['XGBoost'],kind)
    fit('XGBoost_tuned',candidates['XGBoost'],search=True)
    winner=max(validation,key=validation.get); preds={};
    for name,(pipe,cm,t) in fitted.items():
        rp=pipe.predict_proba(x.iloc[test])[:,1]; cp=apply(cm,rp); preds[name]=cp
        for label,prob,cut in [('raw',rp,.5),('calibrated',cp,.5),('optimized',cp,t)]: rows.append({'model':name,'mode':label,'threshold':cut,'validation_ap':validation[name],**scores(y.iloc[test],prob,cut)})
    pd.DataFrame(rows).to_csv(out/'model_metrics.csv',index=False)
    from .evaluation.compare import compare
    compare(y.iloc[test].to_numpy(),g.iloc[test].to_numpy(),preds,winner,out,50 if a.quick else 300)
    from .evaluation.plots import plots
    plots(y.iloc[test],preds,out)
    pipe,cm,t=fitted[winner]; bundle={'pipeline':pipe,'calibrator':cm,'threshold':t,'model':winner,'raw_fields':[c for c in raw if c not in ['encounter_id','patient_nbr','readmitted','weight','examide','citoglipton']],'feature_columns':list(x.columns),'quick':a.quick}
    joblib.dump(bundle,modeldir/'best.joblib'); (out/'winner.json').write_text(json.dumps({'model':winner,'threshold':t,'selection':'validation average precision'},indent=2))
    from .evaluation.explainability import explain
    explain(pipe,x.iloc[train],x.iloc[test],y.iloc[test],out,fitted['XGBoost'][0]); print('Finished',winner,flush=True)
if __name__=='__main__': main()
