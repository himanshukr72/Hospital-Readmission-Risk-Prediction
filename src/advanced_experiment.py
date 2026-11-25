"""Grouped CV feature ablation. Does not touch existing models or test scores."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from .features import features, EXCLUDE

MEDS = 'metformin repaglinide nateglinide chlorpropamide glimepiride acetohexamide glipizide glyburide tolbutamide pioglitazone rosiglitazone acarbose miglitol troglitazone tolazamide insulin glyburide-metformin glipizide-metformin glimepiride-pioglitazone metformin-rosiglitazone metformin-pioglitazone'.split()

def expanded(raw):
    x=features(raw)
    z=x.copy()
    stay=x.time_in_hospital.where(x.time_in_hospital>0)
    for c in ['num_medications','num_procedures','num_lab_procedures']:
        z[c+'_per_day']=x[c]/stay
    z['prior_acute_visits']=x.number_inpatient+x.number_emergency
    z['prior_inpatient_flag']=x.number_inpatient.gt(0).astype(float).where(x.number_inpatient.notna())
    m=raw[MEDS].replace({'?':np.nan,'':np.nan})
    known=m.notna().all(axis=1)
    z['active_med_count']=m.isin(['Steady','Up','Down']).sum(axis=1).where(known)
    z['changed_med_count']=m.isin(['Up','Down']).sum(axis=1).where(known)
    z['increased_med_count']=m.eq('Up').sum(axis=1).where(known)
    z['decreased_med_count']=m.eq('Down').sum(axis=1).where(known)
    z['insulin_a1c']=x.insulin.fillna('Unknown')+' | '+x.A1Cresult.fillna('Unknown')
    z['primary_secondary_diag']=x.diag_1+' | '+x.diag_2
    groups={'intensity':[c+'_per_day' for c in ['num_medications','num_procedures','num_lab_procedures']], 'utilisation':['prior_acute_visits','prior_inpatient_flag'], 'medications':['active_med_count','changed_med_count','increased_med_count','decreased_med_count'], 'interactions':['insulin_a1c','primary_secondary_diag']}
    return z,groups

def pipeline(x,quick):
    num=list(x.select_dtypes(include='number').columns); cat=[c for c in x if c not in num]
    pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Unknown')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cat)])
    return Pipeline([('pre',pre),('model',XGBClassifier(n_estimators=40 if quick else 100,max_depth=4,learning_rate=.08,tree_method='hist',n_jobs=2,random_state=42,eval_metric='logloss'))])

def split(y,g,n,seed):
    return next(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--quick',action='store_true'); p.add_argument('--data',default='data/raw/diabetic_data.csv'); a=p.parse_args()
    raw=pd.read_csv(a.data,dtype=str,keep_default_na=False).drop_duplicates(); raw=raw[~raw.discharge_disposition_id.isin(EXCLUDE)].copy()
    if a.quick:
        ids=raw.patient_nbr.drop_duplicates().sample(min(3000,raw.patient_nbr.nunique()),random_state=42); raw=raw[raw.patient_nbr.isin(ids)]
    raw=raw.reset_index(drop=True); y=raw.readmitted.eq('<30').astype(int); g=raw.patient_nbr
    dev,_=split(y,g,5,42); tr,_=split(y.iloc[dev],g.iloc[dev],4,43); idx=dev[tr]
    raw=raw.iloc[idx].reset_index(drop=True); y=y.iloc[idx].reset_index(drop=True); g=g.iloc[idx].reset_index(drop=True)
    x,groups=expanded(raw); extras=sum(groups.values(),[]); base=[c for c in x if c not in extras]
    configs={'baseline':base,'all_features':list(x)}
    for k,v in groups.items(): configs['add_'+k]=base+v; configs['remove_'+k]=[c for c in x if c not in v]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=45).split(x,y,g)); rows=[]
    for name,cols in configs.items():
        for f,(t,v) in enumerate(folds):
            assert not set(g.iloc[t]) & set(g.iloc[v]); model=pipeline(x[cols],a.quick); model.fit(x.iloc[t][cols],y.iloc[t]); prob=model.predict_proba(x.iloc[v][cols])[:,1]
            rows.append(dict(config=name,fold=f+1,features=len(cols),train_rows=len(t),validation_rows=len(v),ap=average_precision_score(y.iloc[v],prob),roc_auc=roc_auc_score(y.iloc[v],prob)))
        print(name, 'done',flush=True)
    out=Path('reports/advanced_quick' if a.quick else 'reports/advanced'); out.mkdir(parents=True,exist_ok=True)
    d=pd.DataFrame(rows); d.to_csv(out/'ablation_folds.csv',index=False); s=d.groupby('config').agg(ap_mean=('ap','mean'),ap_sd=('ap','std'),auc_mean=('roc_auc','mean'),auc_sd=('roc_auc','std')).reset_index(); s['delta_ap']=s.ap_mean-s.loc[s.config.eq('baseline'),'ap_mean'].iloc[0]; s=s.sort_values('ap_mean',ascending=False); s.to_csv(out/'ablation_summary.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'quick':a.quick,'development_train_rows':len(y),'patients':g.nunique(),'positive_rate':y.mean(),'groups':groups,'selection_metric':'mean validation average precision','test_used':False},indent=2)); print(s.to_string(index=False))
if __name__=='__main__': main()
