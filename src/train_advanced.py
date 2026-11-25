"""Advanced experiments; original baseline files are never overwritten."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib,optuna
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from .advanced_experiment import expanded,pipeline,split
from .features import EXCLUDE
from .models.boosting import models
from .evaluation.calibration import calibrate,apply,threshold
from .evaluation.metrics import scores
from .evaluation.compare import compare
from .evaluation.dashboard import export
from .evaluation.plots import plots
from .models.ensembles import PrefitEnsemble

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--quick',action='store_true'); parser.add_argument('--trials',type=int,default=None); parser.add_argument('--data',default='data/raw/diabetic_data.csv'); a=parser.parse_args()
    out=Path('reports/advanced_quick' if a.quick else 'reports/advanced'); out.mkdir(parents=True,exist_ok=True)
    modeldir=Path('models/advanced_quick' if a.quick else 'models/advanced'); modeldir.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.data,dtype=str,keep_default_na=False).drop_duplicates(); raw=raw[~raw.discharge_disposition_id.isin(EXCLUDE)].copy()
    if a.quick:
        ids=raw.patient_nbr.drop_duplicates().sample(min(3000,raw.patient_nbr.nunique()),random_state=42); raw=raw[raw.patient_nbr.isin(ids)]
    raw=raw.reset_index(drop=True); y=raw.readmitted.eq('<30').astype(int); g=raw.patient_nbr; x,groups=expanded(raw)
    dev,test=split(y,g,5,42); tr,v=split(y.iloc[dev],g.iloc[dev],4,43); train=dev[tr]; valid=dev[v]; c,h=split(y.iloc[valid],g.iloc[valid],2,44); cal=valid[c]; tune=valid[h]
    assert sum(len(set(g.iloc[i])) for i in [train,cal,tune,test])==g.nunique()
    X=x.iloc[train].reset_index(drop=True); Y=y.iloc[train].reset_index(drop=True); G=g.iloc[train].reset_index(drop=True)
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=45).split(X,Y,G)); base=[k for k in X if k not in sum(groups.values(),[])]; configs={'baseline':base,'all_features':list(X)}
    for k,v in groups.items(): configs['add_'+k]=base+v; configs['remove_'+k]=[c for c in X if c not in v]
    def cv(est,cols):
        p=np.zeros(len(Y)); scores_=[]
        for t,v in folds:
            m=clone(est).fit(X.iloc[t][cols],Y.iloc[t]); p[v]=m.predict_proba(X.iloc[v][cols])[:,1]; scores_.append((average_precision_score(Y.iloc[v],p[v]),roc_auc_score(Y.iloc[v],p[v])))
        return p,np.asarray(scores_)
    rows=[]
    for name,cols in configs.items():
        _,s=cv(pipeline(X[cols],a.quick),cols)
        for i,(ap,auc) in enumerate(s): rows.append({'config':name,'fold':i+1,'ap':ap,'roc_auc':auc,'features':len(cols)})
        print('Ablation',name,flush=True)
    d=pd.DataFrame(rows); d.to_csv(out/'ablation_folds.csv',index=False); summary=d.groupby('config').agg(ap_mean=('ap','mean'),ap_sd=('ap','std'),auc_mean=('roc_auc','mean')).reset_index().sort_values('ap_mean',ascending=False); summary.to_csv(out/'ablation_summary.csv',index=False)
    cols=configs[summary.iloc[0]['config']]; selection=[]
    # Grouped fold-only permutation ranking, never uses calibration/test.
    ranks=[]
    for t,v in folds:
        m=pipeline(X[cols],a.quick).fit(X.iloc[t][cols],Y.iloc[t]); sample=np.random.default_rng(42).choice(v,min(len(v),500),replace=False); r=permutation_importance(m,X.iloc[sample][cols],Y.iloc[sample],scoring='average_precision',n_repeats=1,random_state=42); ranks.append(r.importances_mean)
    ranked=list(pd.Series(np.mean(ranks,axis=0),index=cols).sort_values(ascending=False).index)
    # Selection rankings use all development folds: CV estimates here are exploratory, final validation is separate.
    for n in sorted(set([min(20,len(cols)),min(35,len(cols)),len(cols)])):
        subset=ranked[:n] if n<len(cols) else cols; _,s=cv(pipeline(X[subset],a.quick),subset); selection.append({'n':n,'ap':s[:,0].mean(),'columns':subset})
    pd.DataFrame(selection).to_csv(out/'feature_selection.csv',index=False); cols=max(selection,key=lambda r:r['ap'])['columns']; print('Selected features',len(cols),flush=True)
    def objective(trial):
        params={'n_estimators':trial.suggest_int('n_estimators',20 if a.quick else 100,60 if a.quick else 700),'max_depth':trial.suggest_int('max_depth',2,6),'learning_rate':trial.suggest_float('learning_rate',.01,.25,log=True),'min_child_weight':trial.suggest_int('min_child_weight',1,20),'subsample':trial.suggest_float('subsample',.6,1),'colsample_bytree':trial.suggest_float('colsample_bytree',.5,1),'reg_alpha':trial.suggest_float('reg_alpha',0,8),'reg_lambda':trial.suggest_float('reg_lambda',.1,20,log=True),'scale_pos_weight':trial.suggest_float('scale_pos_weight',1,10,log=True)}
        _,s=cv(pipeline(X[cols],a.quick).set_params(**{'model__'+k:v for k,v in params.items()}),cols); return s[:,0].mean()
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42)); study.optimize(objective,n_trials=a.trials or (2 if a.quick else 50)); study.trials_dataframe().to_csv(out/'tuning_trials.csv',index=False)
    tuned=pipeline(X[cols],a.quick).set_params(**{'model__'+k:v for k,v in study.best_params.items()}); candidates={'XGBoost':pipeline(X[cols],a.quick),'XGBoost_tuned':tuned}
    for name,est in {**models(),'RandomForest':RandomForestClassifier(n_estimators=20 if a.quick else 150,max_depth=10,n_jobs=2,random_state=42)}.items():
        if name=='XGBoost':continue
        if a.quick and 'n_estimators' in est.get_params(): est.set_params(n_estimators=12)
        candidates[name]=pipeline(X[cols],a.quick).set_params(model=est)
    oof=[]; fitted={}
    for name,est in candidates.items():
        p,_=cv(est,cols); oof.append(p); fitted[name]=clone(est).fit(X[cols],Y); print('Fit',name,flush=True)
    members=list(fitted.values()); meta=LogisticRegression().fit(np.column_stack(oof),Y); fitted['SoftVoting']=PrefitEnsemble(members); fitted['Stacking']=PrefitEnsemble(members,meta)
    curves=[]
    for frac in [.1,.25,.5,.75,1.]:
        for i,(t,v) in enumerate(folds):
            ids=G.iloc[t].unique(); np.random.default_rng(42+i).shuffle(ids); tt=t[G.iloc[t].isin(ids[:max(1,int(len(ids)*frac))]).to_numpy()]; m=clone(tuned).fit(X.iloc[tt][cols],Y.iloc[tt]); curves.append({'fraction':frac,'fold':i,'train_rows':len(tt),'train_ap':average_precision_score(Y.iloc[tt],m.predict_proba(X.iloc[tt][cols])[:,1]),'validation_ap':average_precision_score(Y.iloc[v],m.predict_proba(X.iloc[v][cols])[:,1])})
    pd.DataFrame(curves).to_csv(out/'learning_curves.csv',index=False)
    validation={}; bundles={}; rows=[]; preds={}
    for name,m in fitted.items():
        cm=calibrate(m.predict_proba(x.iloc[cal][cols])[:,1],y.iloc[cal]); vp=apply(cm,m.predict_proba(x.iloc[tune][cols])[:,1]); cut=threshold(y.iloc[tune],vp); validation[name]=average_precision_score(y.iloc[tune],vp)
        bundles[name]={'pipeline':m,'calibrator':cm,'threshold':cut,'model':name,'raw_fields':[k for k in raw if k not in ['encounter_id','patient_nbr','readmitted','weight','examide','citoglipton']],'feature_columns':cols,'quick':a.quick,'advanced':True}
    winner=max(validation,key=validation.get)
    for name,b in bundles.items():
        rp=b['pipeline'].predict_proba(x.iloc[test][cols])[:,1]; cp=apply(b['calibrator'],rp); preds[name]=cp
        for mode,p,t in [('raw',rp,.5),('calibrated',cp,.5),('optimized',cp,b['threshold'])]: rows.append({'model':name,'mode':mode,'threshold':t,'validation_ap':validation[name],**scores(y.iloc[test],p,t)})
        joblib.dump(b,modeldir/(name+'.joblib'))
    joblib.dump(bundles[winner],modeldir/'best.joblib'); pd.DataFrame(rows).to_csv(out/'model_metrics.csv',index=False)
    (out/'winner.json').write_text(json.dumps({'model':winner,'threshold':bundles[winner]['threshold'],'selection':'validation average precision'},indent=2)); (out/'cohort.json').write_text(json.dumps({'quick':a.quick,'encounters':len(raw),'patients':g.nunique(),'positive_rate':y.mean(),'splits':dict(zip(['train','calibration','threshold','test'],map(len,[train,cal,tune,test]))),'selected_features':cols,'caution':'Original test partition was previously examined; results are exploratory, not independent external validation.'},indent=2))
    compare(y.iloc[test].to_numpy(),g.iloc[test].to_numpy(),preds,winner,out,30 if a.quick else 300); export(y.iloc[test],preds,out); plots(y.iloc[test],preds,out)
    sample=np.random.default_rng(42).choice(test,min(300,len(test)),replace=False); r=permutation_importance(fitted[winner],x.iloc[sample][cols],y.iloc[sample],scoring='average_precision',n_repeats=2,random_state=42); pd.DataFrame({'feature':cols,'importance':r.importances_mean,'std':r.importances_std}).sort_values('importance',ascending=False).to_csv(out/'permutation_importance.csv',index=False)
    import xgboost as xgb
    tree=fitted['XGBoost_tuned']; pre=tree.named_steps['pre']; vals=tree.named_steps['model'].get_booster().predict(xgb.DMatrix(pre.transform(x.iloc[sample][cols])),pred_contribs=True); pd.DataFrame({'encoded_feature':pre.get_feature_names_out(),'mean_absolute_shap':np.abs(vals[:,:-1]).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False).to_csv(out/'xgboost_shap.csv',index=False)
    sub=[]
    for field in ['age','gender','race']:
        for group,idx in raw.iloc[test].groupby(field).groups.items():
            mask=raw.iloc[test][field].eq(group).to_numpy(); yy=y.iloc[test].to_numpy()[mask]
            if len(yy)>=30 and len(np.unique(yy))==2: sub.append({'field':field,'group':group,'n':len(yy),**scores(yy,preds[winner][mask],bundles[winner]['threshold'])})
    pd.DataFrame(sub).to_csv(out/'subgroup_metrics.csv',index=False); print('Finished',winner,flush=True)
if __name__=='__main__':main()
