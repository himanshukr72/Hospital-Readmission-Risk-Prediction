from pathlib import Path
from functools import lru_cache
import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.features import EXCLUDE, diagnosis
from src.predict import load, predict
router=APIRouter(prefix='/api')
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=1)
def raw():
    d=pd.read_csv(ROOT/'data/raw/diabetic_data.csv',dtype=str,keep_default_na=False)
    return d

def records(d): return json.loads(d.to_json(orient='records'))
def run_dir(run):
    if run not in ('quick','full','advanced','advanced_quick'): raise HTTPException(400,'Unknown run')
    p=ROOT/'reports'/('metrics' if run=='full' else run)
    if not (p/'model_metrics.csv').exists(): raise HTTPException(404,'Train this run first')
    return p
@router.get('/runs')
def runs(): return {'runs':[r for r in ('advanced','full','advanced_quick','quick') if (ROOT/'reports'/('metrics' if r=='full' else r)/'model_metrics.csv').exists()]}
@router.get('/overview')
def overview():
    d=raw(); return {'encounters':len(d),'patients':d.patient_nbr.nunique(),'early':int(d.readmitted.eq('<30').sum()),'excluded':int(d.discharge_disposition_id.isin(EXCLUDE).sum()),'missing':records(d.eq('?').mean().mul(100).sort_values(ascending=False).rename('percent').rename_axis('feature').reset_index().query('percent > 0'))}
@router.get('/insights')
def insights(age:str='All',gender:str='All'):
    d=raw(); d=d[~d.discharge_disposition_id.isin(EXCLUDE)].copy(); ages=sorted(d.age.unique()); genders=sorted(d.gender.unique());
    if age!='All': d=d[d.age==age]
    if gender!='All': d=d[d.gender==gender]
    d=d.assign(early=d.readmitted.eq('<30').astype(int),diagnosis=d.diag_1.map(diagnosis))
    def groups(c):
        a=d.groupby(c).early.agg(['size','mean','sum']).reset_index().rename(columns={c:'group','size':'count','mean':'rate','sum':'early'}); return records(a)
    return {'n':len(d),'ages':ages,'genders':genders,'age':groups('age'),'gender':groups('gender'),'diagnosis':groups('diagnosis'),'visits':groups('number_inpatient'),'a1c':groups('A1Cresult'),'stay':groups('time_in_hospital'),'medication':groups('change')}
@router.get('/results/{run}')
def results(run:str):
    p=run_dir(run); out={}
    for name in ['model_metrics','patient_bootstrap','tuning','permutation_importance','xgboost_shap','predictions','curve_points','ablation_summary','feature_selection','tuning_trials','learning_curves','subgroup_metrics']:
        f=p/(name+'.csv'); out[name]=records(pd.read_csv(f)) if f.exists() else []
    for name in ['winner','cohort']:
        f=p/(name+'.json'); out[name]=json.loads(f.read_text()) if f.exists() else {}
    return out
@router.get('/plot/{run}/{name}')
def plot(run:str,name:str):
    from fastapi.responses import FileResponse
    if name not in ('roc','pr','calibration'): raise HTTPException(404)
    f=run_dir(run)/(name+'.png');
    if not f.exists(): raise HTTPException(404)
    return FileResponse(f,media_type='image/png')
@lru_cache(maxsize=2)
def model(run):
    run_dir(run); p=ROOT/'models'/('best.joblib' if run=='full' else run+'/best.joblib')
    if not p.exists(): raise HTTPException(503,'Saved model unavailable; train this run')
    return load(p)
@router.get('/form/{run}')
def form(run:str):
    b=model(run); d=raw(); d=d[~d.discharge_disposition_id.isin(EXCLUDE)]; r=d.iloc[0]; fields=[]
    from src.features import NUM
    for k in b['raw_fields']:
        vals=sorted(d[k].unique().tolist())
        if k=='discharge_disposition_id': vals=[v for v in vals if v not in EXCLUDE]
        fields.append({'name':k,'default':str(r[k]),'options':vals if k not in NUM and len(vals)<=100 else [],'numeric':k in NUM})
    return {'fields':fields,'model':b['model']}
@router.post('/predict/{run}')
def prediction(run:str,record:dict):
    try: return predict(model(run),record)
    except (ValueError,KeyError,TypeError) as e: raise HTTPException(422,str(e))
