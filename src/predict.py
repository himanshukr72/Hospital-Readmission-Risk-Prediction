import joblib,pandas as pd
from .features import features,EXCLUDE
from .evaluation.calibration import apply

def load(path='models/best.joblib'): return joblib.load(path)
def predict(bundle,record):
    missing=set(bundle['raw_fields'])-set(record)
    if missing: raise ValueError('Missing fields: '+', '.join(sorted(missing)))
    if str(record['discharge_disposition_id']) in EXCLUDE: raise ValueError('Encounter outside eligible discharge cohort')
    frame=pd.DataFrame([{k:record[k] for k in bundle['raw_fields']}])
    if bundle.get('advanced'):
        from .advanced_experiment import expanded
        x=expanded(frame)[0]
    else: x=features(frame)
    x=x.reindex(columns=bundle['feature_columns']); p=float(apply(bundle['calibrator'],bundle['pipeline'].predict_proba(x)[:,1])[0]); return {'probability':p,'threshold':bundle['threshold'],'flag':int(p>=bundle['threshold']),'model':bundle['model'],'educational_only':True}
