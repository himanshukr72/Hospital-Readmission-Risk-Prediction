import json
from pathlib import Path
import numpy as np,pandas as pd
from src.features import features,diagnosis
from src.train import split
from src.evaluation.calibration import calibrate,apply,threshold

def test_group_split():
    g=pd.Series(np.repeat(np.arange(50),2)); y=pd.Series(np.tile([0,1],50)); a,b=split(y,g); assert not set(g.iloc[a]) & set(g.iloc[b])
def test_features():
    r=json.loads(Path('configs/example.json').read_text()); r.update(patient_nbr='123',readmitted='<30'); x=features(pd.DataFrame([r])); assert 'patient_nbr' not in x and 'readmitted' not in x; assert diagnosis('250.83')=='Diabetes'; assert x.prior_visits.iloc[0]>=0

def test_calibration():
    p=np.array([.1,.2,.4,.7,.8,.9]); y=np.array([0,0,1,0,1,1]); q=apply(calibrate(p,y),p); assert np.all((q>=0)&(q<=1)); assert 0<=threshold(y,q)<=1
