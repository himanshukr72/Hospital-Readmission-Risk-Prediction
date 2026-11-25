import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from sklearn.calibration import calibration_curve

def export(y, predictions, out):
    curves=[]; thresholds=[]
    for name,p in predictions.items():
        for kind,xy in [('roc',roc_curve(y,p)[:2]),('pr',precision_recall_curve(y,p)[:2][::-1]),('calibration',calibration_curve(y,p,n_bins=10)[::-1])]:
            curves.extend(dict(model=name,kind=kind,x=float(x),y=float(z)) for x,z in zip(*xy))
        for t in np.linspace(0,1,101):
            tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel(); thresholds.append(dict(model=name,threshold=t,tn=tn,fp=fp,fn=fn,tp=tp))
    pd.DataFrame(curves).to_csv(out/'curve_points.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'predictions.csv',index=False)
