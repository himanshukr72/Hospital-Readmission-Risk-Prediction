import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

def logits(p):
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p)).reshape(-1,1)
def calibrate(p,y): return LogisticRegression(C=1e4).fit(logits(p),y)
def apply(model,p): return model.predict_proba(logits(p))[:,1]
def threshold(y,p):
    precision,recall,t=precision_recall_curve(y,p)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(t[np.argmax(f)])
