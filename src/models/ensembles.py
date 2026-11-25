import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
class PrefitEnsemble(ClassifierMixin,BaseEstimator):
    def __init__(self,members,meta=None): self.members=members; self.meta=meta; self.classes_=np.array([0,1])
    def predict_proba(self,X):
        p=np.column_stack([m.predict_proba(X)[:,1] for m in self.members]); q=p.mean(axis=1) if self.meta is None else self.meta.predict_proba(p)[:,1]; return np.column_stack([1-q,q])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=.5).astype(int)
    def fit(self,X,y=None): raise RuntimeError('Prefit ensemble; train base members first')
