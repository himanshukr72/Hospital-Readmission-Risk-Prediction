from imblearn.over_sampling import SMOTENC, ADASYN
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

def sampler(kind, categorical):
    if kind=='SMOTENC': return SMOTENC(categorical_features=categorical,sampling_strategy=.5,random_state=42)
    if kind=='ADASYN': return ADASYN(sampling_strategy=.5,random_state=42)
    return 'passthrough'

def weights(y): return compute_sample_weight('balanced',y)
