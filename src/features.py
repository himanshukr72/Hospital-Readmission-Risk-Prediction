import pandas as pd
import numpy as np
NUM = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses','prior_visits']
DROP = ['encounter_id','patient_nbr','readmitted','target','weight','examide','citoglipton']
EXCLUDE = {'11','13','14','19','20'}
def diagnosis(v):
    if pd.isna(v) or str(v) in ('','?'): return 'Unknown'
    s=str(v)
    if s.startswith(('V','E')): return 'Supplementary'
    try: n=float(s)
    except ValueError: return 'Unknown'
    if 250<=n<251: return 'Diabetes'
    for lo,hi,label in [(390,460,'Circulatory'),(460,520,'Respiratory'),(520,580,'Digestive'),(800,1000,'Injury'),(710,740,'Musculoskeletal'),(580,630,'Genitourinary'),(140,240,'Neoplasm')]:
        if lo<=n<hi: return label
    return 'Other'
def features(df):
    x=df.copy().replace({'?':np.nan,'':np.nan})
    x=x.drop(columns=DROP,errors='ignore')
    for c in ('diag_1','diag_2','diag_3'):
        if c in x: x[c]=x[c].map(diagnosis)
    for c in NUM[:-1]:
        if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
    x['prior_visits']=x[['number_outpatient','number_emergency','number_inpatient']].sum(axis=1,min_count=3)
    for c in x.columns.difference(NUM): x[c]=x[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return x
