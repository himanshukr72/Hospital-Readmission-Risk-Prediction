"""Stage 1: deterministic audit and cohort construction; no fitted preprocessing."""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', type=Path)
    p.add_argument('--out', type=Path, default=Path('outputs/stage1'))
    p.add_argument('--processed', type=Path, default=None)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    processed = a.processed or a.out
    processed.mkdir(parents=True, exist_ok=True)
    # Preserve literal 'None' in lab-result columns: it means no test recorded.
    raw = pd.read_csv(a.csv, dtype=str, keep_default_na=False)
    df = raw.apply(lambda s: s.str.strip()).replace({'?': np.nan, '': np.nan})
    required = {'encounter_id','patient_nbr','readmitted','discharge_disposition_id','A1Cresult','max_glu_serum'}
    if missing := required - set(df): raise ValueError(f'Missing columns: {missing}')
    report = {'raw_shape': list(df.shape), 'unique_patients': int(df.patient_nbr.nunique()), 'exact_duplicate_rows': int(df.duplicated().sum()), 'duplicate_encounter_rows': int(df.encounter_id.duplicated(keep=False).sum()), 'raw_target_counts': df.readmitted.value_counts(dropna=False).to_dict()}
    audit = pd.DataFrame({'missing_count':df.isna().sum(),'missing_percent':df.isna().mean()*100,'unique_nonmissing':df.nunique()})
    audit.sort_values('missing_percent', ascending=False).to_csv(a.out/'column_audit.csv')
    with open(a.out/'category_counts.json','w') as f: json.dump({c:df[c].value_counts(dropna=False).rename(index=lambda x: '<MISSING>' if pd.isna(x) else str(x)).to_dict() for c in df if c not in ['encounter_id','patient_nbr']},f,indent=2)
    df = df.drop_duplicates().copy()
    if df.encounter_id.isna().any() or df.patient_nbr.isna().any(): raise ValueError('Missing identifiers; inspect source.')
    if df.encounter_id.duplicated().any(): raise ValueError('Conflicting encounter IDs; inspect source.')
    if not df.readmitted.isin(['<30','>30','NO']).all(): raise ValueError('Unexpected target labels; inspect source.')
    numeric = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses']
    for c in numeric:
        if c not in df: raise ValueError(f'Missing numeric feature {c}')
        df[c] = pd.to_numeric(df[c], errors='raise')
        if (df[c]<0).any(): raise ValueError(f'Negative count in {c}')
    if not df.time_in_hospital.between(1,14).all(): raise ValueError('Stay outside dataset inclusion criteria')
    df['target'] = df.readmitted.eq('<30').astype('int8')
    # Keep all rows for auditing. Proposed cohort exclusions must be checked against IDs_mapping.csv.
    df.groupby('discharge_disposition_id',dropna=False).target.agg(['size','sum','mean']).to_csv(a.out/'discharge_audit.csv')
    df.drop(columns='readmitted').to_csv(processed/'audited_encounters.csv',index=False)
    report.update({'audited_shape':list(df.shape),'target_counts':df.target.value_counts().to_dict(),'positive_rate':float(df.target.mean()),'patients_with_multiple_encounters':int((df.patient_nbr.value_counts()>1).sum())})
    with open(a.out/'audit_summary.json','w') as f: json.dump(report,f,indent=2,default=str)
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__': main()
