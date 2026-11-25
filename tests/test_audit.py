import subprocess, sys
from pathlib import Path
import pandas as pd

def test_audit_synthetic(tmp_path):
    row = dict(encounter_id='1', patient_nbr='2', readmitted='<30', discharge_disposition_id='1', A1Cresult='None', max_glu_serum='None', time_in_hospital='2', num_lab_procedures='3', num_procedures='0', num_medications='4', number_outpatient='0', number_emergency='0', number_inpatient='0', number_diagnoses='2')
    source = tmp_path / 'raw.csv'
    pd.DataFrame([row, row]).to_csv(source, index=False)
    out = tmp_path / 'audit'
    script = Path(__file__).resolve().parents[1] / 'src/audit.py'
    subprocess.run([sys.executable, str(script), str(source), '--out', str(out)], check=True)
    df = pd.read_csv(out / 'audited_encounters.csv', keep_default_na=False)
    assert len(df) == 1
    assert df.target.iloc[0] == 1
    assert df.A1Cresult.iloc[0] == 'None'
    assert 'readmitted' not in df
