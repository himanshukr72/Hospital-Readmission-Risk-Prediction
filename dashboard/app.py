import os,json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.predict import load,predict
st.set_page_config(page_title='Readmission research',layout='wide'); st.title('Diabetic hospital readmission — boosting comparison'); st.warning('Educational research prototype; not for clinical decisions.')
root=Path(os.getenv('REPORT_DIR','reports/metrics')); path=os.getenv('MODEL_PATH','models/best.joblib')
if not (root/'model_metrics.csv').exists(): st.info('Run python -m src.train first'); st.stop()
if root.name=='quick': st.warning('Subset smoke-run results, not full-dataset estimates.')
tabs=st.tabs(['Cohort & EDA','Comparison','Calibration & thresholds','Explainability','Prediction'])
with tabs[0]:
    st.json(json.loads((root/'cohort.json').read_text())); st.dataframe(pd.read_csv(root/'eda/associations.csv')); files=sorted((root/'eda').glob('*_counts.csv')); f=st.selectbox('Feature',files,format_func=lambda p:p.stem); d=pd.read_csv(f); st.bar_chart(d.set_index(d.columns[0])[['0','1']])
with tabs[1]:
    m=pd.read_csv(root/'model_metrics.csv'); mode=st.selectbox('Evaluation mode',m['mode'].unique()); d=m[m['mode']==mode]; st.dataframe(d); st.bar_chart(d.set_index('model')[['recall','precision','pr_auc','roc_auc']]); st.image(str(root/'roc.png')); st.image(str(root/'pr.png')); st.dataframe(pd.read_csv(root/'patient_bootstrap.csv'))
with tabs[2]:
    st.image(str(root/'calibration.png')); st.dataframe(m[['model','mode','threshold','brier','recall','precision','f1']]); st.caption('Thresholds selected on separate validation patients, not test patients.')
with tabs[3]:
    for f in ['permutation_importance.csv','xgboost_shap.csv']:
        d=pd.read_csv(root/f); st.subheader(f); st.bar_chart(d.head(20).set_index(d.columns[0])[d.columns[1]])
with tabs[4]:
    b=load(path); example=json.loads(Path('configs/example.json').read_text()); text=st.text_area('Example encounter JSON',json.dumps(example,indent=2),height=400)
    if st.button('Predict'):
        try: st.json(predict(b,json.loads(text)))
        except Exception as e: st.error(str(e))
