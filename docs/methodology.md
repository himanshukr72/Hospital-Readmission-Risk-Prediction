# Diabetes hospital readmission: boosting algorithms

End-to-end academic project using UCI Diabetes 130-US Hospitals (1999–2008). Predicts recorded readmission within 30 days at discharge, not diabetes onset. Educational prototype, not clinically validated.

## Run on Windows PowerShell (Python 3.11)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
python src/audit.py data/raw/diabetic_data.csv --out reports/audit --processed data/processed
python -m src.train
```
For a fast integration run use `python -m src.train --quick`. It uses 3,000 patients and fewer estimators. The included `reports/quick` and `models/quick/best.joblib` are from an actual quick run, not full-cohort estimates. Full training writes `reports/metrics` and `models/best.joblib`.

## Launch after full training
```powershell
python -m streamlit run dashboard/app.py
```
In another terminal:
```powershell
python -m uvicorn api.main:app --port 8000
```
Dashboard: http://localhost:8501 ; API docs: http://localhost:8000/docs . POST example JSON from configs/example.json to /predict. Docker alternative: `docker compose up --build` (after training locally). For included quick demo instead, set `$env:MODEL_PATH='models/quick/best.joblib'` and `$env:REPORT_DIR='reports/quick'` before launching both applications. Only load trusted joblib files.

## Methodology
- Exclude mapped death/hospice discharge codes 11,13,14,19,20; keep repeated encounters, separate patients across splits. This defines an eligible discharge cohort, not a competing-risk survival model.
- Remove identifiers, weight (97% missing), and constant drug features. Preserve literal None for unperformed lab tests; group ICD-9 codes coarsely and aggregate previous visits. Impute and encode only within training folds.
- Approximate 60/10/10/20 patient-disjoint train/calibration/threshold-selection/test split. EDA on training patients; association p-values are exploratory and ignore within-patient dependence; effect sizes and Bonferroni adjustments included.
- Compare dummy, logistic regression, tree, random forest, AdaBoost, gradient boosting, XGBoost; compare XGBoost weighted, SMOTENC (mixed-type SMOTE), ADASYN and randomized-search XGBoost. ADASYN interpolates one-hot features and can create fractional categories; treat as experimental, not synthetic clinical records.
- Tune XGBoost with grouped 3-fold cross-validation, average precision objective. Fit sigmoid calibration on separate calibration set; select F1 threshold and champion by average precision on threshold-selection set. Test is used only for final evaluation. Average precision is the reported PR-AUC convention. F1 threshold is illustrative, not a clinical cost policy.
- Evaluate raw/calibrated/threshold-optimized metrics, ROC/PR/reliability plots, paired patient-cluster bootstrap CIs. SHAP uses XGBoost native TreeSHAP on raw margins for base XGBoost, not necessarily the champion; champion permutation importance uses test subset. No causal claims.
- Historical data, no prospective or external validation, no deployment authorization. Missing or unrecorded readmissions may bias labels. No dates for temporal splitting. Dashboard and Docker configurations provided but not launched in the build environment (Streamlit unavailable). API smoke-tested with bundled subset model.

## GitHub
Raw/processed CSVs and model binaries are bundled locally but ignored by Git by default; obtain original data from https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008 . Aggregate reports can be committed. Use `git init`, `git add .`, `git commit -m "Complete boosting research pipeline"` then add your remote and push.
