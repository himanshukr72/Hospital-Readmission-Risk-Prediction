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

## DiaRisk React dashboard update

Preserve your existing `.git`, `.venv`, data, models and reports when copying the update. From the project root, start `python -m uvicorn api.main:app --port 8000`; in a second terminal run `cd frontend`, `npm install`, `npm run dev`. Browse http://localhost:5173. Node 22 LTS recommended. `npm run build` checks TypeScript and builds the frontend. The old Streamlit dashboard is retained. Choose full or demonstration results in the header. Existing results include static ROC/PR/reliability plots and global importance. Run `python -m src.train --quick` (or without `--quick` for full training) to generate aggregate threshold sweeps; restart the API afterward. Only the saved winning model is available for individual predictions. Local SHAP and individual multi-model predictions are not implemented. React uses Recharts and custom CSS rather than additional chart libraries. No patient IDs are served. Docker: `docker compose up --build`, frontend port 5173. Frontend dependencies/build and Docker were not tested in the offline packaging environment.

## Advanced experiment (integrated update)
From repository root, install `python -m pip install -r requirements-dev.txt -r requirements-advanced.txt`.
Run `python -m src.train_advanced --quick` for the subset smoke run, or `python -m src.train_advanced --trials 50` for the entire eligible cohort. The advanced runner performs grouped ablation, exploratory feature selection, Optuna TPE tuning including positive class weight, out-of-fold stacking, soft voting, learning curves, calibration, threshold selection, patient bootstrap, subgroup analysis, permutation importance and XGBoost TreeSHAP. Outputs and model bundles are written separately to `reports/advanced[_quick]` and `models/advanced[_quick]`; original experiments are untouched. The baseline pipeline retains SMOTENC and ADASYN comparisons. The advanced runner tunes class weights but does not retune oversampling ratios. Feature selection and tuning CV scores are exploratory because configurations are selected on the same development folds. Prior test results were inspected; the reused test partition is not independent external validation. Educational use only.

Start API with `python -m uvicorn api.main:app --port 8000` and frontend with `cd frontend; npm.cmd install; npm.cmd run dev`. Choose Advanced full dataset or Advanced demo from the run selector. Restart API after training. The updated frontend source was not built or browser-tested here.

When upgrading an existing checkout, copy source/config files only, not `.git`, `.venv`, `data`, `models`, or `reports`; preserve locally generated results. Install Optuna via requirements-advanced.txt. See reports/advanced_quick/cohort.json for actual bundled demo sizes.


# DiaRisk notebook upgrade

This upgrade replaces nine report-viewer notebooks with fourteen executable analysis notebooks. Existing production training and inference modules are unchanged; `src/notebook_utils.py` is new. Notebook experiments import reusable production estimators and preprocessing, and implement training/evaluation loops in visible cells. They are instructional experiments, not identical to the integrated advanced training run. Notebook 14 optionally runs the exact production command. No previous full-dataset training artifacts are replaced by the notebook experiments.

## Install and run (Windows PowerShell)

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-advanced.txt -r requirements-notebooks.txt
python -m jupyter lab
```

Alternatively install the VS Code Python and Jupyter extensions, open a notebook, select the `.venv` Python kernel, and click Run All. Keep both CSV files under `data/raw/`.

Notebooks 01–14 can be run independently. Each loads its own data and uses the same patient-level partitions; notebook 01 audits all raw rows regardless of QUICK. All modeling notebooks default to `QUICK = True` (3,000 sampled patients, approximately 2,611 training encounters). Change to `False` in each relevant notebook for full eligible cohort. Full tuning and ablation can be computationally expensive. Notebook results go to `reports/notebook_quick/` or `reports/notebook_full/`. These are not the reports consumed by the production dashboard. Notebook 14 reads `reports/advanced_quick/` or `reports/advanced/` and loads the matching saved model. To run actual production training, execute `python -m src.train_advanced --trials 50` from the project root or explicitly enable RUN_TRAINING in notebook 14. This can update the production reports/model for the selected scope. Do not use test performance to select or tune models. The original test partition has already been examined, so results are exploratory and not independent validation.

## Notebook sequence

01 raw data audit, missingness, target and exclusion checks; 02 EDA and associations; 03 leakage-safe preprocessing; 04 baseline training; 05 boosting training; 06 SMOTENC, ADASYN, weights; 07 randomised search; 08 probability calibration and threshold selection; 09 permutation importance and TreeSHAP; 10 advanced feature ablation; 11 selection and Optuna; 12 soft voting and stacking; 13 learning curves and patient bootstrap; 14 production model and prediction verification.

Notebooks contain actual executable code; results are generated when run. Do not publish patient-level data, identifiers, or sensitive outputs. These are academic experiments, not clinically validated predictions.

## Updating an existing project

Extract the small notebook patch and copy its `notebooks/` folder into your project root (replace matching `.ipynb` files), copy `src/notebook_utils.py` into the existing `src/` folder, and copy `requirements-notebooks.txt` and this README to the project root. Do not delete existing Python modules, `.git`, `.venv`, `models/`, `reports/`, or `data/`. Commit notebooks and code only after reviewing `git status`.
