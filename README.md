# Higher Education Attainment Predictor

A Flask web application that analyses student data, predicts whether a student will
continue to higher education, and recommends support for students at risk.

## Features
- Upload a CSV file and save it to an SQL database (SQLite)
- Exploratory data analysis: Chi-Square test, correlation, feature importance
- Model training: XGBoost, LightGBM, HistGradientBoosting, Stacking Ensemble, MLP
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Cohen's Kappa, MCC,
  5-fold stratified cross-validation, IQR outlier detection
- Explanations: LIME and XGBoost TreeSHAP (global and per student)
- Rule-based intervention recommendations, CSV report export
- Dashboard with summary cards and charts

## Requirements
- Python 3.10 or newer
- Packages listed in `requirements.txt`

## Setup (Windows PowerShell)
```powershell
cd D:\higher-ed-predictor
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```
Open http://127.0.0.1:5000 in a browser.

## How to use
1. **Upload Data**: choose the CSV file and click "Upload and save to database".
2. **Data Analysis**: pick a variable to see its relationship with the outcome.
3. **Model Training**: click "Train models" (takes 2 to 4 minutes).
4. **Predict Student**: load a student by ID or fill in the form, then click "Predict".
5. **Interventions**: click "Generate recommendations", then filter, mark done or export.
6. **Dashboard**: overview of students, risk, models and interventions.

## Project structure
```
run.py                  starts the server
config.py               settings (database, folders, upload limits)
app/__init__.py         creates the app and registers the pages
app/models.py           database tables
app/routes/             one file per page
app/services/           data import, analysis, training, prediction, recommender
app/templates/          HTML pages
app/static/             CSS and JavaScript
uploads/                uploaded CSV files
saved_models/           trained models (model_bundle.pkl)
data/                   SQLite database (higher_ed.db)
```

## Notes
- On computers where Windows Application Control blocks compiled libraries, CatBoost,
  TensorFlow and SHAP may not load. This project uses HistGradientBoosting, an MLP
  network and XGBoost TreeSHAP instead. `run.py` also sets
  `DISABLE_SQLALCHEMY_CEXT_RUNTIME=1`.
- The bundled dataset shows no predictive signal (ROC-AUC about 0.50), so predictions
  are a demonstration of the system, not reliable results.
- Delete `data\higher_ed.db` and run the app again to reset the database.