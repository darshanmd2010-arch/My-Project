import os
import pickle

import numpy as np
from flask import current_app
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from app import db
from app.models import ModelResult
from app.services.analysis_service import FEATURES, NUMERIC, _load

BUNDLE_NAME = "model_bundle.pkl"
CV_FOLDS = 5
IQR_FACTOR = 1.5


# ---------------------------------------------------------------- encoding
def build_encoder(data):
    """Learn how to turn student records into numbers (done once, on all data).

    Also finds outliers with the IQR rule: values below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR. Those values are capped (winsorized) when encoding.
    """
    numeric, bounds, outliers = {}, {}, {}
    for name in FEATURES:
        if name not in NUMERIC:
            continue
        present = np.array([v for v in data[name] if v is not None], dtype=float)
        if len(present) == 0:
            numeric[name], outliers[name] = 0.0, 0
            continue
        numeric[name] = float(np.median(present))

        q1, q3 = np.percentile(present, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
            bounds[name] = (float(low), float(high))
            outliers[name] = int(((present < low) | (present > high)).sum())
        else:
            outliers[name] = 0  # no spread, so the rule cannot be applied

    categories = {}
    for name in FEATURES:
        if name not in NUMERIC:
            categories[name] = sorted({v for v in data[name] if v is not None})

    columns = list(numeric) + [
        f"{name}={c}" for name, cats in categories.items() for c in cats
    ]
    return {
        "numeric": numeric,
        "categories": categories,
        "columns": columns,
        "bounds": bounds,
        "outliers": outliers,
    }


def encode(data, enc):
    """data is a dict: feature name -> list of values. Returns a numeric matrix."""
    cols = []
    for name, median in enc["numeric"].items():
        col = np.array(
            [float(v) if v is not None else median for v in data[name]]
        )
        limits = enc.get("bounds", {}).get(name)
        if limits:
            col = np.clip(col, limits[0], limits[1])
        cols.append(col)
    for name, cats in enc["categories"].items():
        values = data[name]
        for c in cats:
            cols.append(np.array([1.0 if v == c else 0.0 for v in values]))
    return np.column_stack(cols)


# ------------------------------------------------------------------ models
def make_models():
    # Imported here so the web app still starts if a library is ever blocked
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier

    def xgb():
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=1, random_state=42,
        )

    def lgbm():
        return LGBMClassifier(
            n_estimators=200, num_leaves=15, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbose=-1,
            n_jobs=1, random_state=42,
        )

    def hgb():
        return HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=42
        )

    return {
        "XGBoost": xgb(),
        "LightGBM": lgbm(),
        "HistGradientBoosting": hgb(),
        "Stacking Ensemble": StackingClassifier(
            estimators=[("xgb", xgb()), ("lgbm", lgbm()), ("hgb", hgb())],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            n_jobs=1,
        ),
        "DNN (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), early_stopping=True,
                max_iter=200, random_state=42,
            ),
        ),
    }


def _metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "kappa": float(cohen_kappa_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def cross_validate_all(X, y):
    """Stratified k-fold cross-validation for every model."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    out = {}
    for name, model in make_models().items():
        try:
            s = cross_validate(
                model, X, y, cv=skf, scoring=["accuracy", "roc_auc"], n_jobs=1
            )
            out[name] = {
                "acc_mean": float(s["test_accuracy"].mean()),
                "acc_std": float(s["test_accuracy"].std()),
                "auc_mean": float(s["test_roc_auc"].mean()),
                "auc_std": float(s["test_roc_auc"].std()),
            }
        except Exception as exc:
            out[name] = {"error": str(exc)[:120]}
    return out


# ---------------------------------------------------------------- training
def train_all():
    """Train every model on 80% of the data, test on 20% (stratified),
    then run stratified cross-validation."""
    data, y = _load()
    if len(y) < 100:
        raise ValueError("Not enough students in the database. Upload the CSV first.")

    enc = build_encoder(data)
    X = encode(data, enc)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    fitted, results = {}, []
    for name, model in make_models().items():
        try:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            results.append({"model": name, **_metrics(y_test, proba)})
            fitted[name] = model
        except Exception as exc:
            results.append({"model": name, "error": str(exc)[:120]})

    good = [r for r in results if "error" not in r]
    if not good:
        raise RuntimeError("No model could be trained.")
    best = max(good, key=lambda r: r["roc_auc"])["model"]

    cv = cross_validate_all(X, y)

    # A small sample of training rows, used later for LIME explanations
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)

    # Save the trained models, the encoder and the validation results together
    folder = current_app.config["MODEL_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, BUNDLE_NAME)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "encoder": enc,
                "models": fitted,
                "best": best,
                "cv": cv,
                "folds": CV_FOLDS,
                "n_rows": int(len(y)),
                "lime_sample": X_train[sample_idx],
            },
            f,
        )

    # Save the metrics in the SQL database (replaces the previous run)
    ModelResult.query.delete()
    for r in good:
        db.session.add(
            ModelResult(
                model_name=r["model"], accuracy=r["accuracy"],
                precision=r["precision"], recall=r["recall"],
                f1_score=r["f1"], roc_auc=r["roc_auc"],
                cohen_kappa=r["kappa"], mcc=r["mcc"],
                is_best=(r["model"] == best), model_path=BUNDLE_NAME,
            )
        )
    db.session.commit()

    share_yes = float(y_test.mean())
    return {
        "results": results,
        "cv": cv,
        "best": best,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "baseline_accuracy": max(share_yes, 1 - share_yes),
    }


def load_bundle():
    """Load the saved models (used by the Predict page)."""
    path = os.path.join(current_app.config["MODEL_FOLDER"], BUNDLE_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)