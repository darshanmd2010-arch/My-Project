import json

import numpy as np

from app import db
from app.models import Prediction, Student
from app.services.analysis_service import FEATURES, pretty
from app.services.ml_service import encode


def form_spec(bundle):
    """Describe the input form: one field per feature, taken from the trained encoder."""
    enc = bundle["encoder"]
    spec = []
    for name in FEATURES:
        if name in enc["numeric"]:
            spec.append({
                "name": name,
                "label": pretty(name),
                "kind": "number",
                "default": round(enc["numeric"][name], 2),
                "options": [],
            })
        else:
            options = enc["categories"][name]
            spec.append({
                "name": name,
                "label": pretty(name),
                "kind": "select",
                "default": options[0] if options else "",
                "options": options,
            })
    return spec


def values_for_student(student_id):
    """Return the saved values of a student, or None if the ID does not exist."""
    student = Student.query.filter_by(student_id=student_id).first()
    if student is None:
        return None
    values = {}
    for name in FEATURES:
        v = getattr(student, name)
        if v is not None:
            values[name] = v
    return values


def risk_level(probability):
    """Risk of NOT continuing to higher education."""
    if probability < 0.40:
        return "High"
    if probability < 0.60:
        return "Medium"
    return "Low"


# ------------------------------------------------------------ explanations
def _contribs(bundle, X):
    """XGBoost's built-in TreeSHAP contributions (one row per student)."""
    model = bundle["models"].get("XGBoost")
    if model is None:
        return None
    try:
        import xgboost as xgb

        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    except Exception:
        return None


def _explain(bundle, X):
    """Top 5 factors for one student (XGBoost contributions)."""
    contribs = _contribs(bundle, X)
    if contribs is None:
        return []

    grouped = {}
    for column, c in zip(bundle["encoder"]["columns"], contribs[0]):
        feature = column.split("=", 1)[0]
        grouped[feature] = grouped.get(feature, 0.0) + float(c)

    ranked = sorted(grouped.items(), key=lambda p: abs(p[1]), reverse=True)[:5]
    return [
        {
            "feature": pretty(f),
            "effect": round(v, 3),
            "direction": "raises" if v > 0 else "lowers",
        }
        for f, v in ranked
    ]


def global_factors(bundle, top=12):
    """Average size of each factor's effect over a sample of students."""
    sample = bundle.get("lime_sample")
    if sample is None:
        return []
    contribs = _contribs(bundle, sample)
    if contribs is None:
        return []

    mean_abs = np.abs(contribs).mean(axis=0)
    grouped = {}
    for column, v in zip(bundle["encoder"]["columns"], mean_abs):
        feature = column.split("=", 1)[0]
        grouped[feature] = grouped.get(feature, 0.0) + float(v)

    ranked = sorted(grouped.items(), key=lambda p: p[1], reverse=True)[:top]
    biggest = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
    return [
        {"feature": pretty(f), "value": round(v, 4), "pct": round(v / biggest * 100)}
        for f, v in ranked
    ]


def _clean(label):
    """Make a LIME label easier to read."""
    suffix = ""
    if label.endswith("=1"):
        label, suffix = label[:-2], " (present)"
    elif label.endswith("=0"):
        label, suffix = label[:-2], " (absent)"
    return label.replace("_", " ") + suffix


def _lime(bundle, x):
    """LIME explanation of the best model's prediction for one student."""
    sample = bundle.get("lime_sample")
    if sample is None:
        return []
    try:
        from lime.lime_tabular import LimeTabularExplainer

        columns = bundle["encoder"]["columns"]
        categorical = [i for i, c in enumerate(columns) if "=" in c]
        explainer = LimeTabularExplainer(
            sample,
            feature_names=columns,
            class_names=["No", "Yes"],
            categorical_features=categorical,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        model = bundle["models"][bundle["best"]]
        exp = explainer.explain_instance(
            x, model.predict_proba, num_features=8, num_samples=1000, labels=(1,)
        )
        items = exp.as_list(label=1)
    except Exception:
        return []

    return [
        {
            "label": _clean(label),
            "weight": round(float(w), 4),
            "direction": "raises" if w > 0 else "lowers",
        }
        for label, w in items
    ]


# -------------------------------------------------------------- prediction
def predict_student(bundle, raw_values, student_id=""):
    """Predict the chance that a student continues to higher education."""
    enc = bundle["encoder"]
    data = {}
    for name in FEATURES:
        raw = raw_values.get(name, "") or ""
        raw = str(raw).strip()
        if name in enc["numeric"]:
            if raw == "":
                data[name] = [None]
            else:
                try:
                    data[name] = [float(raw)]
                except ValueError:
                    raise ValueError(f"{pretty(name)} must be a number.")
        else:
            data[name] = [raw if raw else None]

    X = encode(data, enc)
    best = bundle["best"]
    probability = float(bundle["models"][best].predict_proba(X)[0, 1])
    level = risk_level(probability)
    factors = _explain(bundle, X)
    lime = _lime(bundle, X[0])

    db.session.add(
        Prediction(
            student_id=student_id or "manual",
            probability_continue=probability,
            risk_level=level,
            top_factors=json.dumps(factors),
        )
    )
    db.session.commit()

    return {
        "probability": probability,
        "risk": level,
        "model": best,
        "factors": factors,
        "lime": lime,
    }