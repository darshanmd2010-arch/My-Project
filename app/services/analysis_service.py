import numpy as np
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import Float, Integer

from app import db
from app.models import Student

TARGET = "higher_education_continuation"

# dropout_status is excluded: it is another outcome and would leak the answer
EXCLUDE = {"id", "student_id", "created_at", "dropout_status", TARGET}

FEATURES = [c.name for c in Student.__table__.columns if c.name not in EXCLUDE]
NUMERIC = {
    c.name
    for c in Student.__table__.columns
    if c.name in FEATURES and isinstance(c.type, (Integer, Float))
}

_cache = {}


def pretty(name):
    return name.replace("_", " ").title()


def _load():
    """Load all students from the SQL database as columns of values."""
    cols = FEATURES + [TARGET]
    rows = db.session.query(*[getattr(Student, c) for c in cols]).all()
    rows = [r for r in rows if r[-1] in ("Yes", "No")]
    data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    y = np.array([1 if v == "Yes" else 0 for v in data[TARGET]])
    return data, y


def _label(value):
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def feature_breakdown(name):
    """% of students continuing to higher education in each group of a variable."""
    if name not in FEATURES:
        raise ValueError("Unknown variable")

    data, y = _load()
    vals = data[name]
    idx = [i for i, v in enumerate(vals) if v is not None]
    if not idx:
        raise ValueError("No data for this variable")

    v = [vals[i] for i in idx]
    t = y[idx]

    if name in NUMERIC and len(set(v)) > 10:
        arr = np.array(v, dtype=float)
        edges = np.unique(np.quantile(arr, np.linspace(0, 1, 6)))
        g = np.digitize(arr, edges[1:-1])
        labels = [f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    else:
        cats = sorted(set(v))
        position = {c: i for i, c in enumerate(cats)}
        g = np.array([position[x] for x in v])
        labels = [_label(c) for c in cats]

    n_groups = len(labels)
    counts = np.bincount(g, minlength=n_groups)
    yes = np.bincount(g, weights=t, minlength=n_groups)
    rates = np.where(counts > 0, yes / np.maximum(counts, 1) * 100, 0)

    result = {
        "name": name,
        "title": pretty(name),
        "labels": labels,
        "counts": [int(c) for c in counts],
        "rates": [round(float(r), 1) for r in rates],
        "chi2": None,
        "p": None,
        "dof": None,
        "cramers_v": None,
    }

    # Chi-Square test of independence (group x continues Yes/No)
    table = np.array([counts - yes, yes]).T
    table = table[counts > 0]
    if table.shape[0] >= 2 and table[:, 0].sum() > 0 and table[:, 1].sum() > 0:
        chi2, p, dof, _ = chi2_contingency(table)
        result.update(
            chi2=float(chi2),
            p=float(p),
            dof=int(dof),
            cramers_v=float(np.sqrt(chi2 / table.sum())),
        )
    return result


def correlations():
    """Correlation of each numeric variable with continuing (Yes = 1, No = 0)."""
    data, y = _load()
    out = []
    for name in sorted(NUMERIC):
        idx = [i for i, v in enumerate(data[name]) if v is not None]
        if len(idx) < 3:
            continue
        arr = np.array([data[name][i] for i in idx], dtype=float)
        t = y[idx]
        if arr.std() == 0 or t.std() == 0:
            continue
        r = float(np.corrcoef(arr, t)[0, 1])
        out.append({"feature": pretty(name), "r": round(r, 3)})
    out.sort(key=lambda d: abs(d["r"]), reverse=True)
    return out


def feature_importance(top=15):
    """Random Forest feature importance (cached until the data changes)."""
    data, y = _load()
    n = len(y)
    cached = _cache.get("importance")
    if cached and cached[0] == n:
        return cached[1]

    columns = []
    for name in FEATURES:
        vals = data[name]
        if name in NUMERIC:
            present = [v for v in vals if v is not None]
            fill = float(np.median(present)) if present else 0.0
            columns.append([float(v) if v is not None else fill for v in vals])
        else:
            cats = sorted({v for v in vals if v is not None})
            position = {c: i for i, c in enumerate(cats)}
            columns.append([position.get(v, -1) for v in vals])

    X = np.column_stack(columns).astype(float)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=1
    )
    model.fit(X, y)

    ranked = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda p: p[1], reverse=True
    )[:top]
    result = [
        {"feature": pretty(f), "importance": round(float(s), 4)} for f, s in ranked
    ]
    _cache["importance"] = (n, result)
    return result