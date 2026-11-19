from flask import Blueprint, jsonify, render_template

from app import db
from app.models import ModelResult, Student
from app.services import ml_service
from app.services.analysis_service import pretty
from app.services.predict_service import global_factors

bp = Blueprint("training", __name__)


@bp.route("/training")
def training():
    results = ModelResult.query.order_by(ModelResult.roc_auc.desc()).all()

    total = Student.query.count()
    yes = Student.query.filter_by(higher_education_continuation="Yes").count()
    no = Student.query.filter_by(higher_education_continuation="No").count()
    baseline = max(yes, no) / total if total else 0

    best = next((r for r in results if r.is_best), None)
    weak = best is not None and best.roc_auc < 0.60
    trained_at = results[0].trained_at if results else None

    # Cross-validation, outlier and factor results saved with the trained models
    cv, folds, outlier_rows, factors_global = {}, 0, [], []
    bundle = ml_service.load_bundle()
    if bundle:
        cv = bundle.get("cv", {})
        folds = bundle.get("folds", 0)
        factors_global = global_factors(bundle)
        enc = bundle["encoder"]
        n_rows = bundle.get("n_rows", 0) or 1
        for name, count in enc.get("outliers", {}).items():
            limits = enc.get("bounds", {}).get(name)
            outlier_rows.append({
                "feature": pretty(name),
                "count": count,
                "pct": round(count / n_rows * 100, 2),
                "low": round(limits[0], 2) if limits else None,
                "high": round(limits[1], 2) if limits else None,
            })
        outlier_rows.sort(key=lambda r: r["count"], reverse=True)

    return render_template(
        "training.html",
        active="training",
        results=results,
        total=total,
        baseline=baseline,
        best=best,
        weak=weak,
        trained_at=trained_at,
        cv=cv,
        folds=folds,
        outlier_rows=outlier_rows,
        factors_global=factors_global,
    )


@bp.route("/training/run", methods=["POST"])
def run():
    try:
        out = ml_service.train_all()
        return jsonify({"ok": True, "best": out["best"]})
    except Exception as exc:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(exc)}), 400