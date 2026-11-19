from flask import Blueprint, flash, render_template, request

from app import db
from app.models import ModelResult
from app.services.ml_service import load_bundle
from app.services.predict_service import form_spec, predict_student, values_for_student

bp = Blueprint("predict", __name__)


@bp.route("/predict", methods=["GET", "POST"])
def predict():
    bundle = load_bundle()
    if bundle is None:
        return render_template("predict.html", active="predict", ready=False)

    spec = form_spec(bundle)
    values = {}
    result = None
    student_id = request.values.get("student_id", "").strip()

    if request.method == "POST":
        values = {f["name"]: request.form.get(f["name"], "") for f in spec}
        try:
            result = predict_student(bundle, values, student_id)
        except ValueError as exc:
            db.session.rollback()
            flash(str(exc), "error")
    elif student_id:
        loaded = values_for_student(student_id)
        if loaded is None:
            flash(f"Student {student_id} was not found in the database.", "error")
            student_id = ""
        else:
            values = loaded

    for f in spec:
        values.setdefault(f["name"], f["default"])

    best = ModelResult.query.filter_by(is_best=True).first()
    weak = best is not None and best.roc_auc < 0.60

    return render_template(
        "predict.html",
        active="predict",
        ready=True,
        spec=spec,
        values=values,
        result=result,
        student_id=student_id,
        best=best,
        weak=weak,
    )