from flask import Blueprint, jsonify, render_template, request

from app.models import Student
from app.services import analysis_service as svc

bp = Blueprint("analysis", __name__)


@bp.route("/analysis")
def analysis():
    total = Student.query.count()
    yes = Student.query.filter_by(higher_education_continuation="Yes").count()
    no = Student.query.filter_by(higher_education_continuation="No").count()
    rate_yes = round(yes / total * 100, 1) if total else 0
    rate_no = round(no / total * 100, 1) if total else 0

    features = [(name, svc.pretty(name)) for name in svc.FEATURES]
    return render_template(
        "analysis.html",
        active="analysis",
        total=total,
        yes=yes,
        no=no,
        rate_yes=rate_yes,
        rate_no=rate_no,
        features=features,
    )


@bp.route("/analysis/feature")
def feature():
    try:
        return jsonify(svc.feature_breakdown(request.args.get("name", "")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.route("/analysis/correlation")
def correlation():
    return jsonify(svc.correlations())


@bp.route("/analysis/importance")
def importance():
    return jsonify(svc.feature_importance())