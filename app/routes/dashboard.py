from flask import Blueprint, render_template
from sqlalchemy import func

from app import db
from app.models import Intervention, ModelResult, Prediction, Student

bp = Blueprint("dashboard", __name__)

# Family income bands in INR: (label, from, up to but not including)
BANDS = [
    ("Below ₹20,000", 0, 20000),
    ("₹20,000–40,000", 20000, 40000),
    ("₹40,000–70,000", 40000, 70000),
    ("₹70,000–1,00,000", 70000, 100000),
    ("Above ₹1,00,000", 100000, float("inf")),
]


@bp.route("/dashboard")
def dashboard():
    total = Student.query.count()
    rows = db.session.query(
        Student.family_income_inr, Student.higher_education_continuation
    ).all()

    counts = [0] * len(BANDS)
    yes = [0] * len(BANDS)
    for income, outcome in rows:
        if income is None:
            continue
        for i, (_, low, high) in enumerate(BANDS):
            if low <= income < high:
                counts[i] += 1
                if outcome == "Yes":
                    yes[i] += 1
                break
    rates = [round(y / c * 100, 1) if c else 0 for y, c in zip(yes, counts)]

    continuing = sum(1 for _, outcome in rows if outcome == "Yes")
    baseline = max(continuing, total - continuing) / total if total else 0

    best = ModelResult.query.filter_by(is_best=True).first()

    # Scores created by "Generate recommendations" have no factor text
    batch = Prediction.query.filter(Prediction.top_factors.is_(None))
    at_risk = batch.count()
    riskiest = batch.order_by(Prediction.probability_continue.asc()).limit(8).all()

    barriers_by_student = {}
    if riskiest:
        ids = [p.student_id for p in riskiest]
        for item in Intervention.query.filter(Intervention.student_id.in_(ids)):
            barriers_by_student.setdefault(item.student_id, []).append(item.barrier)

    barrier_counts = (
        db.session.query(Intervention.barrier, func.count(Intervention.id))
        .group_by(Intervention.barrier)
        .order_by(func.count(Intervention.id).desc())
        .all()
    )

    return render_template(
        "dashboard.html",
        active="dashboard",
        total=total,
        continuing_rate=round(continuing / total * 100, 1) if total else 0,
        baseline=baseline,
        best=best,
        at_risk=at_risk,
        pending=Intervention.query.filter_by(status="Pending").count(),
        done=Intervention.query.filter_by(status="Done").count(),
        band_labels=[b[0] for b in BANDS],
        band_rates=rates,
        band_counts=counts,
        riskiest=riskiest,
        barriers_by_student=barriers_by_student,
        barrier_labels=[b for b, _ in barrier_counts],
        barrier_values=[c for _, c in barrier_counts],
    )