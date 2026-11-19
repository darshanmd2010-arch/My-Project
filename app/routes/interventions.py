import csv
import io

from flask import Blueprint, Response, flash, redirect, render_template, request

from app import db
from app.models import Intervention, Prediction
from app.services import recommender

bp = Blueprint("interventions", __name__)

BARRIERS = ["Financial", "Academic", "Family", "Motivation", "Support", "General"]


def _batch_predictions():
    """Scores created by 'Generate recommendations' (they have no factor text)."""
    return Prediction.query.filter(Prediction.top_factors.is_(None))


@bp.route("/interventions")
def interventions():
    barrier = request.args.get("barrier", "")
    status = request.args.get("status", "")

    query = Intervention.query
    if barrier:
        query = query.filter_by(barrier=barrier)
    if status:
        query = query.filter_by(status=status)

    matching = query.count()
    rows = query.order_by(Intervention.id).limit(100).all()

    ids = {r.student_id for r in rows}
    risk = {}
    if ids:
        for p in _batch_predictions().filter(Prediction.student_id.in_(ids)):
            risk[p.student_id] = p

    return render_template(
        "interventions.html",
        active="interventions",
        rows=rows,
        risk=risk,
        matching=matching,
        at_risk=_batch_predictions().count(),
        pending=Intervention.query.filter_by(status="Pending").count(),
        done=Intervention.query.filter_by(status="Done").count(),
        barriers=BARRIERS,
        barrier=barrier,
        status=status,
        threshold=int(recommender.AT_RISK_BELOW * 100),
    )


@bp.route("/interventions/generate", methods=["POST"])
def generate():
    try:
        out = recommender.generate()
        flash(
            f"{out['at_risk']} of {out['students']} students are at risk. "
            f"{out['interventions']} recommendations created.",
            "success",
        )
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "error")
    return redirect("/interventions")


@bp.route("/interventions/<int:item_id>/toggle", methods=["POST"])
def toggle(item_id):
    item = db.session.get(Intervention, item_id)
    if item:
        item.status = "Pending" if item.status == "Done" else "Done"
        db.session.commit()
    return redirect(request.referrer or "/interventions")


@bp.route("/interventions/export")
def export():
    risk = {p.student_id: p for p in _batch_predictions()}
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        ["Student_ID", "Chance_of_continuing_pct", "Risk", "Barrier", "Recommendation", "Status"]
    )
    for i in Intervention.query.order_by(Intervention.id):
        p = risk.get(i.student_id)
        writer.writerow([
            i.student_id,
            round(p.probability_continue * 100, 1) if p else "",
            p.risk_level if p else "",
            i.barrier,
            i.recommendation,
            i.status,
        ])
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=interventions.csv"},
    )