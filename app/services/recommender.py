from app import db
from app.models import Intervention, Prediction, Student
from app.services.analysis_service import FEATURES
from app.services.ml_service import encode, load_bundle
from app.services.predict_service import risk_level

# Settings you can change
AT_RISK_BELOW = 0.50   # at risk if the chance of continuing is below this
LOW_INCOME = 20000     # Family_Income_INR
LOW_GPA = 4.5
MANY_FAILED = 2
LOW_ATTENDANCE = 65


def barriers_for(s):
    """Rule-based barriers for one student: list of (barrier, recommendation)."""
    found = []

    if (
        (s.family_income_inr is not None and s.family_income_inr < LOW_INCOME)
        or s.tuition_fee_status == "Delayed"
        or s.debtor_status == "Yes"
    ):
        found.append(("Financial", "Scholarship, education-loan and fee-support guidance"))

    if (
        (s.cumulative_gpa is not None and s.cumulative_gpa < LOW_GPA)
        or (s.number_of_failed_courses or 0) >= MANY_FAILED
        or (s.attendance_percentage is not None and s.attendance_percentage < LOW_ATTENDANCE)
        or s.academic_performance in ("Low", "Very Low")
    ):
        found.append(("Academic", "Academic mentoring and remedial classes"))

    if s.family_support == "Low" or (
        s.father_qualification == "Below 10th" and s.mother_qualification == "Below 10th"
    ):
        found.append(("Family", "Family counselling session on the value of higher education"))

    if (
        s.motivation == "Low"
        or s.student_engagement == "Low"
        or s.time_management == "Poor"
    ):
        found.append(("Motivation", "Career counselling and motivation workshops"))

    if s.student_support_program == "No" and s.mentoring_support == "No":
        found.append(("Support", "Enrol in a student support and mentoring programme"))

    if not found:
        found.append(("General", "General career counselling"))
    return found


def generate():
    """Score every student, then create recommendations for the at-risk ones."""
    bundle = load_bundle()
    if bundle is None:
        raise ValueError("Train the models first on the Model Training page.")

    students = Student.query.all()
    if not students:
        raise ValueError("No students in the database. Upload the CSV first.")

    data = {name: [getattr(s, name) for s in students] for name in FEATURES}
    X = encode(data, bundle["encoder"])
    proba = bundle["models"][bundle["best"]].predict_proba(X)[:, 1]

    # Keep work already marked Done; rebuild the Pending ones
    done = {(i.student_id, i.barrier) for i in Intervention.query.filter_by(status="Done")}
    Intervention.query.filter_by(status="Pending").delete()
    Prediction.query.filter(Prediction.top_factors.is_(None)).delete()  # old batch scores

    predictions, interventions = [], []
    for s, p in zip(students, proba):
        p = float(p)
        if p >= AT_RISK_BELOW:
            continue
        predictions.append({
            "student_id": s.student_id,
            "probability_continue": p,
            "risk_level": risk_level(p),
        })
        for barrier, text in barriers_for(s):
            if (s.student_id, barrier) in done:
                continue
            interventions.append({
                "student_id": s.student_id,
                "barrier": barrier,
                "recommendation": text,
                "status": "Pending",
            })

    db.session.bulk_insert_mappings(Prediction, predictions)
    db.session.bulk_insert_mappings(Intervention, interventions)
    db.session.commit()

    return {
        "students": len(students),
        "at_risk": len(predictions),
        "interventions": len(interventions),
    }