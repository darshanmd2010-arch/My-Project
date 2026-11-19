from datetime import datetime, timezone

from app import db


def _now():
    return datetime.now(timezone.utc)


def _txt(n=50):
    return db.Column(db.String(n), nullable=True)


def _int():
    return db.Column(db.Integer, nullable=True)


def _flt():
    return db.Column(db.Float, nullable=True)


class Student(db.Model):
    """One row per student (the uploaded CSV is saved here)."""
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False, index=True)

    # Profile and school background
    age = _int()
    gender = _txt(20)
    high_school_type = _txt()
    high_school_grade = _flt()
    entrance_exam_score = _flt()
    previous_qualification = _txt()

    # Current studies and academics
    department = _txt()
    semester = _int()
    semesters_completed = _int()
    cumulative_gpa = _flt()
    number_of_failed_courses = _int()
    attendance_percentage = _flt()
    academic_performance = _txt()

    # Financial factors
    family_income_inr = _flt()
    scholarship_holder = _txt(10)
    financial_assistance = _txt(10)
    tuition_fee_status = _txt()
    debtor_status = _txt(10)

    # Family background
    father_occupation = _txt()
    father_qualification = _txt()
    mother_occupation = _txt()
    mother_qualification = _txt()
    family_support = _txt(20)

    # Support, needs and career
    student_support_program = _txt(10)
    educational_special_needs = _txt(10)
    disability = _txt(10)
    career_change = _txt(10)
    second_career = _txt(10)

    # Behaviour and learning
    motivation = _txt(20)
    student_engagement = _txt(20)
    time_management = _txt(20)
    learning_method = _txt()
    mentoring_support = _txt(10)

    # Outcomes
    higher_education_continuation = _txt(10)  # target: Yes / No
    dropout_status = _txt(30)                 # Continuing/Completed / Dropped Out

    created_at = db.Column(db.DateTime, default=_now)

    def __repr__(self):
        return f"<Student {self.student_id}>"


class UploadLog(db.Model):
    """History of CSV uploads."""
    __tablename__ = "upload_logs"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    rows_total = db.Column(db.Integer, default=0)
    rows_saved = db.Column(db.Integer, default=0)
    rows_skipped = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=_now)


class ModelResult(db.Model):
    """Evaluation metrics of every trained model."""
    __tablename__ = "model_results"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    roc_auc = db.Column(db.Float)
    cohen_kappa = db.Column(db.Float)
    mcc = db.Column(db.Float)
    is_best = db.Column(db.Boolean, default=False)
    model_path = db.Column(db.String(255))  # file inside saved_models/
    trained_at = db.Column(db.DateTime, default=_now)


class Prediction(db.Model):
    """Prediction made for a student."""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), index=True, nullable=False)
    probability_continue = db.Column(db.Float)  # chance of continuing to higher education
    risk_level = db.Column(db.String(20))       # High / Medium / Low
    top_factors = db.Column(db.Text)            # JSON text of SHAP factors
    created_at = db.Column(db.DateTime, default=_now)


class Intervention(db.Model):
    """Recommended support for an at-risk student."""
    __tablename__ = "interventions"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), index=True, nullable=False)
    barrier = db.Column(db.String(50))       # Financial / Academic / Family / Motivation
    recommendation = db.Column(db.String(255))
    status = db.Column(db.String(20), default="Pending")  # Pending / Done
    created_at = db.Column(db.DateTime, default=_now)