import os

from flask import Blueprint, current_app, flash, render_template, request
from werkzeug.utils import secure_filename

from app import db
from app.models import Student, UploadLog
from app.services.data_service import allowed_file, import_csv

bp = Blueprint("upload", __name__)


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    result = None

    if request.method == "POST":
        file = request.files.get("csv_file")

        if not file or file.filename == "":
            flash("Please choose a CSV file.", "error")
        elif not allowed_file(file.filename, current_app.config["ALLOWED_EXTENSIONS"]):
            flash("Only .csv files are allowed.", "error")
        else:
            filename = secure_filename(file.filename)
            path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            try:
                result = import_csv(path, filename)
                if result["ok"]:
                    flash(
                        f"{result['rows_saved']} students saved to the database.",
                        "success",
                    )
                else:
                    flash(result["error"], "error")
            except Exception as exc:
                db.session.rollback()
                flash(f"Could not import the file: {exc}", "error")

    total_students = Student.query.count()
    preview = Student.query.order_by(Student.id).limit(10).all()
    logs = UploadLog.query.order_by(UploadLog.id.desc()).limit(5).all()

    return render_template(
        "upload.html",
        active="upload",
        result=result,
        total_students=total_students,
        preview=preview,
        logs=logs,
    )