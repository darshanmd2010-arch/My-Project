import csv

from sqlalchemy import Float, Integer

from app import db
from app.models import Student, UploadLog

# Columns of the students table that can be filled from the CSV
STUDENT_COLUMNS = {
    c.name: c
    for c in Student.__table__.columns
    if c.name not in ("id", "created_at")
}

# The CSV must contain these columns
REQUIRED = ["student_id", "higher_education_continuation"]


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def _convert(value, column):
    """Convert a CSV text value to the type of the database column."""
    value = (value or "").strip()
    if value == "":
        return None
    if isinstance(column.type, Integer):
        return int(float(value))
    if isinstance(column.type, Float):
        return float(value)
    return value


def import_csv(path, filename):
    """Read the CSV at `path` and save new students to the SQL database.

    Returns a dictionary with the result (rows read, saved, skipped, problems).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        # CSV header -> database column name (Family_Income_INR -> family_income_inr)
        header_map = {h: h.lower().replace(" ", "_") for h in headers}
        known = {h: n for h, n in header_map.items() if n in STUDENT_COLUMNS}
        unknown_columns = [h for h in headers if h not in known]
        missing_required = [n for n in REQUIRED if n not in known.values()]

        if missing_required:
            return {
                "ok": False,
                "error": "Missing required column(s): " + ", ".join(missing_required),
            }

        seen_ids = {sid for (sid,) in db.session.query(Student.student_id).all()}
        to_insert = []
        errors = []
        duplicates = 0
        total = 0

        for line_no, row in enumerate(reader, start=2):
            total += 1
            row = {(k or "").strip(): v for k, v in row.items()}
            try:
                record = {
                    known[h]: _convert(row.get(h), STUDENT_COLUMNS[known[h]])
                    for h in known
                }
            except ValueError:
                errors.append(f"Line {line_no}: invalid number")
                continue

            student_id = record.get("student_id")
            if not student_id:
                errors.append(f"Line {line_no}: Student_ID is empty")
                continue
            if student_id in seen_ids:
                duplicates += 1
                continue

            seen_ids.add(student_id)
            to_insert.append(record)

    if to_insert:
        db.session.bulk_insert_mappings(Student, to_insert)

    skipped = duplicates + len(errors)
    db.session.add(
        UploadLog(
            filename=filename,
            rows_total=total,
            rows_saved=len(to_insert),
            rows_skipped=skipped,
        )
    )
    db.session.commit()

    return {
        "ok": True,
        "rows_total": total,
        "rows_saved": len(to_insert),
        "duplicates_skipped": duplicates,
        "errors": errors[:20],  # show at most 20 problems
        "unknown_columns": unknown_columns,
    }