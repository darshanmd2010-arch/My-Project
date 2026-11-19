import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Used by Flask for sessions and flash messages
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")

    # SQL database (SQLite file inside the data folder).
    # To switch to MySQL later, replace this with:
    # "mysql+pymysql://user:password@localhost/higher_ed_db"
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "data", "higher_ed.db"),
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # CSV uploads
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"csv"}
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB

    # Trained models are saved here
    MODEL_FOLDER = os.path.join(BASE_DIR, "saved_models")

    # Target column used for prediction
    TARGET_COLUMN = "Higher_Education_Continuation"