import os

from flask import Flask, redirect
from flask_sqlalchemy import SQLAlchemy

from config import Config

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Make sure the working folders exist
    for folder in (app.config["UPLOAD_FOLDER"], app.config["MODEL_FOLDER"]):
        os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(app.root_path), "data"), exist_ok=True)

    db.init_app(app)

    # Register pages (blueprints)
    from app.routes.dashboard import bp as dashboard_bp
    from app.routes.upload import bp as upload_bp
    from app.routes.analysis import bp as analysis_bp
    from app.routes.training import bp as training_bp
    from app.routes.predict import bp as predict_bp
    from app.routes.interventions import bp as interventions_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(interventions_bp)

    # The home page opens the Dashboard
    @app.route("/")
    def home():
        return redirect("/dashboard")

    with app.app_context():
        from app import models  # noqa: F401  (registers the tables)
        db.create_all()

    return app