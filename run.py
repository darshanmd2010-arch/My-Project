import os

# Avoids the Windows Application Control block on SQLAlchemy's compiled extension
os.environ["DISABLE_SQLALCHEMY_CEXT_RUNTIME"] = "1"

from app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)