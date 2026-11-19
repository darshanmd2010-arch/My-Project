import os
os.environ["DISABLE_SQLALCHEMY_CEXT_RUNTIME"] = "1"

from app import create_app
from app.services.ml_service import train_all

app = create_app()
with app.app_context():
    out = train_all()
    print()
    print("Train rows:", out["n_train"], "| Test rows:", out["n_test"])
    print("Baseline (always guess the majority): %.3f" % out["baseline_accuracy"])
    print()
    for r in out["results"]:
        if "error" in r:
            print(r["model"], "-> FAILED:", r["error"])
        else:
            print("%-22s acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f kappa=%.3f mcc=%.3f" % (
                r["model"], r["accuracy"], r["precision"], r["recall"],
                r["f1"], r["roc_auc"], r["kappa"], r["mcc"]))
    print()
    print("Best model by ROC-AUC:", out["best"])
