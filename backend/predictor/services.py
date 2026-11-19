import hashlib
import json
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from django.conf import settings

@lru_cache(maxsize=1)
def get_artifact() -> dict:
    path: Path = settings.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError("Model belum tersedia. Jalankan: python ml/train_model.py")
    return joblib.load(path)

def predict_price(jarak_km: float, rating: float, ac: str) -> float:
    artifact = get_artifact()
    row = pd.DataFrame([[jarak_km, rating, artifact["ac_mapping"][ac]]], columns=artifact["features"])
    return float(artifact["model"].predict(row)[0])

def evaluation_summary(artifact: dict | None) -> dict:
    path: Path = settings.EVALUATION_REPORT_PATH
    if not path.exists():
        return {
            "available": False,
            "status": "missing",
            "detail": "File evaluasi belum tersedia.",
        }

    report = json.loads(path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(settings.DATA_PATH.read_bytes()).hexdigest().upper()
    if report["dataset"]["sha256"].upper() != dataset_hash:
        return {
            "available": False,
            "status": "dataset_mismatch",
            "detail": "Dataset berubah sejak evaluasi; metrik perlu diperbarui.",
        }

    if artifact is not None and report["holdout"]["linear_regression"] != artifact["metrics"]:
        return {
            "available": False,
            "status": "model_metrics_mismatch",
            "detail": "Metrik holdout tidak cocok dengan metadata model Joblib.",
        }

    return {"available": True, "status": "current", **report}

def dataset_summary() -> dict:
    frame = pd.read_csv(settings.DATA_PATH)
    frame["ac"] = frame["ac"].astype(str).str.strip().str.lower()
    try:
        artifact = get_artifact()
        model_summary = {
            "algorithm": artifact["algorithm"],
            "features": artifact["features"],
            "training_rows": artifact["training_rows"],
            "metrics": artifact["metrics"],
        }
    except FileNotFoundError:
        artifact = None
        model_summary = None
    return {
        "jumlah_data": int(len(frame)),
        "rata_rata_harga": float(frame["harga"].mean()),
        "harga_minimum": float(frame["harga"].min()),
        "harga_maksimum": float(frame["harga"].max()),
        "rata_rata_harga_ac": float(frame.loc[frame["ac"] == "ya", "harga"].mean()),
        "rata_rata_harga_tanpa_ac": float(frame.loc[frame["ac"] == "tidak", "harga"].mean()),
        "model": model_summary,
        "evaluation": evaluation_summary(artifact),
    }
