from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "kos_madiun.csv"
ARTIFACT_PATH = ROOT / "ml" / "artifacts" / "model_kos.joblib"
EVALUATION_REPORT_PATH = ROOT / "ml" / "artifacts" / "evaluation_metrics.json"
FEATURES = ["jarak_km", "rating", "ac_num"]
AC_MAPPING = {"tidak": 0, "ya": 1}


def canonical_dataset_sha256(path: Path = DATA_PATH) -> str:
    contents = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(contents).hexdigest().upper()


def load_and_prepare_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    required = {"nama", "jarak_km", "harga", "ac", "rating"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing)}")
    frame = frame.copy()
    frame["ac"] = frame["ac"].astype(str).str.strip().str.lower()
    frame["ac_num"] = frame["ac"].map(AC_MAPPING)
    for column in ["jarak_km", "rating", "harga"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FEATURES + ["harga"])
    if frame.empty:
        raise ValueError("Dataset tidak memiliki baris valid.")
    return frame[FEATURES], frame["harga"]


def fit_production_model() -> dict:
    features, target = load_and_prepare_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    evaluator = LinearRegression().fit(x_train, y_train)
    predictions = evaluator.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    model = LinearRegression().fit(features, target)
    artifact = {
        "model": model,
        "features": FEATURES,
        "target": "harga",
        "ac_mapping": AC_MAPPING,
        "metrics": metrics,
        "training_rows": int(len(features)),
        "algorithm": "Linear Regression",
    }
    return artifact


def train() -> dict:
    artifact = fit_production_model()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(json.dumps(artifact["metrics"], indent=2))
    print(f"Model tersimpan di: {ARTIFACT_PATH}")
    return artifact


if __name__ == "__main__":
    train()
