from __future__ import annotations

import json

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split

from ml.train_model import (
    DATA_PATH,
    EVALUATION_REPORT_PATH,
    FEATURES,
    canonical_dataset_sha256,
    load_and_prepare_data,
)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def build_evaluation_report() -> dict:
    features, target = load_and_prepare_data(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    evaluator = LinearRegression().fit(x_train, y_train)
    linear_predictions = evaluator.predict(x_test)
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)

    loo = LeaveOneOut()
    loo_linear_predictions = cross_val_predict(
        LinearRegression(), features, target, cv=loo
    )
    loo_baseline_predictions = cross_val_predict(
        DummyRegressor(strategy="mean"), features, target, cv=loo
    )

    return {
        "source": "ml/evaluate_model.py",
        "algorithm": "Linear Regression",
        "features": FEATURES,
        "dataset": {
            "path": DATA_PATH.relative_to(DATA_PATH.parents[1]).as_posix(),
            "rows": int(len(features)),
            "sha256": canonical_dataset_sha256(DATA_PATH),
        },
        "holdout": {
            "test_size": 0.2,
            "random_state": 42,
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "training_r2": float(evaluator.score(x_train, y_train)),
            "linear_regression": _metrics(y_test.to_numpy(), linear_predictions),
            "dummy_regressor_mean": _metrics(y_test.to_numpy(), baseline_predictions),
        },
        "leave_one_out": {
            "folds": int(len(features)),
            "r2_method": "Pooled out-of-fold predictions",
            "linear_regression": _metrics(target.to_numpy(), loo_linear_predictions),
            "dummy_regressor_mean": _metrics(target.to_numpy(), loo_baseline_predictions),
        },
        "notes": [
            "Holdout metrics are from an evaluator fitted on the training split only.",
            "The production Joblib estimator is separately fitted on all valid rows.",
            "LOO is supplementary and was not used to tune the model.",
            "All metrics use thousand rupiah per month as the target unit.",
        ],
    }


def save_evaluation_report() -> dict:
    report = build_evaluation_report()
    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(save_evaluation_report(), indent=2))
