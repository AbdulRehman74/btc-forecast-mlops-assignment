from datetime import date

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

from app.app import create_app
from src.train_forecast import FEATURE_COLUMNS


def _artifact(path):
    model = DummyRegressor(strategy="constant", constant=70000.0)
    model.fit(np.zeros((2, len(FEATURE_COLUMNS))), [70000.0, 70000.0])
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "max_lag": 30,
            "history": list(np.linspace(60000, 70000, 60)),
            "last_observation_date": date(2026, 1, 1).isoformat(),
            "version": "test-v1",
            "metrics": {"mae": 100.0, "rmse": 120.0, "mape": 1.2},
        },
        path,
    )


def _client(tmp_path):
    path = tmp_path / "model.joblib"
    _artifact(path)
    app = create_app(str(path))
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_reports_model_version(tmp_path):
    response = _client(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_version"] == "test-v1"


def test_predict_returns_seven_dated_prices(tmp_path):
    response = _client(tmp_path).post("/predict", json={})
    body = response.get_json()
    assert response.status_code == 200
    assert body["forecast_days"] == 7
    assert len(body["predictions"]) == 7
    assert body["predictions"][0]["predicted_close_usd"] == 70000.0


def test_predict_accepts_custom_history(tmp_path):
    response = _client(tmp_path).post(
        "/predict", json={"forecast_days": 3, "historical_closes": list(range(100, 130))}
    )
    assert response.status_code == 200
    assert len(response.get_json()["predictions"]) == 3


def test_predict_rejects_invalid_days(tmp_path):
    response = _client(tmp_path).post("/predict", json={"forecast_days": 0})
    assert response.status_code == 400
    assert response.get_json()["error"] == "validation_error"


def test_predict_rejects_short_history(tmp_path):
    response = _client(tmp_path).post("/predict", json={"historical_closes": [60000.0, 61000.0]})
    assert response.status_code == 400
    assert "at least 30" in response.get_json()["message"]


def test_metrics_exposes_validation_values(tmp_path):
    response = _client(tmp_path).get("/metrics")
    assert response.status_code == 200
    assert response.get_json()["mape"] == 1.2


def test_missing_artifact_is_degraded(tmp_path):
    app = create_app(str(tmp_path / "missing.joblib"))
    response = app.test_client().get("/health")
    assert response.status_code == 503
    assert response.get_json()["status"] == "degraded"
