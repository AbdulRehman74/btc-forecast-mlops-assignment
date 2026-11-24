"""Production Flask API for seven-day Bitcoin close-price forecasts."""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("btc_forecast_api")


class ModelUnavailableError(RuntimeError):
    """Raised when no valid model artifact is available."""


class ForecastService:
    """Thread-safe loader and inference facade for the versioned model bundle."""

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self._bundle: dict[str, Any] | None = None
        self._loaded_at: datetime | None = None
        self._lock = threading.RLock()
        self.reload()

    @property
    def ready(self) -> bool:
        return self._bundle is not None

    def reload(self) -> dict[str, Any]:
        with self._lock:
            if not self.model_path.exists():
                self._bundle = None
                LOGGER.warning("Model artifact not found at %s", self.model_path)
                return {"loaded": False, "reason": "artifact_not_found"}
            bundle = joblib.load(self.model_path)
            required = {"model", "feature_columns", "history", "version", "metrics"}
            missing = required.difference(bundle)
            if missing:
                raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
            self._bundle = bundle
            self._loaded_at = datetime.now(timezone.utc)
            LOGGER.info("Loaded model version %s", bundle["version"])
            return {"loaded": True, "model_version": bundle["version"]}

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            if not self._bundle:
                return {"model_version": None, "loaded_at": None}
            return {
                "model_version": self._bundle["version"],
                "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
            }

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            if not self._bundle:
                raise ModelUnavailableError("Model is not loaded")
            return dict(self._bundle["metrics"])

    def forecast(
        self, forecast_days: int = 7, historical_closes: list[float] | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            if not self._bundle:
                raise ModelUnavailableError("Model is not loaded")
            closes = (
                [float(value) for value in historical_closes]
                if historical_closes is not None
                else [float(value) for value in self._bundle["history"]]
            )
            max_lag = int(self._bundle.get("max_lag", 30))
            if len(closes) < max_lag:
                raise ValueError(f"historical_closes requires at least {max_lag} values")
            predictions: list[float] = []
            for _ in range(forecast_days):
                row = self._feature_row(closes)
                frame = pd.DataFrame([row], columns=self._bundle["feature_columns"])
                prediction = max(0.0, float(self._bundle["model"].predict(frame)[0]))
                predictions.append(prediction)
                closes.append(prediction)

            last_date = pd.Timestamp(self._bundle["last_observation_date"])
            dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
            return [
                {"date": date.date().isoformat(), "predicted_close_usd": round(value, 2)}
                for date, value in zip(dates, predictions)
            ]

    @staticmethod
    def _feature_row(closes: list[float]) -> dict[str, float]:
        values = np.asarray(closes, dtype=float)
        returns = pd.Series(values).pct_change().dropna()
        return {
            "lag_1": float(values[-1]),
            "lag_2": float(values[-2]),
            "lag_3": float(values[-3]),
            "lag_7": float(values[-7]),
            "lag_14": float(values[-14]),
            "lag_30": float(values[-30]),
            "ma_7": float(values[-7:].mean()),
            "ma_30": float(values[-30:].mean()),
            "volatility_7": float(returns.tail(7).std(ddof=0)),
            "momentum_7": float(values[-1] / values[-7] - 1.0),
        }


def _validate_prediction_payload(payload: Any) -> tuple[int, list[float] | None]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    forecast_days = payload.get("forecast_days", 7)
    if isinstance(forecast_days, bool) or not isinstance(forecast_days, int):
        raise ValueError("forecast_days must be an integer")
    if not 1 <= forecast_days <= 30:
        raise ValueError("forecast_days must be between 1 and 30")
    history = payload.get("historical_closes")
    if history is not None:
        if not isinstance(history, list) or not history:
            raise ValueError("historical_closes must be a non-empty array")
        try:
            history = [float(value) for value in history]
        except (TypeError, ValueError) as exc:
            raise ValueError("historical_closes must contain only numbers") from exc
        if not all(np.isfinite(value) and value > 0 for value in history):
            raise ValueError("historical_closes must contain positive finite numbers")
    return forecast_days, history


def create_app(model_path: str | None = None) -> Flask:
    app = Flask(__name__)
    started_at = time.monotonic()
    service = ForecastService(
        model_path or os.getenv("MODEL_PATH", "/app/models/btc_forecast.joblib")
    )
    app.config["FORECAST_SERVICE"] = service

    @app.get("/health")
    def health():
        metadata = service.metadata()
        return (
            jsonify(
                status="ok" if service.ready else "degraded",
                uptime_seconds=round(time.monotonic() - started_at, 3),
                model_version=metadata["model_version"],
                model_loaded_at=metadata["loaded_at"],
            ),
            200 if service.ready else 503,
        )

    @app.post("/predict")
    def predict():
        try:
            forecast_days, history = _validate_prediction_payload(request.get_json(silent=True))
            return jsonify(
                generated_at=datetime.now(timezone.utc).isoformat(),
                forecast_days=forecast_days,
                model_version=service.metadata()["model_version"],
                predictions=service.forecast(forecast_days, history),
            )
        except ValueError as exc:
            return jsonify(error="validation_error", message=str(exc)), 400
        except ModelUnavailableError as exc:
            return jsonify(error="model_unavailable", message=str(exc)), 503
        except Exception:
            LOGGER.exception("Prediction failed")
            return jsonify(error="internal_error", message="Prediction could not be completed"), 500

    @app.get("/metrics")
    def metrics():
        try:
            return jsonify(model_version=service.metadata()["model_version"], **service.metrics())
        except ModelUnavailableError as exc:
            return jsonify(error="model_unavailable", message=str(exc)), 503

    @app.post("/internal/reload")
    def reload_model():
        expected = os.getenv("MODEL_RELOAD_TOKEN", "")
        supplied = request.headers.get("X-Reload-Token", "")
        if expected and not hmac.compare_digest(expected, supplied):
            return jsonify(error="unauthorized", message="Invalid reload token"), 401
        try:
            return jsonify(service.reload())
        except Exception:
            LOGGER.exception("Model reload failed")
            return jsonify(error="reload_failed", message="Artifact could not be loaded"), 500

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify(error="not_found", message="Endpoint does not exist"), 404

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
