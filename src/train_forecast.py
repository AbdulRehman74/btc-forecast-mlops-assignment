"""Train, backtest, threshold, and atomically register the BTC model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_30",
    "ma_7",
    "ma_30",
    "volatility_7",
    "momentum_7",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_index()
    close = data["Close"].astype(float)
    for lag in (1, 2, 3, 7, 14, 30):
        data[f"lag_{lag}"] = close.shift(lag)
    data["ma_7"] = close.shift(1).rolling(7).mean()
    data["ma_30"] = close.shift(1).rolling(30).mean()
    data["volatility_7"] = close.pct_change().shift(1).rolling(7).std(ddof=0)
    data["momentum_7"] = close.shift(1) / close.shift(7) - 1.0
    data["target"] = close
    return data.dropna(subset=FEATURE_COLUMNS + ["target"])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "mape": round(float(mape), 3),
    }


def train_and_register(
    data_path: str = "data/raw/btc_usd_daily.csv",
    model_path: str = "models/btc_forecast.joblib",
    metrics_path: str = "models/metrics.json",
    mape_threshold: float = 25.0,
):
    frame = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    features = build_features(frame)
    if len(features) < 90:
        raise ValueError("Insufficient feature rows after preprocessing")
    train, validation = features.iloc[:-7], features.iloc[-7:]
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=24,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    predicted = model.predict(validation[FEATURE_COLUMNS])
    metrics = _metrics(validation["target"].to_numpy(), predicted)
    metrics.update(
        validation_start=validation.index.min().date().isoformat(),
        validation_end=validation.index.max().date().isoformat(),
        validation_rows=7,
    )
    accepted = metrics["mape"] <= mape_threshold
    result = {"accepted": accepted, "threshold_mape": mape_threshold, "metrics": metrics}
    metrics_destination = Path(metrics_path)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not accepted:
        raise RuntimeError(
            f"Candidate rejected: MAPE {metrics['mape']} exceeds threshold {mape_threshold}"
        )
    final_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=24,
    )
    final_model.fit(features[FEATURE_COLUMNS], features["target"])
    timestamp = datetime.now(timezone.utc)
    version = timestamp.strftime("btc-hgbr-%Y%m%dT%H%M%SZ")
    bundle = {
        "model": final_model,
        "model_type": "HistGradientBoostingRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "max_lag": 30,
        "history": frame["Close"].astype(float).tail(60).tolist(),
        "last_observation_date": frame.index.max().date().isoformat(),
        "version": version,
        "trained_at": timestamp.isoformat(),
        "training_rows": int(len(features)),
        "metrics": metrics,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    os.replace(temporary, destination)
    result.update(model_version=version, model_path=str(destination))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/btc_usd_daily.csv")
    parser.add_argument("--model", default="models/btc_forecast.joblib")
    parser.add_argument("--metrics", default="models/metrics.json")
    parser.add_argument("--mape-threshold", type=float, default=25.0)
    args = parser.parse_args()
    print(
        json.dumps(
            train_and_register(args.data, args.model, args.metrics, args.mape_threshold), indent=2
        )
    )


if __name__ == "__main__":
    main()
