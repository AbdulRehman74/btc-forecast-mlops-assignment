"""Portable Kubeflow Pipeline for BTC ingestion, features, and training."""

from kfp import compiler, dsl

PACKAGES = [
    "pandas==2.3.3",
    "numpy==2.2.6",
    "scikit-learn==1.7.2",
    "yfinance==0.2.66",
    "joblib==1.5.2",
]


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
def ingest_btc_data(output_data: dsl.Output[dsl.Dataset], period: str = "5y") -> int:
    import pandas as pd
    import yfinance as yf

    frame = yf.download("BTC-USD", period=period, interval="1d", progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(frame) < 120:
        raise ValueError("Insufficient market history")
    frame.to_csv(output_data.path, index_label="Date")
    return len(frame)


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
def generate_features(
    input_data: dsl.Input[dsl.Dataset], output_data: dsl.Output[dsl.Dataset]
) -> int:
    import pandas as pd

    data = pd.read_csv(input_data.path, parse_dates=["Date"], index_col="Date")
    close = data["Close"].astype(float)
    for lag in (1, 2, 3, 7, 14, 30):
        data[f"lag_{lag}"] = close.shift(lag)
    data["ma_7"] = close.shift(1).rolling(7).mean()
    data["ma_30"] = close.shift(1).rolling(30).mean()
    data["volatility_7"] = close.pct_change().shift(1).rolling(7).std(ddof=0)
    data["momentum_7"] = close.shift(1) / close.shift(7) - 1.0
    data["target"] = close
    data = data.dropna()
    data.to_csv(output_data.path)
    return len(data)


@dsl.component(base_image="python:3.10-slim", packages_to_install=PACKAGES)
def train_model(
    feature_data: dsl.Input[dsl.Dataset], model_artifact: dsl.Output[dsl.Model]
) -> float:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    columns = [
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
    data = pd.read_csv(feature_data.path)
    train, validation = data.iloc[:-7], data.iloc[-7:]
    model = HistGradientBoostingRegressor(max_iter=300, random_state=24)
    model.fit(train[columns], train["target"])
    predicted = model.predict(validation[columns])
    mape = float(np.mean(np.abs((validation["target"] - predicted) / validation["target"])) * 100)
    if mape > 25:
        raise ValueError(f"Model rejected because MAPE {mape:.3f} exceeded 25")
    model.fit(data[columns], data["target"])
    joblib.dump({"model": model, "feature_columns": columns, "mape": mape}, model_artifact.path)
    return mape


@dsl.pipeline(name="btc-seven-day-forecast-training")
def btc_forecast_pipeline(period: str = "5y"):
    ingest = ingest_btc_data(period=period)
    features = generate_features(input_data=ingest.outputs["output_data"])
    train = train_model(feature_data=features.outputs["output_data"])
    train.set_cpu_request("500m").set_memory_request("1Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=btc_forecast_pipeline,
        package_path="kubeflow/compiled_pipeline.yaml",
    )
