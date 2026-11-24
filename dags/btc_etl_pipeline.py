"""Daily Airflow DAG for validated BTC ingestion, retraining, and API reload."""

import os
from datetime import datetime, timedelta

import requests
from airflow.decorators import dag, task


@dag(
    dag_id="btc_daily_mlops_pipeline",
    description="Ingest, validate, train, evaluate, register, and reload BTC model",
    schedule="0 1 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "Abdul Rehman",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=20),
    },
    tags=["mlops", "bitcoin", "production"],
)
def btc_pipeline():
    @task
    def fetch_data():
        from src.etl_fetch import fetch_btc_data

        return fetch_btc_data(os.getenv("BTC_DATA_PATH", "data/raw/btc_usd_daily.csv"))

    @task
    def validate_and_engineer(fetch_summary: dict):
        import pandas as pd

        from src.train_forecast import FEATURE_COLUMNS, build_features

        frame = pd.read_csv(fetch_summary["output_path"], parse_dates=["Date"], index_col="Date")
        engineered = build_features(frame)
        if engineered[FEATURE_COLUMNS].isna().any().any():
            raise ValueError("Feature validation detected null values")
        return {**fetch_summary, "feature_rows": int(len(engineered))}

    @task
    def retrain_and_evaluate(feature_summary: dict):
        from src.train_forecast import train_and_register

        return train_and_register(
            data_path=feature_summary["output_path"],
            model_path=os.getenv("BTC_MODEL_PATH", "models/btc_forecast.joblib"),
            metrics_path="/opt/airflow/project/models/metrics.json",
            mape_threshold=float(os.getenv("MODEL_MAPE_THRESHOLD", "25")),
        )

    @task
    def reload_api(training_result: dict):
        response = requests.post(
            os.getenv("API_RELOAD_URL", "http://api:5000/internal/reload"),
            headers={"X-Reload-Token": os.getenv("MODEL_RELOAD_TOKEN", "")},
            timeout=15,
        )
        response.raise_for_status()
        return {"model_version": training_result["model_version"], "api": response.json()}

    reload_api(retrain_and_evaluate(validate_and_engineer(fetch_data())))


btc_pipeline()
