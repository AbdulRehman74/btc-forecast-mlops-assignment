# Verified Evidence Register

Author: Abdul Rehman  
Roll number: 24L-8010  
Verification date: 24 November 2026

## Model and API

- Dataset: 1,827 daily BTC-USD observations from 24 November 2021 through 24 November 2026.
- Model: HistGradientBoostingRegressor with leakage-safe lag, moving-average, volatility, and momentum features.
- Validation: final seven chronological observations.
- Acceptance threshold: MAPE less than or equal to 25 percent.
- Achieved MAE: 1,713.490 USD.
- Achieved RMSE: 2,004.384 USD.
- Achieved MAPE: 2.836 percent.
- Model version: btc-hgbr-20261124T135324Z before Airflow retraining.
- API checks: health 200, seven-day prediction 200, metrics 200, invalid forecast-days request 400.
- Unit tests: 33 passed with 94 percent line coverage (8 original plus 25 added for API, ETL and training).

## Docker

- Multi-stage image built successfully as `btc-forecast-api:local`.
- Runtime container executed as a non-root user and became healthy.
- Port mapping verified at `5000:5000`.
- Integrated Compose services: Flask/Gunicorn API, PostgreSQL, Airflow webserver, and Airflow scheduler.

## Airflow and Kubeflow

- DAG ID: `btc_daily_mlops_pipeline`.
- DAG import errors: none.
- Scheduled run: success.
- Manually triggered runs: success (14:14 and 17:50 UTC); three successful runs in total.
- Successful task states: fetch_data, validate_and_engineer, retrain_and_evaluate, reload_api.
- Reloaded Airflow model version: btc-hgbr-20261124T141427Z.
- Kubeflow specification compiled successfully to `kubeflow/compiled_pipeline.yaml`.
- Compiled specification size: 9,872 bytes (after switching components to Dataset/Model artifacts).

## Kubernetes baseline

- Cluster: Kind v0.33.0 with one control plane and two workers.
- Deployment manifest begins with three replicas as required.
- HPA range: two to eight replicas at 60 percent target CPU.
- PVC: bound, 1 GiB, mounted at `/app/models`.
- Services: ClusterIP and NodePort 30080, both targeting container port 5000.
- Ingress: NGINX class, host `btc-forecast.local`, address `localhost`.
- Live seven-day prediction succeeded through NodePort.

## Load and resilience

- Load generator: Locust 2.42.1.
- Sustained run: 100 concurrent users for 90 seconds.
- Requests: 3,927 total.
- Successful requests: 3,926.
- Failure rate: 0.03 percent.
- Aggregate throughput: 43.74 requests per second.
- Median response time: 2,000 ms under saturation.
- HPA result: scaled from the baseline to eight replicas.
- Observed CPU: 252 percent against the 60 percent target.
- Post-fix pod restarts during successful load test: zero.
- Chaos test: one explicitly identified pod deleted and automatically replaced in approximately 14 seconds.

## Screenshot capture status

All eight evidence screenshots (01-08) were captured on 24 November 2026 and are stored locally in `report/evidence/` (excluded from git). Screenshot 02 (Airflow) was re-captured from the live Airflow UI because the original file was an undecodable HEIF image. The self-healing screenshot shows pod 6kzgg replaced by jw68f (ready in 7 s); the earlier recorded run replaced 2vljf with g8v7m in about 14 s. The Locust screenshot shows a run in progress (4,058 requests, 0 failures); the recorded final-run totals above are 3,927 requests with 0.03 percent failures.

## Additional verification (24 November 2026, later session)

- Kubeflow pipeline executed end to end with the KFP local runner (`python kubeflow/run_local.py subprocess` in a Linux container): 1,827 rows ingested, 1,797 feature rows, MAPE 2.816 percent, all three components succeeded. The first execution exposed a string-parameter data-passing bug, fixed by using Dataset/Model artifacts. A hosted KFP server was attempted and removed; it is not part of the assignment requirements.
- Airflow: third successful run (manual, 17:50 UTC) retrained btc-hgbr-20261124T175107Z (MAPE 2.873 percent) and the API reported that version.
- Repeat Locust run: 4,161 requests, 0 failures, 46.65 req/s; HPA 2 to 4 to 8 and back to 2.
