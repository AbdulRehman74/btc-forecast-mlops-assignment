from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(4)
    def predict(self):
        with self.client.post(
            "/predict", json={"forecast_days": 7}, name="POST /predict", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
