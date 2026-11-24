import joblib
import pytest

from app.app import create_app
from tests.test_api import _artifact, _client


def _app(tmp_path):
    path = tmp_path / "model.joblib"
    _artifact(path)
    app = create_app(str(path))
    app.config.update(TESTING=True)
    return app, path


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ([1, 2], "must be an object"),
        ({"forecast_days": "7"}, "must be an integer"),
        ({"forecast_days": True}, "must be an integer"),
        ({"forecast_days": 31}, "between 1 and 30"),
        ({"historical_closes": "abc"}, "non-empty array"),
        ({"historical_closes": []}, "non-empty array"),
        ({"historical_closes": ["x"] * 30}, "only numbers"),
        ({"historical_closes": [-1.0] * 30}, "positive finite"),
    ],
)
def test_predict_validation_branches(tmp_path, payload, fragment):
    response = _client(tmp_path).post("/predict", json=payload)
    assert response.status_code == 400
    assert fragment in response.get_json()["message"]


def test_unknown_route_returns_structured_404(tmp_path):
    response = _client(tmp_path).get("/nope")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not_found"


def test_predict_and_metrics_return_503_without_model(tmp_path):
    client = create_app(str(tmp_path / "missing.joblib")).test_client()
    assert client.post("/predict", json={}).status_code == 503
    assert client.get("/metrics").status_code == 503


def test_unexpected_prediction_failure_returns_500(tmp_path, monkeypatch):
    app, _ = _app(tmp_path)
    service = app.config["FORECAST_SERVICE"]

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "forecast", boom)
    response = app.test_client().post("/predict", json={})
    assert response.status_code == 500
    assert "boom" not in response.get_data(as_text=True)


def test_reload_requires_token_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_RELOAD_TOKEN", "secret")
    client = _app(tmp_path)[0].test_client()
    assert client.post("/internal/reload").status_code == 401
    assert client.post("/internal/reload", headers={"X-Reload-Token": "bad"}).status_code == 401
    ok = client.post("/internal/reload", headers={"X-Reload-Token": "secret"})
    assert ok.status_code == 200
    assert ok.get_json() == {"loaded": True, "model_version": "test-v1"}


def test_reload_picks_up_new_version(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_RELOAD_TOKEN", raising=False)
    app, path = _app(tmp_path)
    bundle = joblib.load(path)
    bundle["version"] = "test-v2"
    joblib.dump(bundle, path)
    client = app.test_client()
    assert client.post("/internal/reload").get_json()["model_version"] == "test-v2"
    assert client.get("/health").get_json()["model_version"] == "test-v2"


def test_reload_of_corrupt_artifact_returns_500(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_RELOAD_TOKEN", raising=False)
    app, path = _app(tmp_path)
    joblib.dump({"model": None}, path)
    response = app.test_client().post("/internal/reload")
    assert response.status_code == 500
    assert response.get_json()["error"] == "reload_failed"


def test_reload_reports_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_RELOAD_TOKEN", raising=False)
    app, path = _app(tmp_path)
    path.unlink()
    result = app.test_client().post("/internal/reload").get_json()
    assert result == {"loaded": False, "reason": "artifact_not_found"}
