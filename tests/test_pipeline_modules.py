import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import etl_fetch
from src.train_forecast import train_and_register


def _market(rows=200, **overrides):
    index = pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC", name="Date")
    close = np.linspace(40000, 50000, rows) + np.sin(np.arange(rows)) * 300
    frame = pd.DataFrame(
        {"Open": close, "High": close + 50, "Low": close - 50, "Close": close, "Volume": 1.0},
        index=index,
    )
    for column, value in overrides.items():
        frame[column] = value
    return frame


def _patch_download(monkeypatch, frame):
    monkeypatch.setattr(etl_fetch.yf, "download", lambda *_a, **_k: frame)


def test_fetch_writes_clean_csv_atomically(tmp_path, monkeypatch):
    _patch_download(monkeypatch, _market())
    out = tmp_path / "raw" / "btc.csv"
    summary = etl_fetch.fetch_btc_data(str(out))
    assert summary["rows"] == 200
    assert out.exists() and not out.with_suffix(".tmp").exists()
    assert list(pd.read_csv(out).columns) == ["Date", *etl_fetch.REQUIRED_COLUMNS]


def test_fetch_handles_multiindex_columns(tmp_path, monkeypatch):
    frame = _market()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["BTC-USD"]])
    _patch_download(monkeypatch, frame)
    assert etl_fetch.fetch_btc_data(str(tmp_path / "btc.csv"))["rows"] == 200


@pytest.mark.parametrize(
    "frame,message",
    [
        (pd.DataFrame(), "no rows"),
        (_market().drop(columns=["Volume"]), "missing required columns"),
        (_market(rows=50), "At least 120"),
        (_market(Close=-1.0), "positive"),
        (_market(Volume=-5.0), "negative"),
    ],
)
def test_fetch_rejects_invalid_data(tmp_path, monkeypatch, frame, message):
    _patch_download(monkeypatch, frame)
    with pytest.raises((RuntimeError, ValueError), match=message):
        etl_fetch.fetch_btc_data(str(tmp_path / "btc.csv"))


def _csv(tmp_path, rows=200, noise=0.0):
    frame = _market(rows).tz_convert(None)
    if noise:
        frame["Close"] = frame["Close"] * (1 + np.where(np.arange(rows) % 2, noise, -noise))
    path = tmp_path / "btc.csv"
    frame.to_csv(path)
    return path


def test_training_registers_versioned_model_atomically(tmp_path):
    model, metrics = tmp_path / "m" / "model.joblib", tmp_path / "m" / "metrics.json"
    result = train_and_register(str(_csv(tmp_path)), str(model), str(metrics))
    assert result["accepted"] is True
    assert result["model_version"].startswith("btc-hgbr-")
    bundle = joblib.load(model)
    assert bundle["version"] == result["model_version"] and len(bundle["history"]) == 60
    assert not model.with_suffix(".tmp").exists()
    assert json.loads(metrics.read_text())["metrics"]["validation_rows"] == 7


def test_training_rejects_model_above_threshold_and_keeps_old_artifact(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="rejected"):
        train_and_register(
            str(_csv(tmp_path)), str(model), str(tmp_path / "m.json"), mape_threshold=0.0
        )
    assert model.read_bytes() == b"previous"
    assert json.loads((tmp_path / "m.json").read_text())["accepted"] is False


def test_training_requires_enough_rows(tmp_path):
    with pytest.raises(ValueError, match="Insufficient"):
        train_and_register(str(_csv(tmp_path, rows=60)), str(tmp_path / "m.joblib"))
