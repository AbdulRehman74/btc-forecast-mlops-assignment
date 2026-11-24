"""Fetch and validate historical BTC-USD daily market data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_btc_data(output_path: str = "data/raw/btc_usd_daily.csv", period: str = "5y"):
    frame = yf.download("BTC-USD", period=period, interval="1d", auto_adjust=False, progress=False)
    if frame.empty:
        raise RuntimeError("The BTC-USD provider returned no rows")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None)
    frame.index.name = "Date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index().dropna()
    if len(frame) < 120:
        raise ValueError("At least 120 daily observations are required")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Price columns must contain positive values")
    if (frame["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    frame.to_csv(temporary)
    os.replace(temporary, destination)
    return {
        "rows": int(len(frame)),
        "start_date": frame.index.min().date().isoformat(),
        "end_date": frame.index.max().date().isoformat(),
        "output_path": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/btc_usd_daily.csv")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    print(json.dumps(fetch_btc_data(args.output, args.period), indent=2))


if __name__ == "__main__":
    main()
