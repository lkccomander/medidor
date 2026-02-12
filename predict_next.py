"""Predict the next MEDIDOR metric value using a trained forecast model."""

from __future__ import annotations

import argparse
import pathlib
import unicodedata

import joblib
import numpy as np
import pandas as pd

DEFAULT_INPUT = pathlib.Path(__file__).with_name("internet_speed_data.csv")
DEFAULT_MODEL = pathlib.Path(__file__).with_name("models") / "medidor_forecast.joblib"
TARGET_COLUMNS = ("download_mbps", "upload_mbps", "ping_ms")
ROLLING_WINDOWS = (3, 6)
TARGET_SERVER_NAME = "San José"
TARGET_SERVER_COUNTRY = "Costa Rica"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Predict the next MEDIDOR measurement from latest samples.",
    )
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def load_model(model_path: pathlib.Path) -> dict[str, object]:
    """Load serialized model package."""
    if not model_path.exists():
        message = f"Model file not found: {model_path}"
        raise FileNotFoundError(message)
    payload = joblib.load(model_path)
    required = {"model", "feature_columns", "target", "horizon", "lags"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        message = f"Model file is missing keys: {', '.join(missing)}"
        raise ValueError(message)
    return payload


def load_data(input_path: pathlib.Path) -> pd.DataFrame:
    """Load and sort MEDIDOR CSV."""
    if not input_path.exists():
        message = f"Input CSV not found: {input_path}"
        raise FileNotFoundError(message)

    frame = pd.read_csv(input_path)
    required = {"timestamp_utc", "server_name", "server_country", *TARGET_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        message = f"CSV missing columns: {', '.join(missing)}"
        raise ValueError(message)

    frame = frame[frame["server_name"].map(_normalized_text) == _normalized_text(TARGET_SERVER_NAME)]
    frame = frame[
        frame["server_country"].map(_normalized_text) == _normalized_text(TARGET_SERVER_COUNTRY)
    ]
    if frame.empty:
        message = (
            "No rows match required server filter: "
            f"{TARGET_SERVER_NAME}, {TARGET_SERVER_COUNTRY}"
        )
        raise ValueError(message)

    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    return frame


def _normalized_text(value: object) -> str:
    raw = str(value).strip()
    folded = unicodedata.normalize("NFKD", raw.casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def build_latest_feature_row(
    frame: pd.DataFrame,
    feature_columns: list[str],
    lags: int,
) -> pd.DataFrame:
    """Create one feature row from latest measurements."""
    min_rows = max(lags + 1, max(ROLLING_WINDOWS))
    if len(frame) < min_rows:
        message = (
            "Need at least "
            f"{min_rows} rows in CSV to generate lag and rolling features."
        )
        raise ValueError(message)

    latest = frame.iloc[-1]
    row: dict[str, float] = {}

    hour = int(latest["timestamp_utc"].hour)
    weekday = int(latest["timestamp_utc"].weekday())
    row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    row["weekday_sin"] = float(np.sin(2 * np.pi * weekday / 7))
    row["weekday_cos"] = float(np.cos(2 * np.pi * weekday / 7))

    # Training keeps current numeric metrics as features, so include them.
    for metric in TARGET_COLUMNS:
        row[metric] = float(latest[metric])

    for metric in TARGET_COLUMNS:
        for lag in range(1, lags + 1):
            row[f"{metric}_lag_{lag}"] = float(frame.iloc[-lag][metric])
        series = frame[metric]
        for window in ROLLING_WINDOWS:
            values = series.iloc[-window:]
            row[f"{metric}_roll_mean_{window}"] = float(values.mean())
            row[f"{metric}_roll_std_{window}"] = float(values.std(ddof=1))
            row[f"{metric}_roll_min_{window}"] = float(values.min())
            row[f"{metric}_roll_max_{window}"] = float(values.max())
        row[f"{metric}_diff_1"] = float(series.iloc[-1] - series.iloc[-2])

    feature_row = pd.DataFrame([row])
    missing = [col for col in feature_columns if col not in feature_row.columns]
    if missing:
        message = f"Cannot build inference row. Missing feature columns: {', '.join(missing)}"
        raise ValueError(message)
    return feature_row[feature_columns]


def main() -> None:
    """Predict the next value from the latest measurements."""
    args = parse_args()
    payload = load_model(args.model)
    frame = load_data(args.input)

    model = payload["model"]
    feature_columns = payload["feature_columns"]
    target = payload["target"]
    horizon = payload["horizon"]
    lags = payload["lags"]

    features = build_latest_feature_row(
        frame=frame,
        feature_columns=feature_columns,
        lags=lags,
    )
    prediction = float(model.predict(features)[0])
    latest_ts = frame.iloc[-1]["timestamp_utc"]

    print(f"Target: {target}")
    print(f"Horizon (steps): {horizon}")
    print(f"Latest timestamp (UTC): {latest_ts.isoformat()}")
    print(f"Predicted next value: {prediction:.3f}")


if __name__ == "__main__":
    main()
