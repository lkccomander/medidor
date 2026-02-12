"""Train a simple forecasting model for MEDIDOR internet speed samples."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import unicodedata
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_INPUT = pathlib.Path(__file__).with_name("internet_speed_data.csv")
DEFAULT_MODEL_DIR = pathlib.Path(__file__).with_name("models")
TARGET_COLUMNS = ("download_mbps", "upload_mbps", "ping_ms")
ROLLING_WINDOWS = (3, 6)
TARGET_SERVER_NAME = "San José"
TARGET_SERVER_COUNTRY = "Costa Rica"


@dataclass(slots=True)
class ForecastMetrics:
    """Evaluation metrics for model and baseline."""

    samples_total: int
    samples_train: int
    samples_test: int
    mae_baseline: float
    rmse_baseline: float
    mae_model: float
    rmse_model: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a forecasting model for MEDIDOR measurements.",
    )
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--target", choices=TARGET_COLUMNS, default="download_mbps")
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="How many steps ahead to forecast (default: 1).",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=5,
        help="Number of lag features to generate for each metric (default: 5).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of newest samples reserved for test (default: 0.2).",
    )
    parser.add_argument(
        "--model-output",
        type=pathlib.Path,
        default=DEFAULT_MODEL_DIR / "medidor_forecast.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=pathlib.Path,
        default=DEFAULT_MODEL_DIR / "medidor_forecast_metrics.json",
    )
    return parser.parse_args()


def load_data(input_path: pathlib.Path) -> pd.DataFrame:
    """Load and validate MEDIDOR data."""
    if not input_path.exists():
        message = f"Input CSV not found: {input_path}"
        raise FileNotFoundError(message)

    frame = pd.read_csv(input_path)
    required = {"timestamp_utc", "server_name", "server_country", *TARGET_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        message = f"Missing required columns: {', '.join(missing)}"
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


def build_training_frame(
    frame: pd.DataFrame,
    target: str,
    lags: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create lag/time features and target vector."""
    if lags < 1:
        raise ValueError("--lags must be >= 1")
    if horizon < 1:
        raise ValueError("--horizon must be >= 1")

    work = frame.copy()
    ts = work["timestamp_utc"]
    feature_data: dict[str, pd.Series | np.ndarray] = {
        "hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24),
        "weekday_sin": np.sin(2 * np.pi * ts.dt.weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * ts.dt.weekday / 7),
    }

    for metric in TARGET_COLUMNS:
        series = work[metric]
        for lag in range(1, lags + 1):
            feature_data[f"{metric}_lag_{lag}"] = series.shift(lag)
        for window in ROLLING_WINDOWS:
            rolling = series.rolling(window=window, min_periods=window)
            feature_data[f"{metric}_roll_mean_{window}"] = rolling.mean()
            feature_data[f"{metric}_roll_std_{window}"] = rolling.std()
            feature_data[f"{metric}_roll_min_{window}"] = rolling.min()
            feature_data[f"{metric}_roll_max_{window}"] = rolling.max()

        # First-order trend to capture short-term acceleration/deceleration.
        feature_data[f"{metric}_diff_1"] = series.diff(periods=1)

    work = pd.concat([work, pd.DataFrame(feature_data, index=work.index)], axis=1)
    work["target_next"] = work[target].shift(-horizon)
    model_frame = work.dropna().reset_index(drop=True)

    if len(model_frame) < 20:
        message = (
            f"Not enough rows after feature generation ({len(model_frame)}). "
            "Collect more samples or reduce --lags/--horizon."
        )
        raise ValueError(message)

    # Keep only numeric features to avoid passing categorical text fields
    # such as server_name/server_country to the regressor.
    feature_columns = [
        col
        for col in model_frame.columns
        if col not in {"timestamp_utc", "target_next"}
        and pd.api.types.is_numeric_dtype(model_frame[col])
    ]
    x = model_frame[feature_columns]
    y = model_frame["target_next"]
    return x, y


def train_test_split_time(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data preserving chronology."""
    if not 0 < test_size < 0.5:
        raise ValueError("--test-size must be > 0 and < 0.5")

    split_index = int(len(x) * (1 - test_size))
    if split_index < 10 or (len(x) - split_index) < 5:
        raise ValueError("Not enough rows to create a meaningful time-based split.")

    x_train = x.iloc[:split_index]
    x_test = x.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return x_train, x_test, y_train, y_test


def evaluate_forecast(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target: str,
) -> tuple[RandomForestRegressor, ForecastMetrics]:
    """Train the model and compute metrics against a lag-1 baseline."""
    lag1_column = f"{target}_lag_1"
    if lag1_column not in x_test.columns:
        message = f"Missing baseline feature column: {lag1_column}"
        raise ValueError(message)

    baseline_pred = x_test[lag1_column].to_numpy()
    mae_baseline = mean_absolute_error(y_test, baseline_pred)
    rmse_baseline = mean_squared_error(y_test, baseline_pred) ** 0.5

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(x_train, y_train)
    model_pred = model.predict(x_test)
    mae_model = mean_absolute_error(y_test, model_pred)
    rmse_model = mean_squared_error(y_test, model_pred) ** 0.5

    metrics = ForecastMetrics(
        samples_total=len(x_train) + len(x_test),
        samples_train=len(x_train),
        samples_test=len(x_test),
        mae_baseline=float(mae_baseline),
        rmse_baseline=float(rmse_baseline),
        mae_model=float(mae_model),
        rmse_model=float(rmse_model),
    )
    return model, metrics


def save_outputs(
    model: RandomForestRegressor,
    x_train: pd.DataFrame,
    metrics: ForecastMetrics,
    target: str,
    horizon: int,
    lags: int,
    model_output: pathlib.Path,
    metrics_output: pathlib.Path,
) -> dict[str, str]:
    """Persist trained model package and metrics report."""
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    model_version = (
        f"v{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"_t-{target}_h-{horizon}_l-{lags}"
    )
    versioned_model_output = (
        model_output.parent
        / f"{model_output.stem}_{model_version}{model_output.suffix}"
    )
    versioned_metrics_output = (
        metrics_output.parent
        / f"{metrics_output.stem}_{model_version}{metrics_output.suffix}"
    )

    payload = {
        "model": model,
        "feature_columns": list(x_train.columns),
        "target": target,
        "horizon": horizon,
        "lags": lags,
        "model_version": model_version,
    }
    metrics_payload = asdict(metrics) | {"model_version": model_version}

    # Keep base paths as "latest" aliases and also store immutable versioned artifacts.
    joblib.dump(payload, model_output)
    joblib.dump(payload, versioned_model_output)
    metrics_output.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    versioned_metrics_output.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    return {
        "model_version": model_version,
        "model_output": str(model_output),
        "metrics_output": str(metrics_output),
        "versioned_model_output": str(versioned_model_output),
        "versioned_metrics_output": str(versioned_metrics_output),
    }


def main() -> None:
    """Train and evaluate a forecasting model for MEDIDOR data."""
    args = parse_args()
    frame = load_data(args.input)
    x, y = build_training_frame(
        frame=frame,
        target=args.target,
        lags=args.lags,
        horizon=args.horizon,
    )
    x_train, x_test, y_train, y_test = train_test_split_time(
        x=x,
        y=y,
        test_size=args.test_size,
    )
    model, metrics = evaluate_forecast(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        target=args.target,
    )
    outputs = save_outputs(
        model=model,
        x_train=x_train,
        metrics=metrics,
        target=args.target,
        horizon=args.horizon,
        lags=args.lags,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
    )

    print(f"Target: {args.target}")
    print(
        "Samples used: "
        f"total={metrics.samples_total}, "
        f"train={metrics.samples_train}, "
        f"test={metrics.samples_test}",
    )
    print(
        f"Baseline -> MAE={metrics.mae_baseline:.3f}, "
        f"RMSE={metrics.rmse_baseline:.3f}",
    )
    print(
        f"Model    -> MAE={metrics.mae_model:.3f}, "
        f"RMSE={metrics.rmse_model:.3f}",
    )
    print(f"Model version: {outputs['model_version']}")
    print(f"Model saved: {outputs['model_output']}")
    print(f"Metrics saved: {outputs['metrics_output']}")
    print(f"Versioned model saved: {outputs['versioned_model_output']}")
    print(f"Versioned metrics saved: {outputs['versioned_metrics_output']}")


if __name__ == "__main__":
    main()
