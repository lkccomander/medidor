"""Streamlit web UI for MEDIDOR."""

from __future__ import annotations

import csv
import datetime as dt
import pathlib
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analyze import (
    add_outlier_flags,
    build_report,
    canonical_server_names,
    deduplicate_rows,
    load_data as analyze_load_data,
    write_clean_csv,
)
from main import (
    DEFAULT_DB_HOST,
    DEFAULT_DB_NAME,
    DEFAULT_DB_PASSWORD,
    DEFAULT_DB_PORT,
    DEFAULT_DB_USER,
    DEFAULT_OUTPUT,
    SpeedSample,
    append_prediction_postgres,
    append_sample_csv,
    append_sample_postgres,
    collect_sample,
    psycopg,
)
from predict_next import (
    DEFAULT_MODEL as PREDICT_DEFAULT_MODEL,
    TARGET_SERVER_COUNTRY as PREDICT_TARGET_SERVER_COUNTRY,
    TARGET_SERVER_NAME as PREDICT_TARGET_SERVER_NAME,
    build_latest_feature_row,
    load_data as predict_load_data,
    load_model as predict_load_model,
)
from train_forecast import (
    DEFAULT_INPUT as TRAIN_DEFAULT_INPUT,
    DEFAULT_MODEL_DIR as TRAIN_DEFAULT_MODEL_DIR,
    TARGET_COLUMNS as TRAIN_TARGET_COLUMNS,
    build_training_frame,
    evaluate_forecast,
    load_data as train_load_data,
    save_outputs,
    train_test_split_time,
)

MODEL_REGISTRY_PATH = TRAIN_DEFAULT_MODEL_DIR / "model_registry.csv"


def init_state() -> None:
    defaults: dict[str, object] = {
        "output_path": str(DEFAULT_OUTPUT),
        "timeout_seconds": 20,
        "interval_seconds": 60,
        "storage_mode": "postgres",
        "db_host": DEFAULT_DB_HOST,
        "db_port": DEFAULT_DB_PORT,
        "db_name": DEFAULT_DB_NAME,
        "db_user": DEFAULT_DB_USER,
        "db_password": DEFAULT_DB_PASSWORD,
        "events": [],
        "last_sample": None,
        "train_result": None,
        "predict_result": None,
        "analysis_report": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_event(message: str) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.session_state.events.insert(0, f"[{timestamp}] {message}")
    st.session_state.events = st.session_state.events[:300]


def output_path() -> pathlib.Path:
    return pathlib.Path(str(st.session_state.output_path)).expanduser().resolve()


def clean_output_path() -> pathlib.Path:
    path = output_path()
    return path.with_name(f"{path.stem}_clean{path.suffix}")


def load_model_registry_rows() -> list[dict[str, str]]:
    if not MODEL_REGISTRY_PATH.exists():
        return []
    rows: list[dict[str, str]] = []
    with MODEL_REGISTRY_PATH.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            rows.append(
                {
                    "id": str(row.get("id", "")).strip(),
                    "model_name": str(row.get("model_name", "")).strip(),
                    "model_path": str(row.get("model_path", "")).strip(),
                },
            )
    return rows


def save_model_registry_rows(rows: list[dict[str, str]]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_REGISTRY_PATH.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["id", "model_name", "model_path"])
        writer.writeheader()
        writer.writerows(rows)


def next_download_model_name(rows: list[dict[str, str]]) -> str:
    versions: list[tuple[int, int, int]] = []
    prefix = "Download Model v"
    for row in rows:
        model_name = row.get("model_name", "")
        if not model_name.startswith(prefix):
            continue
        raw_version = model_name.removeprefix(prefix)
        parts = raw_version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        versions.append((int(parts[0]), int(parts[1]), int(parts[2])))

    if not versions:
        return "Download Model v01.00.00"

    major, minor, patch = max(versions)
    patch += 1
    if patch > 99:
        patch = 0
        minor += 1
    if minor > 99:
        minor = 0
        major += 1
    return f"Download Model v{major:02d}.{minor:02d}.{patch:02d}"


def list_model_paths() -> list[pathlib.Path]:
    model_dir = TRAIN_DEFAULT_MODEL_DIR
    model_paths = [path for path in model_dir.glob("*.joblib") if path.is_file()]
    return sorted(model_paths, key=lambda path: path.stat().st_mtime, reverse=True)


def sync_model_registry_with_files() -> None:
    rows = load_model_registry_rows()
    known_paths = {row.get("model_path", "") for row in rows}
    changed = False
    for model_file in list_model_paths():
        resolved = str(model_file.resolve())
        if resolved in known_paths:
            continue
        max_id = 0
        for row in rows:
            if row.get("id", "0").isdigit():
                max_id = max(max_id, int(row["id"]))
        rows.append(
            {
                "id": str(max_id + 1),
                "model_name": next_download_model_name(rows),
                "model_path": resolved,
            },
        )
        known_paths.add(resolved)
        changed = True
    if changed:
        save_model_registry_rows(rows)


def register_model(model_path_raw: str) -> dict[str, str]:
    rows = load_model_registry_rows()
    model_path = str(pathlib.Path(model_path_raw).expanduser().resolve())
    for row in rows:
        if row.get("model_path") == model_path:
            return row

    max_id = 0
    for row in rows:
        if row.get("id", "0").isdigit():
            max_id = max(max_id, int(row["id"]))
    new_row = {
        "id": str(max_id + 1),
        "model_name": next_download_model_name(rows),
        "model_path": model_path,
    }
    rows.append(new_row)
    save_model_registry_rows(rows)
    return new_row


def find_model_by_name(model_name: str) -> dict[str, str] | None:
    sync_model_registry_with_files()
    rows = load_model_registry_rows()
    for row in rows:
        if row.get("model_name") == model_name:
            return row
    return None


def estimate_prediction_timestamp(frame: pd.DataFrame, horizon: int) -> tuple[dt.datetime, int]:
    latest_ts = frame.iloc[-1]["timestamp_utc"]
    if len(frame) >= 2:
        delta = latest_ts - frame.iloc[-2]["timestamp_utc"]
        if delta <= dt.timedelta(0):
            delta = dt.timedelta(minutes=1)
    else:
        delta = dt.timedelta(minutes=1)
    step_seconds = max(int(delta.total_seconds()), 60)
    return latest_ts + (delta * max(horizon, 1)), step_seconds


def build_predicted_sample(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    prediction: float,
) -> tuple[SpeedSample, int]:
    latest = frame.iloc[-1]
    predicted_timestamp, step_seconds = estimate_prediction_timestamp(frame, horizon)

    download = float(latest["download_mbps"])
    upload = float(latest["upload_mbps"])
    ping = float(latest["ping_ms"])
    if target == "download_mbps":
        download = prediction
    elif target == "upload_mbps":
        upload = prediction
    elif target == "ping_ms":
        ping = prediction

    sample = SpeedSample(
        timestamp_utc=predicted_timestamp.isoformat(),
        download_mbps=round(download, 3),
        upload_mbps=round(upload, 3),
        ping_ms=round(ping, 3),
        server_name=PREDICT_TARGET_SERVER_NAME,
        server_country=PREDICT_TARGET_SERVER_COUNTRY,
        isp=str(latest.get("isp", "predicted")),
        public_ip=str(latest.get("public_ip", "127.0.0.1")),
        local_hostname=str(latest.get("local_hostname", "predicted-host")),
        machine=str(latest.get("machine", "predicted-machine")),
    )
    return sample, step_seconds


def load_prediction_history(limit: int = 1000) -> tuple[list[dt.datetime], list[float]]:
    if psycopg is None:
        return [], []
    conninfo = (
        f"host={st.session_state.db_host} "
        f"port={int(st.session_state.db_port)} "
        f"dbname={st.session_state.db_name} "
        f"user={st.session_state.db_user} "
        f"password={st.session_state.db_password}"
    )
    query = """
    WITH latest AS (
        SELECT MAX(timestamp_utc) AS max_ts
        FROM internet_speed_predictions
        WHERE server_name = %s
          AND server_country = %s
    )
    SELECT p.timestamp_utc, p.download_mbps
    FROM internet_speed_predictions p
    CROSS JOIN latest l
    WHERE p.server_name = %s
      AND p.server_country = %s
      AND l.max_ts IS NOT NULL
      AND p.timestamp_utc >= (l.max_ts - interval '1 hour')
    ORDER BY p.timestamp_utc ASC
    LIMIT %s
    """
    with psycopg.connect(conninfo=conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    PREDICT_TARGET_SERVER_NAME,
                    PREDICT_TARGET_SERVER_COUNTRY,
                    PREDICT_TARGET_SERVER_NAME,
                    PREDICT_TARGET_SERVER_COUNTRY,
                    limit,
                ),
            )
            rows = cur.fetchall()
    if not rows:
        return [], []
    return [row[0] for row in rows], [float(row[1]) for row in rows]


def render_settings() -> None:
    st.sidebar.title("Settings")
    st.session_state.output_path = st.sidebar.text_input(
        "Output CSV",
        st.session_state.output_path,
        key="settings_output_csv",
    )
    st.session_state.timeout_seconds = st.sidebar.number_input(
        "Timeout (s)",
        min_value=1,
        value=int(st.session_state.timeout_seconds),
        key="settings_timeout_seconds",
    )
    st.session_state.interval_seconds = st.sidebar.number_input(
        "Interval (s)",
        min_value=1,
        value=int(st.session_state.interval_seconds),
        key="settings_interval_seconds",
    )
    st.session_state.storage_mode = st.sidebar.selectbox(
        "Storage",
        ["csv", "postgres", "both"],
        index=["csv", "postgres", "both"].index(str(st.session_state.storage_mode)),
        key="settings_storage_mode",
    )
    st.sidebar.markdown("### PostgreSQL")
    st.session_state.db_host = st.sidebar.text_input(
        "DB host",
        st.session_state.db_host,
        key="settings_db_host",
    )
    st.session_state.db_port = st.sidebar.number_input(
        "DB port",
        min_value=1,
        value=int(st.session_state.db_port),
        key="settings_db_port",
    )
    st.session_state.db_name = st.sidebar.text_input(
        "DB name",
        st.session_state.db_name,
        key="settings_db_name",
    )
    st.session_state.db_user = st.sidebar.text_input(
        "DB user",
        st.session_state.db_user,
        key="settings_db_user",
    )
    st.session_state.db_password = st.sidebar.text_input(
        "DB password",
        st.session_state.db_password,
        type="password",
        key="settings_db_password",
    )


def monitor_tab() -> None:
    st.subheader("Monitor")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Measure Once", key="monitor_measure_once", width="stretch"):
            try:
                sample = collect_sample(timeout=int(st.session_state.timeout_seconds))
                storage = str(st.session_state.storage_mode)
                path = output_path()
                if storage in {"csv", "both"}:
                    append_sample_csv(sample=sample, output_path=path)
                if storage in {"postgres", "both"}:
                    append_sample_postgres(
                        sample=sample,
                        host=str(st.session_state.db_host),
                        port=int(st.session_state.db_port),
                        database=str(st.session_state.db_name),
                        user=str(st.session_state.db_user),
                        password=str(st.session_state.db_password),
                    )
                st.session_state.last_sample = asdict(sample)
                add_event(f"Sample saved ({storage}) down={sample.download_mbps} Mbps")
                st.success("Sample collected successfully.")
            except Exception as exc:  # noqa: BLE001
                add_event(f"Sample error: {exc}")
                st.error(str(exc))

    with col2:
        batch_samples = st.number_input(
            "Batch samples",
            min_value=1,
            max_value=30,
            value=3,
            key="monitor_batch_samples",
        )
        if st.button("Run Batch", key="monitor_run_batch", width="stretch"):
            try:
                interval = int(st.session_state.interval_seconds)
                for index in range(int(batch_samples)):
                    sample = collect_sample(timeout=int(st.session_state.timeout_seconds))
                    storage = str(st.session_state.storage_mode)
                    path = output_path()
                    if storage in {"csv", "both"}:
                        append_sample_csv(sample=sample, output_path=path)
                    if storage in {"postgres", "both"}:
                        append_sample_postgres(
                            sample=sample,
                            host=str(st.session_state.db_host),
                            port=int(st.session_state.db_port),
                            database=str(st.session_state.db_name),
                            user=str(st.session_state.db_user),
                            password=str(st.session_state.db_password),
                        )
                    st.session_state.last_sample = asdict(sample)
                    add_event(f"[{index + 1}/{batch_samples}] Saved sample ({storage}).")
                    if index < int(batch_samples) - 1:
                        time.sleep(interval)
                st.success("Batch completed.")
            except Exception as exc:  # noqa: BLE001
                add_event(f"Batch error: {exc}")
                st.error(str(exc))

    sample = st.session_state.last_sample
    if isinstance(sample, dict):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Download", f"{sample['download_mbps']} Mbps")
        m2.metric("Upload", f"{sample['upload_mbps']} Mbps")
        m3.metric("Ping", f"{sample['ping_ms']} ms")
        m4.metric("Server", str(sample["server_name"]))
        st.caption(f"Last sample UTC: {sample['timestamp_utc']}")

    st.markdown("### Event Log")
    st.text_area("Events", value="\n".join(st.session_state.events), height=220, label_visibility="collapsed")


def results_tab() -> None:
    st.subheader("Resultados")
    input_path = output_path()
    clean_path = clean_output_path()
    rolling_window = st.number_input("Rolling window", min_value=1, value=5, key="results_rolling_window")
    if st.button("Analyze CSV", key="results_analyze_csv"):
        try:
            rows = analyze_load_data(input_path)
            if not rows:
                raise ValueError("CSV has no valid rows.")
            rows, dedup_dropped = deduplicate_rows(rows)
            server_name_replacements = canonical_server_names(rows)
            outlier_counts = add_outlier_flags(rows)
            write_clean_csv(clean_path, rows)
            report = build_report(
                rows=rows,
                rolling_window=int(rolling_window),
                dedup_dropped=dedup_dropped,
                server_name_replacements=server_name_replacements,
                outlier_counts=outlier_counts,
                output_clean=clean_path,
            )
            st.session_state.analysis_report = report
            add_event(f"Analysis complete. Clean CSV: {clean_path}")
            st.success("Analysis complete.")
        except Exception as exc:  # noqa: BLE001
            add_event(f"Analysis error: {exc}")
            st.error(str(exc))

    st.code(str(st.session_state.analysis_report or "No analysis yet."), language="text")
    if clean_path.exists():
        st.markdown("### Clean CSV Preview")
        st.dataframe(pd.read_csv(clean_path).tail(30), width="stretch")


def train_tab() -> None:
    st.subheader("Entrenar")
    left, right = st.columns(2)
    with left:
        train_input = pathlib.Path(
            st.text_input("Input CSV", str(TRAIN_DEFAULT_INPUT), key="train_input_csv"),
        ).expanduser()
        target = st.selectbox("Target", list(TRAIN_TARGET_COLUMNS), index=0, key="train_target")
        lags = st.number_input("Lags", min_value=1, value=5, key="train_lags")
        horizon = st.number_input("Horizon", min_value=1, value=1, key="train_horizon")
        test_size = st.number_input(
            "Test Size",
            min_value=0.01,
            max_value=0.49,
            value=0.20,
            step=0.01,
            key="train_test_size",
        )
    with right:
        model_output = pathlib.Path(
            st.text_input(
                "Model output",
                str(TRAIN_DEFAULT_MODEL_DIR / "medidor_forecast.joblib"),
                key="train_model_output",
            ),
        ).expanduser()
        metrics_output = pathlib.Path(
            st.text_input(
                "Metrics output",
                str(TRAIN_DEFAULT_MODEL_DIR / "medidor_forecast_metrics.json"),
                key="train_metrics_output",
            ),
        ).expanduser()

    if st.button("Train Model", key="train_model_btn", width="stretch"):
        try:
            frame = train_load_data(train_input)
            x, y = build_training_frame(frame=frame, target=target, lags=int(lags), horizon=int(horizon))
            x_train, x_test, y_train, y_test = train_test_split_time(
                x=x,
                y=y,
                test_size=float(test_size),
            )
            model, metrics = evaluate_forecast(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                target=target,
            )
            outputs = save_outputs(
                model=model,
                x_train=x_train,
                metrics=metrics,
                target=target,
                horizon=int(horizon),
                lags=int(lags),
                model_output=model_output,
                metrics_output=metrics_output,
            )
            registry_row = register_model(outputs["versioned_model_output"])
            sync_model_registry_with_files()
            st.session_state.train_result = {
                "metrics": asdict(metrics),
                "outputs": outputs,
                "registry": registry_row,
            }
            add_event(
                "Training complete "
                f"(model={registry_row['model_name']}, version={outputs['model_version']}).",
            )
            st.success("Model trained successfully.")
        except Exception as exc:  # noqa: BLE001
            add_event(f"Training error: {exc}")
            st.error(str(exc))

    result = st.session_state.train_result
    if isinstance(result, dict):
        metrics = result["metrics"]
        outputs = result["outputs"]
        reg = result["registry"]
        st.markdown("### Training Summary")
        st.write(
            {
                "model_name": reg["model_name"],
                "model_version": outputs["model_version"],
                "samples_total": metrics["samples_total"],
                "samples_train": metrics["samples_train"],
                "samples_test": metrics["samples_test"],
                "mae_baseline": round(metrics["mae_baseline"], 3),
                "mae_model": round(metrics["mae_model"], 3),
                "rmse_baseline": round(metrics["rmse_baseline"], 3),
                "rmse_model": round(metrics["rmse_model"], 3),
            },
        )


def prediction_tab() -> None:
    st.subheader("Predicciones")
    sync_model_registry_with_files()
    registry_rows = load_model_registry_rows()
    model_names = [row["model_name"] for row in sorted(registry_rows, key=lambda r: int(r["id"]), reverse=True)]
    selected_model_name = st.selectbox(
        "Model",
        model_names if model_names else ["No models found"],
        key="predict_model_name",
    )
    predict_input = pathlib.Path(
        st.text_input("Input CSV", str(TRAIN_DEFAULT_INPUT), key="predict_input_csv"),
    ).expanduser()
    save_to_postgres = st.checkbox(
        "Save prediction to PostgreSQL",
        value=False,
        key="predict_save_to_postgres",
    )

    if st.button("Predict Next", key="predict_next_btn", width="stretch"):
        try:
            if selected_model_name == "No models found":
                raise ValueError("No models available. Train first.")
            selected_row = find_model_by_name(selected_model_name)
            if selected_row is None:
                raise ValueError(f"Model not found in registry: {selected_model_name}")
            model_path = pathlib.Path(selected_row["model_path"]).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            payload = predict_load_model(model_path)
            frame = predict_load_data(predict_input)
            model = payload["model"]
            feature_columns = payload["feature_columns"]
            target = str(payload["target"])
            horizon = int(payload["horizon"])
            lags = int(payload["lags"])
            model_version = str(payload.get("model_version", model_path.stem))

            features = build_latest_feature_row(frame=frame, feature_columns=feature_columns, lags=lags)
            prediction = float(model.predict(features)[0])
            latest_ts = frame.iloc[-1]["timestamp_utc"]
            predicted_sample, expected_step_seconds = build_predicted_sample(
                frame=frame,
                target=target,
                horizon=horizon,
                prediction=prediction,
            )
            db_status = "Not saved to PostgreSQL."
            if save_to_postgres:
                append_prediction_postgres(
                    sample=predicted_sample,
                    host=str(st.session_state.db_host),
                    port=int(st.session_state.db_port),
                    database=str(st.session_state.db_name),
                    user=str(st.session_state.db_user),
                    password=str(st.session_state.db_password),
                    predicted_from_timestamp_utc=latest_ts.isoformat(),
                    target_metric=target,
                    horizon_steps=horizon,
                    expected_step_seconds=expected_step_seconds,
                    model_version=model_version,
                    input_source=str(predict_input),
                )
                db_status = "Saved to PostgreSQL."

            st.session_state.predict_result = {
                "target": target,
                "horizon": horizon,
                "prediction": prediction,
                "latest_ts": latest_ts,
                "predicted_ts": predicted_sample.timestamp_utc,
                "model_name": selected_model_name,
                "model_path": str(model_path),
                "input_path": str(predict_input),
                "db_status": db_status,
            }
            add_event(
                f"Prediction complete ({selected_model_name}, target={target}, value={prediction:.3f}).",
            )
            st.success("Prediction completed.")
        except Exception as exc:  # noqa: BLE001
            add_event(f"Prediction error: {exc}")
            st.error(str(exc))

    result = st.session_state.predict_result
    if isinstance(result, dict):
        st.write(
            {
                "model_name": result["model_name"],
                "target": result["target"],
                "horizon": result["horizon"],
                "latest_timestamp_utc": str(result["latest_ts"]),
                "predicted_timestamp_utc": result["predicted_ts"],
                "prediction": round(float(result["prediction"]), 3),
                "db_status": result["db_status"],
            },
        )

        frame = predict_load_data(pathlib.Path(result["input_path"]))
        latest_ts_all = frame.iloc[-1]["timestamp_utc"]
        window_start = latest_ts_all - dt.timedelta(hours=1)
        history = frame[frame["timestamp_utc"] >= window_start].copy()
        if history.empty:
            history = frame.tail(1).copy()

        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(history["timestamp_utc"], history["download_mbps"], color="#0a84ff", linewidth=1.8)
        ax1.set_title("Download speed history (last 1 hour)")
        ax1.set_ylabel("Mbps")
        ax1.grid(alpha=0.25)
        st.pyplot(fig1, width="stretch")

        pred_ts_list, pred_values = load_prediction_history(limit=1000)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        if pred_ts_list:
            ax2.plot(pred_ts_list, pred_values, color="#d62828", linewidth=1.8, marker="o", markersize=3)
            ax2.set_title("Prediction history (PostgreSQL, last 1 hour)")
        else:
            ax2.set_title("Prediction history unavailable (no rows or PostgreSQL not configured)")
        ax2.set_ylabel("Mbps")
        ax2.grid(alpha=0.25)
        st.pyplot(fig2, width="stretch")

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(
            history["timestamp_utc"],
            history["download_mbps"],
            color="#0a84ff",
            linewidth=1.8,
            label="download history",
        )
        if pred_ts_list:
            ax3.plot(
                pred_ts_list,
                pred_values,
                color="#d62828",
                linewidth=1.4,
                marker="o",
                markersize=2.5,
                alpha=0.9,
                label="prediction history",
            )

        latest_ts = history.iloc[-1]["timestamp_utc"]
        latest_download = float(history.iloc[-1]["download_mbps"])
        predicted_ts = dt.datetime.fromisoformat(str(result["predicted_ts"]))
        ax3.plot(
            [latest_ts, predicted_ts],
            [latest_download, float(result["prediction"])],
            color="#ff7f11",
            linewidth=2.0,
            marker="o",
            label="latest prediction",
        )
        ax3.set_title("Combined chart (last 1 hour)")
        ax3.set_ylabel("Mbps")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="upper left")
        st.pyplot(fig3, width="stretch")


def models_tab() -> None:
    st.subheader("Modelos")
    sync_model_registry_with_files()
    rows = sorted(load_model_registry_rows(), key=lambda row: int(row["id"]) if row["id"].isdigit() else 0)
    st.caption(f"Registry file: {MODEL_REGISTRY_PATH.resolve()}")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No models in registry yet.")


def main() -> None:
    st.set_page_config(page_title="MEDIDOR Web UI", layout="wide")
    init_state()
    render_settings()

    st.title("MEDIDOR - Web UI")
    tab_monitor, tab_results, tab_train, tab_predict, tab_models = st.tabs(
        ["Monitor", "Resultados", "Entrenar", "Predicciones", "Modelos"],
    )
    with tab_monitor:
        monitor_tab()
    with tab_results:
        results_tab()
    with tab_train:
        train_tab()
    with tab_predict:
        prediction_tab()
    with tab_models:
        models_tab()


if __name__ == "__main__":
    main()
