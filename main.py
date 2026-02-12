"""Internet speed data collector for future ML forecasting."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import pathlib
import platform
import socket
import time
import urllib.error
from dataclasses import asdict, dataclass


try:
    import speedtest
except ImportError as exc:  # pragma: no cover - runtime dependency check
    speedtest = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    import psycopg
except ImportError as exc:  # pragma: no cover - runtime dependency check
    psycopg = None
    POSTGRES_IMPORT_ERROR = exc
else:
    POSTGRES_IMPORT_ERROR = None


DEFAULT_OUTPUT = pathlib.Path(__file__).with_name("internet_speed_data.csv")
DEFAULT_DB_HOST = os.getenv("MEDIDOR_DB_HOST", "localhost")
DEFAULT_DB_PORT = int(os.getenv("MEDIDOR_DB_PORT", "5432"))
DEFAULT_DB_NAME = os.getenv("MEDIDOR_DB_NAME", "medidor")
DEFAULT_DB_USER = os.getenv("MEDIDOR_DB_USER", "postgres")
DEFAULT_DB_PASSWORD = os.getenv("MEDIDOR_DB_PASSWORD", "")
SAMPLES_TABLE = "internet_speed_samples"
PREDICTIONS_TABLE = "internet_speed_predictions"


@dataclass(slots=True)
class SpeedSample:
    """Single internet speed measurement."""

    timestamp_utc: str
    download_mbps: float
    upload_mbps: float
    ping_ms: float
    server_name: str
    server_country: str
    isp: str
    public_ip: str
    local_hostname: str
    machine: str


def collect_sample(timeout: int) -> SpeedSample:
    """Run one speedtest and normalize output for CSV storage."""
    if speedtest is None:
        message = (
            "Missing dependency 'speedtest-cli'. Install with:\n"
            "  pip install speedtest-cli"
        )
        raise RuntimeError(message) from IMPORT_ERROR

    try:
        st = speedtest.Speedtest(timeout=timeout, secure=True)
        st.get_best_server()
        download_bps = st.download()
        upload_bps = st.upload(pre_allocate=False)
        results = st.results.dict()
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            message = (
                "Speedtest service returned HTTP 403 (Forbidden). "
                "Try upgrading speedtest-cli: pip install -U speedtest-cli"
            )
            raise RuntimeError(message) from exc
        raise
    server_data = results.get("server", {})
    client_data = results.get("client", {})

    return SpeedSample(
        timestamp_utc=dt.datetime.now(dt.UTC).isoformat(),
        download_mbps=round(download_bps / 1_000_000, 3),
        upload_mbps=round(upload_bps / 1_000_000, 3),
        ping_ms=round(float(results.get("ping", 0.0)), 3),
        server_name=str(server_data.get("name", "")),
        server_country=str(server_data.get("country", "")),
        isp=str(client_data.get("isp", "")),
        public_ip=str(client_data.get("ip", "")),
        local_hostname=socket.gethostname(),
        machine=platform.platform(),
    )


def append_sample_csv(sample: SpeedSample, output_path: pathlib.Path) -> None:
    """Append one sample to CSV, writing header if file is new."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(sample)
    file_exists = output_path.exists()
    with output_path.open(mode="a", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_sample_postgres(
    sample: SpeedSample,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
) -> None:
    """Insert one sample into PostgreSQL (creates table/index if needed)."""
    _append_speed_sample_postgres(
        sample=sample,
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        table_name=SAMPLES_TABLE,
        index_name="ux_internet_speed_samples_natural",
    )


def append_prediction_postgres(
    sample: SpeedSample,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    predicted_from_timestamp_utc: str,
    target_metric: str,
    horizon_steps: int,
    expected_step_seconds: int,
    model_version: str,
    input_source: str,
) -> None:
    """Insert one predicted sample into PostgreSQL prediction table."""
    if psycopg is None:
        message = (
            "Missing dependency 'psycopg'. Install with:\n"
            '  pip install "psycopg[binary]"'
        )
        raise RuntimeError(message) from POSTGRES_IMPORT_ERROR

    conninfo = (
        f"host={host} "
        f"port={port} "
        f"dbname={database} "
        f"user={user} "
        f"password={password}"
    )
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        timestamp_utc TIMESTAMPTZ NOT NULL,
        download_mbps DOUBLE PRECISION NOT NULL,
        upload_mbps DOUBLE PRECISION NOT NULL,
        ping_ms DOUBLE PRECISION NOT NULL,
        server_name TEXT NOT NULL,
        server_country TEXT NOT NULL,
        isp TEXT NOT NULL,
        public_ip INET NOT NULL,
        local_hostname TEXT NOT NULL,
        machine TEXT NOT NULL,
        predicted_from_timestamp_utc TIMESTAMPTZ,
        target_metric TEXT,
        horizon_steps INTEGER,
        expected_step_seconds INTEGER,
        model_version TEXT,
        input_source TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    ALTER TABLE {PREDICTIONS_TABLE}
        ADD COLUMN IF NOT EXISTS predicted_from_timestamp_utc TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS target_metric TEXT,
        ADD COLUMN IF NOT EXISTS horizon_steps INTEGER,
        ADD COLUMN IF NOT EXISTS expected_step_seconds INTEGER,
        ADD COLUMN IF NOT EXISTS model_version TEXT,
        ADD COLUMN IF NOT EXISTS input_source TEXT;

    CREATE UNIQUE INDEX IF NOT EXISTS ux_internet_speed_predictions_natural
    ON {PREDICTIONS_TABLE} (timestamp_utc, public_ip, local_hostname);
    """
    insert_sql = f"""
    INSERT INTO {PREDICTIONS_TABLE} (
        timestamp_utc,
        download_mbps,
        upload_mbps,
        ping_ms,
        server_name,
        server_country,
        isp,
        public_ip,
        local_hostname,
        machine,
        predicted_from_timestamp_utc,
        target_metric,
        horizon_steps,
        expected_step_seconds,
        model_version,
        input_source
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (timestamp_utc, public_ip, local_hostname) DO NOTHING;
    """
    row = asdict(sample)
    values = (
        row["timestamp_utc"],
        row["download_mbps"],
        row["upload_mbps"],
        row["ping_ms"],
        row["server_name"],
        row["server_country"],
        row["isp"],
        row["public_ip"],
        row["local_hostname"],
        row["machine"],
        predicted_from_timestamp_utc,
        target_metric,
        horizon_steps,
        expected_step_seconds,
        model_version,
        input_source,
    )
    with psycopg.connect(conninfo=conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(insert_sql, values)
        conn.commit()


def _append_speed_sample_postgres(
    sample: SpeedSample,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    table_name: str,
    index_name: str,
) -> None:
    """Insert a speed sample into the selected PostgreSQL table."""
    if psycopg is None:
        message = (
            "Missing dependency 'psycopg'. Install with:\n"
            '  pip install "psycopg[binary]"'
        )
        raise RuntimeError(message) from POSTGRES_IMPORT_ERROR

    conninfo = (
        f"host={host} "
        f"port={port} "
        f"dbname={database} "
        f"user={user} "
        f"password={password}"
    )
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGSERIAL PRIMARY KEY,
        timestamp_utc TIMESTAMPTZ NOT NULL,
        download_mbps DOUBLE PRECISION NOT NULL,
        upload_mbps DOUBLE PRECISION NOT NULL,
        ping_ms DOUBLE PRECISION NOT NULL,
        server_name TEXT NOT NULL,
        server_country TEXT NOT NULL,
        isp TEXT NOT NULL,
        public_ip INET NOT NULL,
        local_hostname TEXT NOT NULL,
        machine TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
    ON {table_name} (timestamp_utc, public_ip, local_hostname);
    """
    insert_sql = f"""
    INSERT INTO {table_name} (
        timestamp_utc,
        download_mbps,
        upload_mbps,
        ping_ms,
        server_name,
        server_country,
        isp,
        public_ip,
        local_hostname,
        machine
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (timestamp_utc, public_ip, local_hostname) DO NOTHING;
    """
    row = asdict(sample)
    values = (
        row["timestamp_utc"],
        row["download_mbps"],
        row["upload_mbps"],
        row["ping_ms"],
        row["server_name"],
        row["server_country"],
        row["isp"],
        row["public_ip"],
        row["local_hostname"],
        row["machine"],
    )
    with psycopg.connect(conninfo=conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(insert_sql, values)
        conn.commit()


def parse_args() -> argparse.Namespace:
    """CLI arguments for controlled data collection."""
    parser = argparse.ArgumentParser(
        description="Collect internet speed samples and store them in CSV.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to collect (default: 1).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=0,
        help="Seconds to wait between samples (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Timeout for speedtest requests in seconds (default: 20).",
    )
    parser.add_argument(
        "--storage",
        choices=("csv", "postgres", "both"),
        default="csv",
        help="Where to save samples: csv, postgres, or both (default: csv).",
    )
    parser.add_argument("--db-host", default=DEFAULT_DB_HOST)
    parser.add_argument("--db-port", type=int, default=DEFAULT_DB_PORT)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-password", default=DEFAULT_DB_PASSWORD)
    return parser.parse_args()


def main() -> None:
    """Run collection loop and persist samples."""
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds must be >= 0")

    for index in range(args.samples):
        sample = collect_sample(timeout=args.timeout)
        if args.storage in {"csv", "both"}:
            append_sample_csv(sample=sample, output_path=args.output)
        if args.storage in {"postgres", "both"}:
            append_sample_postgres(
                sample=sample,
                host=args.db_host,
                port=args.db_port,
                database=args.db_name,
                user=args.db_user,
                password=args.db_password,
            )
        print(
            f"[{index + 1}/{args.samples}] "
            f"{sample.timestamp_utc} | "
            f"down={sample.download_mbps} Mbps | "
            f"up={sample.upload_mbps} Mbps | "
            f"ping={sample.ping_ms} ms | "
            f"storage={args.storage}",
        )
        is_last = index == args.samples - 1
        if not is_last and args.interval_seconds:
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
