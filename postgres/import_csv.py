"""Import MEDIDOR CSV data into PostgreSQL."""

from __future__ import annotations

import argparse
import csv
import pathlib

try:
    import psycopg
except ImportError as exc:  # pragma: no cover - runtime dependency check
    message = (
        "Missing dependency 'psycopg'. Install with:\n"
        "  pip install \"psycopg[binary]\""
    )
    raise RuntimeError(message) from exc


DEFAULT_CSV = pathlib.Path(__file__).resolve().parent.parent / "internet_speed_data.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Load MEDIDOR internet speed CSV into PostgreSQL.",
    )
    parser.add_argument("--csv-path", type=pathlib.Path, default=DEFAULT_CSV)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="medidor")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="moco23")
    return parser.parse_args()


def load_csv_rows(csv_path: pathlib.Path) -> list[tuple[object, ...]]:
    """Read CSV rows in the same order expected by INSERT."""
    if not csv_path.exists():
        message = f"CSV not found: {csv_path}"
        raise FileNotFoundError(message)

    rows: list[tuple[object, ...]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for raw_row in reader:
            rows.append(
                (
                    raw_row["timestamp_utc"],
                    float(raw_row["download_mbps"]),
                    float(raw_row["upload_mbps"]),
                    float(raw_row["ping_ms"]),
                    raw_row["server_name"],
                    raw_row["server_country"],
                    raw_row["isp"],
                    raw_row["public_ip"],
                    raw_row["local_hostname"],
                    raw_row["machine"],
                ),
            )
    return rows


def ensure_table(conn: psycopg.Connection) -> None:
    """Create target table/index if not present."""
    ddl = """
    CREATE TABLE IF NOT EXISTS internet_speed_samples (
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

    CREATE UNIQUE INDEX IF NOT EXISTS ux_internet_speed_samples_natural
    ON internet_speed_samples (timestamp_utc, public_ip, local_hostname);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def insert_rows(conn: psycopg.Connection, rows: list[tuple[object, ...]]) -> int:
    """Insert rows with deduplication."""
    if not rows:
        return 0

    sql = """
    INSERT INTO internet_speed_samples (
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
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (timestamp_utc, public_ip, local_hostname) DO NOTHING;
    """
    inserted_count = 0
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(sql, row)
            inserted_count += cur.rowcount
    conn.commit()
    return inserted_count


def main() -> None:
    """Import CSV into PostgreSQL."""
    args = parse_args()
    rows = load_csv_rows(args.csv_path)

    conninfo = (
        f"host={args.host} "
        f"port={args.port} "
        f"dbname={args.database} "
        f"user={args.user} "
        f"password={args.password}"
    )
    with psycopg.connect(conninfo=conninfo) as conn:
        ensure_table(conn)
        inserted = insert_rows(conn, rows)

    print(
        f"Rows read: {len(rows)} | Rows inserted: {inserted} | "
        f"Rows skipped (duplicates): {len(rows) - inserted}",
    )


if __name__ == "__main__":
    main()
