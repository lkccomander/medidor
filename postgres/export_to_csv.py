"""Export MEDIDOR samples from PostgreSQL to CSV."""

from __future__ import annotations

import argparse
import csv
import pathlib

try:
    import psycopg
except ImportError as exc:  # pragma: no cover - runtime dependency check
    message = (
        "Missing dependency 'psycopg'. Install with:\n"
        '  pip install "psycopg[binary]"'
    )
    raise RuntimeError(message) from exc


DEFAULT_OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "internet_speed_data.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Export MEDIDOR PostgreSQL data to CSV.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="medidor")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="moco23")
    parser.add_argument(
        "--from-ts",
        default=None,
        help="Optional lower bound timestamp (ISO format), e.g. 2026-02-11T00:00:00+00:00",
    )
    parser.add_argument(
        "--to-ts",
        default=None,
        help="Optional upper bound timestamp (ISO format), e.g. 2026-02-12T00:00:00+00:00",
    )
    return parser.parse_args()


def export_rows(args: argparse.Namespace) -> int:
    """Read rows from PostgreSQL and write them to CSV."""
    args.output.parent.mkdir(parents=True, exist_ok=True)

    conninfo = (
        f"host={args.host} "
        f"port={args.port} "
        f"dbname={args.database} "
        f"user={args.user} "
        f"password={args.password}"
    )
    sql = """
    SELECT
        timestamp_utc,
        download_mbps,
        upload_mbps,
        ping_ms,
        server_name,
        server_country,
        isp,
        host(public_ip) AS public_ip,
        local_hostname,
        machine
    FROM internet_speed_samples
    WHERE (%s IS NULL OR timestamp_utc >= %s::timestamptz)
      AND (%s IS NULL OR timestamp_utc <= %s::timestamptz)
    ORDER BY timestamp_utc ASC;
    """
    total = 0
    with psycopg.connect(conninfo=conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (args.from_ts, args.from_ts, args.to_ts, args.to_ts))
            rows = cur.fetchall()

    header = [
        "timestamp_utc",
        "download_mbps",
        "upload_mbps",
        "ping_ms",
        "server_name",
        "server_country",
        "isp",
        "public_ip",
        "local_hostname",
        "machine",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            total += 1
    return total


def main() -> None:
    """Export from DB and print summary."""
    args = parse_args()
    total = export_rows(args)
    print(f"Rows exported: {total} -> {args.output}")


if __name__ == "__main__":
    main()
