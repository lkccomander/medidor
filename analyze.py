"""Analyze internet speed samples collected by the MEDIDOR script."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import statistics
import unicodedata
from collections import Counter


DEFAULT_INPUT = pathlib.Path(__file__).with_name("internet_speed_data.csv")
DEFAULT_OUTPUT = pathlib.Path(__file__).with_name("internet_speed_data_clean.csv")
TARGET_SERVER_NAME = "San José"
TARGET_SERVER_COUNTRY = "Costa Rica"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze internet speed data collected in CSV format.",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help=f"Path to input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window size for trend estimation (default: 5).",
    )
    parser.add_argument(
        "--output-clean",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write cleaned CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float:
    """Compute percentile using linear interpolation."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile on empty list.")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]

    rank = (len(sorted_values) - 1) * q
    low_index = int(rank)
    high_index = min(low_index + 1, len(sorted_values) - 1)
    fraction = rank - low_index
    low_value = sorted_values[low_index]
    high_value = sorted_values[high_index]
    return low_value + (high_value - low_value) * fraction


def load_data(input_path: pathlib.Path) -> list[dict[str, object]]:
    """Load and validate expected MEDIDOR CSV schema."""
    if not input_path.exists():
        message = f"Input CSV not found: {input_path}"
        raise FileNotFoundError(message)

    rows: list[dict[str, object]] = []
    required_columns = {
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
    }
    with input_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            missing = ", ".join(missing_columns)
            message = f"CSV is missing required columns: {missing}"
            raise ValueError(message)

        for raw_row in reader:
            timestamp_raw = (raw_row.get("timestamp_utc") or "").strip()
            if not timestamp_raw:
                continue
            server_name = str(raw_row.get("server_name", ""))
            server_country = str(raw_row.get("server_country", ""))
            if normalize_text(server_name).casefold() != normalize_text(TARGET_SERVER_NAME).casefold():
                continue
            if (
                normalize_text(server_country).casefold()
                != normalize_text(TARGET_SERVER_COUNTRY).casefold()
            ):
                continue
            try:
                timestamp_utc = dt.datetime.fromisoformat(timestamp_raw)
            except ValueError:
                continue
            if timestamp_utc.tzinfo is None:
                timestamp_utc = timestamp_utc.replace(tzinfo=dt.UTC)
            rows.append(
                {
                    "timestamp_utc": timestamp_utc.astimezone(dt.UTC),
                    "download_mbps": float(raw_row["download_mbps"]),
                    "upload_mbps": float(raw_row["upload_mbps"]),
                    "ping_ms": float(raw_row["ping_ms"]),
                    "server_name": server_name,
                    "server_country": server_country,
                    "isp": str(raw_row["isp"]),
                    "public_ip": str(raw_row["public_ip"]),
                    "local_hostname": str(raw_row["local_hostname"]),
                    "machine": str(raw_row["machine"]),
                },
            )

    rows.sort(key=lambda item: item["timestamp_utc"])
    return rows


def normalize_text(value: str) -> str:
    """Normalize unicode and whitespace in a text field."""
    normalized = unicodedata.normalize("NFC", value.strip())
    return " ".join(normalized.split())


def canonical_server_names(rows: list[dict[str, object]]) -> int:
    """Unify server names that differ only by accents/casing."""
    counts_by_key: dict[str, Counter[str]] = {}
    for row in rows:
        original = normalize_text(str(row["server_name"]))
        row["server_name"] = original
        key = "".join(
            char
            for char in unicodedata.normalize("NFKD", original.casefold())
            if not unicodedata.combining(char)
        )
        counts_by_key.setdefault(key, Counter())[original] += 1

    canonical_by_key = {
        key: counter.most_common(1)[0][0]
        for key, counter in counts_by_key.items()
    }

    replacements = 0
    for row in rows:
        original = str(row["server_name"])
        key = "".join(
            char
            for char in unicodedata.normalize("NFKD", original.casefold())
            if not unicodedata.combining(char)
        )
        canonical = canonical_by_key[key]
        if canonical != original:
            replacements += 1
            row["server_name"] = canonical
    return replacements


def deduplicate_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    """Drop duplicate rows by natural key."""
    deduped: list[dict[str, object]] = []
    seen: set[tuple[dt.datetime, str, str]] = set()
    dropped = 0
    for row in rows:
        key = (
            row["timestamp_utc"],
            str(row["public_ip"]),
            str(row["local_hostname"]),
        )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, dropped


def summarize_series(name: str, values: list[float]) -> list[str]:
    """Build text summary for one metric."""
    sorted_values = sorted(values)
    return [
        f"{name}:",
        f"  avg={statistics.fmean(values):.3f}",
        f"  median={statistics.median(sorted_values):.3f}",
        f"  p95={percentile(sorted_values, 0.95):.3f}",
        f"  min={sorted_values[0]:.3f}",
        f"  max={sorted_values[-1]:.3f}",
        f"  latest={values[-1]:.3f}",
    ]


def find_download_outliers(values: list[float]) -> tuple[int, float, float]:
    """Return outlier count and Tukey bounds for download speed."""
    sorted_values = sorted(values)
    q1 = percentile(sorted_values, 0.25)
    q3 = percentile(sorted_values, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    count = sum(value < lower_bound or value > upper_bound for value in values)
    return count, lower_bound, upper_bound


def iqr_bounds(values: list[float]) -> tuple[float, float]:
    """Return Tukey IQR bounds."""
    sorted_values = sorted(values)
    q1 = percentile(sorted_values, 0.25)
    q3 = percentile(sorted_values, 0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def add_outlier_flags(rows: list[dict[str, object]]) -> dict[str, int]:
    """Add per-metric and global outlier flags in-place."""
    download_values = [float(row["download_mbps"]) for row in rows]
    upload_values = [float(row["upload_mbps"]) for row in rows]
    ping_values = [float(row["ping_ms"]) for row in rows]

    dl_low, dl_high = iqr_bounds(download_values)
    ul_low, ul_high = iqr_bounds(upload_values)
    pg_low, pg_high = iqr_bounds(ping_values)

    counters = {
        "download_outliers": 0,
        "upload_outliers": 0,
        "ping_outliers": 0,
        "any_outliers": 0,
    }
    for row in rows:
        is_dl = int(float(row["download_mbps"]) < dl_low or float(row["download_mbps"]) > dl_high)
        is_ul = int(float(row["upload_mbps"]) < ul_low or float(row["upload_mbps"]) > ul_high)
        is_pg = int(float(row["ping_ms"]) < pg_low or float(row["ping_ms"]) > pg_high)
        is_any = int(bool(is_dl or is_ul or is_pg))

        row["is_outlier_download"] = is_dl
        row["is_outlier_upload"] = is_ul
        row["is_outlier_ping"] = is_pg
        row["is_outlier_any"] = is_any

        counters["download_outliers"] += is_dl
        counters["upload_outliers"] += is_ul
        counters["ping_outliers"] += is_pg
        counters["any_outliers"] += is_any

    return counters


def write_clean_csv(output_path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    """Write cleaned rows with engineered columns."""
    fieldnames = [
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
        "hour_utc",
        "is_outlier_download",
        "is_outlier_upload",
        "is_outlier_ping",
        "is_outlier_any",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp_utc": row["timestamp_utc"].isoformat(),
                    "download_mbps": f"{float(row['download_mbps']):.3f}",
                    "upload_mbps": f"{float(row['upload_mbps']):.3f}",
                    "ping_ms": f"{float(row['ping_ms']):.3f}",
                    "server_name": str(row["server_name"]),
                    "server_country": str(row["server_country"]),
                    "isp": str(row["isp"]),
                    "public_ip": str(row["public_ip"]),
                    "local_hostname": str(row["local_hostname"]),
                    "machine": str(row["machine"]),
                    "hour_utc": int(row["timestamp_utc"].hour),
                    "is_outlier_download": int(row["is_outlier_download"]),
                    "is_outlier_upload": int(row["is_outlier_upload"]),
                    "is_outlier_ping": int(row["is_outlier_ping"]),
                    "is_outlier_any": int(row["is_outlier_any"]),
                },
            )


def build_report(
    rows: list[dict[str, object]],
    rolling_window: int,
    *,
    dedup_dropped: int,
    server_name_replacements: int,
    outlier_counts: dict[str, int],
    output_clean: pathlib.Path,
) -> str:
    """Generate a human-readable report."""
    timestamps = [row["timestamp_utc"] for row in rows]
    download_values = [float(row["download_mbps"]) for row in rows]
    upload_values = [float(row["upload_mbps"]) for row in rows]
    ping_values = [float(row["ping_ms"]) for row in rows]
    server_names = {str(row["server_name"]) for row in rows}
    isp_names = {str(row["isp"]) for row in rows}

    lines: list[str] = []
    lines.append(f"samples={len(rows)}")
    lines.append(
        "period_utc="
        f"{timestamps[0].isoformat()} -> "
        f"{timestamps[-1].isoformat()}",
    )
    lines.append(f"servers={len(server_names)} | isp={len(isp_names)}")
    lines.append(
        "cleaning="
        f"dedup_dropped:{dedup_dropped} "
        f"server_name_replaced:{server_name_replacements}",
    )
    lines.append("")

    lines.extend(summarize_series("download_mbps", download_values))
    lines.append("")
    lines.extend(summarize_series("upload_mbps", upload_values))
    lines.append("")
    lines.extend(summarize_series("ping_ms", ping_values))
    lines.append("")

    hourly_totals: dict[int, float] = {}
    hourly_counts: dict[int, int] = {}
    for row in rows:
        hour = int(row["timestamp_utc"].hour)
        hourly_totals[hour] = hourly_totals.get(hour, 0.0) + float(row["download_mbps"])
        hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
    hourly_avg = sorted(
        (
            (hour, hourly_totals[hour] / hourly_counts[hour])
            for hour in hourly_totals
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    best_hour = hourly_avg[0][0]
    worst_hour = hourly_avg[-1][0]
    lines.append(
        "download_best_hour_utc="
        f"{best_hour:02d}:00 | "
        f"download_worst_hour_utc={worst_hour:02d}:00",
    )

    outlier_count, lower_bound, upper_bound = find_download_outliers(download_values)
    lines.append(
        f"download_outliers={outlier_count} "
        f"(bounds: {lower_bound:.3f}..{upper_bound:.3f})",
    )
    lines.append(
        "outlier_flags="
        f"download:{outlier_counts['download_outliers']} "
        f"upload:{outlier_counts['upload_outliers']} "
        f"ping:{outlier_counts['ping_outliers']} "
        f"any:{outlier_counts['any_outliers']}",
    )

    if rolling_window > 1 and len(download_values) >= rolling_window:
        rolling = [
            statistics.fmean(download_values[idx - rolling_window : idx])
            for idx in range(rolling_window, len(download_values) + 1)
        ]
        lines.append(
            f"download_trend_{rolling_window}="
            f"{rolling[-1]:.3f} Mbps (rolling mean)",
        )

    lines.append(f"clean_csv={output_clean}")
    return "\n".join(lines)


def main() -> None:
    """Load input data and print report."""
    args = parse_args()
    if args.rolling_window < 1:
        raise ValueError("--rolling-window must be >= 1")

    rows = load_data(args.input)
    if not rows:
        raise ValueError("Input CSV has no valid timestamp rows to analyze.")

    rows, dedup_dropped = deduplicate_rows(rows)
    server_name_replacements = canonical_server_names(rows)
    outlier_counts = add_outlier_flags(rows)
    write_clean_csv(args.output_clean, rows)

    report = build_report(
        rows=rows,
        rolling_window=args.rolling_window,
        dedup_dropped=dedup_dropped,
        server_name_replacements=server_name_replacements,
        outlier_counts=outlier_counts,
        output_clean=args.output_clean,
    )
    print(report)


if __name__ == "__main__":
    main()
