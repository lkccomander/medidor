# MEDIDOR

Desktop + CLI toolkit to collect internet speed measurements, store them in CSV/PostgreSQL, and run a basic forecasting pipeline.

## Features

- Collect internet speed samples (`download`, `upload`, `ping`) with `speedtest-cli`.
- Store samples in CSV, PostgreSQL, or both.
- GUI app (`customtkinter`) to run collection, training, prediction, and charts.
- Data cleaning/report generation from collected CSV data.
- Forecast training (`scikit-learn`) and next-value prediction.
- Import/export helpers for PostgreSQL.
- Error logging to `app.log` from GUI/runtime exceptions.

## Requirements

- Python 3.11+ (recommended)
- Windows PowerShell / CMD (project includes `start_gui.bat`)
- Optional: PostgreSQL (if you use DB storage/import/export)

Install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start (GUI)

Use the launcher script:

```bat
start_gui.bat
```

This script:
1. Moves to the project folder
2. Activates `.venv`
3. Runs `python gui.py`

You can also run directly:

```powershell
.\.venv\Scripts\python.exe gui.py
```

## Quick Start (Web UI)

Use the web launcher:

```bat
start_web.bat
```

Or run directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run web_ui.py
```

## CLI Usage

### 1. Collect samples

```powershell
python main.py --samples 5 --interval-seconds 60 --timeout 20 --storage csv
```

Useful flags:
- `--output`
- `--samples`
- `--interval-seconds`
- `--timeout`
- `--storage` (`csv`, `postgres`, `both`)
- `--db-host --db-port --db-name --db-user --db-password`

### 2. Analyze and clean CSV

```powershell
python analyze.py --input internet_speed_data.csv --rolling-window 5 --output-clean internet_speed_data_clean.csv
```

### 3. Train forecasting model

```powershell
python train_forecast.py --input internet_speed_data.csv --target download_mbps --horizon 1 --lags 5 --test-size 0.2
```

Default outputs:
- `models/medidor_forecast.joblib`
- `models/medidor_forecast_metrics.json`

### 4. Predict next value

```powershell
python predict_next.py --input internet_speed_data.csv --model models/medidor_forecast.joblib
```

## PostgreSQL Helpers

Import CSV into PostgreSQL:

```powershell
python postgres/import_csv.py --csv-path internet_speed_data.csv --host localhost --port 5432 --database medidor --user postgres --password YOUR_PASSWORD
```

Export PostgreSQL data to CSV:

```powershell
python postgres/export_to_csv.py --output internet_speed_data.csv --host localhost --port 5432 --database medidor --user postgres --password YOUR_PASSWORD
```

Optional time window on export:
- `--from-ts 2026-02-11T00:00:00+00:00`
- `--to-ts 2026-02-12T00:00:00+00:00`

## Logging

- GUI/runtime errors are logged to: `app.log`
- If something fails silently in the UI, check `app.log` first.

## Project Structure

```text
gui.py                  # Desktop GUI
main.py                 # Sample collector CLI
analyze.py              # CSV analysis/cleaning
train_forecast.py       # Model training
predict_next.py         # Inference on latest data
postgres/import_csv.py  # CSV -> PostgreSQL
postgres/export_to_csv.py # PostgreSQL -> CSV
start_gui.bat           # Windows launcher (activate venv + run GUI)
start_web.bat           # Windows launcher (activate venv + run Streamlit UI)
web_ui.py               # Web UI (Streamlit)
requirements.txt
```

## Notes

- `requirements.txt` is fully pinned from the active environment.
- Keep `.venv` activated when running commands.
- The repository `.gitignore` excludes generated logs, models, and CSV artifacts.
