---
name: model-train-check
description: Run and verify the medidor forecasting workflow end-to-end (analyze, train, predict, and sanity checks). Use when users ask to train the model, validate forecast outputs, debug training failures, or confirm data/model artifacts after changes in main.py, analyze.py, train_forecast.py, predict_next.py, or CSV/model files.
---

# Model Train Check

## Overview

Execute the project forecasting pipeline with repeatable checks.
Detect failures early in data preparation, training, and inference outputs.

## Workflow

1. Verify inputs:
- Confirm required files exist: `internet_speed_data.csv`, `models/` directory.
- Confirm Python environment is available and dependencies are installed (`python -c "import pandas, sklearn, joblib"`).

2. Run data analysis/cleaning:
- Execute `analyze.py` using a small rolling window and explicit output path.
- Stop and report if cleaning fails.

3. Run training:
- Execute `train_forecast.py` with explicit `--input`, `--target`, `--horizon`, `--lags`, and `--test-size`.
- Validate that base artifacts are generated/updated:
  - `models/medidor_forecast.joblib`
  - `models/medidor_forecast_metrics.json`
- Validate that versioned artifacts are generated/updated (timestamped suffix):
  - `models/medidor_forecast_v*_t-*_h-*_l-*.joblib`
  - `models/medidor_forecast_metrics_v*_t-*_h-*_l-*.json`

4. Run prediction:
- Execute `predict_next.py` with explicit model path.
- Report the predicted value and any confidence/metadata printed by the script.

5. Run syntax sanity checks when code changed:
- Execute `python -m py_compile main.py gui.py web_ui.py analyze.py train_forecast.py predict_next.py`.

6. Summarize results:
- State which step failed or passed.
- Include artifact paths and key metric highlights.

## Default Commands

Use these commands as defaults and adjust only if user requests different parameters:

```bash
python analyze.py --input internet_speed_data.csv --rolling-window 5 --output-clean internet_speed_data_clean.csv
python -c "import pandas, sklearn, joblib"
python train_forecast.py --input internet_speed_data_clean.csv --target download_mbps --horizon 1 --lags 5 --test-size 0.2
python predict_next.py --input internet_speed_data_clean.csv --model models/medidor_forecast.joblib
python -m py_compile main.py gui.py web_ui.py analyze.py train_forecast.py predict_next.py
```

## Troubleshooting Rules

- Read `app.log` when the UI pipeline appears to fail silently.
- If training fails due to missing/invalid columns, inspect the CSV headers before changing code.
- Preserve CLI compatibility; if arguments change, update `README.md`.
- Avoid committing generated artifacts unless user explicitly asks.
- Prefer minimal fixes and rerun only the failing step first, then rerun end-to-end.

## References

- `references/medidor-training-checklist.md`: quick checklist and expected artifacts.
