# Medidor Training Checklist

## Preconditions

- `internet_speed_data.csv` exists and has recent rows.
- Python environment has project dependencies installed (`python -c "import pandas, sklearn, joblib"`).
- `models/` directory is writable.

## Pipeline Order

1. Clean/analyze:
`python analyze.py --input internet_speed_data.csv --rolling-window 5 --output-clean internet_speed_data_clean.csv`
2. Train:
`python train_forecast.py --input internet_speed_data_clean.csv --target download_mbps --horizon 1 --lags 5 --test-size 0.2`
3. Predict:
`python predict_next.py --input internet_speed_data_clean.csv --model models/medidor_forecast.joblib`

## Expected Artifacts

- `internet_speed_data_clean.csv`
- `models/medidor_forecast.joblib`
- `models/medidor_forecast_metrics.json`
- `models/medidor_forecast_v*_t-*_h-*_l-*.joblib`
- `models/medidor_forecast_metrics_v*_t-*_h-*_l-*.json`

## Common Failure Signals

- Missing target column: check CSV headers and target arg.
- Empty/short dataset: reduce lags or collect more samples.
- Model file missing: training failed or wrong model path in predict step.
