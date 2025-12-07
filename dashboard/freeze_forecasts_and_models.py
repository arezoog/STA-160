# dashboard/freeze_forecasts_and_models.py

import os
import joblib
import pandas as pd

from model import train_forecast_models, DATA_DIR

if __name__ == "__main__":
    print("[freeze] Training full forecast pipeline locally...")
    result = train_forecast_models()

    data = result["data"]
    comparison_df = result["comparison_df"]
    backtest_df = result["backtest_df"]
    future_df = result["future_df"]

    # Small bundle of models + metadata (this will be tiny)
    models_small = {
        "scaler": result["scaler"],
        "svr_model": result["svr_model"],
        "rf_model": result["rf_model"],
        "feature_cols": result["feature_cols"],
        "target_cols": result["target_cols"],
    }

    os.makedirs(DATA_DIR, exist_ok=True)

    # Save tables as CSVs
    data.to_csv(os.path.join(DATA_DIR, "forecast_data.csv"), index=False)
    comparison_df.to_csv(os.path.join(DATA_DIR, "forecast_comparison.csv"), index=False)
    backtest_df.to_csv(os.path.join(DATA_DIR, "forecast_backtest.csv"), index=False)
    future_df.to_csv(os.path.join(DATA_DIR, "forecast_future.csv"), index=False)

    # Save small model bundle
    models_path = os.path.join(DATA_DIR, "forecast_models_small.joblib")
    joblib.dump(models_small, models_path)

    print("[freeze] Saved:")
    print("  - forecast_data.csv")
    print("  - forecast_comparison.csv")
    print("  - forecast_backtest.csv")
    print("  - forecast_future.csv")
    print(f"  - forecast_models_small.joblib ({models_path})")
