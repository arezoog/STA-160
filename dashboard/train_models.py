import os
import joblib

from model import train_forecast_models, DATA_DIR

if __name__ == "__main__":
    print("Training forecast models...")
    result = train_forecast_models()

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    out_path = os.path.join(DATA_DIR, "forecast_models.joblib")
    joblib.dump(result, out_path)

    print(f"Saved trained models + data to {out_path}")
