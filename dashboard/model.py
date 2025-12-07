import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLEAN_MASTER_PATH = os.path.join(DATA_DIR, "EQR_master_clean_new.csv")
OUTPUT_MASTER_PATH = os.path.join(DATA_DIR, "EQR_master_output_new.csv")


@lru_cache(maxsize=1)
def load_master() -> pd.DataFrame:
    """
    Load the transaction-level EQR master file and build helper columns.

    On Render (or any constrained environment), you can cap the number of
    rows loaded by setting the MAX_EQR_ROWS environment variable. Locally,
    if MAX_EQR_ROWS is not set, the full file is loaded.

    To save memory and speed up I/O, we also restrict to the columns the
    app actually uses via `usecols`.
    """
    if not os.path.exists(CLEAN_MASTER_PATH):
        raise FileNotFoundError(
            f"EQR master file not found at {CLEAN_MASTER_PATH}. "
            "Place EQR_master_clean_new.csv in the dashboard/data folder."
        )

    # 🔹 Columns actually used by the app (modeling + risk/EDA basics)
    needed_cols = [
        # IDs / grouping
        "transaction_unique_id",
        "trade_date_year_mo",

        # dates (for datetime, year, month logic)
        "transaction_begin_date",
        "transaction_end_date",
        "begin_date",
        "delivery_month",

        # quantities / prices / charge
        "standardized_quantity",
        "transaction_quantity",
        "standardized_price",
        "total_transaction_charge",

        # OPTIONAL: add more if your EDA/risk pages need them, e.g.:
        # "product",
        # "class_name",
        # "time_zone",
        # "point_of_delivery_balancing_authority",
    ]

    # Allow row cap via env var (Render safety) + only load needed columns
    max_rows_env = os.environ.get("MAX_EQR_ROWS")
    read_kwargs = {
        "low_memory": False,
        "usecols": needed_cols,
    }

    if max_rows_env:
        try:
            read_kwargs["nrows"] = int(max_rows_env)
        except ValueError:
            # If the env var is not an int, ignore it and load all rows
            pass

    df = pd.read_csv(CLEAN_MASTER_PATH, **read_kwargs)

    # Choose a date column for the derived datetime field
    dt = None
    for col in ["begin_date", "transaction_begin_date", "transaction_end_date", "delivery_month"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            break

    if dt is None:
        raise ValueError(
            "Could not find a suitable date column in EQR_master_clean_new.csv. "
            "Expected one of: begin_date, transaction_begin_date, transaction_end_date, delivery_month."
        )

    df["datetime"] = dt
    df["year"] = df["datetime"].dt.year

    # Quantity proxy
    if "standardized_quantity" in df.columns and df["standardized_quantity"].notna().any():
        df["qty"] = df["standardized_quantity"]
    elif "transaction_quantity" in df.columns:
        df["qty"] = df["transaction_quantity"]
    else:
        df["qty"] = np.nan

    # Charge proxy
    if "total_transaction_charge" in df.columns:
        df["charge"] = df["total_transaction_charge"]
    else:
        df["charge"] = np.nan

    df["month"] = df["datetime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df



# -------------------------------------------------------------------------
# Forecasting pipeline
# -------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_monthly_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build a monthly modeling table from:
      - EQR_master_output_new.csv  (monthly summary)
      - EQR_master_clean_new.csv   (transaction-level details)
    """
    if not os.path.exists(OUTPUT_MASTER_PATH):
        raise FileNotFoundError(
            f"Monthly output file not found at {OUTPUT_MASTER_PATH}. "
            "Place EQR_master_output_new.csv in the dashboard/data folder."
        )

    monthly = pd.read_csv(OUTPUT_MASTER_PATH)

    if "trade_date_year_mo" not in monthly.columns:
        raise ValueError(
            "Expected column 'trade_date_year_mo' in EQR_master_output_new.csv "
            "for the monthly time index."
        )

    monthly = monthly.rename(columns={"trade_date_year_mo": "period"})

    keep_cols = ["period"]
    for col in ["weighted_avg_price", "total_transacted_quantity"]:
        if col in monthly.columns:
            keep_cols.append(col)

    monthly = monthly[keep_cols].copy()
    monthly = monthly.sort_values("period").reset_index(drop=True)

    monthly["weighted_avg_price"] = pd.to_numeric(
        monthly["weighted_avg_price"], errors="coerce"
    )
    monthly["weighted_avg_price"] = monthly["weighted_avg_price"].interpolate()

    if "total_transacted_quantity" in monthly.columns:
        monthly["total_transacted_quantity"] = pd.to_numeric(
            monthly["total_transacted_quantity"], errors="coerce"
        ).fillna(0.0)
    else:
        monthly["total_transacted_quantity"] = 0.0

    # Transaction-level features by trade month (optimized aggregation)
    master = load_master()
    if "trade_date_year_mo" not in master.columns:
        raise ValueError(
            "Expected 'trade_date_year_mo' in EQR_master_clean_new.csv for aggregation."
        )

    # Only keep columns needed for aggregation
    tx = master[
        [
            "trade_date_year_mo",
            "standardized_quantity",
            "transaction_quantity",
            "standardized_price",
            "transaction_unique_id",
        ]
    ].copy()

    tx["trade_date_year_mo"] = tx["trade_date_year_mo"].astype(str)

    agg_tx = (
        tx.groupby("trade_date_year_mo", observed=True)
        .agg(
            total_std_qty=("standardized_quantity", "sum"),
            total_tx_qty=("transaction_quantity", "sum"),
            avg_std_price=("standardized_price", "mean"),
            num_trades=("transaction_unique_id", "count"),
        )
        .reset_index()
        .rename(columns={"trade_date_year_mo": "period"})
    )

    data = monthly.merge(agg_tx, on="period", how="left")

    for col in ["total_std_qty", "total_tx_qty", "avg_std_price", "num_trades"]:
        data[col] = data[col].fillna(0.0)

    # Feature engineering: lags, calendar, targets
    for lag in range(1, 5):
        data[f"lag_{lag}"] = data["weighted_avg_price"].shift(lag)

    dt_period = pd.to_datetime(data["period"])
    data["month"] = dt_period.dt.month
    data["year"] = dt_period.dt.year

    data["target_1"] = data["weighted_avg_price"].shift(-1)
    data["target_2"] = data["weighted_avg_price"].shift(-2)

    data = data.dropna().reset_index(drop=True)

    feature_cols = (
        [f"lag_{i}" for i in range(1, 5)]
        + [
            "month",
            "year",
            "total_transacted_quantity",
            "total_std_qty",
            "total_tx_qty",
            "avg_std_price",
            "num_trades",
        ]
    )
    target_cols = ["target_1", "target_2"]

    return data, feature_cols, target_cols


@lru_cache(maxsize=1)
def train_forecast_models() -> Dict[str, object]:
    """
    Train the monthly forecasting models (tuned SVR + Random Forest),
    perform a 12-month holdout test, a rolling Ridge backtest, and build
    12-month ahead forecasts.

    This is the "from scratch" path used locally or when frozen CSVs are
    not available.
    """
    data, feature_cols, target_cols = build_monthly_dataset()

    # -----------------------------
    # Train/test split (last 12 months as test)
    # -----------------------------
    all_periods = sorted(data["period"].unique())
    if len(all_periods) < 24:
        raise ValueError(
            "Not enough monthly observations to perform a 12-month test split."
        )

    test_periods = all_periods[-12:]
    train_periods = all_periods[:-12]

    train_mask = data["period"].isin(train_periods)
    test_mask = data["period"].isin(test_periods)

    X_train = data.loc[train_mask, feature_cols]
    y_train = data.loc[train_mask, target_cols]
    X_test = data.loc[test_mask, feature_cols]
    y_test = data.loc[test_mask, target_cols]

    # -----------------------------
    # Tuned SVR (multi-output)
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    svr_model = MultiOutputRegressor(svr)
    svr_model.fit(X_train_scaled, y_train)
    y_pred_svr = svr_model.predict(X_test_scaled)

    # -----------------------------
    # Random Forest (multi-output)
    # -----------------------------
    rf = RandomForestRegressor(
        n_estimators=100,       # reduced from 500 for speed
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf_model = MultiOutputRegressor(rf)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # -----------------------------
    # Metrics comparison
    # -----------------------------
    def summarize_model(name: str, y_pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))
        r2 = r2_score(y_test, y_pred, multioutput="raw_values")
        return {
            "Model": name,
            "MAE_t+1": mae[0],
            "MAE_t+2": mae[1],
            "RMSE_t+1": rmse[0],
            "RMSE_t+2": rmse[1],
            "R2_t+1": r2[0],
            "R2_t+2": r2[1],
        }

    rows = [
        summarize_model("SVR_tuned", y_pred_svr),
        summarize_model("RandomForest", y_pred_rf),
    ]
    comparison_df = pd.DataFrame(rows)

    # -----------------------------
    # Rolling backtest (Ridge) for target_1
    # -----------------------------
    X_all = data[feature_cols].values
    y_all = data["target_1"].values
    period_all = data["period"].values

    window = 36
    step = 1  # you can change to 3 or 6 later to save more compute

    mae_bt: List[float] = []
    dates_bt: List[str] = []

    for i in range(window, len(data) - 1, step):
        train_idx = slice(0, i)
        test_idx = i

        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te = X_all[test_idx : test_idx + 1]
        y_te = y_all[test_idx : test_idx + 1]

        scaler_bt = StandardScaler()
        X_tr_s = scaler_bt.fit_transform(X_tr)
        X_te_s = scaler_bt.transform(X_te)

        model_bt = Ridge(alpha=1.0)
        model_bt.fit(X_tr_s, y_tr)
        y_hat = model_bt.predict(X_te_s)

        mae_bt.append(mean_absolute_error(y_te, y_hat))
        dates_bt.append(period_all[test_idx])

    backtest_df = pd.DataFrame(
        {
            "period": pd.to_datetime(dates_bt),
            "mae": mae_bt,
        }
    )
    avg_mae_bt = float(np.mean(mae_bt)) if mae_bt else np.nan

    # -----------------------------
    # 12-month ahead forecast using trained SVR/RF
    # -----------------------------
    horizon = 12
    last_period = pd.to_datetime(data["period"].iloc[-1])
    last_row = data.iloc[-1]

    lag_1 = last_row["lag_1"]
    lag_2 = last_row["lag_2"]
    lag_3 = last_row["lag_3"]
    lag_4 = last_row["lag_4"]

    future_rows: List[Dict] = []

    for h in range(1, horizon + 1):
        new_period = last_period + pd.DateOffset(months=h)
        future_rows.append(
            {
                "period": new_period,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_4": lag_4,
                "month": new_period.month,
                "year": new_period.year,
                "total_transacted_quantity": last_row["total_transacted_quantity"],
                "total_std_qty": last_row["total_std_qty"],
                "total_tx_qty": last_row["total_tx_qty"],
                "avg_std_price": last_row["avg_std_price"],
                "num_trades": last_row["num_trades"],
            }
        )

    future_df = pd.DataFrame(future_rows)

    # fill any missing feature values with last known values
    for col in feature_cols:
        if col in future_df.columns and future_df[col].isna().any():
            future_df[col] = future_df[col].fillna(data[col].iloc[-1])

    X_future = future_df[feature_cols]
    X_future_scaled = scaler.transform(X_future)

    svr_future = svr_model.predict(X_future_scaled)[:, 0]  # t+1 horizon
    rf_future = rf_model.predict(X_future)[:, 0]

    future_df["SVR_forecast"] = svr_future
    future_df["RF_forecast"] = rf_future

    # -----------------------------
    # Return everything needed by Dash
    # -----------------------------
    result: Dict[str, object] = {
        "data": data,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "train_periods": train_periods,
        "test_periods": test_periods,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": {
            "SVR_tuned": y_pred_svr,
            "RandomForest": y_pred_rf,
        },
        "comparison_df": comparison_df,
        "backtest_df": backtest_df,
        "avg_backtest_mae": avg_mae_bt,
        "future_df": future_df,
        # models + scaler for interactive scenarios
        "svr_model": svr_model,
        "rf_model": rf_model,
        "scaler": scaler,
    }

    return result


# -------------------------------------------------------------------------
# Dash-facing helper: prefer frozen CSVs if present
# -------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_forecast_dashboard_data() -> Dict[str, object]:
    """
    Convenience wrapper for Dash pages:

    1. If frozen forecast CSVs exist in dashboard/data, load them and
       refit lightweight SVR/RF models for interactive use.
    2. Otherwise, fall back to training from scratch via
       train_forecast_models().

    In both cases, the result dict has the same keys.
    """
    # Paths for frozen artifacts (CSV-based, GitHub-friendly)
    data_path = os.path.join(DATA_DIR, "forecast_data.csv")
    comp_path = os.path.join(DATA_DIR, "forecast_comparison.csv")
    bt_path = os.path.join(DATA_DIR, "forecast_backtest.csv")
    fut_path = os.path.join(DATA_DIR, "forecast_future.csv")

    have_all_csvs = all(
        os.path.exists(p) for p in [data_path, comp_path, bt_path, fut_path]
    )

    if have_all_csvs:
        # ---------- Fast path: use precomputed CSVs ----------
        print("[model.py] Loading frozen forecast CSVs and refitting lightweight models...")

        data = pd.read_csv(data_path)
        comparison_df = pd.read_csv(comp_path)
        backtest_df = pd.read_csv(bt_path)
        future_df = pd.read_csv(fut_path)

        # Ensure period is datetime where appropriate
        if "period" in backtest_df.columns:
            backtest_df["period"] = pd.to_datetime(backtest_df["period"])
        if "period" in future_df.columns:
            future_df["period"] = pd.to_datetime(future_df["period"])

        # Infer targets and features from the data
        target_cols = [c for c in ["target_1", "target_2"] if c in data.columns]
        if not target_cols:
            raise ValueError(
                "forecast_data.csv must contain at least 'target_1' and/or 'target_2'."
            )

        # Exclude non-feature columns
        feature_cols = [
            c
            for c in data.columns
            if c not in (["period"] + target_cols)
        ]

        # Recreate train/test split (last 12 months as test)
        all_periods = sorted(data["period"].unique())
        if len(all_periods) < 24:
            raise ValueError(
                "Not enough monthly observations in forecast_data.csv for a 12-month test split."
            )

        test_periods = all_periods[-12:]
        train_periods = all_periods[:-12]

        train_mask = data["period"].isin(train_periods)
        test_mask = data["period"].isin(test_periods)

        X_train = data.loc[train_mask, feature_cols]
        y_train = data.loc[train_mask, target_cols]
        X_test = data.loc[test_mask, feature_cols]
        y_test = data.loc[test_mask, target_cols]

        # Refit tuned SVR + RF for interactive use
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svr = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
        svr_model = MultiOutputRegressor(svr)
        svr_model.fit(X_train_scaled, y_train)
        y_pred_svr = svr_model.predict(X_test_scaled)

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        rf_model = MultiOutputRegressor(rf)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        # Backtest average from CSV
        avg_backtest_mae = (
            float(backtest_df["mae"].mean())
            if "mae" in backtest_df.columns and not backtest_df.empty
            else np.nan
        )

        result: Dict[str, object] = {
            "data": data,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "train_periods": train_periods,
            "test_periods": test_periods,
            "train_mask": train_mask,
            "test_mask": test_mask,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_test": {
                "SVR_tuned": y_pred_svr,
                "RandomForest": y_pred_rf,
            },
            "comparison_df": comparison_df,
            "backtest_df": backtest_df,
            "avg_backtest_mae": avg_backtest_mae,
            "future_df": future_df,
            # models + scaler for interactive scenarios
            "svr_model": svr_model,
            "rf_model": rf_model,
            "scaler": scaler,
        }
        return result

    # ---------- Slow path: train from scratch ----------
    print("[model.py] Frozen forecast CSVs not found. Training models from scratch...")
    return train_forecast_models()


    # Fallback: train everything (local dev)
    print("[model.py] Frozen CSVs not found. Training models now...")
    return train_forecast_models()
