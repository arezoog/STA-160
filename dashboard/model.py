import os
from functools import lru_cache
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# -----------------------
# Paths / constants
# -----------------------

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "EQR_master_clean_new.csv")

AVAILABLE_YEARS = list(range(2019, 2025 + 1))
MAX_ROWS = 30000  # cap rows per training run for speed


# -----------------------
# Data loading
# -----------------------

@lru_cache(maxsize=1)
def load_master() -> pd.DataFrame:
    """
    Load the EQR master CSV and build helper columns.
    Uses lru_cache so we only hit disk once per process.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}. Place EQR_master_clean_new.csv in the /data folder."
        )

    usecols = [
        "begin_date",
        "delivery_month",
        "standardized_price",
        "standardized_quantity",
        "transaction_quantity",
        "total_transaction_charge",
    ]

    df = pd.read_csv(CSV_PATH, usecols=usecols, low_memory=False)

    # Choose a primary timestamp: prefer begin_date, else delivery_month
    dt = pd.to_datetime(df.get("begin_date"), errors="coerce", utc=True)
    if dt.notna().sum() == 0 and "delivery_month" in df.columns:
        dt = pd.to_datetime(df.get("delivery_month"), errors="coerce", utc=True)
    if dt.notna().sum() == 0:
        raise ValueError("No usable datetime column found (begin_date or delivery_month).")

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df["year"] = df["datetime"].dt.year

    # Numeric features
    if "standardized_quantity" in df.columns and df["standardized_quantity"].notna().any():
        df["qty"] = df["standardized_quantity"]
    else:
        df["qty"] = df.get("transaction_quantity")

    df["charge"] = df.get("total_transaction_charge")

    # Month cyclic encoding
    df["month"] = df["datetime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def subset_years(years: List[int]) -> pd.DataFrame:
    """Filter master DF to selected years and optionally downsample."""
    df = load_master()
    if years:
        df = df[df["year"].isin([int(y) for y in years])]

    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0)

    return df


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["qty", "charge", "month_sin", "month_cos"]].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def train_test_with_percentile_label(
    df: pd.DataFrame,
    event_percentile: float,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Create a binary 'event' label from TRAIN-SPLIT percentile of standardized_price
    (avoids leakage). High price tail = event.
    """
    if "standardized_price" not in df.columns:
        raise ValueError("Column 'standardized_price' is required for the proxy event label.")

    df_lbl = df.dropna(subset=["standardized_price"]).copy()
    if len(df_lbl) < 50:
        raise ValueError("Not enough rows with 'standardized_price' to define events.")

    X = build_X(df_lbl)
    idx = df_lbl.index.values
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, idx, test_size=test_size, shuffle=True, random_state=random_state
    )

    # Percentile threshold from TRAIN only
    pthr = np.nanpercentile(df_lbl.loc[idx_train, "standardized_price"], event_percentile)
    y_all = (df_lbl["standardized_price"] >= pthr).astype(int)
    y_train = y_all.loc[idx_train]
    y_test = y_all.loc[idx_test]

    return X_train, X_test, y_train, y_test, idx_train, idx_test


def _norm_years(years: List[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(y) for y in years))


# -----------------------
# Train-and-cache helper
# -----------------------

@lru_cache(maxsize=32)
def _train_and_cache_lru(
    years_key: Tuple[int, ...],
    event_percentile: float,
    test_size: float,
    random_state: int,
) -> Dict:
    """
    Internal cached trainer. Keyed by a tuple of years plus hyperparameters.
    Returns both the trained model and all metrics we need.
    """
    years = list(years_key)
    df = subset_years(years)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_with_percentile_label(
        df, event_percentile, test_size, random_state
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds_test)
    try:
        auc = roc_auc_score(y_test, proba_test)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_test, proba_test)
    fpr, tpr, _ = roc_curve(y_test, proba_test)

    ts = pd.DataFrame(
        {"datetime": df.loc[idx_test, "datetime"].values, "p_hat": proba_test, "actual": y_test.values}
    ).sort_values("datetime")

    metrics = {
        "has_labels": True,
        "accuracy": float(acc),
        "auc": float(auc),
        "brier": float(brier),
        "roc_points": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "ts": {
            "datetime": ts["datetime"].astype(str).tolist(),
            "p_hat": ts["p_hat"].tolist(),
            "actual": ts["actual"].tolist(),
        },
        "counts_by_year": df.groupby("year").size().to_dict(),
    }

    return {"model": pipe, "metrics": metrics}


def _train_and_cache(
    years: List[int],
    event_percentile: float,
    test_size: float,
    random_state: int,
) -> Dict:
    years_key = _norm_years(years)
    return _train_and_cache_lru(years_key, float(event_percentile), float(test_size), int(random_state))


# -----------------------
# Public functions used by Dash pages
# -----------------------

def compute_metrics(
    years: List[int],
    event_percentile: float,
    test_size: float,
    random_state: int,
) -> Dict:
    """
    Train (or reuse cached model) and return metrics + time series + ROC info.
    """
    res = _train_and_cache(years, event_percentile, test_size, random_state)
    return res["metrics"]


def predict_probability(
    years: List[int],
    features: Dict[str, float],
    event_percentile: float,
    test_size: float,
    random_state: int,
) -> float:
    """
    Use the cached trained model for this configuration to score a single scenario.
    """
    res = _train_and_cache(years, event_percentile, test_size, random_state)
    model = res["model"]

    m = float(features.get("month", 7))
    x = pd.DataFrame(
        [
            {
                "qty": float(features.get("qty", 100.0)),
                "charge": float(features.get("charge", 100000.0)),
                "month_sin": np.sin(2 * np.pi * m / 12.0),
                "month_cos": np.cos(2 * np.pi * m / 12.0),
            }
        ]
    )
    p = float(model.predict_proba(x)[0, 1])
    return p

