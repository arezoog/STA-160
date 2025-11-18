import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_caching import Cache

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import train_test_split

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Flask backend + cache
# =========================
server = Flask(__name__)
cache = Cache(server, config={"CACHE_TYPE": "SimpleCache"})

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "EQR_master_clean_new.csv")

AVAILABLE_YEARS = list(range(2019, 2025 + 1))
MAX_ROWS = 30000  # cap rows per training run for speed


# ---------- Data helpers ----------
@cache.memoize(timeout=300)
def load_master() -> pd.DataFrame:
    """
    Load the main EQR CSV with only the columns we actually need and
    construct helper columns used by the model and plots.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}. Place EQR_master_clean_new.csv in the /data folder."
        )

    # Only load needed columns (these exist in your CSV)
    usecols = [
        "begin_date",
        "delivery_month",
        "standardized_price",
        "standardized_quantity",
        "transaction_quantity",
        "total_transaction_charge",
    ]

    df = pd.read_csv(CSV_PATH, usecols=usecols, low_memory=False)

    # Choose a primary timestamp for slicing: prefer begin_date, else delivery_month
    dt = pd.to_datetime(df.get("begin_date"), errors="coerce", utc=True)
    if dt.notna().sum() == 0 and "delivery_month" in df.columns:
        dt = pd.to_datetime(df.get("delivery_month"), errors="coerce", utc=True)
    if dt.notna().sum() == 0:
        raise ValueError("No usable datetime column found (begin_date or delivery_month).")

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df["year"] = df["datetime"].dt.year

    # Numeric features (robust defaults)
    if "standardized_quantity" in df.columns and df["standardized_quantity"].notna().any():
        df["qty"] = df["standardized_quantity"]
    else:
        df["qty"] = df.get("transaction_quantity")

    df["charge"] = df.get("total_transaction_charge")

    # Month cyclic encoding for seasonality
    df["month"] = df["datetime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def subset_years(years: List[int]) -> pd.DataFrame:
    """
    Filter the master DF to the selected years and (optionally) downsample
    for speed.
    """
    df = load_master()
    if years:
        df = df[df["year"].isin([int(y) for y in years])]

    # Downsample for speed if needed
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
    (avoids leakage). High price tail = event (proxy until you have a true adequacy label).
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


def model_cache_key(years: List[int], event_percentile: float, test_size: float, random_state: int) -> str:
    yrs = tuple(sorted(int(y) for y in years))
    return f"model:{yrs}:p{int(event_percentile)}:ts{test_size}:rs{random_state}"


def get_or_train_model(years: List[int], event_percentile: float, test_size: float, random_state: int):
    key = model_cache_key(years, event_percentile, test_size, random_state)
    pipe = cache.get(key)
    if pipe is not None:
        return pipe

    df = subset_years(years)
    X_train, X_test, y_train, y_test, *_ = train_test_with_percentile_label(
        df, event_percentile, test_size, random_state
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])
    pipe.fit(X_train, y_train)
    cache.set(key, pipe, timeout=300)
    return pipe


def compute_metrics(years: List[int], event_percentile: float, test_size: float, random_state: int) -> Dict:
    df = subset_years(years)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_with_percentile_label(
        df, event_percentile, test_size, random_state
    )

    pipe = get_or_train_model(years, event_percentile, test_size, random_state)
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

    return {
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


def predict_probability(
    years: List[int],
    features: Dict[str, float],
    event_percentile: float,
    test_size: float,
    random_state: int,
) -> float:
    pipe = get_or_train_model(years, event_percentile, test_size, random_state)
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
    p = float(pipe.predict_proba(x)[0, 1])
    return p


# ---------- Optional JSON API ----------
@server.route("/api/metrics", methods=["POST"])
def api_metrics():
    payload = request.get_json(force=True)
    years = payload.get("years", [])
    test_size = float(payload.get("test_size", 0.2))
    random_state = int(payload.get("random_state", 42))
    event_percentile = float(payload.get("event_percentile", 95))
    if not years:
        return jsonify({"error": "No years provided"}), 400
    try:
        result = compute_metrics(years, event_percentile, test_size, random_state)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400


@server.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    years = payload.get("years", [])
    feats = payload.get("features", {})
    test_size = float(payload.get("test_size", 0.2))
    random_state = int(payload.get("random_state", 42))
    event_percentile = float(payload.get("event_percentile", 95))
    if not years:
        return jsonify({"error": "No years provided"}), 400
    try:
        p = predict_probability(years, feats, event_percentile, test_size, random_state)
        return jsonify({"p": p})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400


# =========================
# Dash front-end (UI)
# =========================
app = dash.Dash(__name__, server=server, title="RA Predictive Dashboard", suppress_callback_exceptions=True)

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#020617",  # slate-950
        "padding": "24px",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "color": "#e5e7eb",
    },
    children=[
        dcc.Store(id="metrics-store"),

        html.Div(
            style={"maxWidth": "1300px", "margin": "0 auto"},
            children=[
                # Header
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "baseline",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H2(
                                    "Resource Adequacy – Predictive Dashboard",
                                    style={"margin": 0, "fontWeight": 600, "letterSpacing": "0.02em"},
                                ),
                                html.P(
                                    "Explore how different data windows and event definitions affect a "
                                    "probabilistic model of high-price (stress) events.",
                                    style={"margin": "6px 0 0 0", "color": "#9ca3af", "fontSize": "0.9rem"},
                                ),
                            ]
                        ),
                        html.Div(
                            "EQR 2019–2025",
                            style={
                                "padding": "4px 10px",
                                "borderRadius": "999px",
                                "border": "1px solid #374151",
                                "fontSize": "0.75rem",
                                "color": "#9ca3af",
                            },
                        ),
                    ],
                ),
                # Main content row
                html.Div(
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                    children=[
                        # Left controls card
                        html.Div(
                            style={
                                "flex": "1 1 280px",
                                "backgroundColor": "#020617",
                                "borderRadius": "16px",
                                "padding": "16px 16px 18px 16px",
                                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                                "border": "1px solid #1f2937",
                            },
                            children=[
                                html.Div(
                                    "Model configuration",
                                    style={
                                        "fontSize": "0.9rem",
                                        "fontWeight": 600,
                                        "marginBottom": "8px",
                                        "color": "#e5e7eb",
                                    },
                                ),
                                html.Label("Datasets (years)", style={"fontSize": "0.8rem", "color": "#9ca3af"}),
                                dcc.Dropdown(
                                    id="year-picker",
                                    options=[{"label": str(y), "value": y} for y in AVAILABLE_YEARS],
                                    value=[2023, 2024, 2025],
                                    multi=True,
                                    placeholder="Select years",
                                    style={"marginTop": "4px"},
                                ),
                                html.Br(),
                                html.Label("Test size (hold-out fraction)", style={"fontSize": "0.8rem", "color": "#9ca3af"}),
                                dcc.Slider(
                                    id="test-size",
                                    min=0.1,
                                    max=0.5,
                                    step=0.05,
                                    value=0.2,
                                    tooltip={"placement": "bottom"},
                                ),
                                html.Div(
                                    "Smaller test sets give more training data; larger test sets give more stable metrics.",
                                    style={"fontSize": "0.7rem", "color": "#6b7280", "marginTop": "4px"},
                                ),
                                html.Br(),
                                html.Label(
                                    "Event definition (price percentile)",
                                    style={"fontSize": "0.8rem", "color": "#9ca3af"},
                                ),
                                dcc.Slider(
                                    id="event-p",
                                    min=50,
                                    max=99,
                                    step=5,
                                    value=95,
                                    marks={p: str(p) for p in range(50, 100, 5)},
                                    tooltip={"placement": "bottom"},
                                ),
                                html.Div(
                                    "Higher percentiles = rarer, more extreme price spikes.",
                                    style={"fontSize": "0.7rem", "color": "#6b7280", "marginTop": "4px"},
                                ),
                                html.Button(
                                    "Train / Refresh model",
                                    id="train-btn",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "12px",
                                        "width": "100%",
                                        "padding": "8px 0",
                                        "borderRadius": "999px",
                                        "border": "none",
                                        "background": "linear-gradient(90deg,#22c55e,#16a34a)",
                                        "color": "#020617",
                                        "fontWeight": 600,
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Div(
                                    id="metrics-box",
                                    style={
                                        "marginTop": "12px",
                                        "fontWeight": 500,
                                        "fontSize": "0.78rem",
                                        "lineHeight": 1.4,
                                        "color": "#d1d5db",
                                    },
                                ),
                            ],
                        ),
                        # Middle charts card
                        html.Div(
                            style={
                                "flex": "2 1 520px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "12px",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "backgroundColor": "#020617",
                                        "borderRadius": "16px",
                                        "padding": "12px",
                                        "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                                        "border": "1px solid #1f2937",
                                    },
                                    children=[
                                        html.Div(
                                            "Out-of-sample predicted probabilities",
                                            style={
                                                "fontSize": "0.82rem",
                                                "fontWeight": 500,
                                                "marginBottom": "4px",
                                                "color": "#e5e7eb",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="ts-graph",
                                            figure=go.Figure(),
                                            style={"height": "320px"},
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "backgroundColor": "#020617",
                                        "borderRadius": "16px",
                                        "padding": "12px",
                                        "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                                        "border": "1px solid #1f2937",
                                    },
                                    children=[
                                        html.Div(
                                            "Discrimination & calibration",
                                            style={
                                                "fontSize": "0.82rem",
                                                "fontWeight": 500,
                                                "marginBottom": "4px",
                                                "color": "#e5e7eb",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="roc-graph",
                                            figure=go.Figure(),
                                            style={"height": "320px"},
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Right scenario panel card
                        html.Div(
                            style={
                                "flex": "1 1 280px",
                                "backgroundColor": "#020617",
                                "borderRadius": "16px",
                                "padding": "16px",
                                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                                "border": "1px solid #1f2937",
                            },
                            children=[
                                html.Div(
                                    "Scenario explorer",
                                    style={
                                        "fontSize": "0.9rem",
                                        "fontWeight": 600,
                                        "marginBottom": "8px",
                                        "color": "#e5e7eb",
                                    },
                                ),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "110px 1fr",
                                        "rowGap": "8px",
                                        "columnGap": "8px",
                                        "fontSize": "0.8rem",
                                    },
                                    children=[
                                        html.Div("Qty"),
                                        dcc.Input(
                                            id="in-qty",
                                            type="number",
                                            value=100.0,
                                            step=1,
                                            style={"width": "100%", "backgroundColor": "#020617", "color": "#e5e7eb"},
                                        ),
                                        html.Div("Charge"),
                                        dcc.Input(
                                            id="in-charge",
                                            type="number",
                                            value=100000.0,
                                            step=1000,
                                            style={"width": "100%", "backgroundColor": "#020617", "color": "#e5e7eb"},
                                        ),
                                        html.Div("Month (1–12)"),
                                        dcc.Input(
                                            id="in-month",
                                            type="number",
                                            value=7,
                                            step=1,
                                            min=1,
                                            max=12,
                                            style={"width": "100%", "backgroundColor": "#020617", "color": "#e5e7eb"},
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Compute probability",
                                    id="predict-btn",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "12px",
                                        "width": "100%",
                                        "padding": "8px 0",
                                        "borderRadius": "999px",
                                        "border": "1px solid #374151",
                                        "background": "#0f172a",
                                        "color": "#e5e7eb",
                                        "fontWeight": 500,
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Div(
                                    id="predict-out",
                                    style={
                                        "marginTop": "10px",
                                        "fontSize": "0.95rem",
                                        "fontWeight": 600,
                                        "color": "#a5b4fc",
                                    },
                                ),
                                html.Hr(style={"borderColor": "#1f2937", "margin": "14px 0"}),
                                html.Div(
                                    "Distribution of model probabilities (test split)",
                                    style={
                                        "fontSize": "0.8rem",
                                        "fontWeight": 500,
                                        "marginBottom": "4px",
                                        "color": "#e5e7eb",
                                    },
                                ),
                                dcc.Graph(
                                    id="hist-graph",
                                    figure=go.Figure(),
                                    style={"height": "260px"},
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    "Prototype dashboard – not for operational decision-making.",
                    style={
                        "marginTop": "12px",
                        "fontSize": "0.7rem",
                        "color": "#6b7280",
                        "textAlign": "right",
                    },
                ),
            ],
        )
    ],
)


# -------- Callbacks --------
@app.callback(
    Output("metrics-store", "data"),
    Output("metrics-box", "children"),
    Output("ts-graph", "figure"),
    Output("roc-graph", "figure"),
    Output("hist-graph", "figure"),
    Input("train-btn", "n_clicks"),
    State("year-picker", "value"),
    State("test-size", "value"),
    State("event-p", "value"),
    prevent_initial_call=False,
)
def train_and_plot(n_clicks, years, test_size, event_p):
    years = years or []
    if not years:
        return dash.no_update, "Select at least one year.", go.Figure(), go.Figure(), go.Figure()
    try:
        result = compute_metrics(years, float(event_p), float(test_size), 42)
    except Exception as e:
        return dash.no_update, f"Training error: {e}", go.Figure(), go.Figure(), go.Figure()

    counts = result.get("counts_by_year", {})
    counts_txt = " | ".join(f"{y}: {n}" for y, n in sorted(counts.items()))

    acc = result.get("accuracy", float("nan"))
    auc = result.get("auc", float("nan"))
    brier = result.get("brier", float("nan"))
    metrics_text = (
        f"Accuracy: {acc:.3f} | AUC: {auc:.3f} | Brier: {brier:.3f} | "
        f"Event percentile: {event_p:.0f} | Points by year: {counts_txt}"
    )

    ts = result["ts"]
    ts_df = pd.DataFrame(
        {"datetime": pd.to_datetime(ts["datetime"]), "p_hat": ts["p_hat"], "actual": ts["actual"]}
    )

    # Time series
    fig_ts = px.line(ts_df, x="datetime", y="p_hat", title=None)
    fig_ts.add_scatter(x=ts_df["datetime"], y=ts_df["actual"], mode="lines", name="Actual (0/1)", opacity=0.6)
    fig_ts.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Datetime",
        yaxis_title="Predicted probability",
    )

    # Histogram
    fig_hist = px.histogram(ts_df, x="p_hat", nbins=30, title=None)
    fig_hist.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb", size=11),
        xaxis_title="Predicted probability",
        yaxis_title="Count",
    )

    # ROC
    roc = result["roc_points"]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#4b5563"))
    fig_roc.update_layout(
        template="plotly_dark",
        title=None,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb", size=11),
        showlegend=False,
    )

    store = {"years": years, "test_size": float(test_size), "event_p": float(event_p)}
    return store, metrics_text, fig_ts, fig_roc, fig_hist


@app.callback(
    Output("predict-out", "children"),
    Input("predict-btn", "n_clicks"),
    State("metrics-store", "data"),
    State("in-qty", "value"),
    State("in-charge", "value"),
    State("in-month", "value"),
    prevent_initial_call=True,
)
def do_predict(n, store, qty, charge, month):
    if not store:
        return "Train a model first."
    years = store.get("years", [])
    test_size = float(store.get("test_size", 0.2))
    event_p = float(store.get("event_p", 95))
    feats = {"qty": qty, "charge": charge, "month": month}
    try:
        p = predict_probability(years, feats, event_p, test_size, 42)
        return f"Predicted high-price (stress) probability: {p:.2%}"
    except Exception as e:
        return f"Prediction error: {e}"


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(debug=False)
