import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import AVAILABLE_YEARS, compute_metrics, predict_probability

dash.register_page(__name__, path="/dashboard", name="Dashboard")


layout = html.Div(
    style={"paddingTop": "8px"},
    children=[
        dcc.Store(id="metrics-store"),

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
                        html.Label(
                            "Test size (hold-out fraction)", style={"fontSize": "0.8rem", "color": "#9ca3af"}
                        ),
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
                            max=95,
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

                # Right scenario panel
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
    ],
)


# ====== Callbacks ======

@callback(
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
    empty_fig = go.Figure()

    if not years:
        return dash.no_update, "Select at least one year.", empty_fig, empty_fig, empty_fig

    try:
        result = compute_metrics(years, float(event_p), float(test_size), 42)
    except Exception as e:
        return dash.no_update, f"Training error: {e}", empty_fig, empty_fig, empty_fig

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

    # Time-series plot
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

    # Histogram plot
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

    # ROC curve
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


@callback(
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


