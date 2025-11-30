import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import AVAILABLE_YEARS, compute_metrics, predict_probability

dash.register_page(__name__, path="/dashboard", name="Dashboard")


def _empty_dark_fig(message=None):
    """Small helper to avoid the white-box effect before data loads."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    if message:
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#9ca3af"),
        )
    return fig


# --- Layout ---------------------------------------------------------------

sorted_years = sorted(AVAILABLE_YEARS) if AVAILABLE_YEARS else []
default_years = sorted_years[-3:] if len(sorted_years) >= 3 else sorted_years

layout = html.Div(
    style={"maxWidth": "1200px", "margin": "16px auto 32px auto", "padding": "0 16px"},
    children=[
        dcc.Store(id="metrics-store"),
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "16px",
                "alignItems": "stretch",
            },
            children=[
                # LEFT: configuration + metrics
                html.Div(
                    className="glass-card",
                    style={
                        "flex": "1 1 260px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
                    children=[
                        html.H2("Model Dashboard", style={"margin": "0 0 4px 0", "fontSize": "1.1rem"}),
                        html.P(
                            "Select training years and thresholds, then fit a simple stress-event model.",
                            className="text-muted",
                            style={"fontSize": "0.85rem", "marginBottom": "4px"},
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Training years",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.14em",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="year-dropdown",
                                    options=[{"label": str(y), "value": y} for y in sorted_years],
                                    value=default_years,
                                    multi=True,
                                    placeholder="Select years…",
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Test size",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.14em",
                                    },
                                ),
                                html.Div(
                                    children=dcc.Slider(
                                        id="test-size-slider",
                                        min=0.1,
                                        max=0.5,
                                        step=0.05,
                                        value=0.3,
                                        marks={0.1: "10%", 0.3: "30%", 0.5: "50%"},
                                        tooltip={"placement": "bottom"},
                                    ),
                                    style={"marginTop": "4px"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Stress event threshold (percentile)",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.14em",
                                    },
                                ),
                                html.Div(
                                    children=dcc.Slider(
                                        id="event-p-slider",
                                        min=80,
                                        max=99,
                                        step=1,
                                        value=95,
                                        marks={80: "80", 90: "90", 95: "95", 99: "99"},
                                        tooltip={"placement": "bottom"},
                                    ),
                                    style={"marginTop": "4px"},
                                ),
                            ]
                        ),
                        html.Button(
                            "⚡ Train / Refresh model",
                            id="train-btn",
                            n_clicks=0,
                            className="primary-button",
                            style={"marginTop": "4px"},
                        ),
                        html.Div(
                            id="model-metrics",
                            className="text-muted",
                            style={"fontSize": "0.85rem", "marginTop": "8px", "lineHeight": "1.6"},
                        ),
                    ],
                ),
                # MIDDLE: graphs
                html.Div(
                    className="glass-card",
                    style={
                        "flex": "2 1 480px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
                    children=[
                        html.H3("Model performance", style={"margin": "0 0 4px 0", "fontSize": "1.0rem"}),
                        dcc.Tabs(
                            id="viz-tabs",
                            value="ts-tab",
                            parent_className="custom-tabs",
                            children=[
                                dcc.Tab(
                                    label="Predictions vs actual",
                                    value="ts-tab",
                                    children=[
                                        dcc.Graph(
                                            id="ts-graph",
                                            className="dash-graph",
                                            figure=_empty_dark_fig("Train the model to see time-series output."),
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="ROC curve",
                                    value="roc-tab",
                                    children=[
                                        dcc.Graph(
                                            id="roc-graph",
                                            className="dash-graph",
                                            figure=_empty_dark_fig("Train the model to see the ROC curve."),
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Probability distribution",
                                    value="hist-tab",
                                    children=[
                                        dcc.Graph(
                                            id="hist-graph",
                                            className="dash-graph",
                                            figure=_empty_dark_fig("Train the model to see probabilities."),
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # RIGHT: live prediction
                html.Div(
                    className="glass-card",
                    style={
                        "flex": "1 1 260px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "10px",
                    },
                    children=[
                        html.H3("Live prediction", style={"margin": "0", "fontSize": "1.0rem"}),
                        html.P(
                            "Use the trained model to estimate stress probability for a single transaction.",
                            className="text-muted",
                            style={"fontSize": "0.8rem"},
                        ),
                        html.Label(
                            "Quantity (MW)",
                            style={
                                "fontSize": "0.8rem",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.14em",
                            },
                        ),
                        dcc.Input(
                            id="in-qty",
                            type="number",
                            debounce=True,
                            placeholder="e.g. 50",
                        ),
                        html.Label(
                            "Total charge ($)",
                            style={
                                "fontSize": "0.8rem",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.14em",
                                "marginTop": "4px",
                            },
                        ),
                        dcc.Input(
                            id="in-charge",
                            type="number",
                            debounce=True,
                            placeholder="e.g. 120000",
                        ),
                        html.Label(
                            "Delivery month (1–12)",
                            style={
                                "fontSize": "0.8rem",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.14em",
                                "marginTop": "4px",
                            },
                        ),
                        dcc.Input(
                            id="in-month",
                            type="number",
                            min=1,
                            max=12,
                            step=1,
                            debounce=True,
                            placeholder="1–12",
                        ),
                        html.Button(
                            "Compute probability",
                            id="predict-btn",
                            n_clicks=0,
                            className="primary-button",
                            style={"marginTop": "6px"},
                        ),
                        html.Div(
                            id="predict-out",
                            style={
                                "marginTop": "10px",
                                "fontSize": "0.9rem",
                            },
                            children="Ready… train the model and enter inputs to see a probability.",
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# --- Callbacks ------------------------------------------------------------


@callback(
    Output("metrics-store", "data"),
    Output("model-metrics", "children"),
    Output("ts-graph", "figure"),
    Output("roc-graph", "figure"),
    Output("hist-graph", "figure"),
    Input("train-btn", "n_clicks"),
    State("year-dropdown", "value"),
    State("test-size-slider", "value"),
    State("event-p-slider", "value"),
)
def update_model(n_clicks, years, test_size, event_p):
    if not sorted_years:
        msg = "No years available from model.AVAILABLE_YEARS."
        empty = _empty_dark_fig(msg)
        return None, [html.Div(msg)], empty, empty, empty

    if not years:
        years = default_years or sorted_years

    # event_p slider is in 80–99; convert to percentile in [0,1]
    if event_p is None:
        event_p = 95
    event_percentile = float(event_p) / 100.0

    if test_size is None:
        test_size = 0.3

    # Compute metrics using shared model helper.
    results = compute_metrics(
        years=years,
        event_percentile=event_percentile,
        test_size=test_size,
        random_state=42,
    )

    acc = results.get("accuracy")
    auc = results.get("auc")
    brier = results.get("brier")

    metrics_text = []
    if acc is not None:
        metrics_text.append(html.Div(f"Accuracy: {acc:.1%}", style={"color": "#00ff9d"}))
    if auc is not None:
        metrics_text.append(html.Div(f"ROC AUC: {auc:.3f}", style={"color": "#38bdf8"}))
    if brier is not None:
        metrics_text.append(html.Div(f"Brier score: {brier:.3f}", style={"color": "#fbbf24"}))

    # Time-series: probability & events
    ts_data = results.get("ts", {})
    df_ts = pd.DataFrame(
        {
            "date": pd.to_datetime(ts_data.get("datetime", [])),
            "prob_pred": ts_data.get("p_hat", []),
            "actual_event": ts_data.get("actual", []),
        }
    )

    fig_ts = _empty_dark_fig()
    if not df_ts.empty:
        fig_ts = go.Figure()

        events_only = df_ts[df_ts["actual_event"] == 1]
        if not events_only.empty:
            fig_ts.add_trace(
                go.Scatter(
                    x=events_only["date"],
                    y=events_only["prob_pred"],
                    mode="markers",
                    name="Stress event",
                    marker=dict(color="#ff0055", size=8, symbol="x", line=dict(width=2, color="#ff0055")),
                )
            )

        fig_ts.add_trace(
            go.Scatter(
                x=df_ts["date"],
                y=df_ts["prob_pred"],
                mode="lines",
                name="Predicted probability",
                line=dict(color="#00f2ff", width=2, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(0, 242, 255, 0.05)",
            )
        )
        fig_ts.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Date",
            yaxis_title="Stress probability",
            hovermode="x unified",
        )

    # ROC curve
    roc_data = results.get("roc_points", {})
    fpr = roc_data.get("fpr", [])
    tpr = roc_data.get("tpr", [])

    fig_roc = _empty_dark_fig()
    if len(fpr) and len(tpr):
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name="ROC",
                line=dict(color="#22d3ee", width=3),
            )
        )
        fig_roc.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(dash="dash", color="#4b5563"),
        )
        fig_roc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )

    # Histogram of predicted probabilities
    fig_hist = _empty_dark_fig()
    if not df_ts.empty:
        fig_hist = px.histogram(df_ts, x="prob_pred", nbins=30)
        fig_hist.update_traces(marker_line_width=0)
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Predicted probability",
            yaxis_title="Count",
            showlegend=False,
        )

    store_data = {"years": years, "test_size": float(test_size), "event_p": float(event_percentile)}
    return store_data, metrics_text, fig_ts, fig_roc, fig_hist


@callback(
    Output("predict-out", "children"),
    Input("predict-btn", "n_clicks"),
    State("metrics-store", "data"),
    State("in-qty", "value"),
    State("in-charge", "value"),
    State("in-month", "value"),
    prevent_initial_call=True,
)
def do_predict(n_clicks, store, qty, charge, month):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if not store:
        return "⚠️ Train the model first."

    if qty is None or charge is None or month is None:
        return "Please fill in quantity, charge, and month."

    try:
        prob = predict_probability(
            store["years"],
            {"qty": float(qty), "charge": float(charge), "month": float(month)},
            store["event_p"],
            store["test_size"],
            42,
        )
    except Exception as e:
        return f"Error computing probability: {e}"

    percentage = float(prob) * 100.0
    color = "#ff0055" if percentage > 50 else "#00ff9d"

    return html.Div(
        [
            html.Span("Estimated stress probability:", style={"fontSize": "0.85rem"}),
            html.Span(
                f" {percentage:.1f}%",
                style={
                    "fontSize": "1.1rem",
                    "fontWeight": "600",
                    "marginLeft": "6px",
                    "color": color,
                    "textShadow": f"0 0 10px {color}",
                },
            ),
        ]
    )
