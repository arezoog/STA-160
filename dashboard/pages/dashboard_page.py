import dash
from dash import html, dcc, callback, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model import get_forecast_dashboard_data

dash.register_page(__name__, path="/dashboard", name="Forecast Dashboard")

# -------------------------------------------------------------------------
# Load all precomputed forecasting outputs once
# -------------------------------------------------------------------------

results = get_forecast_dashboard_data()

data = results["data"].copy()
data["period_dt"] = pd.to_datetime(data["period"])

comparison_df: pd.DataFrame = results["comparison_df"]
future_df: pd.DataFrame = results["future_df"].copy()
backtest_df: pd.DataFrame = results["backtest_df"]

test_mask = results["test_mask"]
test_df = data.loc[test_mask].copy()
test_df["period_dt"] = pd.to_datetime(test_df["period"])

y_test_df: pd.DataFrame = results["y_test"]
y_test_arr = y_test_df.to_numpy()

y_pred_dict = results["y_pred_test"]
y_pred_svr = np.asarray(y_pred_dict["SVR_tuned"])
y_pred_rf = np.asarray(y_pred_dict["RandomForest"])

# Ensure future periods are datetime
if "period_dt" not in future_df.columns:
    future_df["period_dt"] = pd.to_datetime(future_df["period"])

HORIZON_MAX = max(1, len(future_df))  # used for slider

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _empty_fig(message="No data available"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=40),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#9ca3af"),
            )
        ],
    )
    return fig


# -------------------------------------------------------------------------
# Figures (static builders)
# -------------------------------------------------------------------------

def make_full_history_fig() -> go.Figure:
    if data.empty:
        return _empty_fig("No historical data available.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["period_dt"],
            y=data["weighted_avg_price"],
            mode="lines",
            name="Weighted avg price",
        )
    )
    fig.update_layout(
        title="Historical RA Price Index (Monthly)",
        xaxis_title="Period",
        yaxis_title="Weighted avg price ($/kW-month)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_test_1step_fig() -> go.Figure:
    """
    Actual vs predicted 1-step ahead prices on the 12-month test set.
    """
    if test_df.empty or y_test_arr.size == 0:
        return _empty_fig("No test data available for 1-step forecast.")

    x = test_df["period_dt"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_test_arr[:, 0],
            mode="lines+markers",
            name="Actual (t+1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred_svr[:, 0],
            mode="lines+markers",
            name="SVR tuned (t+1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred_rf[:, 0],
            mode="lines+markers",
            name="Random Forest (t+1)",
            opacity=0.7,
        )
    )

    fig.update_layout(
        title="Holdout Test: 1-Step Ahead Forecast (Last 12 Months)",
        xaxis_title="Period",
        yaxis_title="Weighted avg price ($/kW-month)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_backtest_fig() -> go.Figure:
    """
    Rolling 1-step backtest MAE over time (Ridge).
    """
    if backtest_df.empty:
        return _empty_fig("No backtest results available.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["period"],
            y=backtest_df["mae"],
            mode="lines+markers",
            name="1-step MAE (Ridge backtest)",
        )
    )

    avg_mae = results["avg_backtest_mae"]
    if np.isfinite(avg_mae):
        fig.add_hline(
            y=avg_mae,
            line_width=1,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average MAE = {avg_mae:.2f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Rolling Backtest – 1-Step Ahead Errors (Ridge baseline)",
        xaxis_title="Period",
        yaxis_title="MAE (absolute error)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_metrics_table() -> html.Table:
    """
    Simple HTML table for model comparison (SVR baseline, tuned SVR, RF).
    """
    if comparison_df.empty:
        return html.Table(
            html.Tbody(
                html.Tr(html.Td("No model comparison metrics available."))
            )
        )

    df = comparison_df[
        ["Model", "R2_t+1", "R2_t+2", "MAE_t+1", "MAE_t+2", "RMSE_t+1", "RMSE_t+2"]
    ].round(3)

    header = html.Tr(
        [
            html.Th("Model"),
            html.Th("R² (t+1)"),
            html.Th("R² (t+2)"),
            html.Th("MAE (t+1)"),
            html.Th("MAE (t+2)"),
            html.Th("RMSE (t+1)"),
            html.Th("RMSE (t+2)"),
        ]
    )

    body_rows = []
    for _, row in df.iterrows():
        body_rows.append(
            html.Tr(
                [
                    html.Td(row["Model"]),
                    html.Td(row["R2_t+1"]),
                    html.Td(row["R2_t+2"]),
                    html.Td(row["MAE_t+1"]),
                    html.Td(row["MAE_t+2"]),
                    html.Td(row["RMSE_t+1"]),
                    html.Td(row["RMSE_t+2"]),
                ]
            )
        )

    return html.Table(
        [html.Thead(header), html.Tbody(body_rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "0.85rem",
        },
    )


# -------------------------------------------------------------------------
# Layout with controls + graphs
# -------------------------------------------------------------------------

layout = html.Div(
    style={"maxWidth": "1200px", "margin": "24px auto 32px auto", "padding": "0 16px"},
    children=[
        # Top summary + controls
        html.Div(
            className="glass-card",
            style={"padding": "24px", "marginBottom": "24px"},
            children=[
                html.H2("Resource Adequacy Price Forecast Dashboard"),
                html.P(
                    "Use the controls to explore forecasts: choose model, "
                    "adjust forecast horizon, and compare historical vs future prices.",
                    style={"opacity": 0.85},
                ),

                # CONTROL PANEL
                html.Div(
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "16px",
                        "marginTop": "12px",
                        "marginBottom": "8px",
                    },
                    children=[
                        # Model selector
                        html.Div(
                            style={"minWidth": "220px"},
                            children=[
                                html.Label(
                                    "Forecast model",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.04em",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="forecast-model",
                                    options=[
                                        {"label": "SVR (tuned)", "value": "SVR_tuned"},
                                        {"label": "Random Forest", "value": "RandomForest"},
                                        {"label": "Compare both", "value": "both"},
                                    ],
                                    value="both",
                                    clearable=False,
                                ),
                            ],
                        ),
                        # Forecast horizon slider
                        html.Div(
                            style={"minWidth": "260px"},
                            children=[
                                html.Label(
                                    "Forecast horizon (months ahead)",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.04em",
                                    },
                                ),
                                dcc.Slider(
                                    id="forecast-horizon",
                                    min=1,
                                    max=HORIZON_MAX,
                                    step=1,
                                    value=min(12, HORIZON_MAX),
                                    marks={
                                        i: str(i)
                                        for i in range(1, HORIZON_MAX + 1)
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                html.H4("Model Comparison (12-month holdout)"),
                make_metrics_table(),
            ],
        ),

        # ROW 1: full history + interactive forecast
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
                "gap": "16px",
                "marginBottom": "24px",
            },
            children=[
                html.Div(
                    className="glass-card",
                    style={"padding": "16px"},
                    children=[
                        html.H4("Historical Price Index"),
                        dcc.Graph(
                            id="full-history-graph",
                            figure=make_full_history_fig(),
                            style={"height": "360px"},
                        ),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    style={"padding": "16px"},
                    children=[
                        html.H4("Interactive Forecast"),
                        dcc.Graph(
                            id="interactive-forecast-graph",
                            style={"height": "360px"},
                            figure=_empty_fig("Adjust controls to see forecast."),
                        ),
                    ],
                ),
            ],
        ),

        # ROW 2: test set + backtest
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
                "gap": "16px",
            },
            children=[
                html.Div(
                    className="glass-card",
                    style={"padding": "16px"},
                    children=[
                        html.H4("Holdout Test – 1-Step Ahead (Last 12 Months)"),
                        dcc.Graph(
                            id="test-1step-graph",
                            figure=make_test_1step_fig(),
                            style={"height": "360px"},
                        ),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    style={"padding": "16px"},
                    children=[
                        html.H4("Rolling Backtest – 1-Step MAE"),
                        dcc.Graph(
                            id="backtest-graph",
                            figure=make_backtest_fig(),
                            style={"height": "360px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# -------------------------------------------------------------------------
# Callback: interactive forecast graph
# -------------------------------------------------------------------------

@callback(
    Output("interactive-forecast-graph", "figure"),
    Input("forecast-model", "value"),
    Input("forecast-horizon", "value"),
)
def update_forecast_graph(model_choice, horizon):
    """
    Update the forecast plot based on:
      - model_choice: 'SVR_tuned', 'RandomForest', or 'both'
      - horizon: how many months ahead to show (1..len(future_df))
    """
    if future_df.empty or data.empty:
        return _empty_fig("No forecast data available.")

    # Safety: clip horizon
    if horizon is None or horizon < 1:
        horizon = 1
    horizon = min(horizon, len(future_df))

    # Last 36 months of history for context
    hist_tail = data.tail(36).copy()
    hist_tail_x = hist_tail["period_dt"]
    hist_tail_y = hist_tail["weighted_avg_price"]

    # Slice future forecast
    fut = future_df.iloc[:horizon].copy()
    fut_x = fut["period_dt"]
    svr_y = fut["SVR_forecast"] if "SVR_forecast" in fut.columns else None
    rf_y = fut["RF_forecast"] if "RF_forecast" in fut.columns else None

    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=hist_tail_x,
            y=hist_tail_y,
            mode="lines",
            name="Historical price (last 36 months)",
            line=dict(width=2),
        )
    )

    # Forecast lines depending on model_choice
    if model_choice in ["SVR_tuned", "both"] and svr_y is not None:
        fig.add_trace(
            go.Scatter(
                x=fut_x,
                y=svr_y,
                mode="lines+markers",
                name="SVR forecast",
            )
        )

    if model_choice in ["RandomForest", "both"] and rf_y is not None:
        fig.add_trace(
            go.Scatter(
                x=fut_x,
                y=rf_y,
                mode="lines+markers",
                name="RF forecast",
                opacity=0.7,
            )
        )

    # Compute y-range for vertical split marker
    y_vals = list(hist_tail_y.values)
    if model_choice in ["SVR_tuned", "both"] and svr_y is not None:
        y_vals += list(svr_y.values)
    if model_choice in ["RandomForest", "both"] and rf_y is not None:
        y_vals += list(rf_y.values)

    if y_vals:
        y_min = min(y_vals)
        y_max = max(y_vals)
    else:
        y_min, y_max = 0, 1

    # Vertical line at forecast start using shape (safe with datetimes)
    forecast_start = hist_tail_x.iloc[-1]
    fig.add_shape(
        type="line",
        x0=forecast_start,
        x1=forecast_start,
        y0=y_min,
        y1=y_max,
        line=dict(color="gray", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=forecast_start,
        y=y_max,
        text="Forecast start",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="gray"),
    )

    fig.update_layout(
        title=f"{horizon}-Month Ahead Forecast ({model_choice})",
        xaxis_title="Period",
        yaxis_title="Weighted avg price ($/kW-month)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
