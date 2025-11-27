import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import AVAILABLE_YEARS, compute_metrics, predict_probability

dash.register_page(__name__, path="/dashboard", name="Dashboard")

# --- Helper: Transparent Dark Figure ---
# This fixes the "White Box" issue on load
def get_empty_dark_fig():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# --- Styles ---
tab_style = {'borderBottom': '1px solid #374151', 'padding': '12px', 'backgroundColor': 'rgba(0,0,0,0)', 'color': '#94a3b8'}
tab_selected_style = {'borderBottom': '2px solid #00f2ff', 'backgroundColor': 'rgba(0, 242, 255, 0.1)', 'color': '#00f2ff', 'fontWeight': 'bold', 'padding': '12px'}

layout = html.Div(
    style={"paddingTop": "8px"},
    children=[
        dcc.Store(id="metrics-store"),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                # LEFT CARD
                html.Div(
                    className="glass-card", # <--- Applies the CSS blur
                    style={"flex": "1 1 280px", "padding": "20px", "borderRadius": "16px", "display": "flex", "flexDirection": "column", "gap": "20px"},
                    children=[
                        html.Div([
                            html.H4("Configuration", style={"margin": "0 0 10px 0", "fontSize": "1rem"}),
                            html.Label("Training Years:", style={"fontSize": "0.85rem", "color": "#9ca3af"}),
                            dcc.Dropdown(
                                id="year-dropdown",
                                options=[{"label": str(y), "value": y} for y in AVAILABLE_YEARS],
                                value=[2022, 2023, 2024, 2025],
                                multi=True,
                                style={"color": "#000"} 
                            ),
                        ]),
                        html.Div([
                            html.Label("Test Size:", style={"fontSize": "0.85rem", "color": "#9ca3af"}),
                            dcc.Slider(id="test-size-slider", min=0.1, max=0.5, step=0.05, value=0.2, marks={0.1: "10%", 0.5: "50%"}, tooltip={"placement": "bottom"}),
                        ]),
                        html.Div([
                            html.Label("Event Threshold:", style={"fontSize": "0.85rem", "color": "#9ca3af"}),
                            dcc.Slider(id="event-p-slider", min=0.80, max=0.99, step=0.01, value=0.95, marks={0.80: "80%", 0.99: "99%"}, tooltip={"placement": "bottom"}),
                        ]),
                        html.Button(
                            "⚡ Train / Refresh Model", id="train-btn", n_clicks=0, className="glass-card",
                            style={"marginTop": "auto", "color": "#00f2ff", "fontWeight": "bold", "padding": "12px", "cursor": "pointer", "border": "1px solid rgba(0, 242, 255, 0.3)", "background": "rgba(0, 242, 255, 0.1)"}
                        ),
                        html.Div(id="model-metrics", style={"fontSize": "0.85rem", "color": "#d1d5db", "lineHeight": "1.6"})
                    ],
                ),
                # MIDDLE CARD
                html.Div(
                    className="glass-card",
                    style={"flex": "2 1 500px", "padding": "20px", "borderRadius": "16px"},
                    children=[
                        html.H4("Model Performance", style={"margin": "0 0 15px 0"}),
                        dcc.Tabs(
                            id="viz-tabs", value="ts-tab", parent_className="custom-tabs", content_style={"padding": "10px", "border": "none"},
                            children=[
                                dcc.Tab(label="Predictions vs. Actual", value="ts-tab", style=tab_style, selected_style=tab_selected_style,
                                        children=[dcc.Loading(id="loading-1", type="cube", color="#00f2ff", children=dcc.Graph(id="ts-graph", figure=get_empty_dark_fig(), style={"height": "350px"}))]),
                                dcc.Tab(label="ROC Curve", value="roc-tab", style=tab_style, selected_style=tab_selected_style,
                                        children=[dcc.Graph(id="roc-graph", figure=get_empty_dark_fig(), style={"height": "350px"})]),
                                dcc.Tab(label="Prob. Histogram", value="hist-tab", style=tab_style, selected_style=tab_selected_style,
                                        children=[dcc.Graph(id="hist-graph", figure=get_empty_dark_fig(), style={"height": "350px"})]),
                            ]
                        )
                    ],
                ),
                # RIGHT CARD
                html.Div(
                    className="glass-card",
                    style={"flex": "1 1 250px", "padding": "20px", "borderRadius": "16px", "display": "flex", "flexDirection": "column", "gap": "15px"},
                    children=[
                        html.H4("Live Prediction", style={"margin": "0", "fontSize": "1rem"}),
                        # These inputs are styled to be DARK now:
                        html.Div([html.Label("Quantity (MW)", style={"fontSize": "0.8rem"}), dcc.Input(id="in-qty", type="number", value=100, style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #374151", "backgroundColor": "rgba(255,255,255,0.05)", "color": "#e5e7eb"})]),
                        html.Div([html.Label("Total Charge ($)", style={"fontSize": "0.8rem"}), dcc.Input(id="in-charge", type="number", value=5000, style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #374151", "backgroundColor": "rgba(255,255,255,0.05)", "color": "#e5e7eb"})]),
                        html.Div([html.Label("Month (1-12)", style={"fontSize": "0.8rem"}), dcc.Input(id="in-month", type="number", value=7, min=1, max=12, style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #374151", "backgroundColor": "rgba(255,255,255,0.05)", "color": "#e5e7eb"})]),
                        html.Button("Compute Probability", id="predict-btn", n_clicks=0, style={"backgroundColor": "#2563eb", "color": "white", "border": "1px solid #3b82f6", "padding": "10px", "borderRadius": "8px", "fontWeight": "bold", "cursor": "pointer", "marginTop": "10px", "boxShadow": "0 0 10px rgba(59, 130, 246, 0.4)"}),
                        html.Div(id="predict-out", className="neon-border", style={"marginTop": "10px", "padding": "15px", "borderRadius": "8px", "textAlign": "center", "backgroundColor": "rgba(0,0,0,0.3)", "minHeight": "60px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold", "color": "#00f2ff", "border": "1px solid rgba(0,242,255,0.2)"}, children="Ready...")
                    ],
                ),
            ],
        ),
    ],
)

@callback(
    [Output("metrics-store", "data"), Output("model-metrics", "children"), Output("ts-graph", "figure"), Output("roc-graph", "figure"), Output("hist-graph", "figure")],
    Input("train-btn", "n_clicks"),
    [State("year-dropdown", "value"), State("test-size-slider", "value"), State("event-p-slider", "value")]
)
def update_model(n_clicks, years, test_size, event_p):
    if not years: years = [2023, 2024]
    results = compute_metrics(years=years, event_percentile=event_p, test_size=test_size, random_state=42)
    acc, auc, brier = results["accuracy"], results["auc"], results["brier"]
    metrics_text = [html.Div(f"Accuracy: {acc:.1%}", style={"color": "#00ff9d"}), html.Div(f"ROC AUC: {auc:.3f}", style={"color": "#00f2ff"}), html.Div(f"Brier Score: {brier:.3f}", style={"color": "#fbbf24"})]
    
    # NEON GRAPH STYLING
    ts_data = results["ts"]
    df_ts = pd.DataFrame({"date": pd.to_datetime(ts_data["datetime"]), "prob_pred": ts_data["p_hat"], "actual_event": ts_data["actual"]})
    fig_ts = go.Figure()
    events_only = df_ts[df_ts["actual_event"] == 1]
    fig_ts.add_trace(go.Scatter(x=events_only["date"], y=events_only["prob_pred"], mode="markers", name="Stress Event", marker=dict(color="#ff0055", size=8, symbol="x", line=dict(width=2, color="#ff0055"))))
    fig_ts.add_trace(go.Scatter(x=df_ts["date"], y=df_ts["prob_pred"], mode="lines", name="Predicted Prob", line=dict(color="#00f2ff", width=2, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 242, 255, 0.05)'))
    fig_ts.update_layout(template="plotly_dark", title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40), legend=dict(orientation="h", y=1.1, font=dict(family="Rajdhani", size=14)), xaxis=dict(showgrid=False, title="Date"), yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Probability"), hovermode="x unified")
    
    roc = results["roc_points"]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC", line=dict(color="#00ff9d", width=3)))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#4b5563"))
    fig_roc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40), xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

    fig_hist = px.histogram(df_ts, x="prob_pred", nbins=30, color_discrete_sequence=["#00f2ff"])
    fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=10, b=40), xaxis_title="Predicted Probability", yaxis_title="Count", showlegend=False)

    return {"years": years, "test_size": float(test_size), "event_p": float(event_p)}, metrics_text, fig_ts, fig_roc, fig_hist

@callback(Output("predict-out", "children"), Input("predict-btn", "n_clicks"), State("metrics-store", "data"), State("in-qty", "value"), State("in-charge", "value"), State("in-month", "value"), prevent_initial_call=True)
def do_predict(n, store, qty, charge, month):
    if not store: return "⚠️ Train model first"
    prob = predict_probability(store["years"], {"qty": float(qty), "charge": float(charge), "month": float(month)}, store["event_p"], store["test_size"], 42)
    percentage = prob * 100
    color = "#ff0055" if percentage > 50 else "#00ff9d"
    return html.Div([html.Span("Stress Probability: ", style={"color": "#e5e7eb"}), html.Span(f"{percentage:.1f}%", style={"color": color, "fontSize": "1.2rem", "marginLeft": "8px", "textShadow": f"0 0 10px {color}"})])