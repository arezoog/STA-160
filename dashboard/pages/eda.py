import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from model import load_master

dash.register_page(__name__, path="/eda", name="Exploratory Analysis")

# --- Data preparation -----------------------------------------------------

try:
    df = load_master()

    # Try to ensure delivery_month exists and is datetime-like
    if "delivery_month" in df.columns:
        df["delivery_month"] = pd.to_datetime(df["delivery_month"], errors="coerce")
    elif "datetime" in df.columns:
        df["delivery_month"] = pd.to_datetime(df["datetime"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    else:
        df["delivery_month"] = pd.NaT

    available_months = df["delivery_month"].dropna().sort_values().unique()
    available_months_str = [pd.Timestamp(m).strftime("%Y-%m") for m in available_months]
except Exception as e:
    print(f"Error loading data for EDA: {e}")
    df = pd.DataFrame()
    available_months_str = []


def _empty_fig(message="No data available"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
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


# --- Layout ---------------------------------------------------------------

layout = html.Div(
    style={"maxWidth": "1200px", "margin": "16px auto 32px auto", "padding": "0 16px"},
    children=[
        html.Div(
            className="glass-card",
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "10px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H2("Exploratory analysis"),
                                html.P(
                                    "Slice the RA transactions by delivery month and inspect price–quantity patterns "
                                    "in both 2D and 3D.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem"},
                                ),
                            ]
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "16px",
                                "alignItems": "flex-end",
                            },
                            children=[
                                html.Div(
                                    style={"minWidth": "220px"},
                                    children=[
                                        html.Label(
                                            "Delivery month",
                                            style={
                                                "fontSize": "0.8rem",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.14em",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="eda-month-dropdown",
                                            options=[
                                                {"label": m, "value": m}
                                                for m in available_months_str
                                            ],
                                            value=available_months_str[0]
                                            if available_months_str
                                            else None,
                                            clearable=False,
                                            placeholder="Select month…",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "16px",
                "marginTop": "16px",
            },
            children=[
                html.Div(
                    className="glass-card",
                    style={"flex": "1 1 380px"},
                    children=[
                        html.H3("Quantity and price over time", style={"margin": "0 0 6px 0"}),
                        # FIX: Constrain height to 500px
                        dcc.Graph(id="eda-dual-axis", className="dash-graph", style={"height": "500px"}),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    style={"flex": "1 1 380px"},
                    children=[
                        html.H3("3D quantity–price cloud", style={"margin": "0 0 6px 0"}),
                        # FIX: Constrain height to 500px
                        dcc.Graph(id="eda-3d-scatter", className="dash-graph", style={"height": "500px"}),
                    ],
                ),
            ],
        ),
    ],
)


# --- Callback -------------------------------------------------------------


@callback(
    Output("eda-dual-axis", "figure"),
    Output("eda-3d-scatter", "figure"),
    Input("eda-month-dropdown", "value"),
)
def update_eda_graphs(selected_month_str):
    if not selected_month_str or df.empty:
        empty = _empty_fig("Select a month to view detail.")
        return empty, empty

    # Filter data for the selected month
    mask = df["delivery_month"].dt.strftime("%Y-%m") == selected_month_str
    dff = df[mask].sort_values("datetime").copy()

    if dff.empty:
        empty = _empty_fig("No data for this delivery month.")
        return empty, empty

    price_col = (
        "standardized_price"
        if "standardized_price" in dff.columns
        else "charge"
        if "charge" in dff.columns
        else None
    )
    if price_col is None:
        empty = _empty_fig("No price column found.")
        return empty, empty

    # --- 1. Dual-axis chart (bars = quantity, line = price) -------------
    fig2d = make_subplots(specs=[[{"secondary_y": True}]])

    # Quantity bars
    if "qty" in dff.columns:
        fig2d.add_trace(
            go.Bar(
                x=dff["datetime"],
                y=dff["qty"],
                name="Quantity (MW)",
                marker_color="rgba(59,130,246,0.4)",
                marker_line_width=0,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Qty: %{y:.0f} MW<extra></extra>",
            ),
            secondary_y=False,
        )

    # Price line
    fig2d.add_trace(
        go.Scatter(
            x=dff["datetime"],
            y=dff[price_col],
            mode="lines",
            name="Price",
            line=dict(color="#f97316", width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Price: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig2d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
        # Allow plotly to autosize within the container we set
        autosize=True,
    )
    fig2d.update_xaxes(title_text="Datetime")
    fig2d.update_yaxes(title_text="Quantity (MW)", secondary_y=False)
    fig2d.update_yaxes(title_text="Price ($)", secondary_y=True)

    # --- 2. 3D scatter of quantity vs price vs time ----------------------
    if "qty" in dff.columns:
        y_vals = dff["qty"]
    else:
        y_vals = dff.index

    fig3d = go.Figure(
        data=[
            go.Scatter3d(
                x=dff["datetime"].dt.strftime("%Y-%m-%d %H:%M"),
                y=y_vals,
                z=dff[price_col],
                mode="markers",
                marker=dict(
                    size=5,
                    color=dff[price_col],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title="Price", thickness=15, x=0.9),
                ),
                hovertemplate="<b>Time:</b> %{x}<br><b>Qty:</b> %{y} MW<br><b>Price:</b> $%{z:.2f}<extra></extra>",
            )
        ]
    )

    fig3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Quantity",
            zaxis_title="Price",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#374151", color="#e5e7eb"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#374151", color="#e5e7eb"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#374151", color="#e5e7eb"),
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        autosize=True,
    )

    return fig2d, fig3d