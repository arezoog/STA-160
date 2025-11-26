import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Import the shared data loader from your existing model.py
from model import load_master

# Register this page so it appears in the navigation
dash.register_page(__name__, path="/eda", name="Exploratory Analysis")

# --- Data Preparation ---
# We use the existing cached function from model.py to ensure consistency
try:
    df = load_master()
    
    # Ensure delivery_month is datetime for sorting/filtering
    # model.py loads it but might leave it as string depending on CSV format
    df['delivery_month'] = pd.to_datetime(df['delivery_month'], errors='coerce')
    
    # Create list of months for the dropdown (YYYY-MM format)
    # Filter out NaTs just in case
    available_months = df['delivery_month'].dropna().sort_values().unique()
    available_months_str = [pd.Timestamp(m).strftime('%Y-%m') for m in available_months]
except Exception as e:
    print(f"Error loading data for EDA: {e}")
    df = pd.DataFrame()
    available_months_str = []

# --- Layout ---
layout = html.Div(
    style={"paddingTop": "8px"},
    children=[
        # Control Card
        html.Div(
            style={
                "backgroundColor": "#020617",
                "borderRadius": "16px",
                "padding": "18px 20px",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                "border": "1px solid #1f2937",
                "marginBottom": "16px",
            },
            children=[
                html.H3("Visual Exploration", style={"marginTop": 0, "marginBottom": "6px", "fontSize": "1.1rem"}),
                html.P(
                    "Analyze historical price and quantity relationships for specific delivery months.",
                    style={"color": "#9ca3af", "fontSize": "0.9rem", "marginBottom": "12px"}
                ),
                html.Label("Select Delivery Month:", style={"fontSize": "0.8rem", "color": "#e5e7eb"}),
                dcc.Dropdown(
                    id="eda-month-dropdown",
                    options=[{'label': m, 'value': m} for m in available_months_str],
                    value=available_months_str[0] if available_months_str else None,
                    clearable=False,
                    style={
                        "marginTop": "4px", 
                        "color": "#020617",  # Dark text for the dropdown interior
                        "maxWidth": "300px"
                    } 
                ),
            ]
        ),

        # Graphs Container
        html.Div(
            style={"display": "grid", "gap": "16px", "gridTemplateColumns": "1fr"},
            children=[
                
                # 2D Dual-Axis Chart
                html.Div(
                    style={
                        "backgroundColor": "#020617",
                        "borderRadius": "16px",
                        "padding": "12px",
                        "border": "1px solid #1f2937",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                    },
                    children=[
                        html.Div("Price vs. Quantity (Time Series)", style={"color": "#e5e7eb", "fontWeight": 600, "marginBottom": "8px"}),
                        dcc.Graph(id="eda-dual-axis", style={"height": "400px"}, config={"displayModeBar": False})
                    ]
                ),

                # 3D Scatter Chart
                html.Div(
                    style={
                        "backgroundColor": "#020617",
                        "borderRadius": "16px",
                        "padding": "12px",
                        "border": "1px solid #1f2937",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                    },
                    children=[
                        html.Div("Market Surface (3D)", style={"color": "#e5e7eb", "fontWeight": 600, "marginBottom": "4px"}),
                        html.P(
                            "Rotate to view the relationship between Date (X), Quantity (Y), and Price (Z).", 
                            style={"fontSize": "0.8rem", "color": "#9ca3af"}
                        ),
                        dcc.Graph(id="eda-3d-scatter", style={"height": "600px"}, config={"displayModeBar": True})
                    ]
                )
            ]
        )
    ]
)

# --- Callbacks ---
@callback(
    [Output("eda-dual-axis", "figure"),
     Output("eda-3d-scatter", "figure")],
    Input("eda-month-dropdown", "value")
)
def update_eda_graphs(selected_month_str):
    if not selected_month_str or df.empty:
        empty = go.Figure()
        empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return empty, empty

    # Filter data for the selected month
    # We compare formatted strings to handle any timestamp mismatches
    mask = df['delivery_month'].dt.strftime('%Y-%m') == selected_month_str
    dff = df[mask].sort_values('datetime').copy()

    # If 'standardized_price' is missing, fallback to 'charge' / 'qty' or just 0
    # In model.py you used standardized_price, so we stick to that.
    price_col = 'standardized_price' if 'standardized_price' in dff.columns else 'charge'
    
    # --- 1. Dual Axis Chart (Line + Bars) ---
    fig2d = make_subplots(specs=[[{"secondary_y": True}]])

    # Bars for Quantity
    fig2d.add_trace(
        go.Bar(
            x=dff['datetime'],
            y=dff['qty'],
            name="Quantity (MW)",
            marker_color='rgba(59, 130, 246, 0.3)', # Transparent Blue
            marker_line_width=0,
            hoverinfo="x+y"
        ),
        secondary_y=True,
    )

    # Line for Price
    fig2d.add_trace(
        go.Scatter(
            x=dff['datetime'],
            y=dff[price_col],
            name="Price ($)",
            mode='lines',
            line=dict(color='#ef4444', width=2), # Neon Red
            hoverinfo="x+y"
        ),
        secondary_y=False,
    )

    fig2d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified"
    )
    fig2d.update_yaxes(title_text="Price ($)", secondary_y=False, gridcolor="#374151")
    fig2d.update_yaxes(title_text="Quantity (MW)", secondary_y=True, showgrid=False)

    # --- 2. 3D Scatter Chart ---
    # We convert datetime to string for the X-axis so Plotly 3D handles it cleanly
    fig3d = go.Figure(data=[go.Scatter3d(
        x=dff['datetime'].dt.strftime('%Y-%m-%d %H:%M'), 
        y=dff['qty'],
        z=dff[price_col],
        mode='markers',
        marker=dict(
            size=5,
            color=dff[price_col],      # Color points by Price
            colorscale='Viridis',      # Cool-to-Warm gradient
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="Price", thickness=15, x=0.9)
        ),
        hovertemplate="<b>Time:</b> %{x}<br><b>Qty:</b> %{y} MW<br><b>Price:</b> $%{z:.2f}<extra></extra>"
    )])

    fig3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        # No plot_bgcolor in 3D, it uses 'scene'
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Quantity',
            zaxis_title='Price',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#374151", color="#e5e7eb"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#374151", color="#e5e7eb"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#374151", color="#e5e7eb"),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig2d, fig3d
