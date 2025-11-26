import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import shared data loader
from model import load_master

dash.register_page(__name__, path="/risk", name="Risk Analysis")

# --- Data Preparation ---
try:
    df = load_master()
    
    # Ensure datetime for grouping
    df['year'] = df['datetime'].dt.year
    df['month_name'] = df['datetime'].dt.month_name()
    # Sort months correctly (not alphabetical)
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    df['month_cat'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)
    
    # Determine which price column to use
    price_col = 'standardized_price' if 'standardized_price' in df.columns else 'weighted_avg_price'

except Exception as e:
    print(f"Error preparing risk data: {e}")
    df = pd.DataFrame()
    price_col = 'price'

# --- Layout ---
layout = html.Div(
    style={"paddingTop": "8px"},
    children=[
        # 1. The "Stress Grid" (Calendar Heatmap)
        html.Div(
            style={
                "backgroundColor": "#020617",
                "borderRadius": "16px",
                "padding": "18px 20px",
                "border": "1px solid #1f2937",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                "marginBottom": "24px"
            },
            children=[
                html.H3("Market Stress Grid", style={"marginTop": 0, "marginBottom": "6px", "fontSize": "1.1rem"}),
                html.P(
                    "A thermal view of market history. Red squares indicate months with high average prices.",
                    style={"color": "#9ca3af", "fontSize": "0.9rem"}
                ),
                dcc.Graph(id="risk-heatmap", style={"height": "350px"}, config={"displayModeBar": False})
            ]
        ),

        # 2. The "Seasonal Risk" (Violin Plot)
        html.Div(
            style={
                "backgroundColor": "#020617",
                "borderRadius": "16px",
                "padding": "18px 20px",
                "border": "1px solid #1f2937",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Div([
                            html.H3("Seasonal Risk Distribution", style={"marginTop": 0, "marginBottom": "6px", "fontSize": "1.1rem"}),
                            html.P(
                                "Violin plots show the shape of price probability. Wider sections = more common prices.",
                                style={"color": "#9ca3af", "fontSize": "0.9rem", "marginBottom": "0px"}
                            ),
                        ]),
                        # Interactive Toggle
                        dcc.Checklist(
                            id="violin-options",
                            options=[{'label': ' Show Outlier Points', 'value': 'points'}],
                            value=[],
                            style={"color": "#e5e7eb", "fontSize": "0.9rem"}
                        )
                    ]
                ),
                dcc.Graph(id="risk-violin", style={"height": "500px"})
            ]
        )
    ]
)

# --- Callbacks ---
@callback(
    [Output("risk-heatmap", "figure"),
     Output("risk-violin", "figure")],
    Input("violin-options", "value")
)
def update_risk_plots(options):
    if df.empty:
        return go.Figure(), go.Figure()

    # --- 1. Calendar Heatmap ---
    # Pivot data: Rows=Year, Cols=Month, Values=Mean Price
    # We use pivot_table to aggregate
    pivot_df = df.pivot_table(
        index='year', 
        columns='month_cat', 
        values=price_col, 
        aggfunc='mean'
    )
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdBu_r', # Red=High, Blue=Low (Reversed)
        hoverongaps=False,
        hovertemplate="<b>%{x} %{y}</b><br>Avg Price: $%{z:.2f}<extra></extra>"
    ))

    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title=None,
        yaxis=dict(title=None, type='category', dtick=1), # Show every year
    )

    # --- 2. Violin Plot ---
    # Check if user wants to see raw points
    show_points = 'outliers' if 'points' in options else False

    fig_violin = go.Figure()

    # We add one trace per month so they get different colors
    # (Plotly Express does this automatically, but go.Violin gives us more control)
    fig_violin = px.violin(
        df, 
        x="month_cat", 
        y=price_col, 
        color="month_cat", 
        points=show_points, # Interactive part!
        box=True,           # Show box plot inside violin
    )

    fig_violin.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Price Distribution ($)",
    )
    
    # Customize the violin style
    fig_violin.update_traces(
        meanline_visible=True,
        line_color='white', # White outline for contrast
        opacity=0.8
    )

    return fig_heat, fig_violin