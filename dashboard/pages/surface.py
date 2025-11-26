import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from model import load_master

dash.register_page(__name__, path="/surface", name="3D Market Terrain")

# --- Data Prep ---
try:
    df = load_master()
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    
    # We need a grid (Matrix) for a Surface Plot.
    # Pivot: Index=Year, Columns=Month, Values=Price
    price_col = 'standardized_price' if 'standardized_price' in df.columns else 'weighted_avg_price'
    
    # Create the matrix
    matrix = df.pivot_table(index='year', columns='month', values=price_col, aggfunc='mean')
    
    # Fill gaps (interpolation makes the surface smooth instead of jagged/broken)
    matrix_filled = matrix.interpolate(method='linear', axis=1).fillna(0)

    # X and Y labels
    years = matrix_filled.index
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

except Exception as e:
    print(f"Error: {e}")
    matrix_filled = pd.DataFrame()
    years = []
    months = []

# --- Layout ---
layout = html.Div(
    style={"paddingTop": "8px"},
    children=[
        html.Div(
            style={
                "backgroundColor": "#020617",
                "borderRadius": "16px",
                "padding": "18px 20px",
                "border": "1px solid #1f2937",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                "marginBottom": "16px",
            },
            children=[
                html.H3("Market Price Terrain", style={"marginTop": 0, "marginBottom": "6px", "color": "#e5e7eb"}),
                html.P(
                    "Visualizing price stress as a topological landscape. Peaks represent high-stress periods.",
                    style={"color": "#9ca3af", "fontSize": "0.9rem"}
                )
            ]
        ),
        
        html.Div(
            style={
                "backgroundColor": "#020617",
                "borderRadius": "16px",
                "border": "1px solid #1f2937",
                "padding": "10px",
                "height": "80vh"
            },
            children=[
                dcc.Graph(
                    id="surface-plot",
                    style={"height": "100%", "width": "100%"},
                    config={"displayModeBar": False},
                    figure=go.Figure(
                        data=[go.Surface(
                            z=matrix_filled.values,
                            x=months,
                            y=years,
                            colorscale="Viridis", 
                            opacity=0.9,
                            contours_z=dict(
                                show=True, 
                                usecolormap=True, 
                                highlightcolor="limegreen", 
                                project_z=True # This projects a flat heatmap at the bottom!
                            )
                        )],
                        layout=go.Layout(
                            template="plotly_dark",
                            title=None,
                            scene=dict(
                                xaxis=dict(title="Month", gridcolor="#374151", backgroundcolor="rgba(0,0,0,0)"),
                                yaxis=dict(title="Year", gridcolor="#374151", backgroundcolor="rgba(0,0,0,0)"),
                                zaxis=dict(title="Avg Price ($)", gridcolor="#374151", backgroundcolor="rgba(0,0,0,0)"),
                                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Set a nice initial camera angle
                            ),
                            margin=dict(l=0, r=0, b=0, t=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                    )
                )
            ]
        )
    ]
)
