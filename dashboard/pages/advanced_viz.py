import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import numpy as np

# Reuse the data loader from model.py for consistency
from model import load_master

dash.register_page(__name__, path="/animation", name="Market Evolution")

# --- Data Preparation ---
try:
    df = load_master()

    # 1. Create a Time Period column for the animation frame (YYYY-MM)
    # This groups the "frames" of the movie.
    df['period'] = df['datetime'].dt.to_period('M').astype(str)
    
    # 2. Sort by date so the animation plays in order
    df = df.sort_values('period')

    # 3. Handle potential negatives or zeros for Log/Size scaling
    # (Size cannot be negative in bubble charts)
    df['charge_abs'] = df['charge'].abs()
    
    # 4. Define static ranges for axes
    # (Crucial: prevents the axes from jumping around between frames)
    max_qty = 25000
    # Check which price column exists (standardized or raw)
    price_col = 'standardized_price' if 'standardized_price' in df.columns else 'weighted_avg_price'
    max_price = df[price_col].max() * 1.1

except Exception as e:
    print(f"Error preparing animation data: {e}")
    df = pd.DataFrame()
    price_col = 'price'
    max_qty = 100
    max_price = 100

# --- Graph Construction ---
def create_animation():
    if df.empty:
        return {}
    
    fig = px.scatter(
        df,
        x="qty",
        y=price_col,
        animation_frame="period",  # The "Slider" at the bottom
        animation_group="qty",     # Helps smooth transitions
        size="charge_abs",         # Bubble size = Revenue/Charge
        color=price_col,           # Color = Price intensity
        
        # Color Palette: 'Magma' looks great on dark backgrounds (Black->Red->Yellow)
        color_continuous_scale="Magma", 
        
        hover_data=["datetime", "charge"],
        range_x=[0, max_qty],
        range_y=[0, max_price],
        template="plotly_dark"
    )

    # Fine-tuning the layout for your Dark Theme
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(color="#e5e7eb"),
        transition = {'duration': 1000}, # Smoothness of movement
    )
    
    # Update axes to look clean
    fig.update_xaxes(title="Transacted Quantity (MW)", gridcolor="#374151")
    fig.update_yaxes(title="Price ($)", gridcolor="#374151")
    
    # Update the "Play" button and Slider style
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=1.1,
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[dict(
                label="▶ Play Market History",
                method="animate",
                args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True)]
            )]
        )],
        sliders=[dict(
            currentvalue={"prefix": "Month: ", "font": {"size": 20, "color": "#ef4444"}},
            pad={"t": 50}
        )]
    )

    return fig

# --- Layout ---
layout = html.Div(
    style={"paddingTop": "8px"},
    children=[
        # Info Card
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
                html.H3("Market Evolution (Time-Lapse)", style={"marginTop": 0, "marginBottom": "6px", "fontSize": "1.1rem"}),
                html.P(
                    "Press 'Play' to watch 5 years of market data evolve. "
                    "Bubbles represent individual trades. Size = Total Revenue (Charge).",
                    style={"color": "#9ca3af", "fontSize": "0.9rem", "marginBottom": "0px"}
                ),
            ]
        ),

        # The Animated Graph Container
        html.Div(
            style={
                "backgroundColor": "#020617",
                "borderRadius": "16px",
                "padding": "12px",
                "border": "1px solid #1f2937",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
            },
            children=[
                dcc.Graph(
                    id="animation-graph",
                    figure=create_animation(),
                    style={"height": "75vh"}, # Make it tall
                    config={"displayModeBar": False}
                )
            ]
        )
    ]
)