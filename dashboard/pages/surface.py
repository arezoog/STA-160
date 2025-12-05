import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from functools import lru_cache

from model import load_master


@lru_cache(maxsize=1)
def get_surface_data():
    """
    Lazily load and preprocess the master EQR data used by the surface page.

    Returns
    -------
    df : pd.DataFrame
        Data with a parsed 'datetime' column.
    price_col : str or None
        Name of the price column to use.
    """
    try:
        df = load_master()

        # Ensure datetime column
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            dt_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if dt_cols:
                df["datetime"] = pd.to_datetime(df[dt_cols[0]], errors="coerce")
            else:
                df["datetime"] = pd.NaT

        df = df.dropna(subset=["datetime"])

        # Choose price column preference order
        if "standardized_price" in df.columns:
            price_col = "standardized_price"
        elif "weighted_avg_price" in df.columns:
            price_col = "weighted_avg_price"
        elif "charge" in df.columns:
            price_col = "charge"
        else:
            price_col = None

        return df, price_col

    except Exception as e:
        print(f"Error preparing surface data: {e}")
        return pd.DataFrame(), None


def _empty_fig(message: str = "No data available"):
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


# Precompute year options for dropdown (using cached data)
_surface_df, _surface_price_col = get_surface_data()
if not _surface_df.empty:
    _years = sorted(_surface_df["datetime"].dt.year.dropna().unique())
    YEAR_OPTIONS = (
        [{"label": "All years", "value": -1}]
        + [{"label": str(int(y)), "value": int(y)} for y in _years]
    )
else:
    YEAR_OPTIONS = [{"label": "All years", "value": -1}]

DEFAULT_YEAR_VALUE = -1


# -------------------------------------------------------------------------
# Figure builder
# -------------------------------------------------------------------------


@lru_cache(maxsize=16)
def build_surface_figure(selected_year: int | None = None):
    """
    Build the 3D price surface figure.

    Parameters
    ----------
    selected_year : int or None
        If None, uses all years combined.
        Otherwise, filters to that calendar year first.
    """
    df, price_col = get_surface_data()

    if df.empty or price_col is None:
        return _empty_fig()

    df = df.copy()

    # Optional year filter
    if selected_year is not None:
        df = df[df["datetime"].dt.year == int(selected_year)]
        if df.empty:
            return _empty_fig("No data for the selected year")

    # Build month/hour grid
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour

    pivot = df.pivot_table(
        index="hour",
        columns="month",
        values=price_col,
        aggfunc="mean",
    )

    if pivot.empty:
        return _empty_fig("No data to build surface")

    # Interpolate to smooth missing values, then fill any remaining holes
    pivot = pivot.interpolate(method="linear", axis=0).fillna(0)

    x_vals = pivot.columns  # months 1..12
    y_vals = pivot.index    # hours 0..23
    z_vals = pivot.values

    surface = go.Surface(
        z=z_vals,
        x=x_vals,
        y=y_vals,
        colorscale="Electric",
        opacity=0.9,
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True,
        ),
        colorbar=dict(title="Price ($)", thickness=15, x=0.9),
    )

    title_suffix = "All years" if selected_year is None else f"Year {int(selected_year)}"

    fig = go.Figure(data=[surface])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"Hourly Price Terrain (Hologram) – {title_suffix}",
            x=0.5,
            font=dict(family="Orbitron", size=20, color="#22d3ee"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(
                title="Month",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(34, 211, 238, 0.2)",
                showbackground=False,
                tickmode="linear",
                dtick=1,
            ),
            yaxis=dict(
                title="Hour of Day",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(34, 211, 238, 0.2)",
                showbackground=False,
            ),
            zaxis=dict(
                title="Price ($)",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(34, 211, 238, 0.2)",
                showbackground=False,
            ),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
    )

    return fig



def layout():
    return html.Div(
        className="page-container",
        children=[
            html.Div(
                className="glass-card",
                children=[
                    html.H2("Market Price Topography"),
                    html.P(
                        "3D holographic view of average prices across time of day and year. "
                        "Rotate the model to spot structural high-price ridges.",
                        className="text-muted",
                        style={"fontSize": "0.9rem"},
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "flex-start",
                            "alignItems": "center",
                            "marginBottom": "0.75rem",
                            "gap": "0.75rem",
                        },
                        children=[
                            html.Span(
                                "Year filter:",
                                className="text-muted",
                                style={"fontSize": "0.85rem"},
                            ),
                            dcc.Dropdown(
                                id="surface-year-dropdown",
                                options=YEAR_OPTIONS,
                                value=DEFAULT_YEAR_VALUE,
                                clearable=False,
                                className="dropdown-dark",
                                style={"width": "220px"},
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="price-surface",
                        className="dash-graph",
                        style={"height": "75vh", "borderRadius": "16px"},
                        config={"displayModeBar": False},
                        figure=build_surface_figure(None),
                    ),
                ],
            ),
        ],
    )



@callback(
    Output("price-surface", "figure"),
    Input("surface-year-dropdown", "value"),
)
def update_surface(selected_year_value):
    if selected_year_value is None or selected_year_value == -1:
        year = None
    else:
        year = int(selected_year_value)
    return build_surface_figure(year)
