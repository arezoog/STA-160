import dash
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from functools import lru_cache

from model import load_master

#dash.register_page(__name__, path="/risk", name="Risk Analysis")

# --- Data preparation -----------------------------------------------------


@lru_cache(maxsize=1)
def get_risk_data():
    """
    Lazily load and preprocess the master EQR data used by the risk page.

    Runs once per process thanks to lru_cache; subsequent calls are free.
    Returns (df, price_col).
    """
    try:
        df = load_master()

        # Ensure datetime column
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            dt_cols = [
                c for c in df.columns if "date" in c.lower() or "time" in c.lower()
            ]
            if dt_cols:
                df["datetime"] = pd.to_datetime(df[dt_cols[0]], errors="coerce")
            else:
                df["datetime"] = pd.NaT

        df = df.dropna(subset=["datetime"])
        df["year"] = df["datetime"].dt.year
        df["month_num"] = df["datetime"].dt.month

        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        df["month_cat"] = pd.Categorical(
            df["month_num"].map(
                lambda m: month_labels[m - 1] if 1 <= m <= 12 else None
            ),
            categories=month_labels,
            ordered=True,
        )

        # Choose price column
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
        print(f"Error preparing risk data: {e}")
        return pd.DataFrame(), None


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


@lru_cache(maxsize=1)
def build_risk_heatmap():
    """
    Build the year vs month heatmap.

    - Uses mean price per (year, month_cat)
    - Fills months with no transactions as 0 so a tile still appears
    """
    df, price_col = get_risk_data()

    if df.empty or price_col is None:
        return _empty_fig()

    # Pivot: rows = year, columns = month_cat
    pivot_df = df.pivot_table(
        index="year",
        columns="month_cat",
        values=price_col,
        aggfunc="mean",
    ).sort_index(ascending=True)

    # Months with no transactions -> NaN; fill with 0 so we still draw a tile
    no_data_mask = pivot_df.isna()
    pivot_display = pivot_df.fillna(0)  # 0 == "no trades in this month"

    fig = px.imshow(
        pivot_display,
        aspect="auto",
        color_continuous_scale="Turbo",
        origin="lower",
    )
    fig.update_traces(
        hovertemplate="<b>%{x} %{y}</b><br>Avg price: $%{z:.2f}<extra></extra>"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title=None,
        yaxis=dict(title=None, type="category", dtick=1),
        coloraxis_colorbar=dict(title="Avg price"),
        autosize=True,
    )

    # OPTIONAL: if you want to explicitly mark no-data cells with an "×",
    # uncomment the block below.
    #
    # annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    # for year in pivot_df.index:
    #     for month in pivot_df.columns:
    #         if no_data_mask.loc[year, month]:
    #             annotations.append(
    #                 dict(
    #                     x=month,
    #                     y=year,
    #                     text="×",
    #                     showarrow=False,
    #                     font=dict(color="#9ca3af", size=10),
    #                 )
    #             )
    # fig.update_layout(annotations=annotations)

    return fig


# --- Layout ---------------------------------------------------------------


def layout():
    return html.Div(
        className="page-container",
        children=[
            html.Div(
                className="glass-card",
                style={"marginBottom": "16px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.H2("Stress grid – year vs month"),
                                    html.P(
                                        "Average price by month and year. Darker tiles indicate higher average prices. "
                                        "Months with no transactions are shown as 0-price tiles.",
                                        className="text-muted",
                                        style={"fontSize": "0.9rem"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="risk-heatmap",
                        className="dash-graph",
                        style={"height": "450px"},
                        figure=build_risk_heatmap(),
                    ),
                ],
            ),
        ],
    )
