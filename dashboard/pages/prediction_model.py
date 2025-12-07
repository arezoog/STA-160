# dashboard/pages/models.py

import dash
from dash import html

from . import dashboard_page
from . import scenarios

dash.register_page(
    __name__,
    path="/models",
    name="Forecast & Scenario",
)

def layout():
    return html.Div(
        children=[
            # Forecast Dashboard section
            dashboard_page.layout,

            # Scenario Lab section
            scenarios.layout,
        ]
    )
