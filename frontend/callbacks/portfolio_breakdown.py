import logging
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback

logger = logging.getLogger(__name__)

def register_callbacks(app, controller):
    """Register callbacks for portfolio breakdown charts."""

    @app.callback(
        [
            Output("sector-breakdown-chart", "figure"),
            Output("country-breakdown-chart", "figure"),
            Output("currency-breakdown-chart", "figure"),
        ],
        Input("positions-table", "data"),
    )
    def update_breakdown_charts(positions_data):
        """Updates the sector, country, and currency breakdown charts."""
        if not positions_data:
            empty_fig = {"layout": {"title": "No Data Available"}}
            return empty_fig, empty_fig, empty_fig

        try:
            df = pd.DataFrame(positions_data)

            # --- Sector Breakdown Chart ---
            sector_df = df.groupby("sector")["market_value"].sum().reset_index()
            sector_fig = px.pie(
                sector_df,
                values="market_value",
                names="sector",
                title="Portfolio by Sector",
                hole=0.4,
            )
            sector_fig.update_traces(textinfo="percent+label")

            # --- Country Breakdown Chart ---
            country_df = df.groupby("country")["market_value"].sum().reset_index()
            country_fig = px.pie(
                country_df,
                values="market_value",
                names="country",
                title="Portfolio by Country",
                hole=0.4,
            )
            country_fig.update_traces(textinfo="percent+label")

            # --- Currency Breakdown Chart ---
            currency_df = df.groupby("currency")["market_value"].sum().reset_index()
            currency_fig = px.pie(
                currency_df,
                values="market_value",
                names="currency",
                title="Portfolio by Currency",
                hole=0.4,
            )
            currency_fig.update_traces(textinfo="percent+label")

            return sector_fig, country_fig, currency_fig

        except Exception as e:
            logging.error(f"Error creating breakdown charts: {e}")
            error_fig = {"layout": {"title": "Error Loading Chart"}}
            return error_fig, error_fig, error_fig 