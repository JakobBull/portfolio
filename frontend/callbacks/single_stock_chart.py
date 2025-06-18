import logging
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback
import dash
from datetime import date
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the single stock price chart."""

    @app.callback(
        Output("single-stock-selector", "options"),
        [
            Input("url", "pathname"),
            Input("interval-component", "n_intervals"),
        ],
    )
    def update_single_stock_selector_options(pathname, n_intervals):
        """Update single stock selector options with all stocks in database."""
        try:
            all_tickers = controller.get_all_stock_tickers()
            options = [{'label': ticker, 'value': ticker} for ticker in sorted(all_tickers)]
            return options
        except Exception as e:
            logging.error(f"Error updating single stock selector options: {e}")
            return []

    @app.callback(
        Output("single-stock-chart", "figure"),
        [
            Input("single-stock-selector", "value"),
            Input("stock-chart-date-range", "start_date"),
            Input("stock-chart-date-range", "end_date"),
        ],
    )
    def update_single_stock_chart(selected_ticker, start_date_str, end_date_str):
        """Update the single stock price chart."""
        if not selected_ticker:
            # Return empty chart with helpful message
            fig = go.Figure()
            fig.update_layout(
                title="Select a stock to view price history",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                showlegend=False,
                annotations=[
                    dict(
                        text="Please select a stock from the dropdown above",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, xanchor='center', yanchor='middle',
                        showarrow=False,
                        font=dict(size=16, color="gray")
                    )
                ]
            )
            return fig

        try:
            start_date = date.fromisoformat(start_date_str)
            end_date = date.fromisoformat(end_date_str)

            # Get historical stock data
            stock_series = controller.get_stock_historical_data(selected_ticker, start_date, end_date)
            
            if stock_series is None or stock_series.empty:
                # Return empty chart with error message
                fig = go.Figure()
                fig.update_layout(
                    title=f"No data available for {selected_ticker}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white",
                    showlegend=False,
                    annotations=[
                        dict(
                            text=f"No price data found for {selected_ticker} in the selected date range",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, xanchor='center', yanchor='middle',
                            showarrow=False,
                            font=dict(size=14, color="red")
                        )
                    ]
                )
                return fig

            # Create the chart with actual (unnormalized) prices
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=stock_series.index,
                    y=stock_series.values,
                    mode="lines",
                    name=selected_ticker,
                    line=dict(width=2, color='#2E86AB')
                )
            )

            # Get currency information for the y-axis label
            try:
                stock_info = controller.db_manager.get_stock(selected_ticker)
                currency = stock_info.currency if stock_info else 'USD'
                currency_symbol = '€' if currency == 'EUR' else ('£' if currency == 'GBP' else '$')
                yaxis_title = f"Price ({currency_symbol})"
            except:
                yaxis_title = "Price"

            fig.update_layout(
                title=f"{selected_ticker} - Historical Price",
                xaxis_title="Date",
                yaxis_title=yaxis_title,
                template="plotly_white",
                showlegend=False,
                hovermode='x unified'
            )
            
            # Format hover template
            fig.update_traces(
                hovertemplate=f'<b>{selected_ticker}</b><br>' +
                             'Date: %{x}<br>' +
                             'Price: %{y:.2f}<br>' +
                             '<extra></extra>'
            )

            return fig

        except Exception as e:
            logging.error(f"Error updating single stock chart: {e}")
            # Return error chart
            fig = go.Figure()
            fig.update_layout(
                title=f"Error loading data for {selected_ticker}",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                showlegend=False,
                annotations=[
                    dict(
                        text="An error occurred while loading the chart",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, xanchor='center', yanchor='middle',
                        showarrow=False,
                        font=dict(size=14, color="red")
                    )
                ]
            )
            return fig 