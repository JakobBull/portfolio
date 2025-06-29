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
        """Update the single stock price chart with fundamental valuation model."""
        if not selected_ticker:
            # Return empty chart with helpful message
            fig = go.Figure()
            fig.update_layout(
                title="Select a stock to view price history and fundamental analysis",
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

            # Get stock data with fundamental values (this will fit the model)
            data_result = controller.get_stock_data_with_fundamental_values(selected_ticker, start_date, end_date)
            
            stock_series = data_result.get('stock_series')
            fundamental_series = data_result.get('fundamental_series')
            model_info = data_result.get('model_info', {})
            
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

            # Create the chart with actual prices
            fig = go.Figure()
            
            # Add actual stock price
            fig.add_trace(
                go.Scatter(
                    x=stock_series.index,
                    y=stock_series.values,
                    mode="lines",
                    name=f"{selected_ticker} (Actual)",
                    line=dict(width=2, color='#2E86AB'),
                    hovertemplate=f'<b>{selected_ticker} Actual</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Price: %{y:.2f}<br>' +
                                 '<extra></extra>'
                )
            )
            
            # Add fundamental value if available
            if fundamental_series is not None and not fundamental_series.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fundamental_series.index,
                        y=fundamental_series.values,
                        mode="lines",
                        name=f"{selected_ticker} (Fundamental)",
                        line=dict(width=2, color='#FF6B6B', dash='dash'),
                        hovertemplate=f'<b>{selected_ticker} Fundamental</b><br>' +
                                     'Date: %{x}<br>' +
                                     'Value: %{y:.2f}<br>' +
                                     '<extra></extra>'
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

            # Create title with model info
            title_parts = [f"{selected_ticker} - Price vs Fundamental Value"]
            if model_info.get('success'):
                r2_score = model_info.get('r2_score', 0)
                title_parts.append(f"(Model R² = {r2_score:.3f})")
            elif fundamental_series is None:
                title_parts.append("(Model: No Earnings Data)")
            else:
                title_parts.append("(Model: Failed to Fit)")
            
            title = " ".join(title_parts)

            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title=yaxis_title,
                template="plotly_white",
                showlegend=True,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Add annotation with model info if available
            annotations = []
            if model_info.get('success'):
                model_summary = f"""Model Parameters:
PE×d coefficient: {model_info.get('pe_d_coefficient', 0):.2e}
Earnings trend (γ): {model_info.get('gamma', 0):.2e}
Constant: {model_info.get('constant', 0):.2f}
Samples: {model_info.get('n_samples', 0)}"""
                
                annotations.append(
                    dict(
                        text=model_summary,
                        xref="paper", yref="paper",
                        x=0.02, y=0.7, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                )
            elif fundamental_series is None and not model_info.get('success'):
                error_msg = model_info.get('error', 'Unknown error')
                annotations.append(
                    dict(
                        text=f"Fundamental model could not be fitted:\n{error_msg}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.3, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="red"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red",
                        borderwidth=1
                    )
                )
            
            fig.update_layout(annotations=annotations)

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