import logging
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback
import dash
from datetime import date
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the portfolio performance chart."""

    @app.callback(
        Output("performance-chart", "figure"),
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("benchmark-selector", "value"),
            Input("interval-component", "n_intervals"),
            Input("transaction-result-store", "data"),
        ],
    )
    def update_performance_chart(start_date_str, end_date_str, selected_benchmarks, n_intervals, transaction_result):
        """Fetches performance data and updates the chart."""
        if transaction_result and transaction_result.get('status') != 'success':
            return dash.no_update

        try:
            start_date = date.fromisoformat(start_date_str)
            end_date = date.fromisoformat(end_date_str)

            # --- Portfolio Data ---
            portfolio_hist = controller.get_historical_portfolio_value(start_date, end_date)
            portfolio_df = pd.DataFrame(
                list(portfolio_hist.items()), columns=["Date", "Value"]
            ).set_index("Date")
            
            # Normalize portfolio value
            portfolio_df["Normalized"] = (portfolio_df["Value"] / portfolio_df["Value"].iloc[0]) * 100

            # --- Chart Creation ---
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df["Normalized"],
                    mode="lines",
                    name="Portfolio",
                )
            )

            # --- Benchmark Data ---
            benchmark_map = {"NASDAQ": "^IXIC", "S&P 500": "^GSPC", "DAX 30": "^GDAXI"}
            for benchmark_name in selected_benchmarks:
                ticker = benchmark_map.get(benchmark_name)
                if ticker:
                    benchmark_series = controller.get_benchmark_data(ticker, start_date, end_date)
                    if benchmark_series is not None and not benchmark_series.empty:
                        # Normalize benchmark data
                        benchmark_series_normalized = (benchmark_series / benchmark_series.iloc[0]) * 100
                        fig.add_trace(
                            go.Scatter(
                                x=benchmark_series_normalized.index,
                                y=benchmark_series_normalized,
                                mode="lines",
                                name=benchmark_name,
                            )
                        )

            fig.update_layout(
                title="Portfolio Performance vs. Benchmarks",
                xaxis_title="Date",
                yaxis_title="Value (Normalized to 100)",
                template="plotly_white",
            )
            return fig

        except Exception as e:
            logging.error(f"Error updating performance chart: {e}")
            # Return an empty figure on error
            return go.Figure().update_layout(title="Error loading chart") 