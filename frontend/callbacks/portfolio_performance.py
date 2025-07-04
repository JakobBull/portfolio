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
        Output("stock-selector", "options"),
        [
            Input("url", "pathname"),
            Input("interval-component", "n_intervals"),
            Input("transaction-result-store", "data"),
        ],
    )
    def update_stock_selector_options(pathname, n_intervals, transaction_result):
        """Update stock selector options with current portfolio tickers."""
        try:
            current_tickers = controller.get_current_portfolio_tickers()
            options = [{'label': ticker, 'value': ticker} for ticker in sorted(current_tickers)]
            return options
        except Exception as e:
            logging.error(f"Error updating stock selector options: {e}")
            return []

    @app.callback(
        Output("performance-chart", "figure"),
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("benchmark-selector", "value"),
            Input("stock-selector", "value"),
            Input("interval-component", "n_intervals"),
            Input("transaction-result-store", "data"),
        ],
    )
    def update_performance_chart(start_date_str, end_date_str, selected_benchmarks, selected_stocks, n_intervals, transaction_result):
        """Fetches performance data and updates the chart."""
        if transaction_result and transaction_result.get('status') != 'success':
            return dash.no_update

        #try:
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)

        # --- Portfolio Data ---
        portfolio_perf = controller.get_portfolio_performance_twrr(start_date, end_date)
        
        fig = go.Figure()

        if not portfolio_perf.empty:
            portfolio_df = pd.DataFrame(portfolio_perf).rename(columns={"return_factor": "Normalized"})
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df["Normalized"],
                    mode="lines",
                    name="Portfolio",
                    line=dict(width=3)  # Make portfolio line slightly thicker
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
                            line=dict(dash='dash')  # Make benchmarks dashed
                        )
                    )

        # --- Individual Stock Data ---
        if selected_stocks:
            for stock_ticker in selected_stocks:
                stock_series = controller.get_stock_historical_data(stock_ticker, start_date, end_date)
                if stock_series is not None and not stock_series.empty:
                    # Normalize stock data
                    stock_series_normalized = (stock_series / stock_series.iloc[0]) * 100
                    fig.add_trace(
                        go.Scatter(
                            x=stock_series_normalized.index,
                            y=stock_series_normalized,
                            mode="lines",
                            name=stock_ticker,
                            line=dict(width=1, dash='dot')  # Make stocks thinner and dotted
                        )
                    )

        fig.update_layout(
            title="Portfolio Performance vs. Benchmarks & Individual Stocks",
            xaxis_title="Date",
            yaxis_title="Value (Normalized to 100)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig

        # except Exception as e:
        #     logging.error(f"Error updating performance chart: {e}")
        #     # Return an empty figure on error
        #     return go.Figure().update_layout(title="Error loading chart") 