import logging
import dash
from dash import Input, Output, callback, html
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the portfolio summary section."""

    @app.callback(
        [
            Output("total-value", "children"),
            Output("unrealized-pl", "children"),
            Output("dividend-yield", "children"),
            Output("total-cost", "children"),
            Output("return-pct", "children"),
            Output("dividend-income", "children"),
        ],
        [Input("url", "pathname"), 
         Input("interval-component", "n_intervals"),
         Input("transaction-result-store", "data")],
    )
    def update_portfolio_summary(pathname, n_intervals, transaction_result):
        """Fetches data and updates the portfolio summary card."""
        if transaction_result and transaction_result.get('status') != 'success':
            return [dash.no_update] * 6

        try:
            metrics = controller.get_performance_metrics()
            
            total_value = metrics.get('total_value', 0.0)
            unrealized_pl = metrics.get('unrealized_pl', 0.0)
            dividend_yield = metrics.get('dividend_yield', 0.0) * 100 # convert to percentage
            cost_basis = metrics.get('cost_basis', 0.0)
            return_pct = metrics.get('total_return_pct', 0.0)
            dividend_income = metrics.get('dividend_income', 0.0)

            # Formatting
            total_value_str = f"€ {total_value:.2f}"
            cost_basis_str = f"€ {cost_basis:.2f}"
            dividend_income_str = f"€ {dividend_income:.2f}"
            dividend_yield_str = f"{dividend_yield:.2f}%"

            pl_color = "green" if unrealized_pl >= 0 else "red"
            unrealized_pl_str = html.Span(f"€ {unrealized_pl:.2f}", style={"color": pl_color})
            
            return_pct_color = "green" if return_pct >= 0 else "red"
            return_pct_str = html.Span(f"{return_pct:.2f}%", style={"color": return_pct_color})

            return (
                total_value_str,
                unrealized_pl_str,
                dividend_yield_str,
                cost_basis_str,
                return_pct_str,
                dividend_income_str,
            )

        except Exception as e:
            logging.error(f"Error updating portfolio summary: {e}")
            return ("€ 0.00", "€ 0.00", "0.00%", "€ 0.00", "0.00%", "€ 0.00")