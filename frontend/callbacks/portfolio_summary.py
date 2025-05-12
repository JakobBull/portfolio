import logging
from dash import Input, Output, callback
import dash_html_components as html

def register_callbacks(app, controller):
    """Register callbacks for portfolio summary section"""
    
    @app.callback(
        Output('summary-data-store', 'data'),
        [Input('url', 'pathname'),
         Input('interval-component', 'n_intervals')]  # Load on start and refresh periodically
    )
    def update_summary_data(pathname, n_intervals):
        """Update the summary data store with current portfolio metrics"""
        if n_intervals is not None and n_intervals > 0:
            logging.info(f"Refreshing summary data (interval {n_intervals})")
        else:
            logging.info("Initial load of summary data")
            
        # Default values in case of error
        default_metrics = {
            'total_value': 0.0,
            'cost_basis': 0.0,
            'unrealized_pl': 0.0,
            'unrealized_pl_percent': 0.0,
            'dividend_yield': 0.0,
            'dividend_income': 0.0
        }
        
        try:
            metrics = controller.get_performance_metrics()
            logging.info("Successfully retrieved performance metrics")
            return metrics
        except Exception as e:
            logging.error(f"Error retrieving performance metrics: {str(e)}")
            return default_metrics

    @app.callback(
        [Output('total-value', 'children'),
         Output('total-cost', 'children'),
         Output('unrealized-pl', 'children'),
         Output('return-pct', 'children'),
         Output('dividend-yield', 'children'),
         Output('dividend-income', 'children')],
        [Input('summary-data-store', 'data')]
    )
    def update_portfolio_summary(metrics):
        """Update the portfolio summary UI components with data from the store"""
        if not metrics:
            logging.warning("No metrics data available for summary")
            return "€ 0.00", "€ 0.00", "€ 0.00", "0.00%", "0.00%", "€ 0.00"
        
        logging.info("Updating portfolio summary display")
        
        # Format values
        total_value = f"€ {metrics['total_value']:.2f}"
        total_cost = f"€ {metrics['cost_basis']:.2f}"
        
        unrealized_pl = metrics['unrealized_pl']
        unrealized_pl_str = f"€ {unrealized_pl:.2f}"
        if unrealized_pl > 0:
            unrealized_pl_str = html.Span(unrealized_pl_str, style={'color': 'green'})
        elif unrealized_pl < 0:
            unrealized_pl_str = html.Span(unrealized_pl_str, style={'color': 'red'})
        
        return_pct = metrics['unrealized_pl_percent']
        return_pct_str = f"{return_pct:.2f}%"
        if return_pct > 0:
            return_pct_str = html.Span(return_pct_str, style={'color': 'green'})
        elif return_pct < 0:
            return_pct_str = html.Span(return_pct_str, style={'color': 'red'})
        
        dividend_yield = f"{metrics['dividend_yield']:.2f}%"
        dividend_income = f"€ {metrics['dividend_income']:.2f}"
        
        return total_value, total_cost, unrealized_pl_str, return_pct_str, dividend_yield, dividend_income