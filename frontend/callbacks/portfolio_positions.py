import logging
from dash import Input, Output, callback
import dash
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the portfolio positions table."""

    @app.callback(
        Output("positions-table", "data"),
        [Input("url", "pathname"), 
         Input("interval-component", "n_intervals"),
         Input("transaction-result-store", "data")],
    )
    def update_positions_table(pathname, n_intervals, transaction_result):
        """Fetches position data from the controller and updates the table."""
        if transaction_result and transaction_result.get('status') != 'success':
            return dash.no_update
            
        try:
            positions_data = controller.get_positions_data_for_table()
            return positions_data
        except Exception as e:
            logging.error(f"Error updating positions table: {e}")
            return []

# This function is kept for backward compatibility but is no longer used
def get_sector(ticker):
    """Get sector for a ticker (placeholder function)"""
    logging.warning(f"get_sector function called for {ticker} but is deprecated")
    return 'Unknown' 