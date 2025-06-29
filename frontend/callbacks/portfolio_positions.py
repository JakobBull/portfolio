import logging
from dash import Input, Output, callback, State
import dash
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the portfolio positions table."""

    @app.callback(
        Output("positions-table", "data"),
        [
            Input("url", "pathname"), 
            Input("interval-component", "n_intervals"),
            Input("transaction-status", "children") # Refresh when transaction is added
        ],
    )
    def update_positions_table(pathname, n_intervals, transaction_status):
        """Fetches position data from the controller and updates the table."""
        try:
            positions_data = controller.get_positions_data_for_table()
            return positions_data
        except Exception as e:
            logging.error(f"Error updating positions table: {e}")
            return []

    @app.callback(
        Output("positions-table", "data", allow_duplicate=True),
        Input("positions-table", "data_timestamp"),
        State("positions-table", "data"),
        prevent_initial_call=True
    )
    def update_target_price(timestamp, rows):
        """Handle target price updates when user edits the table."""
        if not rows:
            return dash.no_update
            
        try:
            # This assumes the controller can handle a list of updates
            # or we can iterate and call the controller for each updated row.
            for row in rows:
                ticker = row.get('ticker')
                target_price = row.get('target_price')
                
                # A simple check to see if target_price might be a valid number
                if ticker and target_price is not None:
                    try:
                        # Convert to float, this might fail if input is not a number
                        price_float = float(target_price)
                        controller.update_stock_target_price(ticker, price_float)
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid target price '{target_price}' for {ticker}. Skipping update.")
            
            return dash.no_update # Don't return rows to avoid callback loops
        except Exception as e:
            logging.error(f"Error updating target price: {e}")
            return dash.no_update

# This function is kept for backward compatibility but is no longer used
def get_sector(ticker):
    """Get sector for a ticker (placeholder function)"""
    logging.warning(f"get_sector function called for {ticker} but is deprecated")
    return 'Unknown' 
