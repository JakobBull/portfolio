import logging
from dash import Input, Output, State, callback, html, ctx
from datetime import date
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the watchlist section."""

    @app.callback(
        Output("watchlist-table", "data", allow_duplicate=True),
        Output("watchlist-result", "children"),
        Input("add-to-watchlist-button", "n_clicks"),
        Input("add-transaction-button", "n_clicks"),
        State("watchlist-ticker-input", "value"),
        State("watchlist-target-price", "value"),
        State("ticker-input", "value"),
        prevent_initial_call=True,
    )
    def handle_watchlist_updates(add_watch_clicks, add_trans_clicks, watch_ticker, target_price, trans_ticker):
        """Adds a new stock to the watchlist and provides feedback."""
        triggered_id = ctx.triggered_id
        ticker_to_add = None
        msg = ""

        if triggered_id == "add-to-watchlist-button":
            ticker_to_add = watch_ticker
        elif triggered_id == "add-transaction-button":
            ticker_to_add = trans_ticker

        if ticker_to_add:
            try:
                if controller.add_to_watchlist(
                    ticker=ticker_to_add.upper(), 
                    target_price=target_price, 
                    date_added=date.today()
                ):
                    msg = html.Div(f"Added {ticker_to_add.upper()} to watchlist.", className="text-success")
                else:
                    msg = html.Div(f"Could not add {ticker_to_add.upper()}. Ensure it's a valid stock.", className="text-danger")
            except Exception as e:
                logging.error(f"Error adding to watchlist: {e}")
                msg = html.Div(f"An error occurred: {e}", className="text-danger")
        
        return controller.get_watchlist_data_for_table(), msg

    @app.callback(
        Output("watchlist-table", "data"),
        Input("url", "pathname"),
        Input("interval-component", "n_intervals"),
    )
    def refresh_watchlist_table(pathname, n_intervals):
        """Fetches watchlist data and updates the table on load and interval."""
        try:
            return controller.get_watchlist_data_for_table()
        except Exception as e:
            logging.error(f"Error loading watchlist table: {e}")
            return []

    # Note: A callback for removing from the watchlist would need a different UI trigger,
    # for instance, a button per row. This is a simplified implementation. 