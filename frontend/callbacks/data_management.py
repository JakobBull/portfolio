import logging
from dash import Input, Output, State, callback, html, ctx, ALL, MATCH
from datetime import date, datetime
from backend.controller import Controller
import json

logger = logging.getLogger(__name__)

def register_callbacks(app, controller: Controller):
    """Register callbacks for the data management section."""

    # Watchlist AgGrid callbacks
    @app.callback(
        Output("watchlist-aggrid", "rowData"),
        [Input("url", "pathname"), 
         Input("interval-component", "n_intervals"),
         Input("add-to-watchlist-button", "n_clicks"),
         Input("transaction-result-store", "data")],
        prevent_initial_call=False,
    )
    def update_watchlist_grid(pathname, n_intervals, add_clicks, transaction_result):
        """Update watchlist AgGrid data."""
        try:
            return controller.get_watchlist_data_for_table()
        except Exception as e:
            logger.error(f"Error updating watchlist grid: {e}")
            return []

    # Transactions AgGrid callbacks
    @app.callback(
        Output("transactions-aggrid", "rowData"),
        [Input("url", "pathname"), 
         Input("interval-component", "n_intervals"),
         Input("transaction-result-store", "data")],
        prevent_initial_call=False,
    )
    def update_transactions_grid(pathname, n_intervals, transaction_result):
        """Update transactions AgGrid data."""
        try:
            return controller.get_transactions_data_for_table()
        except Exception as e:
            logger.error(f"Error updating transactions grid: {e}")
            return []

    # Dividends AgGrid callbacks
    @app.callback(
        Output("dividends-aggrid", "rowData"),
        [Input("url", "pathname"), 
         Input("interval-component", "n_intervals"),
         Input("transaction-result-store", "data")],
        prevent_initial_call=False,
    )
    def update_dividends_grid(pathname, n_intervals, transaction_result):
        """Update dividends AgGrid data."""
        try:
            return controller.get_dividends_data_for_table()
        except Exception as e:
            logger.error(f"Error updating dividends grid: {e}")
            return []

    # Handle watchlist edits
    @app.callback(
        [Output("data-management-result", "children"),
         Output("watchlist-aggrid", "rowData", allow_duplicate=True)],
        Input("watchlist-aggrid", "cellValueChanged"),
        prevent_initial_call=True,
    )
    def handle_watchlist_edit(cell_changed):
        """Handle cell edits in the watchlist AgGrid."""
        if not cell_changed:
            return "", []
        
        try:
            for change in cell_changed:
                row_id = change['data']['id']
                field = change['colId']
                new_value = change['newValue']
                
                # Convert values if needed
                if field in ['strike_price'] and new_value is not None:
                    new_value = float(new_value)
                
                # Update in database
                success = controller.update_watchlist_item(row_id, **{field: new_value})
                
                if not success:
                    return html.Div("Failed to update watchlist item.", className="alert alert-danger"), []
            
            # Return updated data
            updated_data = controller.get_watchlist_data_for_table()
            return html.Div("Watchlist updated successfully.", className="alert alert-success"), updated_data
            
        except Exception as e:
            logger.error(f"Error handling watchlist edit: {e}")
            return html.Div(f"Error updating watchlist: {str(e)}", className="alert alert-danger"), []

    # Handle transaction edits
    @app.callback(
        [Output("data-management-result", "children", allow_duplicate=True),
         Output("transactions-aggrid", "rowData", allow_duplicate=True)],
        Input("transactions-aggrid", "cellValueChanged"),
        prevent_initial_call=True,
    )
    def handle_transaction_edit(cell_changed):
        """Handle cell edits in the transactions AgGrid."""
        if not cell_changed:
            return "", []
        
        try:
            for change in cell_changed:
                row_id = change['data']['id']
                field = change['colId']
                new_value = change['newValue']
                
                # Convert values if needed
                if field in ['amount', 'price', 'cost'] and new_value is not None:
                    new_value = float(new_value)
                elif field == 'date' and new_value is not None:
                    # Handle date conversion if needed
                    if isinstance(new_value, str):
                        new_value = new_value.split('T')[0]  # Remove time part if present
                
                # Update in database
                success = controller.update_transaction_record(row_id, **{field: new_value})
                
                if not success:
                    return html.Div("Failed to update transaction.", className="alert alert-danger"), []
            
            # Return updated data
            updated_data = controller.get_transactions_data_for_table()
            return html.Div("Transaction updated successfully.", className="alert alert-success"), updated_data
            
        except Exception as e:
            logger.error(f"Error handling transaction edit: {e}")
            return html.Div(f"Error updating transaction: {str(e)}", className="alert alert-danger"), []

    # Handle dividend edits
    @app.callback(
        [Output("data-management-result", "children", allow_duplicate=True),
         Output("dividends-aggrid", "rowData", allow_duplicate=True)],
        Input("dividends-aggrid", "cellValueChanged"),
        prevent_initial_call=True,
    )
    def handle_dividend_edit(cell_changed):
        """Handle cell edits in the dividends AgGrid."""
        if not cell_changed:
            return "", []
        
        try:
            for change in cell_changed:
                row_id = change['data']['id']
                field = change['colId']
                new_value = change['newValue']
                
                # Convert values if needed
                if field in ['amount_per_share', 'tax_withheld'] and new_value is not None:
                    new_value = float(new_value)
                elif field == 'date' and new_value is not None:
                    # Handle date conversion if needed
                    if isinstance(new_value, str):
                        new_value = new_value.split('T')[0]  # Remove time part if present
                
                # Update in database
                success = controller.update_dividend_record(row_id, **{field: new_value})
                
                if not success:
                    return html.Div("Failed to update dividend.", className="alert alert-danger"), []
            
            # Return updated data
            updated_data = controller.get_dividends_data_for_table()
            return html.Div("Dividend updated successfully.", className="alert alert-success"), updated_data
            
        except Exception as e:
            logger.error(f"Error handling dividend edit: {e}")
            return html.Div(f"Error updating dividend: {str(e)}", className="alert alert-danger"), []

    # Handle add to watchlist
    @app.callback(
        [Output("data-management-result", "children", allow_duplicate=True),
         Output("watchlist-aggrid", "rowData", allow_duplicate=True),
         Output("watchlist-ticker-input", "value"),
         Output("watchlist-target-price", "value")],
        Input("add-to-watchlist-button", "n_clicks"),
        [State("watchlist-ticker-input", "value"),
         State("watchlist-target-price", "value")],
        prevent_initial_call=True,
    )
    def handle_add_to_watchlist(n_clicks, ticker, target_price):
        """Handle adding a new item to the watchlist."""
        if not n_clicks or not ticker:
            return "", [], "", None
        
        try:
            success = controller.add_to_watchlist(
                ticker=ticker.upper(), 
                target_price=target_price, 
                date_added=date.today()
            )
            
            if success:
                updated_data = controller.get_watchlist_data_for_table()
                return (
                    html.Div(f"Added {ticker.upper()} to watchlist.", className="alert alert-success"),
                    updated_data,
                    "",  # Clear ticker input
                    None  # Clear target price input
                )
            else:
                return (
                    html.Div(f"Could not add {ticker.upper()}. Ensure it's a valid stock.", className="alert alert-danger"),
                    [],
                    ticker,
                    target_price
                )
                
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
            return (
                html.Div(f"An error occurred: {e}", className="alert alert-danger"),
                [],
                ticker,
                target_price
            ) 