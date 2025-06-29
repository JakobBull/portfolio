from __future__ import annotations
import logging
from dash import Input, Output, State, callback, html, no_update
from datetime import datetime, date
from backend.database import TransactionType
from backend.controller import Controller

def register_callbacks(app, controller: Controller):
    """Register callbacks for the 'add transaction' section."""

    @app.callback(
        Output("ticker-input", "value"),
        Input("positions-table", "active_cell"),
        State("positions-table", "data"),
        prevent_initial_call=True
    )
    def update_ticker_input_from_table(active_cell, table_data):
        """Fills the ticker input when a user clicks on a row in the positions table."""
        if not active_cell or not table_data:
            return ""
        
        row_index = active_cell["row"]
        if row_index < len(table_data):
            ticker = table_data[row_index].get("ticker", "")
            return ticker
        return ""

    @callback(
        [Output('transaction-status', 'children'),
         Output('transaction-result-store', 'data')],
        Input('add-transaction-button', 'n_clicks'),
        [State('ticker-input', 'value'),
         State('transaction-type-input', 'value'),
         State('shares-input', 'value'),
         State('price-input', 'value'),
         State('cost-input', 'value'),
         State('transaction-date-input', 'date')],
        prevent_initial_call=True
    )
    def add_transaction(n_clicks, ticker, transaction_type, shares, price, cost, date_str):
        if not n_clicks:
            return "", None

        try:
            if not all([ticker, transaction_type, shares, price, date_str]):
                return html.Div("Please fill in all required fields.", className="alert alert-danger"), None
            
            shares = float(shares)
            price = float(price)
            cost = float(cost) if cost else 0.0
            transaction_date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d').date()

            success = controller.add_transaction(
                ticker=ticker.upper(),
                transaction_type=transaction_type,
                shares=shares,
                price=price,
                transaction_date=transaction_date,
                transaction_cost=cost
            )

            if success:
                msg = f"Transaction '{transaction_type.capitalize()} {shares} {ticker.upper()}' added successfully!"
                return html.Div(msg, className="alert alert-success"), {'status': 'success', 'ticker': ticker}
            else:
                msg = f"Failed to add transaction for {ticker.upper()}. Check ticker and logs."
                return html.Div(msg, className="alert alert-danger"), {'status': 'failed'}

        except ValueError as e:
            logging.error(f"Error adding transaction: {e}")
            return html.Div(f"Error: Invalid input. {e}", className="alert alert-danger"), None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return html.Div("An unexpected error occurred.", className="alert alert-danger"), None 