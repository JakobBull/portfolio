import logging
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, State, no_update, ctx
import dash
from datetime import date, datetime
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
            fig = go.Figure()
            fig.update_layout(title="Select a stock to view price history and fundamental analysis")
            return fig

        try:
            start_date = date.fromisoformat(start_date_str)
            end_date = date.fromisoformat(end_date_str)
            data_result = controller.get_stock_data_with_fundamental_values(selected_ticker, start_date, end_date)
            stock_series = data_result.get('stock_series')
            fundamental_series = data_result.get('fundamental_series')
            model_info = data_result.get('model_info', {})
            
            fig = go.Figure()
            if stock_series is None or stock_series.empty:
                fig.update_layout(title=f"No data available for {selected_ticker}")
                return fig

            fig.add_trace(go.Scatter(x=stock_series.index, y=stock_series.values, mode="lines", name=f"{selected_ticker} (Actual)"))
            if fundamental_series is not None and not fundamental_series.empty:
                fig.add_trace(go.Scatter(x=fundamental_series.index, y=fundamental_series.values, mode="lines", name=f"{selected_ticker} (Fundamental)", line=dict(dash='dash')))

            return fig
        except Exception as e:
            logging.error(f"Error updating single stock chart: {e}")
            fig = go.Figure()
            fig.update_layout(title=f"Error loading data for {selected_ticker}")
            return fig

    @app.callback(
        Output("earnings-table-container", "style"),
        Input("single-stock-selector", "value"),
    )
    def toggle_earnings_table_visibility(selected_ticker):
        return {'display': 'block'} if selected_ticker else {'display': 'none'}

    @app.callback(
        Output("earnings-table", "rowData"),
        Input("single-stock-selector", "value"),
        Input("save-earning-button", "n_clicks"),
    )
    def update_earnings_table(selected_ticker, save_clicks):
        if not selected_ticker:
            return []
        return controller.get_earnings_data_for_table(selected_ticker)

    @app.callback(
        [
            Output("earning-modal", "is_open"),
            Output("earning-modal-header", "children"),
            Output("earning-date-picker", "date"),
            Output("earning-eps-input", "value"),
            Output("earning-currency-dropdown", "value"),
            Output("earning-type-radios", "value"),
            Output("earning-store", "data"),
            Output("earnings-table-status", "children")
        ],
        [
            Input("add-earning-button", "n_clicks"),
            Input("save-earning-button", "n_clicks"),
            Input("cancel-earning-button", "n_clicks"),
        ],
        [
            State("single-stock-selector", "value"),
            State("earning-date-picker", "date"),
            State("earning-eps-input", "value"),
            State("earning-currency-dropdown", "value"),
            State("earning-type-radios", "value"),
            State("earning-store", "data"),
            State("earning-modal", "is_open"),
        ],
        prevent_initial_call=True,
    )
    def handle_earning_modal(add_clicks, save_clicks, cancel_clicks, ticker, new_date, new_eps, new_currency, new_type, earning_data, is_open):
        triggered_id = ctx.triggered_id
        
        if triggered_id == "add-earning-button":
            return True, "Add New Earning", date.today(), None, "USD", "quarterly", {}, no_update
            
        if triggered_id == "save-earning-button" and ticker:
            earning_id = (earning_data or {}).get('id')
            if earning_id: # Edit mode
                success = controller.update_earning_record(earning_id, date=new_date, eps=new_eps, currency=new_currency, type=new_type)
                status_message = "Earning updated successfully." if success else "Failed to update earning."
            else: # Add mode
                success = controller.add_earning(ticker, new_date, new_eps, new_type, new_currency)
                status_message = "Earning added successfully." if success else "Failed to add earning."
            return False, no_update, no_update, no_update, no_update, no_update, None, status_message
            
        if triggered_id == "cancel-earning-button":
            return False, no_update, no_update, no_update, no_update, no_update, None, ""
            
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update 

    @app.callback(
        Output("earnings-table-status", "children", allow_duplicate=True),
        Input("earnings-table", "cellValueChanged"),
        prevent_initial_call=True,
    )
    def save_inline_edit(changed_data):
        if not changed_data:
            return dash.no_update
        
        # The event is a list containing one dictionary for the changed cell
        event_data = changed_data[0]
        
        # The full data for the row is in the 'data' key
        row_data = event_data.get('data')
        
        if not row_data:
            return "Could not save change: Row data is missing from the event."

        earning_id = row_data.get('id')

        if not earning_id:
            return "Could not save change: missing earning ID."

        try:
            # The date from the grid will be a string, needs parsing
            if isinstance(row_data.get('date'), str):
                row_data['date'] = date.fromisoformat(row_data['date'].split('T')[0])

            # Pass the full dictionary of row data for the update
            success = controller.update_earning_record(earning_id, **row_data)

            if success:
                return f"Successfully updated earning {earning_id}."
            else:
                return f"Failed to update earning {earning_id}."
        except Exception as e:
            logging.error(f"Error saving inline edit for earning {earning_id}: {e}")
            return f"Error saving changes: {e}" 