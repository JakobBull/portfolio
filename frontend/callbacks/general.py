import logging
from dash import Input, Output, State, callback, html

def register_callbacks(app, controller):
    """Register general callbacks for the app"""
    
    @app.callback(
        Output('loading-output', 'children'),
        [Input('url', 'pathname'),
         Input('interval-component', 'n_intervals')],
        [State('positions-data-store', 'data'),
         State('summary-data-store', 'data'),
         State('performance-data-store', 'data')]
    )
    def update_loading_state(pathname, n_intervals, positions_data, summary_data, performance_data):
        """Update the loading state based on data availability"""
        # This callback doesn't actually change anything visible
        # It's just used to trigger the loading indicator
        if n_intervals is not None and n_intervals > 0:
            logging.info(f"Periodic refresh triggered (interval {n_intervals})")
        return ""

    @app.callback(
        [Output('global-error-display', 'children'),
         Output('global-error-display', 'style')],
        [Input('url', 'pathname'),
         Input('interval-component', 'n_intervals')],
        [State('positions-data-store', 'data'),
         State('summary-data-store', 'data'),
         State('performance-data-store', 'data'),
         State('search-results-store', 'data'),
         State('transaction-result-store', 'data'),
         State('tax-result-store', 'data')]
    )
    def update_error_display(pathname, n_intervals, positions_data, summary_data, performance_data, 
                           search_results, transaction_result, tax_result):
        """Update the global error display based on data store contents"""
        
        # Check each data store for errors
        error_messages = []
        
        # Helper function to check for errors in store data
        def check_store(data, store_name):
            if data and isinstance(data, dict) and 'error' in data:
                error_messages.append(f"{store_name}: {data['error']}")
        
        # Check each store
        check_store(positions_data, "Positions")
        check_store(summary_data, "Summary")
        check_store(performance_data, "Performance")
        check_store(search_results, "Search")
        
        # For stores with success/message format
        for store_data, store_name in [
            (transaction_result, "Transaction"), 
            (tax_result, "Tax Settings")
        ]:
            if store_data and isinstance(store_data, dict) and 'success' in store_data and not store_data['success']:
                error_messages.append(f"{store_name}: {store_data.get('message', 'Unknown error')}")
        
        if error_messages:
            return [
                html.Div([
                    html.H5("Errors:"),
                    html.Ul([html.Li(msg) for msg in error_messages])
                ]),
                {"display": "block"}
            ]
        else:
            return ["", {"display": "none"}] 