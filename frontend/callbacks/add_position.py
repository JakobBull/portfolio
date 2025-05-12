import logging
import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from datetime import datetime
from backend.controller import Controller
from backend.stock import Stock

def register_callbacks(app, controller: Controller):
    """Register callbacks for add position section"""
    
    @app.callback(
        Output('search-results-store', 'data'),
        [Input('search-button', 'n_clicks')],
        [State('stock-search', 'value')]
    )
    def search_stocks(n_clicks, search_term):
        """Search for stocks based on user input and store results"""
        if not n_clicks or not search_term:
            return None
        
        try:
            # For simplicity, we'll use a predefined list of popular stocks
            # In a real app, you'd use a proper search API
            popular_stocks = {
                'AAPL': 'Apple Inc.',
                'MSFT': 'Microsoft Corporation',
                'AMZN': 'Amazon.com, Inc.',
                'GOOGL': 'Alphabet Inc.',
                'META': 'Meta Platforms, Inc.',
                'TSLA': 'Tesla, Inc.',
                'NVDA': 'NVIDIA Corporation',
                'JPM': 'JPMorgan Chase & Co.',
                'V': 'Visa Inc.',
                'JNJ': 'Johnson & Johnson',
                'WMT': 'Walmart Inc.',
                'PG': 'Procter & Gamble Co.',
                'MA': 'Mastercard Incorporated',
                'UNH': 'UnitedHealth Group Incorporated',
                'HD': 'The Home Depot, Inc.',
                'BAC': 'Bank of America Corporation',
                'XOM': 'Exxon Mobil Corporation',
                'DIS': 'The Walt Disney Company',
                'NFLX': 'Netflix, Inc.',
                'ADBE': 'Adobe Inc.'
            }
            
            # Filter based on search term
            filtered_stocks = {k: v for k, v in popular_stocks.items() 
                                if search_term.upper() in k or search_term.lower() in v.lower()}
            
            if not filtered_stocks:
                return {'error': 'No stocks found matching your search term.'}
            
            return {'stocks': filtered_stocks}
        
        except Exception as e:
            logging.error(f"Error searching for stocks: {str(e)}")
            return {'error': f"Error searching for stocks: {str(e)}"}

    @app.callback(
        Output('search-results', 'children'),
        [Input('search-results-store', 'data')]
    )
    def update_search_results(data):
        """Update the search results UI with data from the store"""
        if not data:
            return html.Div()
        
        if 'error' in data:
            return html.Div(data['error'], className="text-danger")
        
        search_results = []
        for ticker, name in data['stocks'].items():
            search_results.append(
                dbc.ListGroupItem(
                    [
                        html.Div(f"{ticker} - {name}"),
                        dbc.Button(
                            "Select",
                            id={"type": "select-stock", "ticker": ticker},
                            color="primary",
                            size="sm",
                            className="float-end"
                        )
                    ],
                    className="d-flex justify-content-between align-items-center"
                )
            )
        
        return dbc.ListGroup(search_results)

    @app.callback(
        Output('ticker-input', 'value'),
        [Input({'type': 'select-stock', 'ticker': dash.dependencies.ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def select_stock(n_clicks_list):
        """Handle stock selection from search results"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Get the ticker from the triggered button
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ticker = eval(button_id)['ticker']
        
        return ticker

    @app.callback(
        Output('transaction-result-store', 'data'),
        [Input('add-transaction-button', 'n_clicks')],
        [State('ticker-input', 'value'),
         State('transaction-type', 'value'),
         State('shares-input', 'value'),
         State('price-input', 'value'),
         State('currency-input', 'value'),
         State('date-input', 'date'),
         State('transaction-cost-input', 'value'),
         State('sell-target-input', 'value')]
    )
    def add_transaction(n_clicks, ticker, transaction_type, shares, price, currency, transaction_date, transaction_cost, sell_target):
        """Add a transaction to the portfolio"""
        if not n_clicks or not ticker or not transaction_type or not shares or not price or not currency or not transaction_date:
            return None
        
        try:
            # Convert inputs to appropriate types
            ticker = ticker.strip().upper()
            shares = float(shares)
            price = float(price)
            transaction_cost = float(transaction_cost) if transaction_cost is not None else 7.5
            transaction_date = datetime.strptime(transaction_date, '%Y-%m-%d').date()
            
            # Create stock object to get real data
            stock = Stock(ticker)
            name = stock.get_name() or 'Unknown'
            sector = stock.get_sector() or 'Unknown'
            country = stock.get_country() or 'Unknown'
            
            # Add transaction
            if transaction_type == 'buy':
                # For buy transactions, also update the position with sell target if provided
                success = controller.add_transaction(transaction_type, ticker, shares, price, currency, transaction_date, transaction_cost)
                
                # If sell target is provided, update the position
                if success and sell_target is not None and sell_target > 0:
                    # Add position with sell target (strike price)
                    controller.db_manager.add_watchlist_item(
                        ticker=ticker,
                        strike_price=float(sell_target),
                        name=name,
                        sector=sector,
                        country=country,
                        notes=f"Sell target for position added on {transaction_date}"
                    )
                    
                return {
                    'success': success,
                    'message': f"Successfully added {transaction_type} transaction for {shares} shares of {ticker} at {price} {currency}" +
                              (f" with sell target of {sell_target} {currency}" if sell_target else "") +
                              f" (Transaction cost: {transaction_cost} {currency})"
                }
            else:
                # For sell and dividend transactions
                success = controller.add_transaction(transaction_type, ticker, shares, price, currency, transaction_date, transaction_cost)
                return {
                    'success': success,
                    'message': f"Successfully added {transaction_type} transaction for {shares} shares of {ticker} at {price} {currency}" +
                              f" (Transaction cost: {transaction_cost} {currency})"
                }
        except Exception as e:
            logging.error(f"Error adding transaction: {str(e)}")
            return {
                'success': False,
                'message': f"Error adding transaction: {str(e)}"
            }

    @app.callback(
        Output('transaction-result', 'children'),
        [Input('transaction-result-store', 'data')]
    )
    def update_transaction_result(data):
        """Update the transaction result UI with data from the store"""
        if not data:
            return ""
        
        if data['success']:
            return html.Div(data['message'], className="text-success")
        else:
            return html.Div(data['message'], className="text-danger") 