import logging
import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from datetime import datetime, date
from backend.stock import Stock
from backend.controller import Controller
def register_callbacks(app, controller: Controller):
    """Register callbacks for watchlist section"""
    
    @app.callback(
        Output('watchlist-data-store', 'data'),
        [Input('url', 'pathname'),
         Input('interval-component', 'n_intervals')]  # Load on start and refresh periodically
    )
    def update_watchlist_data(pathname, n_intervals):
        """Update the watchlist data store with current watchlist data"""
        if n_intervals is not None and n_intervals > 0:
            logging.info(f"Refreshing watchlist data (interval {n_intervals})")
        else:
            logging.info("Initial load of watchlist data")
            
        try:
            # Get watchlist items from database
            watchlist_items = controller.db_manager.get_all_watchlist_items()
            
            # Add current price for each item
            for item in watchlist_items:
                try:
                    # Get the latest price (in a real app, you'd fetch this from an API)
                    latest_price = controller.db_manager.get_last_known_price(item['ticker'])
                    item['current_price'] = latest_price if latest_price else Stock(item['ticker']).get_price(item['currency'], date.today())
                    
                    # Add action button for removing from watchlist
                    item['actions'] = f"[Remove](remove:{item['ticker']})"
                except Exception as e:
                    logging.error(f"Error getting price for {item['ticker']}: {str(e)}")
                    item['current_price'] = 0.0
            
            logging.info(f"Successfully retrieved {len(watchlist_items)} watchlist items")
            return watchlist_items
        except Exception as e:
            logging.error(f"Error retrieving watchlist data: {str(e)}")
            return []
    
    @app.callback(
        Output('watchlist-table', 'data'),
        [Input('watchlist-data-store', 'data')]
    )
    def update_watchlist_table(data):
        """Update the watchlist table with data from the store"""
        if data:
            logging.info(f"Updating watchlist table with {len(data)} items")
        else:
            logging.warning("No data available for watchlist table")
        return data or []
    
    @app.callback(
        Output('watchlist-result', 'children'),
        [Input('add-to-watchlist-button', 'n_clicks'),
         Input('watchlist-table', 'active_cell')],
        [State('watchlist-ticker-input', 'value'),
         State('watchlist-name-input', 'value'),
         State('watchlist-target-price', 'value'),
         State('watchlist-table', 'data')]
    )
    def manage_watchlist(n_clicks, active_cell, ticker, name, target_price, table_data):
        """Add to or remove from watchlist based on user action"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return None
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            if trigger_id == 'add-to-watchlist-button' and n_clicks:
                if not ticker:
                    return dbc.Alert("Please enter a ticker symbol", color="warning")
                
                ticker = ticker.strip().upper()
                
                # Create stock object to get real data
                try:
                    stock = Stock(ticker)
                    name = stock.get_name() or 'Unknown'
                    sector = stock.get_sector() or 'Unknown'
                    country = stock.get_country() or 'Unknown'
                    
                    # Add to watchlist
                    success = controller.db_manager.add_watchlist_item(
                        ticker=ticker,
                        strike_price=float(target_price) if target_price else None,
                        name=name,
                        sector=sector,
                        country=country
                    )
                    
                    if success:
                        return dbc.Alert(f"Added {ticker} to watchlist", color="success")
                    else:
                        return dbc.Alert(f"Failed to add {ticker} to watchlist", color="danger")
                except Exception as e:
                    logging.error(f"Error adding {ticker} to watchlist: {str(e)}")
                    return dbc.Alert(f"Error: {str(e)}", color="danger")
            
            elif trigger_id == 'watchlist-table' and active_cell:
                # Check if the click was on an action cell
                row = active_cell['row']
                col = active_cell['column_id']
                
                if col == 'actions' and row < len(table_data):
                    cell_value = table_data[row]['actions']
                    if cell_value.startswith('[Remove](remove:'):
                        # Extract ticker from the action link
                        ticker = cell_value.split('remove:')[1].rstrip(')')
                        
                        # Remove from watchlist
                        success = controller.db_manager.delete_watchlist_item(ticker)
                        
                        if success:
                            return dbc.Alert(f"Removed {ticker} from watchlist", color="success")
                        else:
                            return dbc.Alert(f"Failed to remove {ticker} from watchlist", color="danger")
            
            return None
        except Exception as e:
            logging.error(f"Error managing watchlist: {str(e)}")
            return dbc.Alert(f"Error: {str(e)}", color="danger") 