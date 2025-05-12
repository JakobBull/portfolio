import logging
from dash import Input, Output, callback
from backend.stock import Stock

def register_callbacks(app, controller):
    """Register callbacks for portfolio positions section"""
    
    @app.callback(
        Output('positions-data-store', 'data'),
        [Input('url', 'pathname'),
         Input('interval-component', 'n_intervals')]  # Load on start and refresh periodically
    )
    def update_positions_data(pathname, n_intervals):
        """Update the positions data store with current portfolio data"""
        if n_intervals is not None and n_intervals > 0:
            logging.info(f"Refreshing positions data (interval {n_intervals})")
        else:
            logging.info("Initial load of positions data")
            
        try:
            df = controller.get_position_breakdown()
            logging.info("Successfully retrieved position breakdown")
            
            # Convert DataFrame to dict for JSON serialization
            if df is not None and not df.empty:
                # Get sector information from the database for each position
                position_data = df.to_dict('records')
                
                # Add name, sector, and country information from database
                for position in position_data:
                    ticker = position['ticker']
                    # Get position details from database
                    db_position = controller.db_manager.get_position(ticker)
                    if db_position:
                        # Add sector information
                        if db_position.get('sector') and 'sector' not in position:
                            position['sector'] = db_position['sector']
                        elif 'sector' not in position or not position['sector'] or position['sector'] == 'Unknown':
                            position['sector'] = db_position.get('sector', 'Unknown')
                            logging.info(f"Using sector information from database for {ticker}: {position['sector']}")
                        
                        # Add country information
                        if db_position.get('country') and 'country' not in position:
                            position['country'] = db_position['country']
                        elif 'country' not in position or not position['country'] or position['country'] == 'Unknown':
                            position['country'] = db_position.get('country', 'Unknown')
                            logging.info(f"Using country information from database for {ticker}: {position['country']}")
                        
                        # Add name information
                        if db_position.get('name') and 'name' not in position:
                            position['name'] = db_position['name']
                        elif 'name' not in position:
                            position['name'] = ticker
                            logging.info(f"No name information found for {ticker}, using ticker symbol")
                            
                        # Calculate current price per share if market value and shares are available
                        if 'market_value' in position and 'shares' in position and position['shares'] > 0:
                            position['current_price'] = position['market_value'] / position['shares']
                            logging.info(f"Calculated current price for {ticker}: {position['current_price']}")
                    else:
                        # If position not in database, use defaults
                        if 'sector' not in position:
                            position['sector'] = 'Unknown'
                        if 'country' not in position:
                            position['country'] = 'Unknown'
                        if 'name' not in position:
                            position['name'] = ticker
                        logging.info(f"No database information found for {ticker}, using defaults")
                    
                    # If country is still Unknown, try to get it from Stock class
                    if position.get('country') == 'Unknown':
                        try:
                            stock = Stock(ticker)
                            country = stock.get_country()
                            if country:
                                position['country'] = country
                                logging.info(f"Retrieved country for {ticker} from Stock class: {country}")
                        except Exception as e:
                            logging.warning(f"Error getting country for {ticker} from Stock class: {str(e)}")
                    
                    # Get target price from watchlist if available
                    try:
                        watchlist_item = controller.db_manager.get_watchlist_item(ticker)
                        if watchlist_item and watchlist_item.get('strike_price'):
                            position['target_price'] = watchlist_item['strike_price']
                            logging.info(f"Retrieved target price for {ticker} from watchlist: {position['target_price']}")
                    except Exception as e:
                        logging.warning(f"Error getting target price for {ticker} from watchlist: {str(e)}")
                
                return position_data
            else:
                # Return empty list with correct structure
                logging.warning("No position data available")
                return []
        except Exception as e:
            logging.error(f"Error retrieving position breakdown: {str(e)}")
            # Return empty list with correct structure
            return []

    @app.callback(
        Output('positions-table', 'data'),
        [Input('positions-data-store', 'data')]
    )
    def update_positions_table(data):
        """Update the positions table with data from the store"""
        if data:
            logging.info(f"Updating positions table with {len(data)} positions")
        else:
            logging.warning("No data available for positions table")
        return data or []

# This function is kept for backward compatibility but is no longer used
def get_sector(ticker):
    """Get sector for a ticker (placeholder function)"""
    logging.warning(f"get_sector function called for {ticker} but is deprecated")
    return 'Unknown' 