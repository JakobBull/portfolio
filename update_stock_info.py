from backend.database import db_manager
from backend.stock import Stock
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_stock_info():
    """Update all existing positions and watchlist items with name and sector information"""
    
    # Update portfolio positions
    positions = db_manager.get_all_positions()
    logger.info(f"Updating {len(positions)} portfolio positions")
    
    for pos in positions:
        ticker = pos['ticker']
        
        # Skip if already has name and sector
        if pos['name'] and pos['sector']:
            logger.info(f"Position {ticker} already has name and sector information")
            continue
            
        # Create stock object to get name and sector
        stock = Stock(ticker)
        stock_name = stock.get_name()
        stock_sector = stock.get_sector()
        
        if stock_name or stock_sector:
            # Update position in database
            db_manager.add_position(
                ticker,
                pos['amount'],
                pos['purchase_price'],
                pos['purchase_currency'],
                pos['purchase_date'],
                pos['cost_basis'],
                stock_name,
                stock_sector
            )
            logger.info(f"Updated position {ticker} with name: {stock_name}, sector: {stock_sector}")
        else:
            logger.warning(f"Could not get name and sector information for position {ticker}")
    
    # Update watchlist items
    watchlist = db_manager.get_all_watchlist_items()
    logger.info(f"Updating {len(watchlist)} watchlist items")
    
    for item in watchlist:
        ticker = item['ticker']
        
        # Skip if already has name and sector
        if item['name'] and item['sector']:
            logger.info(f"Watchlist item {ticker} already has name and sector information")
            continue
            
        # Create stock object to get name and sector
        stock = Stock(ticker)
        stock_name = stock.get_name()
        stock_sector = stock.get_sector()
        
        if stock_name or stock_sector:
            # Update watchlist item in database
            db_manager.add_watchlist_item(
                ticker,
                item['strike_price'],
                item['notes'],
                stock_name,
                stock_sector
            )
            logger.info(f"Updated watchlist item {ticker} with name: {stock_name}, sector: {stock_sector}")
        else:
            logger.warning(f"Could not get name and sector information for watchlist item {ticker}")

if __name__ == "__main__":
    update_stock_info()
    
    # Check the database after updating
    positions = db_manager.get_all_positions()
    print('\nUpdated portfolio positions:')
    for pos in positions:
        print(f"Ticker: {pos['ticker']}, Name: {pos['name']}, Sector: {pos['sector']}")
    
    watchlist = db_manager.get_all_watchlist_items()
    print('\nUpdated watchlist items:')
    for item in watchlist:
        print(f"Ticker: {item['ticker']}, Name: {item['name']}, Sector: {item['sector']}") 