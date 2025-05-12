from datetime import date, timedelta
import logging
from backend.database import db_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_last_known_market_data(ticker, target_date):
    """
    Get the last known market data for a ticker before the target date
    
    Args:
        ticker: Stock ticker symbol
        target_date: The date for which we need data
        
    Returns:
        Dictionary with market data or None if not found
    """
    # Try to find the closest date with data
    closest_date_data = db_manager.find_closest_date_data("market_data", ticker, target_date)
    
    if closest_date_data:
        closest_date, _ = closest_date_data
        # Get the full market data for this date
        market_data = db_manager.get_market_data(ticker, closest_date)
        if market_data:
            logger.info(f"Found last known data for {ticker} on {closest_date}")
            return market_data
    
    logger.warning(f"No historical data found for {ticker} before {target_date}")
    return None

def add_missing_market_data(tickers, target_date):
    """
    Add market data for the target date using the last known data
    
    Args:
        tickers: List of ticker symbols
        target_date: The date for which to add data
        
    Returns:
        Number of tickers updated
    """
    updated_count = 0
    
    for ticker in tickers:
        # Check if data already exists for this date
        existing_data = db_manager.get_market_data(ticker, target_date)
        if existing_data:
            logger.info(f"Market data already exists for {ticker} on {target_date}")
            continue
        
        # Get the last known data
        last_known_data = get_last_known_market_data(ticker, target_date)
        if not last_known_data:
            logger.warning(f"No historical data found for {ticker}, skipping")
            continue
        
        # Add the data with the is_synthetic flag set to True
        success = db_manager.add_market_data(
            ticker=ticker,
            date=target_date,
            close_price=last_known_data['close_price'],
            currency=last_known_data['currency'],
            open_price=last_known_data['open_price'],
            high_price=last_known_data['high_price'],
            low_price=last_known_data['low_price'],
            volume=last_known_data['volume'],
            is_synthetic=True
        )
        
        if success:
            logger.info(f"Added synthetic market data for {ticker} on {target_date} using data from {last_known_data['date']}")
            updated_count += 1
        else:
            logger.error(f"Failed to add synthetic market data for {ticker} on {target_date}")
    
    return updated_count

if __name__ == "__main__":
    # List of tickers to update
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'ASC.L', '^IXIC', '^GSPC', '^GDAXI']
    
    # Target date (today)
    today = date.today()
    
    logger.info(f"Adding missing market data for {len(tickers)} tickers on {today}")
    updated_count = add_missing_market_data(tickers, today)
    logger.info(f"Added synthetic market data for {updated_count} tickers")
    
    # You can also add data for a range of dates
    # For example, to fill in the last 7 days:
    for i in range(1, 8):
        past_date = today - timedelta(days=i)
        logger.info(f"Adding missing market data for {len(tickers)} tickers on {past_date}")
        updated_count = add_missing_market_data(tickers, past_date)
        logger.info(f"Added synthetic market data for {updated_count} tickers on {past_date}") 