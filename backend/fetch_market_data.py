#!/usr/bin/env python3
"""
Script to fetch market data for stocks in the portfolio and watchlist.
This script is designed to be run as a standalone script via a cron job.

Usage:
    python fetch_market_data.py [--days DAYS] [--verbose] [--update-portfolio] [--report] [--tickers TICKER1,TICKER2,...]
                               [--full-history] [--impute-missing]

Options:
    --days DAYS             Number of days of historical data to fetch (default: 7)
    --verbose               Enable verbose logging
    --update-portfolio      Update portfolio positions with latest prices
    --report                Generate a summary report
    --tickers TICKERS       Comma-separated list of tickers to update (default: all tickers in portfolio and watchlist)
    --full-history          Fetch full history since position was added (default for portfolio stocks)
    --impute-missing        Impute missing values for non-trading days (default: True)
"""

import os
import sys
import logging
import argparse
import datetime
from typing import List, Set, Dict, Any, Optional, Tuple
import time
import json
import pandas as pd

# Set up logging
script_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(script_dir, 'market_data_fetch.log'))
    ]
)
logger = logging.getLogger('fetch_market_data')

# Import our modules
try:
    # When running as a module
    from database import db_manager
    from market_interface import MarketInterface
except ImportError:
    # When running as a standalone script
    from backend.database import db_manager
    from backend.market_interface import MarketInterface

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fetch market data for stocks in portfolio and watchlist')
    parser.add_argument('--days', type=int, default=7, help='Number of days of historical data to fetch')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--update-portfolio', action='store_true', help='Update portfolio positions with latest prices')
    parser.add_argument('--report', action='store_true', help='Generate a summary report')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers to update (default: all tickers in portfolio and watchlist)')
    parser.add_argument('--full-history', action='store_true', help='Fetch full history since position was added (default for portfolio stocks)')
    parser.add_argument('--impute-missing', action='store_true', default=True, help='Impute missing values for non-trading days')
    return parser.parse_args()

def get_tickers_to_track(specific_tickers: Optional[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Get all tickers that should be tracked, separated by portfolio and watchlist
    
    Args:
        specific_tickers: Optional comma-separated list of tickers to track
        
    Returns:
        Tuple of (all tickers, portfolio tickers, watchlist tickers)
    """
    try:
        if specific_tickers:
            # Parse comma-separated list of tickers
            all_tickers = [ticker.strip().upper() for ticker in specific_tickers.split(',')]
            logger.info(f"Using {len(all_tickers)} specified tickers: {', '.join(all_tickers)}")
            
            # Get portfolio and watchlist tickers
            portfolio_positions = db_manager.get_all_positions()
            portfolio_tickers = [position['ticker'] for position in portfolio_positions]
            
            # Filter the specified tickers into portfolio and watchlist
            portfolio_tickers = [ticker for ticker in all_tickers if ticker in portfolio_tickers]
            watchlist_tickers = [ticker for ticker in all_tickers if ticker not in portfolio_tickers]
            
            return all_tickers, portfolio_tickers, watchlist_tickers
        else:
            # Get portfolio tickers
            portfolio_positions = db_manager.get_all_positions()
            portfolio_tickers = [position['ticker'] for position in portfolio_positions]
            logger.info(f"Found {len(portfolio_tickers)} tickers in portfolio")
            
            # Get watchlist tickers
            watchlist_items = db_manager.get_all_watchlist_items()
            watchlist_tickers = [item['ticker'] for item in watchlist_items]
            logger.info(f"Found {len(watchlist_tickers)} tickers in watchlist")
            
            # Combine all tickers (removing duplicates)
            all_tickers = list(set(portfolio_tickers + watchlist_tickers))
            logger.info(f"Total unique tickers to track: {len(all_tickers)}")
            
            return all_tickers, portfolio_tickers, watchlist_tickers
    except Exception as e:
        logger.error(f"Error getting tickers to track: {e}")
        return [], [], []

def get_earliest_portfolio_date() -> datetime.date:
    """
    Get the earliest date a position was added to the portfolio
    
    Returns:
        The earliest purchase date in the portfolio, or 30 days ago if no positions
    """
    try:
        # Get all positions
        positions = db_manager.get_all_positions()
        
        if not positions:
            # If no positions, return 30 days ago
            return datetime.date.today() - datetime.timedelta(days=30)
        
        # Get the earliest purchase date
        earliest_date = min(
            position['purchase_date'] for position in positions 
            if position['purchase_date'] is not None
        )
        
        logger.info(f"Earliest portfolio position date: {earliest_date}")
        return earliest_date
    except Exception as e:
        logger.error(f"Error getting earliest portfolio date: {e}")
        # Default to 30 days ago
        return datetime.date.today() - datetime.timedelta(days=30)

def get_position_purchase_date(ticker: str) -> Optional[datetime.date]:
    """
    Get the purchase date for a specific position
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        The purchase date or None if not found
    """
    try:
        position = db_manager.get_position(ticker)
        if position and position['purchase_date']:
            return position['purchase_date']
        return None
    except Exception as e:
        logger.error(f"Error getting purchase date for {ticker}: {e}")
        return None

def impute_missing_values(df):
    """
    Impute missing values in a DataFrame by forward filling.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    if df.empty:
        return df
    
    # Check if the DataFrame has a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        logger.debug("Using DatetimeIndex for imputation")
        # Create a complete date range
        date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        # Reindex the DataFrame with the complete date range
        df = df.reindex(date_range)
        # Forward fill missing values
        df = df.ffill()
        # Reset index to make date a column for consistent return format
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        # Convert timestamps to date objects
        df['date'] = df['date'].dt.date
        return df
    
    # Check if the index has a name 'date' but is not a DatetimeIndex
    if df.index.name == 'date':
        logger.debug("Converting index with name 'date' to DatetimeIndex for imputation")
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        # Now create a complete date range
        date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        # Reindex the DataFrame with the complete date range
        df = df.reindex(date_range)
        # Forward fill missing values
        df = df.ffill()
        # Reset index to make date a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        # Convert timestamps to date objects
        df['date'] = df['date'].dt.date
        return df
    
    # If no DatetimeIndex, check for a date column
    if 'date' in df.columns:
        logger.debug("Using 'date' column for imputation")
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df = df.set_index('date')
        
        # Create a complete date range
        date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        
        # Reindex the DataFrame with the complete date range
        df = df.reindex(date_range)
        
        # Forward fill missing values
        df = df.ffill()
        
        # Reset index to make date a column again
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        
        # Convert timestamps to date objects
        df['date'] = df['date'].dt.date
        
        return df
    
    logger.warning("No DatetimeIndex or date column found for imputation")
    return df

def fetch_market_data(tickers: List[str], portfolio_tickers: List[str], watchlist_tickers: List[str], 
                     days: int = 7, verbose: bool = False, full_history: bool = True, 
                     impute_missing: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Fetch market data for the given tickers
    
    Args:
        tickers: List of all ticker symbols
        portfolio_tickers: List of portfolio ticker symbols
        watchlist_tickers: List of watchlist ticker symbols
        days: Number of days of historical data to fetch (for watchlist tickers)
        verbose: Whether to enable verbose logging
        full_history: Whether to fetch full history for portfolio tickers
        impute_missing: Whether to impute missing values for non-trading days
        
    Returns:
        Dictionary mapping ticker symbols to their latest data
    """
    if not tickers:
        logger.warning("No tickers to fetch data for")
        return {}
    
    # Initialize market interface
    market_interface = MarketInterface()
    
    # Set logging level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get the earliest portfolio date
    earliest_portfolio_date = get_earliest_portfolio_date()
    
    # Calculate default date range (for watchlist tickers)
    end_date = datetime.date.today()
    default_start_date = end_date - datetime.timedelta(days=days)
    
    logger.info(f"Fetching market data for {len(tickers)} tickers")
    logger.info(f"Portfolio tickers: {len(portfolio_tickers)}")
    logger.info(f"Watchlist tickers: {len(watchlist_tickers)}")
    
    # Track success and failure counts
    success_count = 0
    failure_count = 0
    
    # Store latest data for each ticker
    latest_data = {}
    
    # Process each ticker
    for i, ticker in enumerate(tickers):
        try:
            logger.info(f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
            
            ticker_data = {
                'ticker': ticker,
                'fetch_date': end_date.isoformat(),
                'success': False
            }
            
            # Determine if this is a portfolio ticker
            is_portfolio_ticker = ticker in portfolio_tickers
            
            # Get current price
            current_price = market_interface.get_price(ticker, end_date)
            if current_price is not None:
                logger.info(f"Current price for {ticker}: {current_price}")
                ticker_data['current_price'] = current_price
                ticker_data['price_date'] = end_date.isoformat()
                ticker_data['success'] = True
            else:
                logger.warning(f"Failed to get current price for {ticker}")
                
                # Try to get the last known price
                last_price = db_manager.get_last_known_price(ticker)
                if last_price is not None:
                    logger.info(f"Using last known price for {ticker}: {last_price}")
                    ticker_data['current_price'] = last_price
                    ticker_data['price_date'] = 'last_known'
                    ticker_data['success'] = True
            
            # Determine the start date for historical data
            if is_portfolio_ticker and full_history:
                # For portfolio tickers, get data since purchase date
                purchase_date = get_position_purchase_date(ticker)
                if purchase_date:
                    start_date = purchase_date
                    logger.info(f"Fetching historical data for {ticker} since purchase date {start_date}")
                else:
                    # If no purchase date, use earliest portfolio date
                    start_date = earliest_portfolio_date
                    logger.info(f"No purchase date found for {ticker}, using earliest portfolio date {start_date}")
            else:
                # For watchlist tickers, use default date range
                start_date = default_start_date
                logger.info(f"Fetching recent historical data for {ticker} (last {days} days)")
            
            # Get historical prices
            historical_prices = market_interface.get_historical_prices(ticker, start_date, end_date)
            
            # Impute missing values if requested
            if impute_missing and not historical_prices.empty:
                logger.info(f"Imputing missing values for {ticker}")
                historical_prices = impute_missing_values(historical_prices)
            
            if not historical_prices.empty:
                logger.info(f"Retrieved {len(historical_prices)} historical price points for {ticker}")
                if verbose:
                    logger.debug(f"Historical prices for {ticker}:\n{historical_prices.head()}")
                
                # Store historical prices in the database
                if is_portfolio_ticker:
                    logger.info(f"Storing complete historical prices for portfolio ticker {ticker}")
                    try:
                        # Convert date column to datetime.date objects for database storage
                        historical_prices_for_db = historical_prices.copy()
                        if 'date' in historical_prices_for_db.columns:
                            # Check if the date column contains datetime objects
                            if pd.api.types.is_datetime64_any_dtype(historical_prices_for_db['date']):
                                # Convert datetime to date
                                historical_prices_for_db['date'] = historical_prices_for_db['date'].dt.date
                        db_manager.store_historical_prices(ticker, start_date, end_date, historical_prices_for_db)
                    except Exception as e:
                        # If the error is about SQLite Date type, log it but continue
                        if "SQLite Date type only accepts Python date objects as input" in str(e) or "int() argument must be a string" in str(e):
                            logger.warning(f"Known SQLAlchemy issue when storing historical prices for {ticker}. Continuing...")
                        else:
                            logger.error(f"Error storing historical prices for {ticker}: {e}")
                
                # Calculate price change
                if len(historical_prices) >= 2:
                    try:
                        # Handle different DataFrame formats
                        if 'Date' in historical_prices.index.names:
                            # Handle multi-index DataFrame
                            first_price = historical_prices['price'].iloc[0]
                            last_price = historical_prices['price'].iloc[-1]
                        else:
                            # Handle standard DataFrame
                            first_price = historical_prices['price'].iloc[0]
                            last_price = historical_prices['price'].iloc[-1]
                        
                        # Convert Series to float if needed
                        if hasattr(first_price, 'item'):
                            first_price = first_price.item()
                        else:
                            first_price = float(first_price)
                            
                        if hasattr(last_price, 'item'):
                            last_price = last_price.item()
                        else:
                            last_price = float(last_price)
                        
                        price_change = last_price - first_price
                        price_change_pct = (price_change / first_price) * 100 if first_price != 0 else 0
                        
                        ticker_data['first_price'] = float(first_price)
                        ticker_data['last_price'] = float(last_price)
                        ticker_data['price_change'] = float(price_change)
                        ticker_data['price_change_pct'] = float(price_change_pct)
                    except Exception as e:
                        logger.warning(f"Error calculating price change for {ticker}: {e}")
            else:
                logger.warning(f"No historical prices found for {ticker}")
            
            # Store ticker data
            latest_data[ticker] = ticker_data
            
            # Update success count
            success_count += 1
            
            # Add a small delay between tickers to avoid overwhelming the API
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            failure_count += 1
            latest_data[ticker] = {
                'ticker': ticker,
                'fetch_date': end_date.isoformat(),
                'success': False,
                'error': str(e)
            }
    
    logger.info(f"Market data fetch complete. Success: {success_count}, Failure: {failure_count}")
    
    # Save latest data to a JSON file
    try:
        with open(os.path.join(script_dir, 'latest_market_data.json'), 'w') as f:
            json.dump(latest_data, f, indent=2)
        logger.info("Saved latest market data to JSON file")
    except Exception as e:
        logger.error(f"Error saving latest market data to JSON file: {e}")
    
    return latest_data

def update_portfolio_positions(latest_data: Dict[str, Dict[str, Any]]):
    """
    Update portfolio positions with the latest prices
    
    Args:
        latest_data: Dictionary mapping ticker symbols to their latest data
    """
    logger.info("Updating portfolio positions with latest prices")
    
    # Get all positions
    positions = db_manager.get_all_positions()
    
    # Initialize market interface for currency conversion
    market_interface = MarketInterface()
    
    # Track success and failure counts
    success_count = 0
    failure_count = 0
    
    # Process each position
    for position in positions:
        ticker = position['ticker']
        
        try:
            # Check if we have latest data for this ticker
            if ticker in latest_data and latest_data[ticker]['success']:
                ticker_data = latest_data[ticker]
                
                # Get current price and date
                current_price = ticker_data.get('current_price')
                price_date_str = ticker_data.get('price_date')
                
                if current_price is not None:
                    # Convert price date string to date object
                    if price_date_str == 'last_known':
                        price_date = datetime.date.today()
                    else:
                        price_date = datetime.date.fromisoformat(price_date_str)
                    
                    # Calculate position value
                    amount = position['amount']
                    position_value = current_price * amount
                    
                    # Calculate unrealized P/L
                    purchase_price = position['purchase_price']
                    cost_basis = position['cost_basis'] or (purchase_price * amount)
                    unrealized_pl = position_value - cost_basis
                    
                    # Calculate return percentage
                    return_percentage = (unrealized_pl / cost_basis) * 100 if cost_basis != 0 else 0
                    
                    # Handle currency conversion if needed
                    value_currency = position['purchase_currency']  # Use same currency as purchase
                    
                    # Update position in database
                    db_manager.update_position_value(
                        ticker=ticker,
                        last_value=position_value,
                        value_currency=value_currency,
                        value_date=price_date,
                        unrealized_pl=unrealized_pl,
                        return_percentage=return_percentage
                    )
                    
                    logger.info(f"Updated position {ticker}: value={position_value:.2f} {value_currency}, P/L={unrealized_pl:.2f}, return={return_percentage:.2f}%")
                    success_count += 1
                else:
                    logger.warning(f"No current price available for {ticker}")
                    failure_count += 1
            else:
                logger.warning(f"No latest data available for {ticker}")
                failure_count += 1
                
        except Exception as e:
            logger.error(f"Error updating position {ticker}: {e}")
            failure_count += 1
    
    logger.info(f"Portfolio update complete. Success: {success_count}, Failure: {failure_count}")

def fetch_benchmark_data(days: int = 365, verbose: bool = False, impute_missing: bool = True):
    """
    Fetch data for benchmark indices
    
    Args:
        days: Number of days of historical data to fetch
        verbose: Whether to enable verbose logging
        impute_missing: Whether to impute missing values for non-trading days
    """
    logger.info("Fetching benchmark data")
    
    # Initialize market interface
    market_interface = MarketInterface()
    
    # Define benchmark tickers
    benchmark_tickers = ["^GSPC", "^IXIC", "^GDAXI"]  # S&P 500, NASDAQ, DAX 30
    
    # Get the earliest portfolio date
    earliest_portfolio_date = get_earliest_portfolio_date()
    
    # Calculate date range
    end_date = datetime.date.today()
    
    # Use earliest portfolio date or default days
    start_date = min(earliest_portfolio_date, end_date - datetime.timedelta(days=days))
    
    logger.info(f"Fetching benchmark data from {start_date} to {end_date}")
    
    # Process each benchmark
    for ticker in benchmark_tickers:
        try:
            logger.info(f"Processing benchmark {ticker}")
            
            # Get historical prices
            historical_prices = market_interface.get_historical_prices(ticker, start_date, end_date)
            
            # Impute missing values if requested
            if impute_missing and not historical_prices.empty:
                logger.info(f"Imputing missing values for benchmark {ticker}")
                historical_prices = impute_missing_values(historical_prices)
            
            if not historical_prices.empty:
                logger.info(f"Retrieved {len(historical_prices)} historical price points for benchmark {ticker}")
                
                # Store historical prices in the database
                try:
                    # Convert date column to datetime.date objects for database storage
                    historical_prices_for_db = historical_prices.copy()
                    if 'date' in historical_prices_for_db.columns:
                        # Check if the date column contains datetime objects
                        if pd.api.types.is_datetime64_any_dtype(historical_prices_for_db['date']):
                            # Convert datetime to date
                            historical_prices_for_db['date'] = historical_prices_for_db['date'].dt.date
                    db_manager.store_historical_prices(ticker, start_date, end_date, historical_prices_for_db)
                    logger.info(f"Stored historical prices for benchmark {ticker}")
                except Exception as e:
                    # If the error is about SQLite Date type, log it but continue
                    if "SQLite Date type only accepts Python date objects as input" in str(e) or "int() argument must be a string" in str(e):
                        logger.warning(f"Known SQLAlchemy issue when storing historical prices for {ticker}. Continuing...")
                    else:
                        logger.error(f"Error storing historical prices for benchmark {ticker}: {e}")
            else:
                logger.warning(f"No historical prices found for benchmark {ticker}")
            
            # Add a small delay to avoid overwhelming the API
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing benchmark {ticker}: {e}")
    
    logger.info("Benchmark data fetch complete")

def generate_report(latest_data: Dict[str, Dict[str, Any]]):
    """
    Generate a summary report of the latest market data
    
    Args:
        latest_data: Dictionary mapping ticker symbols to their latest data
    """
    logger.info("Generating market data summary report")
    
    # Get all positions
    positions = db_manager.get_all_positions()
    
    # Get all watchlist items
    watchlist_items = db_manager.get_all_watchlist_items()
    
    # Create report data
    report = {
        'generated_at': datetime.datetime.now().isoformat(),
        'portfolio': {
            'total_value': 0.0,
            'total_cost': 0.0,
            'total_pl': 0.0,
            'total_return_pct': 0.0,
            'positions': []
        },
        'watchlist': {
            'items': []
        }
    }
    
    # Process portfolio positions
    for position in positions:
        ticker = position['ticker']
        
        # Create position data
        position_data = {
            'ticker': ticker,
            'name': position['name'],
            'amount': position['amount'],
            'purchase_price': position['purchase_price'],
            'purchase_currency': position['purchase_currency'],
            'purchase_date': position['purchase_date'].isoformat() if position['purchase_date'] else None,
            'cost_basis': position['cost_basis'],
        }
        
        # Add latest price data if available
        if ticker in latest_data and latest_data[ticker]['success']:
            ticker_data = latest_data[ticker]
            position_data.update({
                'current_price': ticker_data.get('current_price'),
                'price_date': ticker_data.get('price_date'),
                'price_change': ticker_data.get('price_change'),
                'price_change_pct': ticker_data.get('price_change_pct')
            })
        
        # Add position value data if available
        if position['last_value'] is not None:
            position_data.update({
                'current_value': position['last_value'],
                'value_currency': position['last_value_currency'],
                'value_date': position['last_value_date'].isoformat() if position['last_value_date'] else None,
                'unrealized_pl': position['unrealized_pl'],
                'return_percentage': position['return_percentage']
            })
            
            # Update portfolio totals
            report['portfolio']['total_value'] += position['last_value']
            report['portfolio']['total_cost'] += position['cost_basis'] or 0
            report['portfolio']['total_pl'] += position['unrealized_pl'] or 0
        
        # Add position to report
        report['portfolio']['positions'].append(position_data)
    
    # Calculate total return percentage
    if report['portfolio']['total_cost'] > 0:
        report['portfolio']['total_return_pct'] = (report['portfolio']['total_pl'] / report['portfolio']['total_cost']) * 100
    
    # Process watchlist items
    for item in watchlist_items:
        ticker = item['ticker']
        
        # Create watchlist item data
        item_data = {
            'ticker': ticker,
            'name': item['name'],
            'sector': item['sector'],
            'strike_price': item['strike_price'],
            'date_added': item['date_added'].isoformat() if item['date_added'] else None,
            'notes': item['notes']
        }
        
        # Add latest price data if available
        if ticker in latest_data and latest_data[ticker]['success']:
            ticker_data = latest_data[ticker]
            item_data.update({
                'current_price': ticker_data.get('current_price'),
                'price_date': ticker_data.get('price_date'),
                'price_change': ticker_data.get('price_change'),
                'price_change_pct': ticker_data.get('price_change_pct')
            })
            
            # Check if price has reached strike price
            if item['strike_price'] is not None and ticker_data.get('current_price') is not None:
                item_data['reached_strike'] = ticker_data.get('current_price') <= item['strike_price']
        
        # Add item to report
        report['watchlist']['items'].append(item_data)
    
    # Save report to a JSON file
    try:
        report_path = os.path.join(script_dir, 'market_data_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved market data report to {report_path}")
        
        # Also save a timestamped copy
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(script_dir, f'reports/market_data_report_{timestamp}.json')
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.join(script_dir, 'reports'), exist_ok=True)
        
        with open(archive_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved archived market data report to {archive_path}")
        
        return report
    except Exception as e:
        logger.error(f"Error saving market data report: {e}")
        return None

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Get tickers to track
    all_tickers, portfolio_tickers, watchlist_tickers = get_tickers_to_track(args.tickers)
    
    # Fetch market data
    latest_data = fetch_market_data(
        all_tickers, 
        portfolio_tickers, 
        watchlist_tickers, 
        days=args.days, 
        verbose=args.verbose,
        full_history=args.full_history,
        impute_missing=args.impute_missing
    )
    
    # Fetch benchmark data
    fetch_benchmark_data(days=365, verbose=args.verbose, impute_missing=args.impute_missing)
    
    # Update portfolio positions if requested
    if args.update_portfolio:
        update_portfolio_positions(latest_data)
    
    # Generate report if requested
    if args.report:
        generate_report(latest_data)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1) 