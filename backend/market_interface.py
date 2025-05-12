import datetime
import yfinance as yf
import pandas as pd
from typing import Dict, Optional, Union, Any, Tuple, List, cast, Callable
from functools import lru_cache, wraps
import logging
from requests.exceptions import RequestException
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from .database import db_manager
from .rate_limiter import execute_function, no_rate_limit
from .constants import (
    DEFAULT_EXCHANGE_RATES, 
    YFINANCE_MAX_REQUESTS, 
    YFINANCE_TIME_WINDOW,
    YFINANCE_BURST_LIMIT,
    YFINANCE_BURST_TIME,
    YFINANCE_MAX_RETRIES,
    YFINANCE_RETRY_DELAY,
    CACHE_EXPIRY_EXCHANGE_RATES,
    CACHE_EXPIRY_STOCK_PRICES,
    CACHE_EXPIRY_HISTORICAL,
    YFINANCE_BATCH_SIZE
)

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a decorator function that simply passes through to the function
def scheduled(func: Callable) -> Callable:
    """Decorator for functions that were previously scheduled"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            raise
    return wrapper

class MarketInterface:
    """
    Enhanced interface to market data via Yahoo Finance with robust error handling,
    rate limiting, and persistent caching
    """
    
    def __init__(self) -> None:
        """Initialize the market interface with database and API scheduler"""
        # Initialize database for persistent caching
        self.db = db_manager
        
        # In-memory cache for frequently accessed data
        self._exchange_rate_cache: Dict[Tuple[str, str, datetime.date], Tuple[float, datetime.datetime]] = {}
        self._stock_price_cache: Dict[Tuple[str, datetime.date], Tuple[Optional[float], datetime.datetime]] = {}
        self._historical_prices_cache: Dict[Tuple[str, datetime.date, datetime.date], Tuple[pd.DataFrame, datetime.datetime]] = {}
        
        logger.info("Market interface initialized with database and rate limiting")
        
    def convert_currency(self, amount: float, from_currency: str, to_currency: str, 
                        exchange_date: Optional[datetime.date] = None) -> float:
        """
        Convert an amount from one currency to another
        
        Args:
            amount: The amount to convert
            from_currency: Source currency code (e.g., 'USD')
            to_currency: Target currency code (e.g., 'EUR')
            exchange_date: Date for the exchange rate (defaults to today)
            
        Returns:
            The converted amount in the target currency
            
        Raises:
            ValueError: If invalid currency codes are provided
        """
        if not isinstance(amount, (int, float)):
            raise TypeError(f"Amount must be a number, got {type(amount)}")
            
        if amount < 0:
            raise ValueError("Amount cannot be negative")
            
        if not from_currency or not to_currency:
            raise ValueError("Currency codes cannot be empty")
            
        if exchange_date is None:
            exchange_date = datetime.date.today()
            
        # No conversion needed if currencies are the same
        if from_currency == to_currency:
            return amount
            
        # Get exchange rate
        try:
            rate = self.get_rate(from_currency, to_currency, exchange_date)
            return amount * rate
        except Exception as e:
            logger.warning(f"Error converting {amount} from {from_currency} to {to_currency}: {e}")
            # Fall back to default rate in case of error
            rate = self._get_default_rate(from_currency, to_currency)
            logger.info(f"Using fallback exchange rate: {rate}")
            return amount * rate
    
    def get_rate(self, from_currency: str, to_currency: str, date: datetime.date) -> float:
        """
        Get the exchange rate from one currency to another for a specific date
        
        Args:
            from_currency: The source currency code (e.g., 'USD')
            to_currency: The target currency code (e.g., 'EUR')
            date: The date for which to get the exchange rate
            
        Returns:
            The exchange rate as a float
        """
        # If currencies are the same, return 1.0
        if from_currency == to_currency:
            return 1.0
        
        # Check memory cache first
        cache_key = f"{from_currency}_{to_currency}_{date}"
        if cache_key in self._exchange_rate_cache:
            logger.debug(f"Exchange rate found in memory cache: {from_currency} to {to_currency} on {date}")
            return self._exchange_rate_cache[cache_key][0]
        
        # Check database cache
        db_rate = self.db.get_exchange_rate(from_currency, to_currency, date)
        if db_rate is not None:
            logger.debug(f"Exchange rate found in database cache: {from_currency} to {to_currency} on {date}")
            # Store in memory cache
            self._exchange_rate_cache[cache_key] = (db_rate, datetime.datetime.now())
            return db_rate
        
        # Try to fetch from API
        try:
            logger.info(f"Fetching exchange rate from API: {from_currency} to {to_currency} on {date}")
            rate = self._fetch_exchange_rate(from_currency, to_currency, date)
            
            # Store in database cache
            self.db.store_exchange_rate(from_currency, to_currency, date, rate)
            
            # Store in memory cache
            self._exchange_rate_cache[cache_key] = (rate, datetime.datetime.now())
            
            return rate
        except Exception as e:
            logger.warning(f"Error fetching exchange rate: {e}")
            
            # Try to find a close date in the database
            close_date_data = self.db.find_closest_date_data(
                "exchange_rates", 
                f"{from_currency}_{to_currency}", 
                date
            )
            
            if close_date_data:
                close_date, close_rate = close_date_data
                logger.info(f"Using exchange rate from close date {close_date}")
                return close_rate
            
            # Fall back to default rates
            logger.info(f"Using default exchange rate for {from_currency} to {to_currency}")
            return self._get_default_rate(from_currency, to_currency)
    
    @scheduled
    def _fetch_exchange_rate(self, from_currency: str, to_currency: str, 
                           date: datetime.date) -> float:
        """
        Fetch exchange rate from Yahoo Finance
        
        This method is decorated with the scheduler to ensure rate limiting
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Date for the exchange rate (should already be adjusted to trading day)
            
        Returns:
            The exchange rate
            
        Raises:
            Exception: If the API request fails
        """
        # Check API status before making the call
        api_status = self.db.get_api_status('exchange_rate')
        if api_status['status'] == 'down':
            logger.warning(f"Exchange rate API is down, using default rate for {from_currency}/{to_currency}")
            raise ValueError(f"Exchange rate API is down")
        
        if api_status['is_in_cooldown']:
            cooldown_until = datetime.datetime.fromisoformat(api_status['cooldown_until'])
            logger.warning(f"Exchange rate API is in cooldown until {cooldown_until}, using default rate")
            raise ValueError(f"Exchange rate API is in cooldown")
        
        # Construct ticker for currency pair
        ticker = f"{from_currency}{to_currency}=X"
        
        try:
            # Get data from Yahoo Finance
            data = yf.download(ticker, start=date, end=date + datetime.timedelta(days=5), 
                             progress=False, timeout=10)
            
            # If no data for exact date, get closest date
            if data.empty:
                # Try getting historical data for a wider range
                end_date = min(date + datetime.timedelta(days=7), datetime.date.today())
                data = yf.download(ticker, start=date - datetime.timedelta(days=7), 
                                 end=end_date, progress=False, timeout=10)
                
                if data.empty:
                    # If still no data, use a default rate
                    raise ValueError(f"No exchange rate data found for {from_currency}/{to_currency}")
                else:
                    # Use the closest date - handle different data structures
                    close_value = data['Close'].iloc[-1]
                    # Extract the value safely
                    if hasattr(close_value, 'item'):
                        rate = float(close_value.item())
                    elif isinstance(close_value, (pd.Series, pd.DataFrame)):
                        # If it's a Series or DataFrame, get the first value
                        rate = float(close_value.iloc[0])
                    else:
                        # If it's already a scalar
                        rate = float(close_value)
                    
                    logger.info(f"Using closest available rate from {data.index[-1].date()} for {from_currency}/{to_currency}")
            else:
                # Use the rate from the requested date - handle different data structures
                close_value = data['Close'].iloc[0]
                # Extract the value safely
                if hasattr(close_value, 'item'):
                    rate = float(close_value.item())
                elif isinstance(close_value, (pd.Series, pd.DataFrame)):
                    # If it's a Series or DataFrame, get the first value
                    rate = float(close_value.iloc[0])
                else:
                    # If it's already a scalar
                    rate = float(close_value)
                
            return rate
        except Exception as e:
            error_msg = str(e)
            # Check if this is a rate limit error
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                # Update API status to rate limited
                self.db.log_api_request("exchange_rate", False, f"Rate limit exceeded: {error_msg}")
                logger.warning(f"Exchange rate API rate limit exceeded, cooling down")
            else:
                # Log the error
                self.db.log_api_request("exchange_rate", False, error_msg)
                logger.error(f"Error fetching exchange rate: {error_msg}")
            
            # Re-raise the exception to be handled by the caller
            raise
    
    def _get_default_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get default exchange rate for currency pairs
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            A default exchange rate
        """
        # Check if we have a direct rate
        key = f"{from_currency}_{to_currency}"
        if key in DEFAULT_EXCHANGE_RATES:
            return DEFAULT_EXCHANGE_RATES[key]
        
        # Try to calculate via USD as a bridge
        usd_from_key = f"{from_currency}_USD"
        usd_to_key = f"USD_{to_currency}"
        
        if usd_from_key in DEFAULT_EXCHANGE_RATES and usd_to_key in DEFAULT_EXCHANGE_RATES:
            # Convert via USD
            return DEFAULT_EXCHANGE_RATES[usd_from_key] * DEFAULT_EXCHANGE_RATES[usd_to_key]
        
        # Try to calculate via EUR as a bridge
        eur_from_key = f"{from_currency}_EUR"
        eur_to_key = f"EUR_{to_currency}"
        
        if eur_from_key in DEFAULT_EXCHANGE_RATES and eur_to_key in DEFAULT_EXCHANGE_RATES:
            # Convert via EUR
            return DEFAULT_EXCHANGE_RATES[eur_from_key] * DEFAULT_EXCHANGE_RATES[eur_to_key]
        
        # If all else fails, return 1.0 (not ideal but prevents errors)
        logger.warning(f"No default rate found for {from_currency}/{to_currency}, using 1.0")
        return 1.0
    
    def get_price(self, ticker: str, date: datetime.date) -> Optional[float]:
        """
        Get the stock price for a specific ticker on a specific date
        
        Args:
            ticker: The stock ticker symbol
            date: The date for which to get the price
            
        Returns:
            The stock price as a float, or None if not available
        """
        # Check memory cache first
        cache_key = f"{ticker}_{date}"
        if cache_key in self._stock_price_cache:
            logger.debug(f"Stock price found in memory cache: {ticker} on {date}")
            return self._stock_price_cache[cache_key][0]
        
        # Check database cache
        db_price = self.db.get_stock_price(ticker, date)
        if db_price is not None:
            logger.debug(f"Stock price found in database cache: {ticker} on {date}")
            # Store in memory cache
            self._stock_price_cache[cache_key] = (db_price, datetime.datetime.now())
            return db_price
        
        # Try to find a close date in the database
        close_date_data = self.db.find_closest_date_data(
            "market_data", 
            ticker, 
            date
        )
        
        if close_date_data:
            # Unpack the tuple returned by find_closest_date_data
            if isinstance(close_date_data, tuple) and len(close_date_data) == 2:
                close_date, close_price = close_date_data
                logger.info(f"Using stock price from close date {close_date} for {ticker}")
                # Store in memory cache
                self._stock_price_cache[cache_key] = (close_price, datetime.datetime.now())
                return close_price
        
        # If not found in database, try to fetch from Yahoo Finance API
        try:
            logger.info(f"Fetching price from Yahoo Finance API for {ticker} on {date}")
            
            # Adjust date to trading day
            adjusted_date = self._adjust_to_trading_day(date)
            if adjusted_date != date:
                logger.info(f"Adjusted date from {date} to last trading day {adjusted_date}")
            
            # Add a small buffer to ensure we get data
            buffer_end_date = adjusted_date + datetime.timedelta(days=5)
            
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=adjusted_date, end=buffer_end_date, progress=False, timeout=15)
            
            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance API for {ticker}")
                return None
            
            # Get the first available price - handle different data structures robustly
            close_value = data['Close'].iloc[0]
            # Extract the value safely
            if hasattr(close_value, 'item'):
                price = float(close_value.item())
            elif isinstance(close_value, (pd.Series, pd.DataFrame)):
                # If it's a Series or DataFrame, get the first value
                price = float(close_value.iloc[0])
            else:
                # If it's already a scalar
                price = float(close_value)
            
            # Store in database
            self.db.store_stock_price(ticker, adjusted_date, price)
            
            # Store in memory cache
            self._stock_price_cache[cache_key] = (price, datetime.datetime.now())
            
            return price
        except Exception as e:
            logger.error(f"Error fetching stock price from API: {e}")
            return None
    
    @scheduled
    def _fetch_stock_price(self, ticker: str, date: datetime.date) -> Optional[float]:
        """
        Fetch stock price from database only
        
        Args:
            ticker: Stock ticker symbol
            date: Date to get price for
            
        Returns:
            Stock price or None if not available
        """
        logger.debug(f"Fetching price for {ticker} on {date} from database only")
        
        # This method is kept for backward compatibility but should not be used
        # to fetch data from Yahoo Finance anymore
        return self.db.get_stock_price(ticker, date)
    
    def _adjust_to_trading_day(self, date: datetime.date) -> datetime.date:
        """
        Adjust a date to the last trading day if it falls on a weekend or holiday
        
        Args:
            date: The date to adjust
            
        Returns:
            The adjusted date (same date if it's a trading day, or the last trading day before it)
        """
        # Check if the date is a weekend
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            # Adjust to the previous Friday
            days_to_subtract = date.weekday() - 4  # 5 -> 1, 6 -> 2
            date = date - datetime.timedelta(days=days_to_subtract)
            
        # TODO: Add holiday checking if a calendar of market holidays is available
        
        return date

    def get_historical_prices(self, ticker: str, start_date: datetime.date, 
                            end_date: Optional[datetime.date] = None) -> pd.DataFrame:
        """
        Get historical prices for a ticker over a date range
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)
            
        Returns:
            DataFrame with dates as index and 'price' column with closing prices
        """
        if not ticker:
            raise ValueError("Ticker cannot be empty")
            
        if end_date is None:
            end_date = datetime.date.today()
        
        # Adjust dates to trading days
        adjusted_start_date = self._adjust_to_trading_day(start_date)
        adjusted_end_date = self._adjust_to_trading_day(end_date)
        
        if adjusted_start_date != start_date:
            logger.info(f"Adjusted start date from {start_date} to last trading day {adjusted_start_date}")
            
        if adjusted_end_date != end_date:
            logger.info(f"Adjusted end date from {end_date} to last trading day {adjusted_end_date}")
            
        if adjusted_start_date > adjusted_end_date:
            raise ValueError(f"Start date ({adjusted_start_date}) cannot be after end date ({adjusted_end_date})")
            
        # Check database cache
        db_prices = self.db.get_historical_prices(ticker, adjusted_start_date, adjusted_end_date)
        if db_prices is not None and not db_prices.empty:
            return db_prices
            
        # Check API status before attempting to fetch
        api_status = self.db.get_api_status('historical_prices')
        if api_status['status'] == 'down' or api_status['is_in_cooldown']:
            logger.warning(f"Historical prices API is unavailable, building from cache for {ticker}")
            
            # Try to build historical prices from individual price points
            result = self._build_historical_prices_from_cache(ticker, adjusted_start_date, adjusted_end_date)
            
            # If we have some data, return it
            if not result.empty and not result['price'].isna().all():
                logger.info(f"Successfully built historical prices from cache for {ticker}")
                return result
            
            # If we couldn't build from cache, try to find partial historical data
            partial_data = self.db.get_historical_prices(ticker, adjusted_start_date, adjusted_end_date, partial_match=True)
            if partial_data is not None and not partial_data.empty:
                logger.info(f"Using partial historical data for {ticker}")
                return partial_data
            
            # If all else fails, return empty DataFrame
            logger.warning(f"No historical price data found for {ticker} from {adjusted_start_date} to {adjusted_end_date}")
            return pd.DataFrame(columns=['price'])
        
        # If not in cache and API is available, fetch from API
        try:
            # Use the scheduler to respect rate limits
            data = self._fetch_historical_prices(ticker, adjusted_start_date, adjusted_end_date)
            
            if not data.empty:
                # Store in database
                self.db.store_historical_prices(ticker, adjusted_start_date, adjusted_end_date, data)
                
                # Log API request
                self.db.log_api_request("historical_prices", True)
            
            return data
            
        except Exception as e:
            # Log failed API request
            self.db.log_api_request("historical_prices", False, str(e))
            
            # Try to build historical prices from individual price points
            logger.info(f"Building historical prices for {ticker} from individual price points")
            result = self._build_historical_prices_from_cache(ticker, adjusted_start_date, adjusted_end_date)
            
            # If we have some data, return it
            if not result.empty and not result['price'].isna().all():
                return result
            
            # If we couldn't build from cache, try to find partial historical data
            partial_data = self.db.get_historical_prices(ticker, adjusted_start_date, adjusted_end_date, partial_match=True)
            if partial_data is not None and not partial_data.empty:
                logger.info(f"Using partial historical data for {ticker}")
                return partial_data
            
            # If all else fails, return empty DataFrame
            logger.warning(f"No historical price data found for {ticker} from {adjusted_start_date} to {adjusted_end_date}")
            return pd.DataFrame(columns=['price'])
    
    @scheduled
    def _fetch_historical_prices(self, ticker: str, start_date: datetime.date, 
                               end_date: datetime.date) -> pd.DataFrame:
        """
        Fetch historical prices from Yahoo Finance API
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical prices
        """
        logger.debug(f"Fetching historical prices for {ticker} from {start_date} to {end_date} from Yahoo Finance API")
        
        try:
            # Add a small buffer to the end date to ensure we get data for the end date
            buffer_end_date = end_date + datetime.timedelta(days=5)
            
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=buffer_end_date, progress=False, timeout=15)
            
            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance API for {ticker}")
                return pd.DataFrame(columns=['date', 'price'])
            
            # Create a DataFrame with date and price columns
            try:
                # Handle the case where data['Close'] might be multi-dimensional
                if isinstance(data['Close'], pd.Series):
                    # If it's a Series, we can use it directly
                    result = pd.DataFrame({
                        'price': data['Close'].values
                    }, index=data.index)
                else:
                    # If it's a DataFrame or another structure, convert to Series first
                    close_series = pd.Series(data['Close'])
                    result = pd.DataFrame({
                        'price': close_series.values
                    }, index=data.index)
            except Exception as e:
                logger.error(f"Error processing Yahoo Finance data: {e}")
                # Try an alternative approach
                result_data = []
                for idx, row in data.iterrows():
                    # Extract the price value as a float
                    if hasattr(row['Close'], 'item'):
                        price = row['Close'].item()
                    else:
                        price = float(row['Close'])
                        
                    result_data.append({
                        'date': idx.date(),
                        'price': price
                    })
                result = pd.DataFrame(result_data)
                # Set date as index
                if 'date' in result.columns:
                    result['date'] = pd.to_datetime(result['date'])
                    result = result.set_index('date')
            
            # Filter to only include dates within our requested range
            result = result[result.index >= pd.Timestamp(start_date)]
            result = result[result.index <= pd.Timestamp(end_date)]
            
            if result.empty:
                logger.warning(f"No data within requested date range for {ticker}")
                return pd.DataFrame(columns=['price'])
            
            logger.info(f"Successfully fetched {len(result)} historical prices for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical prices from Yahoo Finance API: {e}")
            # If API call fails, return empty DataFrame
            return pd.DataFrame(columns=['price'])
    
    def _build_historical_prices_from_cache(self, ticker: str, start_date: datetime.date, 
                                         end_date: datetime.date) -> pd.DataFrame:
        """
        Build a historical prices DataFrame from individual price points
        
        Args:
            ticker: The stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical prices
        """
        # Create a date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create an empty DataFrame
        prices = []
        dates = []
        
        # Get price for each date
        for date in date_range:
            price = self.get_price(ticker, date.date())
            prices.append(price)
            dates.append(date.date())  # Store as date objects, not datetime objects
        
        # Create DataFrame with date column
        df = pd.DataFrame({'date': dates, 'price': prices})
        
        # If all prices are None, return an empty DataFrame
        if df['price'].isna().all():
            return pd.DataFrame(columns=['date', 'price'])
            
        return df
    
    def prefetch_data(self, tickers: List[str], start_date: datetime.date, 
                    end_date: Optional[datetime.date] = None) -> None:
        """
        Prefetch data for multiple tickers to populate cache
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)
        """
        if end_date is None:
            end_date = datetime.date.today()
            
        logger.info(f"Prefetching data for {len(tickers)} tickers")
        
        # Process tickers in batches to avoid overwhelming the API
        for i in range(0, len(tickers), YFINANCE_BATCH_SIZE):
            batch = tickers[i:i+YFINANCE_BATCH_SIZE]
            
            # Process each ticker sequentially instead of in parallel
            for ticker in batch:
                try:
                    # Fetch historical prices
                    self.get_historical_prices(ticker, start_date, end_date)
                    
                    # Fetch current price
                    self.get_price(ticker, end_date)
                    
                    logger.debug(f"Prefetched data for {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to prefetch data for {ticker}: {str(e)}")
                    
            # Small delay between batches to avoid rate limiting
            time.sleep(1)
            
        logger.info("Prefetching complete")
    
    def clear_cache(self) -> None:
        """Clear in-memory cache"""
        self._exchange_rate_cache.clear()
        self._stock_price_cache.clear()
        self._historical_prices_cache.clear()
        logger.info("In-memory cache cleared")

    def get_batch_prices(self, tickers: List[str], date: datetime.date) -> Dict[str, Optional[float]]:
        """
        Get prices for multiple tickers on a specific date
        
        Args:
            tickers: List of ticker symbols
            date: Date to get prices for
            
        Returns:
            Dictionary mapping ticker symbols to prices
        """
        result: Dict[str, Optional[float]] = {}
        
        # Process each ticker sequentially instead of in parallel
        for ticker in tickers:
            try:
                price = self.get_price(ticker, date)
                result[ticker] = price
            except Exception as e:
                logger.warning(f"Failed to get price for {ticker}: {str(e)}")
                result[ticker] = None
                
        return result

    def get_stock_info(self, ticker: str) -> Dict[str, Optional[str]]:
        """
        Get stock information including name, sector, and country
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information (name, sector, country)
        """
        # Default result with None values
        result = {
            'name': None,
            'sector': None,
            'country': None
        }
        
        try:
            # Check if we have this info in the database first
            # For portfolio positions
            position = self.db.get_position(ticker)
            if position and (position.get('name') or position.get('sector') or position.get('country')):
                if position.get('name'):
                    result['name'] = position.get('name')
                if position.get('sector'):
                    result['sector'] = position.get('sector')
                if position.get('country'):
                    result['country'] = position.get('country')
                
                # If we have all information, return early
                if result['name'] and result['sector'] and result['country']:
                    logger.debug(f"Complete stock info for {ticker} found in portfolio database")
                    return result
            
            # For watchlist items
            watchlist_items = self.db.get_all_watchlist_items()
            for item in watchlist_items:
                if item['ticker'] == ticker and (item.get('name') or item.get('sector') or item.get('country')):
                    if item.get('name') and not result['name']:
                        result['name'] = item.get('name')
                    if item.get('sector') and not result['sector']:
                        result['sector'] = item.get('sector')
                    if item.get('country') and not result['country']:
                        result['country'] = item.get('country')
                    
                    # If we have all information, return early
                    if result['name'] and result['sector'] and result['country']:
                        logger.debug(f"Complete stock info for {ticker} found in watchlist database")
                        return result
            
            # If we don't have complete info in the database, fetch from Yahoo Finance
            logger.info(f"Fetching stock info from Yahoo Finance API for {ticker}")
            
            # Check API status before attempting to fetch
            api_status = self.db.get_api_status('stock_info')
            if api_status.get('is_in_cooldown', False):
                cooldown_until = api_status.get('cooldown_until')
                if cooldown_until and datetime.datetime.fromisoformat(cooldown_until) > datetime.datetime.now():
                    logger.warning(f"API is in cooldown until {cooldown_until}")
                    return result
            
            # Use yfinance to get stock info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info:
                # Extract name (longName or shortName)
                if 'longName' in info:
                    result['name'] = info['longName']
                elif 'shortName' in info:
                    result['name'] = info['shortName']
                
                # Extract sector
                if 'sector' in info:
                    result['sector'] = info['sector']
                
                # Extract country
                if 'country' in info:
                    result['country'] = info['country']
                elif 'exchange' in info:
                    # Try to determine country from exchange
                    exchange = info['exchange']
                    exchange_country_map = {
                        'NYSE': 'United States',
                        'NASDAQ': 'United States',
                        'AMEX': 'United States',
                        'LSE': 'United Kingdom',
                        'XETRA': 'Germany',
                        'FSX': 'Germany',
                        'PAR': 'France',
                        'MIL': 'Italy',
                        'MCE': 'Spain',
                        'AMS': 'Netherlands',
                        'BRU': 'Belgium',
                        'STO': 'Sweden',
                        'CPH': 'Denmark',
                        'HEL': 'Finland',
                        'OSL': 'Norway',
                        'SWX': 'Switzerland',
                        'TSE': 'Canada',
                        'ASX': 'Australia',
                        'HKEX': 'Hong Kong',
                        'TSE': 'Japan',
                        'KRX': 'South Korea',
                        'TWSE': 'Taiwan',
                        'SZSE': 'China',
                        'SSE': 'China',
                    }
                    result['country'] = exchange_country_map.get(exchange)
                
                # Log API request success
                self.db.log_api_request("stock_info", True)
                
                logger.info(f"Successfully fetched stock info for {ticker}")
            else:
                logger.warning(f"No info returned from Yahoo Finance API for {ticker}")
                # Log API request failure
                self.db.log_api_request("stock_info", False, "No info returned")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching stock info from Yahoo Finance API: {e}")
            # Log API request failure
            self.db.log_api_request("stock_info", False, str(e))
            return result

    # Note: Frontend should use db_manager.get_historical_prices and db_manager.get_stock_price directly
    # instead of going through MarketInterface