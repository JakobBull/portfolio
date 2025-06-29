import yfinance as yf
from datetime import datetime, date, timedelta
from typing import overload
import pandas as pd
import logging
from zoneinfo import ZoneInfo
import time
from curl_cffi import requests
import random
from requests.cookies import create_cookie

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataError(Exception):
    """Custom exception for market data fetching errors."""
    pass

def _wrap_cookie(cookie, session):
    """
    If cookie is just a str (cookie name), look up its value
    in session.cookies and wrap it into a real Cookie object.
    This fixes compatibility issues between curl_cffi and yfinance.
    """
    if isinstance(cookie, str):
        value = session.cookies.get(cookie)
        return create_cookie(name=cookie, value=value)
    return cookie

def _patch_yfinance_cookie_handling():
    """
    Monkey-patch YfData._get_cookie_basic so that
    it always returns a proper Cookie object,
    even when response.cookies is a simple dict.
    This solves the 'str' object has no attribute 'name' error.
    """
    import yfinance.data as _data
    
    if hasattr(_data.YfData, '_get_cookie_basic'):
        original = _data.YfData._get_cookie_basic
        
        def _patched(self, proxy=None, timeout=30):
            cookie = original(self, proxy)
            return _wrap_cookie(cookie, self._session)
        
        _data.YfData._get_cookie_basic = _patched
        logger.info("Applied yfinance cookie compatibility patch")

class MarketInterface:
    def __init__(self, cache_ttl_seconds=300):
        """
        Initialize the MarketInterface class.
        
        Args:
            cache_ttl_seconds (int): Time-to-live for the cache in seconds.
        """
        self._info_cache = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        
        # Apply the compatibility patch for curl_cffi
        _patch_yfinance_cookie_handling()
        
        # Create curl_cffi session with browser impersonation to avoid rate limiting
        self.session = requests.Session(impersonate="chrome")
        
        # Add rate limiting to be respectful to Yahoo's servers
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum 1 second between requests
        
        logger.info("MarketInterface initialized with rate limiting and compatibility fixes")

    def _rate_limit(self):
        """Implement rate limiting to avoid overwhelming Yahoo's servers."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()

    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if the cache for a ticker is still valid."""
        if ticker not in self._info_cache:
            return False
        
        timestamp, _ = self._info_cache[ticker]
        return (datetime.now() - timestamp) < self._cache_ttl

    def get_stock_info(self, ticker: str) -> dict | None:
        """
        Get information for a given stock ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
            
        Returns:
            dict | None: A dictionary with stock information or None if not found.
        """
        if self._is_cache_valid(ticker):
            logger.info(f"Returning cached info for {ticker}")
            _, data = self._info_cache[ticker]
            return data

        self._rate_limit()
        
        try:
            stock = yf.Ticker(ticker, session=self.session)
            info = stock.info
            
            # yfinance can return a dict with 'regularMarketPrice': None for invalid tickers
            if not info or info.get('regularMarketPrice') is None:
                logger.warning(f"Could not retrieve info for ticker: {ticker}")
                return None
            
            stock_info = {
                'name': info.get('shortName'),
                'currency': info.get('currency', 'USD'),
                'sector': info.get('sector'),
                'country': info.get('country')
            }
            logger.info(f"Successfully retrieved info for {ticker}, caching result.")
            self._info_cache[ticker] = (datetime.now(), stock_info)
            return stock_info
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            # Cache the failure for a short period to avoid spamming the API
            self._info_cache[ticker] = (datetime.now(), None)
            return None

    def get_stock_info_for_tickers(self, tickers: list[str]) -> dict[str, dict] | None:
        """
        Get information for a list of stock tickers using a single batch request.
        """
        if not tickers:
            return {}

        all_info = {}
        logger.info(f"Fetching info for {len(tickers)} tickers in a single batch...")

        self._rate_limit()
        
        try:
            # Use yf.Tickers for a batch request with our compatible session
            ticker_objects = yf.Tickers(tickers, session=self.session)
            
            # yfinance's batch info fetching can be all or nothing or partially successful.
            # We need to iterate through the original ticker list and access each Ticker object
            for ticker_str in tickers:
                try:
                    # Get the individual Ticker object from the Tickers collection
                    ticker_obj = ticker_objects.tickers[ticker_str]
                    info = ticker_obj.info
                    
                    # Basic validation of the returned info dict
                    if not info or info.get('regularMarketPrice') is None:
                        logger.warning(f"No valid info found for {ticker_str} in batch request.")
                        all_info[ticker_str] = None
                        continue

                    stock_info = {
                        'name': info.get('shortName'),
                        'currency': info.get('currency', 'USD'),
                        'sector': info.get('sector'),
                        'country': info.get('country')
                    }
                    all_info[ticker_str] = stock_info
                    
                except Exception as e:
                    # Sometimes individual tickers within the batch fail
                    logger.error(f"Error processing {ticker_str} within batch: {e}")
                    all_info[ticker_str] = None
                
                # A small delay even in batch processing can be helpful
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"An error occurred during the batch info fetch: {e}")
            # As a fallback, you might get some partial data, so return what we have
            return all_info

        logger.info(f"Successfully processed info for {len(tickers)} tickers.")
        return all_info

    def get_historical_prices_for_tickers(self, tickers: list[str], start_date: date | None, end_date: date, target_currency: str | None = None) -> pd.DataFrame | None:
        """
        Get historical prices for a list of tickers with a robust retry mechanism.
        If the batch download fails, it will attempt to download tickers individually.
        
        Args:
            tickers (list[str]): List of stock ticker symbols
            start_date (date | None): Start date for historical data. If None, fetches full history.
            end_date (date): End date for historical data
            target_currency (str, optional): Currency to convert all prices to. If None, keeps original currencies.
            
        Returns:
            pd.DataFrame | None: DataFrame with historical prices, optionally converted to target currency
        """
        self._rate_limit()
        
        try:
            # Use yf.Tickers for a batch request with our compatible session
            ticker_objects = yf.Tickers(tickers, session=self.session)
            
            # Fetch historical data using the .history() method on the Tickers object
            if start_date:
                data = ticker_objects.history(start=start_date, end=end_date, auto_adjust=True, group_by='ticker')
            else:
                print(f"Fetching full history for {tickers}")
                data = ticker_objects.history(period="max", end=end_date, auto_adjust=True, group_by='ticker')

            # Check for failure: yfinance sometimes returns a DataFrame with NaNs or is empty on failure.
            if data.empty:
                raise MarketDataError("Empty dataframe returned, possibly rate-limited.")
            
            # More robust check for all-NaN data that works with both single and multiple tickers
            try:
                # Check if all values in the dataframe are NaN
                all_nan = data.isnull().all().all() if len(tickers) == 1 else data.isnull().values.all()
                if all_nan:
                    raise MarketDataError("All-NaN dataframe returned, possibly rate-limited.")
            except Exception as nan_check_error:
                # If the NaN check itself fails, log it but don't fail the entire operation
                logger.warning(f"Could not perform NaN check on data: {nan_check_error}. Proceeding with data.")
            
            # If only some tickers failed, yfinance logs errors but doesn't raise an exception.
            # We can proceed with the partial data, and the calling script will handle missing tickers.
            
            # Convert currencies if target_currency is specified
            if target_currency:
                data = self._convert_dataframe_currencies(data, tickers, target_currency)
            
            return data
        except Exception as e:
            error_str = str(e)
            if "Too Many Requests" in error_str or "YFRateLimitError" in error_str or isinstance(e, MarketDataError):
                logger.warning(f"Batch request for {tickers} failed: {e}. Falling back to individual fetching.")
                fallback_data = self._fetch_tickers_individually(tickers, start_date, end_date)
                if fallback_data is not None and target_currency:
                    fallback_data = self._convert_dataframe_currencies(fallback_data, tickers, target_currency)
                return fallback_data
            else:
                logger.error(f"An unexpected error occurred fetching historical data for {tickers}: {e}")
                return None

    def _fetch_tickers_individually(self, tickers: list[str], start_date: date | None, end_date: date) -> pd.DataFrame | None:
        """Fallback to fetch tickers one by one if batch request fails."""
        logger.info(f"Batch request failed for {tickers}. Falling back to individual fetching.")
        all_data = []
        max_retries = 3
        backoff_factor = 5

        for ticker in tickers:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching individually: {ticker} (Attempt {attempt + 1}/{max_retries})")
                    
                    # Rate limit each individual request
                    self._rate_limit()
                    
                    # Use yf.Ticker(...).history() for individual fetching
                    stock = yf.Ticker(ticker, session=self.session)
                    if start_date:
                        data = stock.history(start=start_date, end=end_date, auto_adjust=True)
                    else:
                        data = stock.history(period="max", end=end_date, auto_adjust=True)
                    
                    if not data.empty:
                        # For consistency with batch download, create a MultiIndex
                        data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
                        all_data.append(data)
                        break  # Success, exit retry loop for this ticker
                    
                    # If data is empty, it's a failure (either rate limit or no data). Let's retry.
                    logger.warning(f"No data returned for {ticker} on attempt {attempt + 1}.")
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"Waiting {wait_time} seconds before next attempt for {ticker}...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Could not fetch {ticker} after {max_retries} attempts. It may be delisted or have no data for the period.")

                except Exception as e:
                    logger.error(f"An exception occurred while fetching {ticker} on attempt {attempt+1}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"Retrying {ticker} in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Could not fetch {ticker} after {max_retries} attempts due to exception.")
            
            # Pause between fetching different tickers to be respectful
            time.sleep(random.uniform(2, 4))

        if not all_data:
            return None
        
        # Concatenate all individual dataframes
        combined_df = pd.concat(all_data, axis=1)
        return combined_df

    def _convert_dataframe_currencies(self, data: pd.DataFrame, tickers: list[str], target_currency: str) -> pd.DataFrame:
        """
        Convert all prices in a multi-ticker DataFrame to the target currency.
        
        Args:
            data (pd.DataFrame): DataFrame with multi-level columns (ticker, price_type)
            tickers (list[str]): List of ticker symbols
            target_currency (str): Target currency for conversion
            
        Returns:
            pd.DataFrame: DataFrame with prices converted to target currency
        """
        if data.empty:
            return data
            
        # Get stock info for all tickers to determine their currencies
        stock_info = self.get_stock_info_for_tickers(tickers)
        if not stock_info:
            logger.warning("Could not retrieve currency info for tickers, returning unconverted data")
            return data
        
        converted_data = data.copy()
        
        for ticker in tickers:
            if ticker not in stock_info or stock_info[ticker] is None:
                logger.warning(f"No currency info available for {ticker}, skipping conversion")
                continue
                
            ticker_currency = stock_info[ticker].get('currency', 'USD')
            
            # Skip conversion if already in target currency
            if ticker_currency == target_currency:
                continue
                
            logger.info(f"Converting {ticker} prices from {ticker_currency} to {target_currency}")
            
            # Convert all price columns for this ticker
            try:
                # Check if we have multi-level columns (multiple tickers)
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = converted_data[ticker]
                        
                        # Convert each price column (Open, High, Low, Close, Volume)
                        for col in ticker_data.columns:
                            if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                                for date_idx, price in ticker_data[col].items():
                                    if pd.notna(price):  # Skip NaN values
                                        date_for_conversion = date_idx.date() if hasattr(date_idx, 'date') else date_idx
                                        converted_price = self.convert_currency(price, ticker_currency, target_currency, date_for_conversion)
                                        converted_data.loc[date_idx, (ticker, col)] = converted_price
                else:
                    # Single ticker case - columns are just price types
                    for col in converted_data.columns:
                        if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                            for date_idx, price in converted_data[col].items():
                                if pd.notna(price):  # Skip NaN values
                                    date_for_conversion = date_idx.date() if hasattr(date_idx, 'date') else date_idx
                                    converted_price = self.convert_currency(price, ticker_currency, target_currency, date_for_conversion)
                                    converted_data.loc[date_idx, col] = converted_price
                                    
            except Exception as e:
                logger.error(f"Error converting currency for {ticker}: {e}")
                # Continue with other tickers even if one fails
                continue
        
        logger.info(f"Successfully converted prices to {target_currency} for {len(tickers)} tickers")
        return converted_data

    def get_historical_dividends_for_tickers(self, tickers: list[str], start_date: date | None, end_date: date, target_currency: str | None = None) -> dict[str, pd.Series] | None:
        """
        Get historical dividends for a list of tickers with a robust retry mechanism.
        If the batch request fails, it will attempt to download tickers individually.
        
        Args:
            tickers (list[str]): List of stock ticker symbols
            start_date (date | None): Start date for historical dividend data. If None, fetches full history.
            end_date (date): End date for historical dividend data
            target_currency (str, optional): Currency to convert all dividends to. If None, keeps original currencies.
            
        Returns:
            dict[str, pd.Series] | None: Dictionary mapping ticker -> dividend Series, optionally converted to target currency
        """
        if not tickers:
            return {}
            
        logger.info(f"Fetching dividend data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        self._rate_limit()
        
        try:
            # Get stock info for currency conversion if needed
            stock_info = None
            if target_currency:
                stock_info = self.get_stock_info_for_tickers(tickers)
            
            # Fetch dividends for each ticker individually (yfinance doesn't support batch dividend fetching)
            all_dividends = {}
            
            for ticker in tickers:
                try:
                    logger.info(f"Fetching dividends for {ticker}")
                    self._rate_limit()
                    
                    stock = yf.Ticker(ticker, session=self.session)
                    dividends = stock.dividends
                    
                    if dividends.empty:
                        logger.info(f"No dividend data found for {ticker}")
                        all_dividends[ticker] = pd.Series(dtype=float)
                        continue
                    
                    # Filter by date range
                    mask = (dividends.index.date <= end_date)
                    if start_date:
                        mask &= (dividends.index.date >= start_date)
                    filtered_dividends = dividends[mask]
                    
                    if filtered_dividends.empty:
                        logger.info(f"No dividend data in date range for {ticker}")
                        all_dividends[ticker] = pd.Series(dtype=float)
                        continue
                    
                    # Convert currency if target_currency is specified
                    if target_currency and stock_info and ticker in stock_info and stock_info[ticker]:
                        ticker_currency = stock_info[ticker].get('currency', 'USD')
                        
                        if ticker_currency != target_currency:
                            logger.info(f"Converting {ticker} dividends from {ticker_currency} to {target_currency}")
                            converted_dividends = pd.Series(index=filtered_dividends.index, dtype=float)
                            
                            for date_idx, dividend_amount in filtered_dividends.items():
                                date_for_conversion = date_idx.date() if hasattr(date_idx, 'date') else date_idx
                                converted_amount = self.convert_currency(dividend_amount, ticker_currency, target_currency, date_for_conversion)
                                converted_dividends[date_idx] = converted_amount
                            
                            all_dividends[ticker] = converted_dividends
                        else:
                            all_dividends[ticker] = filtered_dividends
                    else:
                        all_dividends[ticker] = filtered_dividends
                    
                    logger.info(f"Successfully fetched {len(filtered_dividends)} dividend records for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error fetching dividends for {ticker}: {e}")
                    all_dividends[ticker] = pd.Series(dtype=float)
                
                # Small delay between ticker requests
                time.sleep(random.uniform(1, 2))
            
            logger.info(f"Successfully fetched dividend data for {len(tickers)} tickers")
            return all_dividends
            
        except Exception as e:
            logger.error(f"An error occurred during dividend data fetching: {e}")
            return None

    @overload
    def get_stock_price(self, ticker: str, target_date: date, target_currency: str | None = None) -> tuple[float, str]:
        """Get stock price for a single date."""
        ...

    @overload
    def get_stock_price(self, ticker: str, start_date: date, end_date: date, target_currency: str | None = None) -> tuple[pd.Series, str]:
        """Get stock prices for a date range."""
        ...

    def get_stock_price(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date | None = None,
        target_currency: str | None = None
    ) -> tuple[float, str] | tuple[pd.Series, str]:
        """
        Get stock price(s) and currency for a given ticker and date(s).
        
        Args:
            ticker (str): The stock ticker symbol
            start_date (date): The target date or start date of the range
            end_date (date, optional): The end date of the range. If None, returns single price.
            target_currency (str, optional): Currency to convert prices to. If None, returns original currency.
            
        Returns:
            tuple[float, str] | tuple[pd.Series, str]: 
                - For single date: Tuple of (price, currency)
                - For date range: Tuple of (prices_series, currency)
                - Currency returned is the target_currency if conversion was performed
            
        Raises:
            ValueError: If the ticker is invalid or data is not available
            ConnectionError: If there are issues connecting to the API
        """
        self._rate_limit()
        
        try:
            stock = yf.Ticker(ticker, session=self.session)
            info = stock.info
            currency = info.get('currency', 'USD')
            
            if end_date is None:
                # Single date query
                # Create timezone-aware datetime in UTC
                target_datetime = datetime.combine(start_date, datetime.min.time(), tzinfo=ZoneInfo("UTC"))
                # Get a few days around the target date to handle weekends/holidays
                query_start = target_datetime - timedelta(days=5)
                query_end = target_datetime + timedelta(days=5)
                
                hist = stock.history(start=query_start, end=query_end)
                
                if hist.empty:
                    raise ValueError(f"No price data available for {ticker} on {start_date}")
                
                # Convert index to UTC timezone if it's not already
                hist.index = hist.index.tz_localize('UTC') if hist.index.tz is None else hist.index.tz_convert('UTC')
                
                # Get the closest date to our target date
                closest_date = min(hist.index, key=lambda x: abs((x - target_datetime).total_seconds()))
                price = hist.loc[closest_date, 'Close']
                
                # Convert currency if target_currency is specified
                if target_currency and target_currency != currency:
                    price = self.convert_currency(price, currency, target_currency, start_date)
                    final_currency = target_currency
                    logger.info(f"Successfully retrieved and converted price for {ticker} on {start_date}: {price} {final_currency} (from {currency})")
                else:
                    final_currency = currency
                    logger.info(f"Successfully retrieved price for {ticker} on {start_date}: {price} {final_currency}")
                
                return price, final_currency
            else:
                # Date range query
                start_datetime = datetime.combine(start_date, datetime.min.time(), tzinfo=ZoneInfo("UTC"))
                end_datetime = datetime.combine(end_date, datetime.min.time(), tzinfo=ZoneInfo("UTC")) + timedelta(days=1)
                
                hist = stock.history(start=start_datetime, end=end_datetime)
                
                if hist.empty:
                    raise ValueError(f"No price data available for {ticker} between {start_date} and {end_date}")
                
                # Convert index to UTC timezone if it's not already
                hist.index = hist.index.tz_localize('UTC') if hist.index.tz is None else hist.index.tz_convert('UTC')
                
                # Return the closing prices series
                prices = hist['Close']
                
                # Convert currency if target_currency is specified
                if target_currency and target_currency != currency:
                    # For series, we need to convert each price using the date from the index
                    converted_prices = pd.Series(index=prices.index, dtype=float)
                    for date_idx, price in prices.items():
                        date_for_conversion = date_idx.date()
                        converted_prices[date_idx] = self.convert_currency(price, currency, target_currency, date_for_conversion)
                    final_currency = target_currency
                    logger.info(f"Successfully retrieved and converted prices for {ticker} between {start_date} and {end_date}: {final_currency} (from {currency})")
                    return converted_prices, final_currency
                else:
                    logger.info(f"Successfully retrieved prices for {ticker} between {start_date} and {end_date}")
                    return prices, currency
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            raise ConnectionError(f"Failed to fetch price data for {ticker}: {str(e)}")
        
    def convert_currency(self, amount: float, from_currency: str, to_currency: str, 
                        exchange_date: date | None = None) -> float:
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
            exchange_date = date.today()
            
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
            rate = 1.0
            logger.info(f"Using fallback exchange rate: {rate}")
            return amount * rate
    
    def get_rate(self, from_currency: str, to_currency: str, date: date) -> float:
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
        
        # Handle common currency conversions with hardcoded rates
        # TODO: Replace with proper API integration
        conversion_rates = {
            ("GBp", "USD"): 0.0117,  # British pence to USD
            ("GBp", "GBP"): 0.01,    # British pence to British pounds
            ("GBP", "USD"): 1.17,    # British pounds to USD (approximate)
            ("EUR", "USD"): 1.08,    # Euro to USD (approximate)
            ("USD", "EUR"): 0.93,    # USD to Euro (approximate)
            ("USD", "GBP"): 0.85,    # USD to British pounds (approximate)
            ("GBp", "EUR"): 0.0117,  # British pence to EUR
        }
        
        # Check for direct conversion rate
        rate_key = (from_currency, to_currency)
        if rate_key in conversion_rates:
            rate = conversion_rates[rate_key]
            # logger.info(f"Using hardcoded exchange rate: {from_currency} to {to_currency} = {rate}")
            return rate
        
        # Check for reverse conversion rate (1/rate)
        reverse_key = (to_currency, from_currency)
        if reverse_key in conversion_rates:
            rate = 1.0 / conversion_rates[reverse_key]
            logger.info(f"Using reverse hardcoded exchange rate: {from_currency} to {to_currency} = {rate}")
            return rate
        
        # Fallback to 1.0 for unknown currency pairs
        logger.warning(f"No exchange rate available for {from_currency} to {to_currency}, using fallback rate of 1.0")
        return 1.0