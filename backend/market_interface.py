import yfinance as yf
from datetime import datetime, date, timedelta
from typing import overload
import pandas as pd
import logging
from zoneinfo import ZoneInfo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketInterface:
    def __init__(self):
        """Initialize the MarketInterface class."""
        pass

    @overload
    def get_stock_price(self, ticker: str, target_date: date) -> tuple[float, str]:
        """Get stock price for a single date."""
        ...

    @overload
    def get_stock_price(self, ticker: str, start_date: date, end_date: date) -> tuple[pd.Series, str]:
        """Get stock prices for a date range."""
        ...

    def get_stock_price(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date = None
    ) -> tuple[float, str] | tuple[pd.Series, str]:
        """
        Get stock price(s) and currency for a given ticker and date(s).
        
        Args:
            ticker (str): The stock ticker symbol
            start_date (date): The target date or start date of the range
            end_date (date, optional): The end date of the range. If None, returns single price.
            
        Returns:
            Union[Tuple[float, str], Tuple[pd.Series, str]]: 
                - For single date: Tuple of (price, currency)
                - For date range: Tuple of (prices_series, currency)
            
        Raises:
            ValueError: If the ticker is invalid or data is not available
            ConnectionError: If there are issues connecting to the API
        """
        try:
            stock = yf.Ticker(ticker)
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
                
                logger.info(f"Successfully retrieved price for {ticker} on {start_date}: {price} {currency}")
                return price, currency
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
                logger.info(f"Successfully retrieved prices for {ticker} between {start_date} and {end_date}")
                return prices, currency
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            raise ConnectionError(f"Failed to fetch price data for {ticker}: {str(e)}")