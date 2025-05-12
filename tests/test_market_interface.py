import pytest
import datetime
import pandas as pd
import numpy as np
from unittest.mock import Mock
from pandas.testing import assert_frame_equal
from backend.market_interface import MarketInterface

@pytest.fixture
def market_interface(monkeypatch):
    """Create a market interface instance for testing"""
    # Create a mock database class using unittest.mock
    mock_db_instance = Mock()
    
    # Configure the mock database to return appropriate values
    mock_db_instance.get_api_status.return_value = {
        'status': 'up',
        'is_in_cooldown': False,
        'cooldown_until': None
    }
    
    # Patch the db_manager import in the market_interface module
    monkeypatch.setattr('backend.market_interface.db_manager', mock_db_instance)
    
    # Create the market interface
    interface = MarketInterface()
    
    # Store the mock db for test access
    interface._mock_db = mock_db_instance
    
    # Patch the _adjust_to_trading_day method to return the same date
    # This is important because our tests assume the date doesn't change
    def identity_adjust(date):
        return date
    
    interface._adjust_to_trading_day = identity_adjust
    
    return interface

@pytest.fixture
def sample_date():
    """Sample date for testing"""
    return datetime.date(2023, 1, 15)

@pytest.fixture
def sample_historical_df():
    """Sample DataFrame for testing historical prices"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    return pd.DataFrame({'price': prices}, index=dates)

class TestMarketInterface:
    """Test the MarketInterface class"""
    
    def test_initialization(self, market_interface):
        """Test market interface initialization"""
        assert hasattr(market_interface, 'db')
        assert isinstance(market_interface._exchange_rate_cache, dict)
        assert isinstance(market_interface._stock_price_cache, dict)
    
    def test_convert_currency_same_currency(self, market_interface):
        """Test converting between the same currency"""
        result = market_interface.convert_currency(100.0, 'USD', 'USD')
        assert result == 100.0
    
    def test_convert_currency_different_currency(self, market_interface, mocker):
        """Test converting between different currencies"""
        mocker.patch.object(market_interface, 'get_rate', return_value=0.85)
        result = market_interface.convert_currency(100.0, 'USD', 'EUR')
        assert result == 85.0
    
    def test_convert_currency_error_handling(self, market_interface, mocker):
        """Test error handling in currency conversion"""
        mocker.patch.object(market_interface, 'get_rate', side_effect=Exception('API error'))
        mocker.patch.object(market_interface, '_get_default_rate', return_value=0.9)
        
        # Should use the default rate when the API fails
        result = market_interface.convert_currency(100.0, 'USD', 'EUR')
        assert result == 90.0
    
    def test_get_rate_from_memory_cache(self, market_interface, sample_date):
        """Test getting exchange rate from memory cache"""
        # Add rate to memory cache
        cache_key = f"USD_EUR_{sample_date}"
        market_interface._exchange_rate_cache[cache_key] = (0.85, datetime.datetime.now())
        
        # Get rate
        rate = market_interface.get_rate('USD', 'EUR', sample_date)
        assert rate == 0.85
    
    def test_get_rate_from_db_cache(self, market_interface, sample_date):
        """Test getting exchange rate from database cache"""
        # Configure mock database to return a rate
        market_interface._mock_db.get_exchange_rate.return_value = 0.85
        
        # Get rate
        rate = market_interface.get_rate('USD', 'EUR', sample_date)
        assert rate == 0.85
        
        # Verify database was queried
        market_interface._mock_db.get_exchange_rate.assert_called_once_with('USD', 'EUR', sample_date)
        
        # Verify rate was added to memory cache
        cache_key = f"USD_EUR_{sample_date}"
        assert cache_key in market_interface._exchange_rate_cache
        assert market_interface._exchange_rate_cache[cache_key][0] == 0.85
    
    def test_get_rate_from_api(self, market_interface, sample_date, mocker):
        """Test getting exchange rate from API"""
        # Configure mock database to return None (no cached rate)
        market_interface._mock_db.get_exchange_rate.return_value = None
        
        # Mock the _fetch_exchange_rate method
        mocker.patch.object(market_interface, '_fetch_exchange_rate', return_value=0.85)
        
        # Get rate
        rate = market_interface.get_rate('USD', 'EUR', sample_date)
        assert rate == 0.85
        
        # Verify rate was stored in memory cache
        cache_key = f"USD_EUR_{sample_date}"
        assert cache_key in market_interface._exchange_rate_cache
        assert market_interface._exchange_rate_cache[cache_key][0] == 0.85
    
    def test_get_rate_api_error(self, market_interface, sample_date, mocker):
        """Test handling API errors when getting exchange rate"""
        # Configure mock database to return None (no cached rate)
        market_interface._mock_db.get_exchange_rate.return_value = None
        
        # Mock find_closest_date_data to return None (no close date found)
        market_interface._mock_db.find_closest_date_data.return_value = None
        
        # Mock the _fetch_exchange_rate method to raise an exception
        mocker.patch.object(market_interface, '_fetch_exchange_rate', side_effect=Exception('API error'))
        
        # Mock the _get_default_rate method
        mocker.patch.object(market_interface, '_get_default_rate', return_value=0.9)
        
        # Get rate
        rate = market_interface.get_rate('USD', 'EUR', sample_date)
        
        # Should use default rate
        assert rate == 0.9
    
    def test_get_price_from_memory_cache(self, market_interface, sample_date):
        """Test getting stock price from memory cache"""
        # Add price to memory cache
        cache_key = f"AAPL_{sample_date}"
        market_interface._stock_price_cache[cache_key] = (150.0, datetime.datetime.now())
        
        # Get price
        price = market_interface.get_price('AAPL', sample_date)
        assert price == 150.0
        
        # Verify database was not queried
        market_interface._mock_db.get_stock_price.assert_not_called()
    
    def test_get_price_from_db_cache(self, market_interface, sample_date):
        """Test getting stock price from database cache"""
        # Configure mock database to return a price
        market_interface._mock_db.get_stock_price.return_value = 150.0
        
        # Get price
        price = market_interface.get_price('AAPL', sample_date)
        assert price == 150.0
        
        # Verify database was queried
        market_interface._mock_db.get_stock_price.assert_called_once_with('AAPL', sample_date)
        
        # Verify price was stored in memory cache
        cache_key = f"AAPL_{sample_date}"
        assert cache_key in market_interface._stock_price_cache
        assert market_interface._stock_price_cache[cache_key][0] == 150.0
    
    def test_get_price_from_api(self, market_interface, sample_date, mocker):
        """Test getting stock price from API"""
        # Configure mock database to return None (no cached price)
        market_interface._mock_db.get_stock_price.return_value = None
        
        # Mock find_closest_date_data to return None (no close date found)
        market_interface._mock_db.find_closest_date_data.return_value = None
        
        # Mock the _fetch_stock_price method
        mocker.patch.object(market_interface, '_fetch_stock_price', return_value=150.0)
        
        # Mock yf.download to prevent real API calls
        mock_df = pd.DataFrame({'Close': [150.0]})
        mocker.patch('yfinance.download', return_value=mock_df)
        
        # Get price
        price = market_interface.get_price('AAPL', sample_date)
        assert price == 150.0
        
        # Verify price was stored in database
        market_interface._mock_db.store_stock_price.assert_called_once_with('AAPL', sample_date, 150.0)
        
        # Verify price was stored in memory cache
        cache_key = f"AAPL_{sample_date}"
        assert cache_key in market_interface._stock_price_cache
        assert market_interface._stock_price_cache[cache_key][0] == 150.0
    
    def test_get_price_api_error(self, market_interface, sample_date, mocker):
        """Test handling API errors when getting stock price"""
        # Configure mock database to return None (no cached price)
        market_interface._mock_db.get_stock_price.return_value = None
        
        # Mock find_closest_date_data to return None (no close date found)
        market_interface._mock_db.find_closest_date_data.return_value = None
        
        # Mock the _fetch_stock_price method to raise an exception
        mocker.patch.object(market_interface, '_fetch_stock_price', side_effect=Exception('API error'))
        
        # Mock yf.download to prevent real API calls and raise an exception
        mocker.patch('yfinance.download', side_effect=Exception('API error'))
        
        # Get price - this should return None since there's no default price method
        price = market_interface.get_price('AAPL', sample_date)
        
        # Should return None
        assert price is None
    
    def test_get_historical_prices_from_db_cache(self, market_interface, sample_historical_df):
        """Test getting historical prices from database cache"""
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 10)
        
        # Configure mock database to return historical prices
        market_interface._mock_db.get_historical_prices.return_value = sample_historical_df
        
        # Get historical prices
        df = market_interface.get_historical_prices('AAPL', start_date, end_date)
        assert df is sample_historical_df
        
        # Verify database was queried
        market_interface._mock_db.get_historical_prices.assert_called_once_with(
            'AAPL', start_date, end_date
        )
    
    def test_get_historical_prices_from_api(self, market_interface, sample_historical_df, mocker):
        """Test getting historical prices from API"""
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 10)
        
        # Configure mock database to return None (no cached data)
        market_interface._mock_db.get_historical_prices.return_value = None
        
        # Mock the _fetch_historical_prices method
        mocker.patch.object(market_interface, '_fetch_historical_prices', return_value=sample_historical_df)
        
        # Get historical prices
        df = market_interface.get_historical_prices('AAPL', start_date, end_date)
        assert df is sample_historical_df
        
        # Verify data was stored in database
        market_interface._mock_db.store_historical_prices.assert_called_once_with(
            'AAPL', start_date, end_date, sample_historical_df
        )
    
    def test_get_historical_prices_api_error(self, market_interface, mocker):
        """Test handling API errors when getting historical prices"""
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 10)
        
        # Configure mock database to return None (no cached data)
        market_interface._mock_db.get_historical_prices.return_value = None
        
        # Mock find_closest_date_data to return None (no close date found)
        market_interface._mock_db.find_closest_date_data.return_value = None
        
        # Mock the _fetch_historical_prices method to raise an exception
        mocker.patch.object(market_interface, '_fetch_historical_prices', side_effect=Exception('API error'))
        
        # Mock get_price to return None
        mocker.patch.object(market_interface, 'get_price', return_value=None)
        
        # Get historical prices
        df = market_interface.get_historical_prices('AAPL', start_date, end_date)
        
        # Should return an empty DataFrame
        assert isinstance(df, pd.DataFrame)
        assert df.empty
    
    def test_prefetch_data(self, market_interface, mocker):
        """Test prefetching data for multiple tickers"""
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 10)
        tickers = ['AAPL', 'MSFT', 'GOOG']
        
        # Mock the get_historical_prices method
        mock_get = mocker.patch.object(market_interface, 'get_historical_prices')
        mock_get_price = mocker.patch.object(market_interface, 'get_price')
        
        # Prefetch data
        market_interface.prefetch_data(tickers, start_date, end_date)
        
        # Check that get_historical_prices was called for each ticker
        assert mock_get.call_count == len(tickers)
        assert mock_get_price.call_count == len(tickers)
    
    def test_clear_cache(self, market_interface):
        """Test clearing the in-memory cache"""
        # Add some items to the cache
        market_interface._exchange_rate_cache['test_key'] = (0.85, datetime.datetime.now())
        market_interface._stock_price_cache['test_key'] = (150.0, datetime.datetime.now())
        
        # Clear cache
        market_interface.clear_cache()
        
        # Verify cache is empty
        assert len(market_interface._exchange_rate_cache) == 0
        assert len(market_interface._stock_price_cache) == 0 