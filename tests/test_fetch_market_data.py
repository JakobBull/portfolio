import pytest
import pandas as pd
import datetime
import numpy as np
from unittest.mock import Mock, patch
from pandas.testing import assert_frame_equal

# Import the functions to test
from backend.fetch_market_data import (
    impute_missing_values,
    get_tickers_to_track,
    get_earliest_portfolio_date,
    get_position_purchase_date,
    fetch_market_data,
    fetch_benchmark_data
)

@pytest.fixture
def sample_historical_df():
    """Sample DataFrame for testing historical prices with date as column"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    return pd.DataFrame({
        'date': dates,
        'price': prices
    })

@pytest.fixture
def sample_historical_df_with_gaps():
    """Sample DataFrame with gaps for testing imputation"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='2D')  # Every other day
    prices = [100.0, 102.0, 104.0, 106.0, 108.0]
    return pd.DataFrame({
        'date': dates,
        'price': prices
    })

@pytest.fixture
def sample_historical_df_with_index():
    """Sample DataFrame for testing historical prices with DatetimeIndex"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    return pd.DataFrame({
        'price': prices
    }, index=dates)

@pytest.fixture
def sample_historical_df_empty():
    """Empty DataFrame for testing edge cases"""
    return pd.DataFrame()

class TestImputeMissingValues:
    """Test the impute_missing_values function"""
    
    def test_impute_missing_values_with_date_column(self, sample_historical_df_with_gaps):
        """Test imputing missing values in a DataFrame with date column"""
        # Get the original DataFrame with gaps
        df = sample_historical_df_with_gaps.copy()
        
        # Impute missing values
        result = impute_missing_values(df)
        
        # Check that all dates in the range are present
        expected_dates = pd.date_range(start='2023-01-01', end='2023-01-09')  # End date is inclusive in the original data
        assert len(result) == len(expected_dates)
        
        # Convert expected dates to date objects for comparison
        expected_dates_as_date = [d.date() for d in expected_dates]
        
        # Check that the dates in the result match the expected dates
        result_dates = [d for d in result['date']]
        assert result_dates == expected_dates_as_date
        
        # Check that missing values were forward filled
        # Find rows by date
        for i, date in enumerate(expected_dates_as_date):
            row = result[result['date'] == date]
            if i % 2 == 0:  # Original dates (0, 2, 4, 6, 8)
                expected_price = 100.0 + i
            else:  # Imputed dates (1, 3, 5, 7, 9)
                expected_price = 100.0 + (i - 1)
            
            assert len(row) == 1, f"Expected one row for date {date}"
            assert row['price'].iloc[0] == expected_price, f"Price mismatch for date {date}"
    
    def test_impute_missing_values_with_datetime_index(self, sample_historical_df_with_index):
        """Test imputing missing values in a DataFrame with DatetimeIndex"""
        # Create a DataFrame with gaps using DatetimeIndex
        dates = pd.date_range(start='2023-01-01', end='2023-01-09', freq='2D')  # Every other day
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        df = pd.DataFrame({'price': prices}, index=dates)
        
        # Impute missing values
        result = impute_missing_values(df)
        
        # Check that all dates in the range are present
        expected_dates = pd.date_range(start='2023-01-01', end='2023-01-09')  # End date is inclusive in the original data
        assert len(result) == len(expected_dates)
        
        # Convert expected dates to date objects for comparison
        expected_dates_as_date = [d.date() for d in expected_dates]
        
        # Check that the dates in the result match the expected dates
        result_dates = [d for d in result['date']]
        assert result_dates == expected_dates_as_date
        
        # Check that missing values were forward filled
        # Find rows by date
        for i, date in enumerate(expected_dates_as_date):
            row = result[result['date'] == date]
            if i % 2 == 0:  # Original dates (0, 2, 4, 6, 8)
                expected_price = 100.0 + i
            else:  # Imputed dates (1, 3, 5, 7, 9)
                expected_price = 100.0 + (i - 1)
            
            assert len(row) == 1, f"Expected one row for date {date}"
            assert row['price'].iloc[0] == expected_price, f"Price mismatch for date {date}"
    
    def test_impute_missing_values_empty_dataframe(self, sample_historical_df_empty):
        """Test imputing missing values in an empty DataFrame"""
        # Impute missing values in an empty DataFrame
        result = impute_missing_values(sample_historical_df_empty)
        
        # Should return the empty DataFrame unchanged
        assert result.empty
    
    def test_impute_missing_values_no_date_column(self):
        """Test imputing missing values in a DataFrame without a date column"""
        # Create a DataFrame without a date column
        df = pd.DataFrame({'price': [100.0, 101.0, 102.0]})
        
        # Impute missing values
        result = impute_missing_values(df)
        
        # Should return the DataFrame unchanged
        assert_frame_equal(result, df)

@patch('backend.fetch_market_data.db_manager')
class TestGetTickersToTrack:
    """Test the get_tickers_to_track function"""
    
    def test_get_tickers_to_track_specific(self, mock_db_manager):
        """Test getting specific tickers to track"""
        # Configure mock database to return positions and watchlist items
        mock_db_manager.get_all_positions.return_value = [
            {'ticker': 'AAPL'}, 
            {'ticker': 'MSFT'}, 
            {'ticker': 'GOOGL'}
        ]
        
        # Call the function with specific tickers
        all_tickers, portfolio_tickers, watchlist_tickers = get_tickers_to_track('AAPL,TSLA')
        
        # Check the results
        assert all_tickers == ['AAPL', 'TSLA']
        assert portfolio_tickers == ['AAPL']
        assert watchlist_tickers == ['TSLA']
    
    def test_get_tickers_to_track_all(self, mock_db_manager):
        """Test getting all tickers to track"""
        # Configure mock database to return positions and watchlist items
        mock_db_manager.get_all_positions.return_value = [
            {'ticker': 'AAPL'}, 
            {'ticker': 'MSFT'}
        ]
        mock_db_manager.get_all_watchlist_items.return_value = [
            {'ticker': 'GOOGL'}, 
            {'ticker': 'TSLA'}
        ]
        
        # Call the function without specific tickers
        all_tickers, portfolio_tickers, watchlist_tickers = get_tickers_to_track()
        
        # Check the results
        assert set(all_tickers) == {'AAPL', 'MSFT', 'GOOGL', 'TSLA'}
        assert set(portfolio_tickers) == {'AAPL', 'MSFT'}
        assert set(watchlist_tickers) == {'GOOGL', 'TSLA'}

@patch('backend.fetch_market_data.db_manager')
class TestGetEarliestPortfolioDate:
    """Test the get_earliest_portfolio_date function"""
    
    def test_get_earliest_portfolio_date(self, mock_db_manager):
        """Test getting the earliest portfolio date"""
        # Configure mock database to return positions with purchase dates
        mock_db_manager.get_all_positions.return_value = [
            {'purchase_date': datetime.date(2022, 1, 15)},
            {'purchase_date': datetime.date(2022, 3, 10)},
            {'purchase_date': datetime.date(2021, 12, 5)}
        ]
        
        # Call the function
        result = get_earliest_portfolio_date()
        
        # Check the result
        assert result == datetime.date(2021, 12, 5)
    
    def test_get_earliest_portfolio_date_no_positions(self, mock_db_manager):
        """Test getting the earliest portfolio date with no positions"""
        # Configure mock database to return no positions
        mock_db_manager.get_all_positions.return_value = []
        
        # Call the function
        result = get_earliest_portfolio_date()
        
        # Check the result (should be 30 days ago)
        expected = datetime.date.today() - datetime.timedelta(days=30)
        assert result == expected
    
    def test_get_earliest_portfolio_date_none_dates(self, mock_db_manager):
        """Test getting the earliest portfolio date with None dates"""
        # Configure mock database to return positions with some None dates
        mock_db_manager.get_all_positions.return_value = [
            {'purchase_date': None},
            {'purchase_date': datetime.date(2022, 3, 10)},
            {'purchase_date': datetime.date(2021, 12, 5)}
        ]
        
        # Call the function
        result = get_earliest_portfolio_date()
        
        # Check the result (should ignore None dates)
        assert result == datetime.date(2021, 12, 5)

@patch('backend.fetch_market_data.db_manager')
class TestGetPositionPurchaseDate:
    """Test the get_position_purchase_date function"""
    
    def test_get_position_purchase_date(self, mock_db_manager):
        """Test getting the purchase date for a position"""
        # Configure mock database to return a position
        mock_db_manager.get_position.return_value = {
            'purchase_date': datetime.date(2022, 1, 15)
        }
        
        # Call the function
        result = get_position_purchase_date('AAPL')
        
        # Check the result
        assert result == datetime.date(2022, 1, 15)
    
    def test_get_position_purchase_date_not_found(self, mock_db_manager):
        """Test getting the purchase date for a position that doesn't exist"""
        # Configure mock database to return None
        mock_db_manager.get_position.return_value = None
        
        # Call the function
        result = get_position_purchase_date('UNKNOWN')
        
        # Check the result
        assert result is None
    
    def test_get_position_purchase_date_no_date(self, mock_db_manager):
        """Test getting the purchase date for a position without a date"""
        # Configure mock database to return a position without a purchase date
        mock_db_manager.get_position.return_value = {
            'purchase_date': None
        }
        
        # Call the function
        result = get_position_purchase_date('AAPL')
        
        # Check the result
        assert result is None

@patch('backend.fetch_market_data.MarketInterface')
@patch('backend.fetch_market_data.db_manager')
class TestFetchMarketData:
    """Test the fetch_market_data function"""
    
    def test_fetch_market_data_portfolio_ticker(self, mock_db_manager, mock_market_interface, sample_historical_df):
        """Test fetching market data for a portfolio ticker"""
        # Configure mocks
        market_instance = mock_market_interface.return_value
        market_instance.get_price.return_value = 150.0
        market_instance.get_historical_prices.return_value = sample_historical_df
        
        mock_db_manager.get_position.return_value = {
            'purchase_date': datetime.date(2023, 1, 1)
        }
        
        # Call the function
        tickers = ['AAPL']
        portfolio_tickers = ['AAPL']
        watchlist_tickers = []
        
        with patch('backend.fetch_market_data.impute_missing_values', return_value=sample_historical_df):
            result = fetch_market_data(tickers, portfolio_tickers, watchlist_tickers, days=7, verbose=True, full_history=True)
        
        # Check the result
        assert 'AAPL' in result
        assert result['AAPL']['current_price'] == 150.0
        assert result['AAPL']['success'] is True
        
        # Verify that get_historical_prices was called with the purchase date
        market_instance.get_historical_prices.assert_called_once()
        args = market_instance.get_historical_prices.call_args[0]
        assert args[0] == 'AAPL'
        assert args[1] == datetime.date(2023, 1, 1)
    
    def test_fetch_market_data_watchlist_ticker(self, mock_db_manager, mock_market_interface, sample_historical_df):
        """Test fetching market data for a watchlist ticker"""
        # Configure mocks
        market_instance = mock_market_interface.return_value
        market_instance.get_price.return_value = 150.0
        market_instance.get_historical_prices.return_value = sample_historical_df
        
        # Call the function
        tickers = ['TSLA']
        portfolio_tickers = []
        watchlist_tickers = ['TSLA']
        
        with patch('backend.fetch_market_data.impute_missing_values', return_value=sample_historical_df):
            result = fetch_market_data(tickers, portfolio_tickers, watchlist_tickers, days=7, verbose=True, full_history=False)
        
        # Check the result
        assert 'TSLA' in result
        assert result['TSLA']['current_price'] == 150.0
        assert result['TSLA']['success'] is True
        
        # Verify that get_historical_prices was called with the default date range
        market_instance.get_historical_prices.assert_called_once()
        args = market_instance.get_historical_prices.call_args[0]
        assert args[0] == 'TSLA'
        # The start date should be 7 days before today
        expected_start = datetime.date.today() - datetime.timedelta(days=7)
        assert args[1].year == expected_start.year
        assert args[1].month == expected_start.month
        assert args[1].day == expected_start.day

@patch('backend.fetch_market_data.MarketInterface')
@patch('backend.fetch_market_data.db_manager')
class TestFetchBenchmarkData:
    """Test the fetch_benchmark_data function"""
    
    def test_fetch_benchmark_data(self, mock_db_manager, mock_market_interface, sample_historical_df):
        """Test fetching benchmark data"""
        # Configure mocks
        market_instance = mock_market_interface.return_value
        market_instance.get_historical_prices.return_value = sample_historical_df
        
        # Mock get_earliest_portfolio_date
        with patch('backend.fetch_market_data.get_earliest_portfolio_date', return_value=datetime.date(2023, 1, 1)):
            # Call the function
            with patch('backend.fetch_market_data.impute_missing_values', return_value=sample_historical_df):
                fetch_benchmark_data(days=365, verbose=True, impute_missing=True)
        
        # Verify that get_historical_prices was called for each benchmark
        assert market_instance.get_historical_prices.call_count == 3  # For ^GSPC, ^IXIC, ^GDAXI
        
        # Check the first call (^GSPC)
        args = market_instance.get_historical_prices.call_args_list[0][0]
        assert args[0] == '^GSPC'
        assert args[1] == datetime.date(2023, 1, 1) 