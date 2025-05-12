import pytest
import os
import datetime
import pandas as pd
import tempfile
from backend.database import DatabaseManager

@pytest.fixture
def temp_db_path():
    """Create a temporary database file path"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Clean up after test
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def test_db(temp_db_path):
    """Create a test database instance"""
    db_url = f"sqlite:///{temp_db_path}"
    # Create a database manager
    db_manager = DatabaseManager(db_url)
    
    # Create all tables in the database
    from backend.database import Base
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    
    return db_manager

@pytest.fixture
def sample_date():
    """Sample date for testing"""
    return datetime.date(2023, 1, 15)

@pytest.fixture
def sample_df():
    """Sample DataFrame for testing historical prices"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    return pd.DataFrame({'price': prices}, index=dates)

class TestDatabaseManager:
    """Test the DatabaseManager class"""
    
    def test_initialization(self, test_db, temp_db_path):
        """Test database initialization"""
        assert os.path.exists(temp_db_path)
        assert test_db.engine is not None
    
    def test_store_and_get_exchange_rate(self, test_db, sample_date):
        """Test storing and retrieving exchange rates"""
        # Store exchange rate
        result = test_db.store_exchange_rate('USD', 'EUR', sample_date, 0.85)
        assert result is True
        
        # Retrieve exchange rate
        rate = test_db.get_exchange_rate('USD', 'EUR', sample_date)
        assert rate == 0.85
        
        # Test non-existent rate
        rate = test_db.get_exchange_rate('USD', 'JPY', sample_date)
        assert rate is None
    
    def test_store_and_get_stock_price(self, test_db, sample_date):
        """Test storing and retrieving stock prices"""
        # Store stock price
        result = test_db.store_stock_price('AAPL', sample_date, 150.0)
        assert result is True
        
        # Retrieve stock price
        price = test_db.get_stock_price('AAPL', sample_date)
        assert price == 150.0
        
        # Test non-existent price
        price = test_db.get_stock_price('MSFT', sample_date)
        assert price is None
    
    def test_store_and_get_historical_prices(self, test_db, sample_df):
        """Test storing and retrieving historical prices"""
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 1, 10)
        
        # Store historical prices
        result = test_db.store_historical_prices('AAPL', start_date, end_date, sample_df)
        assert result is True
        
        # Retrieve historical prices
        df = test_db.get_historical_prices('AAPL', start_date, end_date)
        assert df is not None
        assert 'price' in df.columns
        assert len(df) == len(sample_df)
        assert df['price'].iloc[0] == sample_df['price'].iloc[0]
        
        # Test non-existent historical prices
        df = test_db.get_historical_prices('MSFT', start_date, end_date)
        assert df is None
    
    def test_log_api_request(self, test_db):
        """Test logging API requests"""
        # Log successful request
        result = test_db.log_api_request('exchange_rate', True)
        assert result is True
        
        # Log failed request
        result = test_db.log_api_request('stock_price', False, 'API error')
        assert result is True
    
    def test_get_recent_api_requests(self, test_db):
        """Test getting recent API requests"""
        # Log some requests
        test_db.log_api_request('exchange_rate', True)
        test_db.log_api_request('stock_price', True)
        test_db.log_api_request('historical_prices', False, 'API error')
        
        # Get recent requests
        count = test_db.get_recent_api_requests(minutes=60)
        assert count == 3 