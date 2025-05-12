import pytest
import os
import datetime
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database import Base, Portfolio, Transaction, Watchlist, MarketData, ExchangeRate, DatabaseManager

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
def test_db_manager(temp_db_path):
    """Create a test database manager instance"""
    db_url = f"sqlite:///{temp_db_path}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    return DatabaseManager(db_url)

@pytest.fixture
def sample_date():
    """Sample date for testing"""
    return datetime.date(2023, 1, 15)

class TestDatabaseManager:
    """Test the DatabaseManager class"""
    
    def test_initialization(self, test_db_manager, temp_db_path):
        """Test database initialization"""
        assert os.path.exists(temp_db_path)
        assert test_db_manager.engine is not None
        
    def test_add_and_get_position(self, test_db_manager, sample_date):
        """Test adding and retrieving a position"""
        # Add position
        result = test_db_manager.add_position(
            "AAPL", 10, 150.0, "USD", sample_date, 1500.0
        )
        assert result is True
        
        # Get position
        position = test_db_manager.get_position("AAPL")
        assert position is not None
        assert position['ticker'] == "AAPL"
        assert position['amount'] == 10
        assert position['purchase_price'] == 150.0
        assert position['purchase_currency'] == "USD"
        assert position['purchase_date'] == sample_date
        assert position['cost_basis'] == 1500.0
        
        # Get all positions
        positions = test_db_manager.get_all_positions()
        assert len(positions) == 1
        assert positions[0]['ticker'] == "AAPL"
        
    def test_update_position_value(self, test_db_manager, sample_date):
        """Test updating a position's value"""
        # Add position
        test_db_manager.add_position(
            "AAPL", 10, 150.0, "USD", sample_date, 1500.0
        )
        
        # Update position value
        value_date = datetime.date(2023, 2, 15)
        result = test_db_manager.update_position_value(
            "AAPL", 1600.0, "USD", value_date, 100.0, 6.67
        )
        assert result is True
        
        # Get position
        position = test_db_manager.get_position("AAPL")
        assert position['last_value'] == 1600.0
        assert position['last_value_currency'] == "USD"
        assert position['last_value_date'] == value_date
        assert position['unrealized_pl'] == 100.0
        assert position['return_percentage'] == 6.67
        
    def test_delete_position(self, test_db_manager, sample_date):
        """Test deleting a position"""
        # Add position
        test_db_manager.add_position(
            "AAPL", 10, 150.0, "USD", sample_date, 1500.0
        )
        
        # Delete position
        result = test_db_manager.delete_position("AAPL")
        assert result is True
        
        # Get position
        position = test_db_manager.get_position("AAPL")
        assert position is None
        
        # Get all positions
        positions = test_db_manager.get_all_positions()
        assert len(positions) == 0
        
    def test_add_and_get_transaction(self, test_db_manager, sample_date):
        """Test adding and retrieving a transaction"""
        # Add transaction
        result = test_db_manager.add_transaction(
            "buy", "AAPL", 10, 150.0, "USD", sample_date, 5.0, False
        )
        assert result is True
        
        # Get all transactions
        transactions = test_db_manager.get_all_transactions()
        assert len(transactions) == 1
        assert transactions[0]['type'] == "buy"
        assert transactions[0]['ticker'] == "AAPL"
        assert transactions[0]['amount'] == 10
        assert transactions[0]['price'] == 150.0
        assert transactions[0]['currency'] == "USD"
        assert transactions[0]['transaction_date'] == sample_date
        assert transactions[0]['cost'] == 5.0
        assert transactions[0]['is_dividend'] is False
        
        # Get transactions by ticker
        ticker_transactions = test_db_manager.get_transactions_by_ticker("AAPL")
        assert len(ticker_transactions) == 1
        assert ticker_transactions[0]['ticker'] == "AAPL"
        
        # Get transactions by type
        buy_transactions = test_db_manager.get_all_transactions("buy")
        assert len(buy_transactions) == 1
        assert buy_transactions[0]['type'] == "buy"
        
    def test_add_and_get_dividend_transaction(self, test_db_manager, sample_date):
        """Test adding and retrieving a dividend transaction"""
        # Add dividend transaction
        result = test_db_manager.add_transaction(
            "dividend", "AAPL", 10, 0.5, "USD", sample_date, 0.0, True
        )
        assert result is True
        
        # Get dividend transactions
        dividends = test_db_manager.get_dividend_transactions()
        assert len(dividends) == 1
        assert dividends[0]['ticker'] == "AAPL"
        assert dividends[0]['amount'] == 10
        assert dividends[0]['dividend_per_share'] == 0.5
        assert dividends[0]['currency'] == "USD"
        assert dividends[0]['transaction_date'] == sample_date
        
    def test_add_and_get_watchlist_item(self, test_db_manager):
        """Test adding and retrieving a watchlist item"""
        # Add watchlist item
        result = test_db_manager.add_watchlist_item(
            "AAPL", 150.0, "Buy if price drops below this level"
        )
        assert result is True
        
        # Get all watchlist items
        items = test_db_manager.get_all_watchlist_items()
        assert len(items) == 1
        assert items[0]['ticker'] == "AAPL"
        assert items[0]['strike_price'] == 150.0
        assert items[0]['notes'] == "Buy if price drops below this level"
        assert items[0]['date_added'] == datetime.date.today()
        
        # Update watchlist item
        result = test_db_manager.add_watchlist_item(
            "AAPL", 160.0, "Updated target price"
        )
        assert result is True
        
        # Get updated watchlist items
        items = test_db_manager.get_all_watchlist_items()
        assert len(items) == 1
        assert items[0]['strike_price'] == 160.0
        assert items[0]['notes'] == "Updated target price"
        
        # Delete watchlist item
        result = test_db_manager.delete_watchlist_item("AAPL")
        assert result is True
        
        # Get all watchlist items
        items = test_db_manager.get_all_watchlist_items()
        assert len(items) == 0
        
    def test_add_and_get_market_data(self, test_db_manager, sample_date):
        """Test adding and retrieving market data"""
        # Add market data
        result = test_db_manager.add_market_data(
            "AAPL", sample_date, 150.0, "USD", 149.0, 151.0, 148.0, 1000000
        )
        assert result is True
        
        # Get market data
        data = test_db_manager.get_market_data("AAPL", sample_date)
        assert data is not None
        assert data['ticker'] == "AAPL"
        assert data['date'] == sample_date
        assert data['close_price'] == 150.0
        assert data['open_price'] == 149.0
        assert data['high_price'] == 151.0
        assert data['low_price'] == 148.0
        assert data['volume'] == 1000000
        assert data['currency'] == "USD"
        
        # Get latest market data
        latest = test_db_manager.get_latest_market_data("AAPL")
        assert latest is not None
        assert latest['ticker'] == "AAPL"
        assert latest['date'] == sample_date
        
        # Add another data point
        next_date = datetime.date(2023, 1, 16)
        test_db_manager.add_market_data(
            "AAPL", next_date, 155.0, "USD", 152.0, 156.0, 151.0, 1200000
        )
        
        # Get historical market data
        historical = test_db_manager.get_historical_market_data(
            "AAPL", sample_date, next_date
        )
        assert len(historical) == 2
        assert historical[0]['date'] == sample_date
        assert historical[1]['date'] == next_date
        
    def test_add_and_get_exchange_rate(self, test_db_manager, sample_date):
        """Test adding and retrieving an exchange rate"""
        # Add exchange rate
        result = test_db_manager.add_exchange_rate(
            "USD", "EUR", sample_date, 0.85
        )
        assert result is True
        
        # Get exchange rate
        rate = test_db_manager.get_exchange_rate("USD", "EUR", sample_date)
        assert rate == 0.85
        
        # Update exchange rate
        result = test_db_manager.add_exchange_rate(
            "USD", "EUR", sample_date, 0.86
        )
        assert result is True
        
        # Get updated exchange rate
        rate = test_db_manager.get_exchange_rate("USD", "EUR", sample_date)
        assert rate == 0.86
        
    def test_get_tickers_to_track(self, test_db_manager, sample_date):
        """Test getting tickers to track"""
        # Add position
        test_db_manager.add_position(
            "AAPL", 10, 150.0, "USD", sample_date, 1500.0
        )
        
        # Add watchlist item
        test_db_manager.add_watchlist_item("MSFT", 250.0)
        
        # Add another watchlist item that's also in portfolio
        test_db_manager.add_watchlist_item("AAPL", 160.0)
        
        # Get tickers to track
        tickers = test_db_manager.get_tickers_to_track()
        assert len(tickers) == 2
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        
    def test_clear_all_data(self, test_db_manager, sample_date):
        """Test clearing all data"""
        # Add position
        test_db_manager.add_position(
            "AAPL", 10, 150.0, "USD", sample_date, 1500.0
        )
        
        # Add transaction
        test_db_manager.add_transaction(
            "buy", "AAPL", 10, 150.0, "USD", sample_date, 5.0, False
        )
        
        # Add watchlist item
        test_db_manager.add_watchlist_item("MSFT", 250.0)
        
        # Add market data
        test_db_manager.add_market_data(
            "AAPL", sample_date, 150.0, "USD", 149.0, 151.0, 148.0, 1000000
        )
        
        # Add exchange rate
        test_db_manager.add_exchange_rate(
            "USD", "EUR", sample_date, 0.85
        )
        
        # Clear all data
        result = test_db_manager.clear_all_data()
        assert result is True
        
        # Check that all data is cleared
        assert len(test_db_manager.get_all_positions()) == 0
        assert len(test_db_manager.get_all_transactions()) == 0
        assert len(test_db_manager.get_all_watchlist_items()) == 0
        assert test_db_manager.get_market_data("AAPL", sample_date) is None
        assert test_db_manager.get_exchange_rate("USD", "EUR", sample_date) is None 