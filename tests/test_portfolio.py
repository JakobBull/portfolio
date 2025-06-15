import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock
from backend.portfolio import Portfolio
from backend.database import Stock, Position, Transaction, TransactionType

@pytest.fixture
def sample_stock():
    """Fixture for a sample stock."""
    return Stock(ticker="AAPL", name="Apple", currency="USD", sector="Technology", prices=[])

@pytest.fixture
def sample_portfolio(sample_stock):
    """Fixture for a sample portfolio."""
    positions = [
        Position(stock=sample_stock, amount=10, purchase_price=150.0, purchase_date=date(2023, 1, 1), purchase_currency="USD")
    ]
    transactions = [
        Transaction(stock=sample_stock, amount=10, price=150.0, date=date(2023, 1, 1), currency="USD", type=TransactionType.BUY)
    ]
    return Portfolio(positions, transactions)

def test_get_position(sample_portfolio: Portfolio):
    """Test retrieving a position from the portfolio."""
    position = sample_portfolio.get_position("AAPL")
    assert position is not None
    assert position.stock.ticker == "AAPL"

    assert sample_portfolio.get_position("GOOG") is None

def test_get_transaction_history(sample_portfolio: Portfolio):
    """Test retrieving transaction history."""
    history = sample_portfolio.get_transaction_history()
    assert len(history) == 1

    ticker_history = sample_portfolio.get_transaction_history(transaction_type="buy")
    assert len(ticker_history) == 1
    assert ticker_history[0].price == 150.0
    
    empty_history = sample_portfolio.get_transaction_history(transaction_type="sell")
    assert len(empty_history) == 0

def test_get_total_value(sample_portfolio: Portfolio):
    """Test calculating the total value of the portfolio."""
    # Add a mock price to the stock for value calculation
    sample_portfolio.positions[0].stock.prices.append(MagicMock(close_price=200.0))
    
    value = sample_portfolio.get_total_value()
    assert value == 2000.0 # 10 shares * $200

def test_get_performance(sample_portfolio: Portfolio):
    """Test calculating portfolio performance."""
    # Mock initial and final values for performance calculation
    # This is a simplification; a real test would involve more complex setup
    # to simulate price changes over time.
    initial_value = 1500.0  # 10 * 150
    final_value = 2000.0    # 10 * 200

    # We can't easily test the date-based values without more complex mocking,
    # so we focus on the logic with assumed values.
    sample_portfolio.get_total_value = MagicMock(side_effect=[initial_value, final_value])
    performance = sample_portfolio.get_performance(start_date=date.today() - timedelta(days=1), end_date=date.today())

    assert performance["absolute_return"] == 500.0
    assert performance["percentage_return"] == pytest.approx(33.333, abs=1e-3)