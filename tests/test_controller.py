import pytest
from datetime import date
from backend.controller import Controller
from backend.database import Stock, Position, Transaction, TransactionType

def test_add_stock(controller: Controller):
    """Test adding a stock"""
    stock_data = {"ticker": "AAPL", "name": "Apple Inc.", "currency": "USD"}
    
    result = controller.add_stock(**stock_data)
    
    assert result is not None
    assert result.ticker == "AAPL"
    
    retrieved_stock = controller.db_manager.get_stock("AAPL")
    assert retrieved_stock is not None
    assert retrieved_stock.name == "Apple Inc."

def test_add_position(controller: Controller):
    """Test adding a position"""
    stock_data = {"ticker": "AAPL", "name": "Apple Inc.", "currency": "USD"}
    controller.add_stock(**stock_data)
    
    position_data = {
        "ticker": "AAPL",
        "amount": 10,
        "purchase_price": 150.0,
        "purchase_date": date.today()
    }
    
    result = controller.add_position(**position_data)
    
    assert result is not None
    assert result.amount == 10
    
    positions = controller.db_manager.get_all_positions()
    assert len(positions) == 1
    assert positions[0].stock.ticker == "AAPL"

def test_add_transaction(controller: Controller):
    """Test adding a transaction"""
    stock_data = {"ticker": "AAPL", "name": "Apple Inc.", "currency": "USD"}
    controller.add_stock(**stock_data)
    
    position_data = {
        "ticker": "AAPL",
        "amount": 10,
        "purchase_price": 150.0,
        "purchase_date": date.today()
    }
    controller.add_position(**position_data)
    
    transaction_data = {
        "ticker": "AAPL",
        "transaction_type": TransactionType.BUY,
        "amount": 5,
        "price": 160.0,
        "transaction_date": date.today()
    }
    
    result = controller.add_transaction(**transaction_data)
    
    assert result is not None
    assert result.price == 160.0
    
    transactions = controller.db_manager.get_all_transactions()
    assert len(transactions) == 1
    assert transactions[0].price == 160.0

def test_get_portfolio_value(controller: Controller):
    """Test getting portfolio value"""
    # This test needs market data to calculate value, which is mocked in the controller fixture.
    # For now, we just test that it returns 0 for an empty portfolio.
    value = controller.get_portfolio_value()
    assert value == 0.0

def test_get_watchlist(controller: Controller):
    """Test retrieving the watchlist."""
    stock_data = {"ticker": "GOOG", "name": "Alphabet", "currency": "USD"}
    controller.add_stock(**stock_data)
    controller.add_to_watchlist("GOOG")

    watchlist = controller.get_watchlist()
    
    assert len(watchlist) == 1
    assert watchlist[0].ticker == "GOOG"