import pytest
from datetime import date
from backend.database import DatabaseManager, Stock, Position, Transaction, Watchlist, TransactionType

def test_add_and_get_stock(test_db: DatabaseManager):
    """Test adding and retrieving a stock."""
    stock = test_db.add_stock("AAPL", "Apple Inc.", "USD")
    assert stock is not None
    retrieved_stock = test_db.get_stock(stock.ticker)
    assert retrieved_stock is not None
    assert retrieved_stock.ticker == "AAPL"
    assert retrieved_stock.name == "Apple Inc."

def test_add_and_get_stock_price(test_db: DatabaseManager):
    """Test adding and retrieving a stock price."""
    stock = test_db.add_stock("GOOG", "Alphabet Inc.", "USD")
    assert stock is not None

    price_date = date(2023, 1, 1)
    stock_price = test_db.add_stock_price(stock.ticker, price_date, 150.0)
    assert stock_price is not None
    assert stock_price.price == 150.0

    retrieved_price = test_db.get_stock_price(stock.ticker, price_date)
    assert retrieved_price is not None
    assert retrieved_price.price == 150.0

def test_add_and_get_position(test_db: DatabaseManager):
    """Test adding and retrieving a position."""
    stock = test_db.add_stock("TSLA", "Tesla, Inc.", "USD")
    assert stock is not None

    position = test_db.add_position(stock.ticker, 10, 200.0, "USD", date(2023, 1, 1))
    assert position is not None
    assert position.amount == 10

    positions = test_db.get_all_positions()
    assert len(positions) == 1
    assert positions[0].stock.ticker == "TSLA"

def test_add_and_get_transaction(test_db: DatabaseManager):
    """Test adding and retrieving a transaction."""
    stock = test_db.add_stock("MSFT", "Microsoft", "USD")
    assert stock is not None
    position = test_db.add_position(stock.ticker, 5, 300.0, "USD", date(2023, 1, 1))
    assert position is not None

    transaction = test_db.add_transaction(
        transaction_type=TransactionType.BUY,
        ticker=stock.ticker,
        amount=5,
        price=310.0,
        currency="USD",
        transaction_date=date(2023, 1, 2)
    )
    assert transaction is not None
    assert transaction.type == TransactionType.BUY

    transactions = test_db.get_all_transactions()
    assert len(transactions) == 1
    assert transactions[0].price == 310.0

def test_add_and_get_watchlist_item(test_db: DatabaseManager):
    """Test adding and retrieving a watchlist item."""
    stock = test_db.add_stock("NVDA", "NVIDIA", "USD")
    assert stock is not None
    
    item = test_db.add_watchlist_item(stock.ticker)
    assert item is not None

    watchlist = test_db.get_all_watchlist_items()
    assert len(watchlist) == 1
    assert watchlist[0].stock.ticker == "NVDA"

def test_get_tickers_to_track(test_db: DatabaseManager):
    """Test retrieving unique tickers from positions and watchlist."""
    stock1 = test_db.add_stock("AMZN", "Amazon", "USD")
    stock2 = test_db.add_stock("META", "Meta", "USD")
    assert stock1 is not None and stock2 is not None

    test_db.add_position(stock1.ticker, 2, 130.0, "USD", date(2023, 1, 1))
    test_db.add_watchlist_item(stock2.ticker)

    tickers = test_db.get_tickers_to_track()
    assert set(tickers) == {"AMZN", "META"} 