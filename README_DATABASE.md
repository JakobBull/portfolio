# Portfolio Database Documentation

This document provides information about the SQLAlchemy-based database implementation for the portfolio application.

## Database Structure

The database consists of the following tables:

1. **Portfolio** - Stores information about portfolio positions
2. **Transaction** - Records all buy, sell, and dividend transactions
3. **Watchlist** - Tracks stocks of interest with target prices
4. **MarketData** - Stores historical market data for stocks
5. **ExchangeRate** - Tracks currency exchange rates

## Database Setup

The database is automatically initialized when the application starts. By default, it creates a SQLite database file named `portfolio.db` in the application directory.

## Database Manager

All interactions with the database should be done through the `DatabaseManager` class, which provides methods for:

- Adding, updating, and retrieving portfolio positions
- Recording and querying transactions
- Managing watchlist items
- Storing and retrieving market data
- Tracking exchange rates

## Usage Examples

### Initializing the Database Manager

```python
from backend.database import DatabaseManager

# Default initialization (creates portfolio.db in the current directory)
db_manager = DatabaseManager()

# Custom database path
db_manager = DatabaseManager("sqlite:///path/to/custom.db")
```

### Managing Portfolio Positions

```python
# Add a new position
db_manager.add_position(
    ticker="AAPL",
    amount=10,
    purchase_price=150.0,
    purchase_currency="USD",
    purchase_date=datetime.date(2023, 1, 15),
    cost_basis=1500.0
)

# Get a specific position
position = db_manager.get_position("AAPL")

# Get all positions
positions = db_manager.get_all_positions()

# Update a position's current value
db_manager.update_position_value(
    ticker="AAPL",
    current_value=1600.0,
    value_currency="USD",
    value_date=datetime.date(2023, 2, 15),
    unrealized_pl=100.0,
    return_percentage=6.67
)

# Delete a position
db_manager.delete_position("AAPL")
```

### Recording Transactions

```python
# Add a buy transaction
db_manager.add_transaction(
    type="buy",
    ticker="AAPL",
    amount=10,
    price=150.0,
    currency="USD",
    transaction_date=datetime.date(2023, 1, 15),
    cost=5.0,
    is_dividend=False
)

# Add a sell transaction
db_manager.add_transaction(
    type="sell",
    ticker="AAPL",
    amount=5,
    price=160.0,
    currency="USD",
    transaction_date=datetime.date(2023, 2, 15),
    cost=5.0,
    is_dividend=False
)

# Add a dividend transaction
db_manager.add_transaction(
    type="dividend",
    ticker="AAPL",
    amount=10,  # shares owned
    price=0.5,  # dividend per share
    currency="USD",
    transaction_date=datetime.date(2023, 3, 15),
    cost=0.0,
    is_dividend=True
)

# Get all transactions
transactions = db_manager.get_all_transactions()

# Get transactions by type
buy_transactions = db_manager.get_all_transactions("buy")

# Get transactions for a specific ticker
ticker_transactions = db_manager.get_transactions_by_ticker("AAPL")

# Get dividend transactions
dividends = db_manager.get_dividend_transactions()
```

### Managing Watchlist

```python
# Add a watchlist item
db_manager.add_watchlist_item(
    ticker="MSFT",
    strike_price=250.0,
    notes="Buy if price drops below this level"
)

# Get all watchlist items
items = db_manager.get_all_watchlist_items()

# Delete a watchlist item
db_manager.delete_watchlist_item("MSFT")
```

### Storing Market Data

```python
# Add market data for a ticker
db_manager.add_market_data(
    ticker="AAPL",
    date=datetime.date(2023, 1, 15),
    close_price=150.0,
    currency="USD",
    open_price=149.0,
    high_price=151.0,
    low_price=148.0,
    volume=1000000
)

# Get market data for a specific date
data = db_manager.get_market_data("AAPL", datetime.date(2023, 1, 15))

# Get latest market data for a ticker
latest = db_manager.get_latest_market_data("AAPL")

# Get historical market data for a date range
historical = db_manager.get_historical_market_data(
    ticker="AAPL",
    start_date=datetime.date(2023, 1, 1),
    end_date=datetime.date(2023, 1, 31)
)
```

### Managing Exchange Rates

```python
# Add an exchange rate
db_manager.add_exchange_rate(
    from_currency="USD",
    to_currency="EUR",
    date=datetime.date(2023, 1, 15),
    rate=0.85
)

# Get an exchange rate
rate = db_manager.get_exchange_rate(
    from_currency="USD",
    to_currency="EUR",
    date=datetime.date(2023, 1, 15)
)
```

### Utility Methods

```python
# Get all tickers to track (from portfolio and watchlist)
tickers = db_manager.get_tickers_to_track()

# Clear all data (use with caution!)
db_manager.clear_all_data()
```

## Testing

The database implementation includes comprehensive tests in `tests/test_sqlalchemy_db.py`. Run the tests using pytest:

```bash
pytest tests/test_sqlalchemy_db.py
```

## Database Schema

### Portfolio Table
- ticker (primary key): Stock symbol
- amount: Number of shares
- purchase_price: Average purchase price per share
- purchase_currency: Currency used for purchase
- purchase_date: Date of purchase
- cost_basis: Total cost including fees
- last_value: Current value of the position
- last_value_currency: Currency of the current value
- last_value_date: Date of the last valuation
- unrealized_pl: Unrealized profit/loss
- return_percentage: Return percentage

### Transaction Table
- id (primary key): Unique identifier
- type: Transaction type (buy, sell, dividend)
- ticker: Stock symbol
- amount: Number of shares
- price: Price per share
- currency: Transaction currency
- transaction_date: Date of transaction
- cost: Transaction cost/fee
- is_dividend: Whether this is a dividend transaction

### Watchlist Table
- ticker (primary key): Stock symbol
- strike_price: Target price
- notes: Additional notes
- date_added: Date when added to watchlist

### MarketData Table
- id (primary key): Unique identifier
- ticker: Stock symbol
- date: Date of the data point
- close_price: Closing price
- currency: Price currency
- open_price: Opening price
- high_price: Highest price of the day
- low_price: Lowest price of the day
- volume: Trading volume

### ExchangeRate Table
- id (primary key): Unique identifier
- from_currency: Source currency
- to_currency: Target currency
- date: Date of the exchange rate
- rate: Exchange rate value 