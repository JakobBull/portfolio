# Portfolio Manager

A comprehensive portfolio management system with tax calculations, dividend tracking, and benchmark comparison.

## Features

- **Portfolio Management**: Track positions, transactions, and performance
- **German Tax Calculations**: Calculate taxes according to German tax rules
- **Dividend Tracking**: Monitor dividend income and yield
- **Benchmark Comparison**: Compare portfolio performance against major indices
- **Interactive Dashboard**: Visualize portfolio data and performance
- **Enhanced Market Interface**: Robust API handling with rate limiting and persistent caching

## Getting Started

### Prerequisites

- Docker and Docker Compose

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/portfolio-manager.git
   cd portfolio-manager
   ```

2. Build and start the Docker container:
   ```
   docker-compose up -d
   ```

3. Access the dashboard at http://localhost:8050

## Usage

### Portfolio Summary

The dashboard provides a comprehensive overview of your portfolio, including:
- Total value
- Unrealized profit/loss
- Dividend yield
- Total cost
- Return percentage
- Dividend income

### Performance Tracking

Compare your portfolio performance against major benchmarks:
- NASDAQ Composite
- S&P 500
- DAX 30

Adjust the date range to view performance over different periods.

### Position Management

View detailed information about each position in your portfolio:
- Market value
- Cost basis
- Unrealized profit/loss
- Return percentage
- Weight in portfolio
- Dividend yield
- Sector

### Adding Transactions

Add new transactions to your portfolio:
1. Search for a stock by ticker or name
2. Select the transaction type (buy, sell, dividend)
3. Enter the number of shares and price per share
4. Select the currency and date
5. Click "Add Transaction"

### Tax Settings

Configure tax settings for German tax calculations:
- Married (joint filing)
- Church tax
- Partial exemption for funds

## Testing

The application includes a comprehensive test suite using pytest to ensure all components work correctly.

### Running Tests

You can run the tests in several ways:

1. **Using the test runner script**:
   ```
   python run_tests.py
   ```

2. **Using pytest directly**:
   ```
   pytest tests/
   ```

3. **Using Docker Compose**:
   ```
   docker-compose run test
   ```

4. **Running specific tests**:
   ```
   pytest tests/test_transaction.py
   ```

5. **Running tests with coverage**:
   ```
   pytest --cov=backend tests/
   ```

### Test Coverage

The test suite covers:
- Transaction and Dividend classes
- Position class
- Portfolio class
- Controller class
- MoneyAmount class
- Dash application

### Adding New Tests

To add new tests:
1. Create a new test file in the `tests` directory
2. Follow the naming convention `test_*.py`
3. Use pytest fixtures for test setup
4. Run the tests to ensure they pass

## Docker Deployment

The application is containerized using Docker for easy deployment:

```
# Build the Docker image
docker build -t portfolio-manager .

# Run the container
docker run -p 8050:8050 portfolio-manager
```

Or use Docker Compose:

```
docker-compose up -d
```

## Architecture

The system consists of several components:

- **Portfolio**: Manages positions and transactions
- **Position**: Tracks holdings in specific stocks
- **Transaction**: Represents buy, sell, and dividend transactions
- **Controller**: Coordinates between components and provides analysis
- **Tax Calculator**: Implements German tax rules
- **Benchmark**: Compares portfolio performance against indices
- **Market Interface**: Provides robust access to market data with fallbacks and caching
- **Database**: Stores market data for offline access and reduced API calls
- **Rate Limiter**: Ensures API rate limits are respected
- **Dashboard**: Provides interactive visualization and management

## Market Interface

The enhanced market interface provides robust access to market data with the following features:

### Persistent Caching

- SQLite database for storing exchange rates, stock prices, and historical data
- Reduces API calls and provides offline access to previously fetched data
- Automatic fallback to cached data when API calls fail

### Rate Limiting

- Token bucket algorithm to respect API rate limits
- Configurable limits for requests per hour and burst requests
- Queuing system for handling requests when limits are reached

### Error Handling

- Multiple fallback mechanisms when API calls fail
- Default exchange rates for common currency pairs
- Automatic retries with exponential backoff
- Logging of API errors for troubleshooting

### Parallel Processing

- Thread pool for prefetching data for multiple tickers
- Efficient batch processing of historical data

### Usage

```python
from backend.market_interface import MarketInterface

# Initialize the market interface
market = MarketInterface()

# Get current stock price
price = market.get_price('AAPL', datetime.date.today())

# Get historical prices
historical = market.get_historical_prices('MSFT', 
                                        start_date=datetime.date(2023, 1, 1),
                                        end_date=datetime.date(2023, 12, 31))

# Convert currency
eur_amount = market.convert_currency(100.0, 'USD', 'EUR')

# Prefetch data for multiple tickers
market.prefetch_data(['AAPL', 'MSFT', 'GOOG'], 
                    start_date=datetime.date(2023, 1, 1))
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.