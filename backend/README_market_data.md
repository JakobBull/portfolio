# Market Data Fetcher

This script fetches market data for stocks in your portfolio and watchlist using the Yahoo Finance API.

## Features

- Fetches current and historical prices for stocks in your portfolio and watchlist
- For portfolio stocks: Retrieves complete historical data since purchase date
- For watchlist stocks: Retrieves only the latest prices
- Imputes missing values for non-trading days using forward fill
- Updates portfolio positions with the latest prices
- Calculates unrealized profit/loss and return percentages
- Generates a summary report of your portfolio and watchlist
- Fetches benchmark data (S&P 500, NASDAQ, DAX 30) from the earliest portfolio date
- Handles API rate limiting and errors gracefully
- Caches data to avoid unnecessary API calls
- Supports specifying specific tickers to update

## Usage

```bash
python fetch_market_data.py [--days DAYS] [--verbose] [--update-portfolio] [--report] [--tickers TICKER1,TICKER2,...] [--full-history] [--impute-missing]
```

### Options

- `--days DAYS`: Number of days of historical data to fetch (default: 7)
- `--verbose`: Enable verbose logging
- `--update-portfolio`: Update portfolio positions with latest prices
- `--report`: Generate a summary report
- `--tickers TICKERS`: Comma-separated list of tickers to update (default: all tickers in portfolio and watchlist)
- `--full-history`: Fetch full history since position was added (default for portfolio stocks)
- `--impute-missing`: Impute missing values for non-trading days (default: True)

### Examples

Fetch data for all stocks in portfolio and watchlist:
```bash
python fetch_market_data.py
```

Fetch data and update portfolio positions:
```bash
python fetch_market_data.py --update-portfolio
```

Fetch data, update portfolio positions, and generate a report:
```bash
python fetch_market_data.py --update-portfolio --report
```

Fetch 30 days of historical data:
```bash
python fetch_market_data.py --days 30
```

Fetch data for specific tickers:
```bash
python fetch_market_data.py --tickers AAPL,MSFT,GOOGL
```

Fetch data without imputing missing values:
```bash
python fetch_market_data.py --no-impute-missing
```

## Data Handling

The script handles data differently based on the type of ticker:

1. **Portfolio Stocks**: 
   - Fetches complete historical data from the purchase date to today
   - Imputes missing values for non-trading days (weekends, holidays)
   - Stores all historical data points in the database

2. **Watchlist Stocks**:
   - Fetches only recent data (last N days specified by `--days`)
   - Primarily focused on current price for monitoring

3. **Benchmark Indices**:
   - Fetches data from the earliest portfolio position date
   - Includes S&P 500, NASDAQ, and DAX 30 by default
   - Used for performance comparison

## Output Files

The script generates the following output files:

- `market_data_fetch.log`: Log file with detailed information about the script execution
- `latest_market_data.json`: JSON file with the latest market data for all tickers
- `market_data_report.json`: JSON file with a summary report of your portfolio and watchlist
- `reports/market_data_report_YYYYMMDD_HHMMSS.json`: Archived reports with timestamps

## Scheduling with Cron

You can schedule the script to run automatically using cron. See `cron_example.txt` for examples.

## Dependencies

The script depends on the following modules:

- `database.py`: Database interface for storing and retrieving data
- `market_interface.py`: Interface to the Yahoo Finance API
- `yfinance`: Python library for accessing Yahoo Finance data
- `pandas`: Data manipulation library used for imputing missing values

## Error Handling

The script handles various error conditions:

- API rate limiting: Uses cached data when API limits are reached
- Network errors: Falls back to cached data when network errors occur
- Missing data: Uses last known prices when current prices are not available
- Weekend/holiday data: Adjusts dates to trading days and imputes missing values
- Missing purchase dates: Falls back to earliest portfolio date

## Logging

The script logs detailed information about its execution to both the console and a log file. Use the `--verbose` option for more detailed logging. 