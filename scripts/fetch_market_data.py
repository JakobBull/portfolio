import os
import sys
from datetime import date, datetime, timedelta
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# NOTE: This script stores ALL stock prices in EUR for consistent currency handling
# Prices are automatically converted from their original currencies to EUR

from backend.database import DatabaseManager, Stock
from backend.market_interface import MarketInterface
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def seed_benchmarks(db_manager: DatabaseManager):
    """Ensures benchmark stocks are present in the database."""
    benchmarks = [
        {'ticker': '^GSPC', 'name': 'S&P 500', 'currency': 'USD', 'sector': 'Index', 'country': 'USA'},
        {'ticker': '^IXIC', 'name': 'NASDAQ Composite', 'currency': 'USD', 'sector': 'Index', 'country': 'USA'},
        {'ticker': '^GDAXI', 'name': 'DAX 30', 'currency': 'EUR', 'sector': 'Index', 'country': 'Germany'},
    ]
    logging.info("Seeding benchmark data...")
    for bm in benchmarks:
        db_manager.add_stock(
            ticker=bm['ticker'],
            name=bm['name'],
            currency=bm['currency'],
            sector=bm['sector'],
            country=bm['country']
        )
    logging.info("Benchmark seeding complete.")

def _process_and_store(data: pd.DataFrame, ticker: str, db_manager: DatabaseManager):
    """Helper to process and store data for a single ticker (prices are in EUR)."""
    for index_date, row in data.iterrows():
        if isinstance(index_date, pd.Timestamp):
            price_date = index_date.date()
        else:
            price_date = index_date
        
        close_price = row.get('Close')
        if pd.notna(close_price):
            # Store the EUR-converted price
            db_manager.add_stock_price(ticker, price_date, float(close_price))

def update_stock_details(db_manager: DatabaseManager, market_interface: MarketInterface):
    """
    Fetches and updates missing details (name, currency, sector, country) for all stocks.
    """
    logging.info("Starting to update stock details (name, currency, sector, country)...")
    
    with db_manager.session_scope() as session:
        stocks_to_update = session.query(Stock).filter(
            (Stock.name.is_(None)) |
            (Stock.currency.is_(None)) |
            (Stock.sector.is_(None)) |
            (Stock.country.is_(None))
        ).all()

    if not stocks_to_update:
        logging.info("All stocks have complete details. No updates needed.")
        return

    tickers_to_fetch = [stock.ticker for stock in stocks_to_update]
    logging.info(f"Found {len(tickers_to_fetch)} stocks missing details. Fetching in a single batch...")

    try:
        all_stock_info = market_interface.get_stock_info_for_tickers(tickers_to_fetch)

        if not all_stock_info:
            logging.warning("Could not retrieve any stock info from the batch request.")
            return

        for stock in stocks_to_update:
            stock_info = all_stock_info.get(stock.ticker)

            if stock_info:
                db_manager.add_stock(
                    ticker=stock.ticker,
                    name=stock_info.get('name') or stock.name,
                    currency=stock_info.get('currency') or stock.currency,
                    sector=stock_info.get('sector') or stock.sector,
                    country=stock_info.get('country') or stock.country
                )
                logging.info(f"Updated details for {stock.ticker}.")
            else:
                logging.warning(f"No details found for {stock.ticker} in the batch response.")

    except Exception as e:
        logging.error(f"An error occurred during the batch update of stock details: {e}")

def fetch_and_store_market_data():
    """
    Fetches historical market data for all stocks in the database and stores it in EUR.
    All stock prices are automatically converted to EUR for consistent currency storage.
    Also updates missing stock details like sector and country.
    """
    db_manager = DatabaseManager()
    market_interface = MarketInterface()
    
    logging.info("🌍 Market Data Fetcher: All prices will be stored in EUR for consistent currency handling")

    try:
        # Step 1: Seed benchmarks to ensure they are in the database
        seed_benchmarks(db_manager)

        # Step 2: Update details for all stocks missing information
        update_stock_details(db_manager, market_interface)

        # Step 3: Fetch historical prices for all stocks in the database
        all_tickers = db_manager.get_all_stock_tickers()

        if not all_tickers:
            logging.info("No tickers found in the database. Nothing to fetch.")
            return

        default_start_date_str = os.environ.get("START_DATE", "2020-01-01")
        default_start_date = datetime.strptime(default_start_date_str, "%Y-%m-%d").date()
        end_date = date.today()

        tickers_by_start_date = {}
        logging.info("Checking for latest price data for each ticker...")
        for ticker in all_tickers:
            latest_price = db_manager.get_latest_stock_price(ticker)
            start_date = default_start_date
            if latest_price and latest_price.date:
                # If we have data, start fetching from the next day
                start_date = latest_price.date + timedelta(days=1)
            
            # Group tickers by the calculated start date
            if start_date not in tickers_by_start_date:
                tickers_by_start_date[start_date] = []
            tickers_by_start_date[start_date].append(ticker)

        logging.info("Finished grouping tickers by fetch start date.")

        for start_date, tickers_in_group in tickers_by_start_date.items():
            if start_date >= end_date:
                logging.info(f"Tickers {tickers_in_group} are already up-to-date. Skipping.")
                continue

            logging.info(f"Fetching data for {len(tickers_in_group)} tickers from {start_date} to {end_date} (converting to EUR).")
            
            batch_size = 10
            for i in range(0, len(tickers_in_group), batch_size):
                batch_tickers = tickers_in_group[i:i + batch_size]
                logging.info(f"Fetching batch: {batch_tickers} (will convert all prices to EUR)")

                historical_data = market_interface.get_historical_prices_for_tickers(batch_tickers, start_date, end_date, target_currency="EUR")

                if historical_data is None or historical_data.empty:
                    logging.warning(f"No data for batch: {batch_tickers}. Skipping.")
                    time.sleep(2) # Still sleep to be nice to the API
                    continue

                if isinstance(historical_data.columns, pd.MultiIndex):
                    for ticker in batch_tickers:
                        if ticker in historical_data.columns.get_level_values(0):
                            _process_and_store(historical_data[ticker], ticker, db_manager)
                else: # Only one ticker in batch
                    if len(batch_tickers) == 1:
                        _process_and_store(historical_data, batch_tickers[0], db_manager)
                
                logging.info(f"Successfully processed batch for: {batch_tickers} (stored prices in EUR)")
                time.sleep(2)
        
        logging.info("Successfully fetched and stored all new historical market data in EUR.")

    except Exception as e:
        logging.error(f"An error occurred during market data fetching: {e}")

if __name__ == "__main__":
    fetch_and_store_market_data() 