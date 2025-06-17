from __future__ import annotations
import os
import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, Date, ForeignKey, DateTime, Text, func, Enum, UniqueConstraint
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base, joinedload
import pandas as pd
from io import StringIO
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import enum
from contextlib import contextmanager
from backend.money_amount import MoneyAmount

# Set up logging
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'portfolio.db')
DB_URL = f"sqlite:///{DB_PATH}"

# Create base class for models
Base = declarative_base()

class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"

class Stock(Base):
    """Stock model for storing stock information"""
    __tablename__ = 'stocks'
    
    ticker = Column(String, primary_key=True)
    name = Column(String)
    currency = Column(String, default='USD')
    sector = Column(String)
    country = Column(String)
    target_price = Column(Float, default=None)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="stock", cascade="all, delete-orphan")
    dividends = relationship("Dividend", back_populates="stock", cascade="all, delete-orphan")
    watchlist_item = relationship("Watchlist", back_populates="stock", uselist=False, cascade="all, delete-orphan")

class StockPrice(Base):
    """Stock price model for storing historical prices"""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey('stocks.ticker'))
    date = Column(Date, nullable=False)
    price = Column(Float, nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")
    
    __table_args__ = (
        # Ensure we don't have duplicate prices for the same stock and date
        {'sqlite_autoincrement': True},
    )

class Transaction(Base):
    """Transaction model for storing portfolio transactions"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    type = Column(Enum("buy", "sell", "dividend"), nullable=False)
    ticker = Column(String, ForeignKey('stocks.ticker'))
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    currency = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    cost = Column(Float, default=0.0)
    
    # Relationships
    stock = relationship("Stock", back_populates="transactions")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )

class Dividend(Base):
    """Dividend payments table"""
    __tablename__ = 'dividends'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey('stocks.ticker'), nullable=False)
    date = Column(Date, nullable=False)  # Ex-dividend date
    amount_per_share = Column(Float, nullable=False)
    tax_withheld = Column(Float, default=0.0)
    currency = Column(String, nullable=False, default='USD')
    
    # Relationships
    stock = relationship("Stock", back_populates="dividends")
    
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='unique_dividend_per_stock_per_date'),
        {'sqlite_autoincrement': True},
    )

class Watchlist(Base):
    """Watchlist model for storing stocks to watch"""
    __tablename__ = 'watchlist'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey('stocks.ticker'), unique=True)
    strike_price = Column(Float)
    notes = Column(Text)
    date_added = Column(Date, default=datetime.date.today)
    
    # Relationships
    stock = relationship("Stock", back_populates="watchlist_item")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables
Base.metadata.create_all(bind=engine)

class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self, db_url: str = DB_URL):
        """Initialize database manager"""
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine,
            expire_on_commit=False
        )
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Database error occurred: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    # Stock operations
    def add_stock(self, ticker: str, name: str = None, currency: str = 'USD', sector: str = None, country: str = None) -> Stock | None:
        """Add or update a stock in the database"""
        with self.session_scope() as session:
            stock = session.query(Stock).filter(Stock.ticker == ticker).first()
            if stock:
                stock.name = name if name is not None else stock.name
                stock.currency = currency if currency is not None else stock.currency
                stock.sector = sector if sector is not None else stock.sector
                stock.country = country if country is not None else stock.country
            else:
                stock = Stock(ticker=ticker, name=name, currency=currency, sector=sector, country=country)
                session.add(stock)
            session.flush()
            session.refresh(stock)
            return stock
    
    def get_stock(self, ticker: str) -> Stock | None:
        """Get stock information"""
        with self.session_scope() as session:
            return session.query(Stock).filter(Stock.ticker == ticker).first()

    def update_stock_target_price(self, ticker: str, target_price: float) -> bool:
        """Update the target price for a stock"""
        with self.session_scope() as session:
            stock = session.query(Stock).filter(Stock.ticker == ticker).first()
            if stock:
                stock.target_price = target_price
                return True
            return False

    def get_all_stock_tickers(self) -> List[str]:
        """Get all stock tickers from the database."""
        with self.session_scope() as session:
            tickers = session.query(Stock.ticker).all()
            return [ticker[0] for ticker in tickers]
    
    # Stock price operations
    def add_stock_price(self, ticker: str, date: datetime.date, price: float) -> StockPrice | None:
        """Add or update a stock price"""
        with self.session_scope() as session:
            stock_price = session.query(StockPrice).filter_by(ticker=ticker, date=date).first()
            if stock_price:
                stock_price.price = price
            else:
                stock_price = StockPrice(ticker=ticker, date=date, price=price)
                session.add(stock_price)
            session.flush()
            session.refresh(stock_price)
            return stock_price
    
    def get_stock_price(self, ticker: str, date: datetime.date) -> StockPrice | None:
        """Get stock price for a specific date"""
        with self.session_scope() as session:
            return session.query(StockPrice).filter_by(ticker=ticker, date=date).first()
    
    def get_latest_stock_price(self, ticker: str) -> StockPrice | None:
        """Get the latest stock price"""
        with self.session_scope() as session:
            return session.query(StockPrice).filter_by(ticker=ticker).order_by(StockPrice.date.desc()).first()
    
    def get_historical_stock_prices(self, ticker: str, start_date: datetime.date,
                                  end_date: datetime.date) -> List[StockPrice]:
        """Get historical stock prices"""
        with self.session_scope() as session:
            return session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).order_by(StockPrice.date).all()
    
    def get_stock_price_at_date(self, ticker: str, target_date: datetime.date) -> Optional[float]:
        """Get stock price for a specific date, or closest available date"""
        with self.session_scope() as session:
            # Try exact date first
            exact_price = session.query(StockPrice).filter_by(ticker=ticker, date=target_date).first()
            if exact_price:
                return exact_price.price
            
            # Find closest date with available price (before or at target date)
            closest_price = (session.query(StockPrice)
                           .filter(StockPrice.ticker == ticker, StockPrice.date <= target_date)
                           .order_by(StockPrice.date.desc())
                           .first())
            
            if closest_price:
                return closest_price.price
            
            # If no historical price found, try getting any available price
            any_price = (session.query(StockPrice)
                        .filter(StockPrice.ticker == ticker)
                        .order_by(StockPrice.date.desc())
                        .first())
            
            return any_price.price if any_price else None

    # Transaction-based position calculations
    def get_portfolio_positions_at_date(self, target_date: datetime.date) -> Dict[str, float]:
        """
        Calculate portfolio positions (shares held) at a specific date from transactions.
        
        Args:
            target_date: Date to calculate positions for
            
        Returns:
            Dictionary mapping ticker -> shares held
        """
        with self.session_scope() as session:
            # Get all buy and sell transactions up to target date
            transactions = (session.query(Transaction)
                          .filter(Transaction.date <= target_date)
                          .filter(Transaction.type.in_(['buy', 'sell']))
                          .order_by(Transaction.date, Transaction.id)
                          .all())
            
            positions = {}
            for transaction in transactions:
                ticker = transaction.ticker
                if ticker not in positions:
                    positions[ticker] = 0.0
                
                if transaction.type == 'buy':
                    positions[ticker] += transaction.amount
                elif transaction.type == 'sell':
                    positions[ticker] -= transaction.amount
            
            # Remove positions with zero or negative shares
            return {ticker: shares for ticker, shares in positions.items() if shares > 0}
    
    def get_portfolio_value_at_date(self, target_date: datetime.date, currency: str = 'USD') -> float:
        """
        Calculate market value of portfolio positions at a specific date.
        
        Args:
            target_date: Date to calculate value for
            currency: Currency to return value in
            
        Returns:
            Total market value of stock positions (excludes cash/dividends)
        """
        positions = self.get_portfolio_positions_at_date(target_date)
        total_value = 0.0
        
        for ticker, shares in positions.items():
            price = self.get_stock_price_at_date(ticker, target_date)
            if price:
                # TODO: Add currency conversion if needed
                total_value += shares * price
        
        return total_value
    
    def get_total_return_at_date(self, target_date: datetime.date, currency: str = 'USD') -> float:
        """
        Calculate total return (portfolio value + dividends received) at a specific date.
        
        Args:
            target_date: Date to calculate total return for
            currency: Currency to return value in
            
        Returns:
            Total return (market value + cumulative dividends)
        """
        market_value = self.get_portfolio_value_at_date(target_date, currency)
        dividend_income = self.get_dividend_income_up_to_date(target_date, currency)
        return market_value + dividend_income
    
    def get_portfolio_values_over_time(self, start_date: datetime.date, end_date: datetime.date, 
                                     currency: str = 'USD') -> pd.Series:
        """
        Calculate portfolio values over a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            currency: Currency to return values in
            
        Returns:
            Pandas Series with dates as index and portfolio values
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        values = []
        
        for date in date_range:
            date_obj = date.date()
            value = self.get_portfolio_value_at_date(date_obj, currency)
            values.append(value)
        
        return pd.Series(values, index=date_range)
    
    def get_total_return_values_over_time(self, start_date: datetime.date, end_date: datetime.date, 
                                        currency: str = 'USD') -> pd.Series:
        """
        Calculate total return values (portfolio + dividends) over a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            currency: Currency to return values in
            
        Returns:
            Pandas Series with dates as index and total return values
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        values = []
        
        for date in date_range:
            date_obj = date.date()
            value = self.get_total_return_at_date(date_obj, currency)
            values.append(value)
        
        return pd.Series(values, index=date_range)
    
    def get_position_cost_basis(self, ticker: str, target_date: datetime.date, currency: str = 'USD') -> float:
        """
        Calculate the cost basis for a position at a specific date using FIFO method.
        
        Args:
            ticker: Stock ticker
            target_date: Date to calculate cost basis for
            currency: Currency to return cost basis in
            
        Returns:
            Total cost basis for the position
        """
        with self.session_scope() as session:
            # Get all transactions for this ticker up to target date
            transactions = (session.query(Transaction)
                          .filter(Transaction.ticker == ticker)
                          .filter(Transaction.date <= target_date)
                          .filter(Transaction.type.in_(['buy', 'sell']))
                          .order_by(Transaction.date, Transaction.id)
                          .all())
            
            if not transactions:
                return 0.0
            
            # Track purchases and calculate FIFO cost basis
            purchases = []  # List of (shares, price_per_share, cost)
            total_cost_basis = 0.0
            
            for transaction in transactions:
                if transaction.type == 'buy':
                    # Add to purchases
                    cost_per_share = (transaction.price * transaction.amount + transaction.cost) / transaction.amount
                    purchases.append({
                        'shares': transaction.amount,
                        'cost_per_share': cost_per_share,
                        'total_cost': transaction.price * transaction.amount + transaction.cost
                    })
                elif transaction.type == 'sell':
                    # Remove from purchases using FIFO
                    shares_to_sell = transaction.amount
                    while shares_to_sell > 0 and purchases:
                        oldest_purchase = purchases[0]
                        if oldest_purchase['shares'] <= shares_to_sell:
                            # Sell entire oldest purchase
                            shares_to_sell -= oldest_purchase['shares']
                            purchases.pop(0)
                        else:
                            # Partially sell oldest purchase
                            oldest_purchase['shares'] -= shares_to_sell
                            oldest_purchase['total_cost'] = oldest_purchase['shares'] * oldest_purchase['cost_per_share']
                            shares_to_sell = 0
            
            # Sum up remaining cost basis
            total_cost_basis = sum(purchase['total_cost'] for purchase in purchases)
            
            # TODO: Add currency conversion if needed
            return total_cost_basis

    # Transaction operations
    def add_transaction(self, transaction_type: str, ticker: str, amount: float, 
                       price: float, currency: str, transaction_date: datetime.date,
                       cost: float = 0.0) -> None:
        """Add a transaction to the database"""
        with self.session_scope() as session:
            transaction = Transaction(
                type=TransactionType(transaction_type),
                ticker=ticker,
                amount=amount,
                price=price,
                currency=currency,
                date=transaction_date,
                cost=cost
            )
            session.add(transaction)
    
    def get_all_transactions(self, transaction_type: str = None) -> List[Transaction]:
        """Get all transactions of a given type"""
        with self.session_scope() as session:
            query = session.query(Transaction)
            if transaction_type:
                query = query.filter(Transaction.type == transaction_type)
            return query.order_by(Transaction.date, Transaction.id).all()

    def get_positions_data_for_table(self, target_date: datetime.date = None) -> list[dict]:
        """
        Returns position data calculated from transactions for the frontend table.
        
        Args:
            target_date: Date to calculate positions for (defaults to today)
        """
        if target_date is None:
            target_date = datetime.date.today()
            
        positions = self.get_portfolio_positions_at_date(target_date)
        positions_data = []
        total_portfolio_value = 0.0
        
        # First pass: calculate individual position values for correct portfolio weights
        position_values = {}
        for ticker, shares in positions.items():
            latest_price_info = self.get_latest_stock_price(ticker)
            current_price = latest_price_info.price if latest_price_info else 0
            market_value = current_price * shares
            position_values[ticker] = market_value
            total_portfolio_value += market_value
        
        # Second pass: create position data with corrected calculations
        for ticker, shares in positions.items():
            stock = self.get_stock(ticker)
            if not stock:
                logger.warning(f"Stock data not found for ticker {ticker}")
                continue
                
            latest_price_info = self.get_latest_stock_price(ticker)
            current_price = latest_price_info.price if latest_price_info else 0
            market_value = position_values[ticker]
            cost_basis = self.get_position_cost_basis(ticker, target_date)
            
            # Calculate unrealized P/L (without dividends for this field)
            unrealized_pl = market_value - cost_basis
            
            # Calculate time-weighted return percentage
            return_pct = self.get_time_weighted_return_for_ticker(ticker, target_date, current_price)
            
            # Calculate portfolio weight as decimal (AG Grid will multiply by 100 for %)
            weight_pct = (market_value / total_portfolio_value) if total_portfolio_value > 0 else 0
            
            # Calculate dividend yield based on last 12 months dividends vs current price
            dividend_yield = self.get_dividend_yield_for_ticker(ticker, target_date)
            
            positions_data.append({
                'ticker': ticker,
                'name': stock.name,
                'shares': shares,
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'unrealized_pl': unrealized_pl,
                'return_pct': return_pct,
                'weight_pct': weight_pct,
                'target_price': stock.target_price,
                'dividend_yield': dividend_yield,
                'currency': stock.currency,
                'sector': stock.sector,
                'country': stock.country
            })
            
        return positions_data

    # Watchlist operations
    def add_watchlist_item(self, ticker: str, strike_price: float = None, 
                          notes: str = None, date_added: Optional[datetime.date] = None) -> Watchlist | None:
        """Add an item to the watchlist, or return existing if found."""
        with self.session_scope() as session:
            existing_item = session.query(Watchlist).filter_by(ticker=ticker).first()
            if existing_item:
                logger.warning(f"Ticker {ticker} is already in the watchlist.")
                return existing_item

            item = Watchlist(
                ticker=ticker,
                strike_price=strike_price,
                notes=notes,
                date_added=date_added or datetime.date.today()
            )
            session.add(item)
            session.flush()
            session.refresh(item)
            return item
    
    def get_all_watchlist_items(self) -> List[Watchlist]:
        """Get all items from the watchlist"""
        with self.session_scope() as session:
            return session.query(Watchlist).options(joinedload(Watchlist.stock)).all()
    
    def get_watchlist_data_for_table(self) -> list[dict]:
        """Returns a list of dictionaries with watchlist data for the frontend table."""
        with self.session_scope() as session:
            watchlist_items = session.query(Watchlist).options(joinedload(Watchlist.stock)).all()
            watchlist_data = []
            for item in watchlist_items:
                stock = item.stock
                if not stock:
                    logger.warning(f"Watchlist item with ticker {item.ticker} is missing stock data.")
                    continue

                latest_price_info = self.get_latest_stock_price(item.ticker)
                current_price = latest_price_info.price if latest_price_info else 0
                
                watchlist_data.append({
                    'id': item.id,
                    'ticker': stock.ticker,
                    'name': stock.name,
                    'sector': stock.sector,
                    'country': stock.country,
                    'currency': stock.currency,
                    'current_price': current_price,
                    'strike_price': item.strike_price,
                    'notes': item.notes,
                    'date_added': item.date_added.isoformat(),
                    'actions': f'[Remove](/remove-from-watchlist/{stock.ticker})'
                })
            return watchlist_data

    def get_transactions_data_for_table(self) -> list[dict]:
        """Returns a list of dictionaries with transaction data for the frontend table."""
        with self.session_scope() as session:
            transactions = session.query(Transaction).options(joinedload(Transaction.stock)).order_by(Transaction.date.desc(), Transaction.id.desc()).all()
            transactions_data = []
            for transaction in transactions:
                stock = transaction.stock
                stock_name = stock.name if stock else transaction.ticker
                
                transactions_data.append({
                    'id': transaction.id,
                    'type': transaction.type,
                    'ticker': transaction.ticker,
                    'name': stock_name,
                    'amount': transaction.amount,
                    'price': transaction.price,
                    'currency': transaction.currency,
                    'date': transaction.date.isoformat(),
                    'cost': transaction.cost,
                    'total_value': transaction.amount * transaction.price
                })
            return transactions_data

    def get_dividends_data_for_table(self) -> list[dict]:
        """Returns a list of dictionaries with dividend data for the frontend table."""
        with self.session_scope() as session:
            dividends = session.query(Dividend).options(joinedload(Dividend.stock)).order_by(Dividend.date.desc(), Dividend.id.desc()).all()
            dividends_data = []
            for dividend in dividends:
                stock = dividend.stock
                stock_name = stock.name if stock else dividend.ticker
                
                # Get shares held at dividend date to calculate total dividend received
                shares_held = self.get_shares_held_at_date(dividend.ticker, dividend.date)
                total_dividend = dividend.amount_per_share * shares_held if shares_held > 0 else 0
                
                dividends_data.append({
                    'id': dividend.id,
                    'ticker': dividend.ticker,
                    'name': stock_name,
                    'date': dividend.date.isoformat(),
                    'amount_per_share': dividend.amount_per_share,
                    'shares_held': shares_held,
                    'total_dividend': total_dividend,
                    'tax_withheld': dividend.tax_withheld,
                    'currency': dividend.currency
                })
            return dividends_data

    def update_watchlist_item(self, item_id: int, **kwargs) -> bool:
        """Update a watchlist item"""
        with self.session_scope() as session:
            item = session.query(Watchlist).filter(Watchlist.id == item_id).first()
            if item:
                for key, value in kwargs.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                return True
            return False

    def update_transaction(self, transaction_id: int, **kwargs) -> bool:
        """Update a transaction"""
        with self.session_scope() as session:
            transaction = session.query(Transaction).filter(Transaction.id == transaction_id).first()
            if transaction:
                for key, value in kwargs.items():
                    if hasattr(transaction, key):
                        # Handle date conversion
                        if key == 'date' and isinstance(value, str):
                            value = datetime.datetime.strptime(value, '%Y-%m-%d').date()
                        setattr(transaction, key, value)
                return True
            return False

    def update_dividend(self, dividend_id: int, **kwargs) -> bool:
        """Update a dividend"""
        with self.session_scope() as session:
            dividend = session.query(Dividend).filter(Dividend.id == dividend_id).first()
            if dividend:
                for key, value in kwargs.items():
                    if hasattr(dividend, key):
                        # Handle date conversion
                        if key == 'date' and isinstance(value, str):
                            value = datetime.datetime.strptime(value, '%Y-%m-%d').date()
                        setattr(dividend, key, value)
                return True
            return False

    def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction"""
        with self.session_scope() as session:
            transaction = session.query(Transaction).filter(Transaction.id == transaction_id).first()
            if transaction:
                session.delete(transaction)
                return True
            return False

    def delete_dividend(self, dividend_id: int) -> bool:
        """Delete a dividend"""
        with self.session_scope() as session:
            dividend = session.query(Dividend).filter(Dividend.id == dividend_id).first()
            if dividend:
                session.delete(dividend)
                return True
            return False

    def delete_watchlist_item(self, item_id: int) -> bool:
        """Delete a watchlist item by ID"""
        with self.session_scope() as session:
            item = session.query(Watchlist).filter(Watchlist.id == item_id).first()
            if item:
                session.delete(item)
                return True
            return False

    def get_tickers_to_track(self) -> List[str]:
        """Get all unique tickers from transactions and watchlist"""
        with self.session_scope() as session:
            transaction_tickers = session.query(Transaction.ticker).distinct().all()
            watchlist_tickers = session.query(Watchlist.ticker).distinct().all()
            
            tickers = {t[0] for t in transaction_tickers}
            tickers.update({w[0] for w in watchlist_tickers})
            
            return list(tickers)

    def remove_watchlist_item(self, ticker: str) -> bool:
        """Remove an item from the watchlist by ticker"""
        with self.session_scope() as session:
            item = session.query(Watchlist).filter_by(ticker=ticker).first()
            if item:
                session.delete(item)
                return True
            return False

    # Dividend operations
    def add_dividend(self, ticker: str, date: datetime.date, amount_per_share: float, 
                    tax_withheld: float = 0.0, currency: str = 'USD') -> Dividend | None:
        """Add or update a dividend record"""
        with self.session_scope() as session:
            dividend = session.query(Dividend).filter_by(ticker=ticker, date=date).first()
            if dividend:
                dividend.amount_per_share = amount_per_share
                dividend.tax_withheld = tax_withheld
                dividend.currency = currency
            else:
                dividend = Dividend(
                    ticker=ticker,
                    date=date,
                    amount_per_share=amount_per_share,
                    tax_withheld=tax_withheld,
                    currency=currency
                )
                session.add(dividend)
            session.flush()
            session.refresh(dividend)
            return dividend

    def get_latest_dividend_date(self, ticker: str) -> datetime.date | None:
        """Get the latest dividend date for a ticker"""
        with self.session_scope() as session:
            latest_dividend = session.query(Dividend).filter_by(ticker=ticker).order_by(Dividend.date.desc()).first()
            return latest_dividend.date if latest_dividend else None

    def get_all_dividends(self, ticker: str = None) -> List[Dividend]:
        """Get all dividends, optionally filtered by ticker"""
        with self.session_scope() as session:
            query = session.query(Dividend)
            if ticker:
                query = query.filter(Dividend.ticker == ticker)
            return query.order_by(Dividend.date).all()

    def get_dividend_income_up_to_date(self, target_date: datetime.date, currency: str = 'USD') -> float:
        """
        Calculate total dividend income received up to a specific date.
        
        Args:
            target_date: Date to calculate dividend income up to
            currency: Currency to return income in
            
        Returns:
            Total dividend income received up to the target date
        """
        with self.session_scope() as session:
            # Get all dividends up to target date
            dividends = (session.query(Dividend)
                        .filter(Dividend.date <= target_date)
                        .all())
            
            total_dividend_income = 0.0
            
            for dividend in dividends:
                # Get shares held at the dividend ex-date to calculate total dividend received
                shares_held = self.get_shares_held_at_date(dividend.ticker, dividend.date)
                if shares_held > 0:
                    dividend_received = dividend.amount_per_share * shares_held
                    # TODO: Add currency conversion if needed
                    total_dividend_income += dividend_received
            
            return total_dividend_income

    def get_shares_held_at_date(self, ticker: str, target_date: datetime.date) -> float:
        """
        Calculate shares held for a specific ticker at a specific date.
        
        Args:
            ticker: Stock ticker
            target_date: Date to calculate shares held for
            
        Returns:
            Number of shares held at the target date
        """
        with self.session_scope() as session:
            # Get all buy and sell transactions for this ticker up to target date
            transactions = (session.query(Transaction)
                          .filter(Transaction.ticker == ticker)
                          .filter(Transaction.date <= target_date)
                          .filter(Transaction.type.in_(['buy', 'sell']))
                          .order_by(Transaction.date, Transaction.id)
                          .all())
            
            shares_held = 0.0
            for transaction in transactions:
                if transaction.type == 'buy':
                    shares_held += transaction.amount
                elif transaction.type == 'sell':
                    shares_held -= transaction.amount
            
            return max(0.0, shares_held)  # Don't return negative shares

    def get_dividend_income_for_ticker_up_to_date(self, ticker: str, target_date: datetime.date, currency: str = 'USD') -> float:
        """
        Calculate total dividend income received for a specific ticker up to a specific date.
        
        Args:
            ticker: Stock ticker
            target_date: Date to calculate dividend income up to
            currency: Currency to return income in
            
        Returns:
            Total dividend income received for the ticker up to the target date
        """
        with self.session_scope() as session:
            # Get all dividends for this ticker up to target date
            dividends = (session.query(Dividend)
                        .filter(Dividend.ticker == ticker)
                        .filter(Dividend.date <= target_date)
                        .all())
            
            total_dividend_income = 0.0
            
            for dividend in dividends:
                # Get shares held at the dividend ex-date to calculate total dividend received
                shares_held = self.get_shares_held_at_date(ticker, dividend.date)
                if shares_held > 0:
                    dividend_received = dividend.amount_per_share * shares_held
                    # TODO: Add currency conversion if needed
                    total_dividend_income += dividend_received
            
            return total_dividend_income

    def get_time_weighted_return_for_ticker(self, ticker: str, target_date: datetime.date, current_price: float) -> float:
        """
        Calculate time-weighted return for a ticker, considering individual purchase timing.
        Uses proper CAGR (Compound Annual Growth Rate) formula.
        Only considers shares that are still held (uses FIFO to determine which purchases are still active).
        
        Args:
            ticker: Stock ticker
            target_date: Date to calculate return for
            current_price: Current stock price
            
        Returns:
            Annualized return percentage weighted by purchase values
        """
        with self.session_scope() as session:
            # Get all buy and sell transactions for this ticker up to target date
            transactions = (session.query(Transaction)
                          .filter(Transaction.ticker == ticker)
                          .filter(Transaction.type.in_(['buy', 'sell']))
                          .filter(Transaction.date <= target_date)
                          .order_by(Transaction.date, Transaction.id)
                          .all())
            
            if not transactions or current_price <= 0:
                return 0.0
            
            # Track remaining shares from each purchase using FIFO
            active_purchases = []  # List of dicts with purchase info
            
            for transaction in transactions:
                if transaction.type == 'buy':
                    # Include transaction costs in the cost basis for consistency
                    cost_per_share = (transaction.price * transaction.amount + transaction.cost) / transaction.amount
                    active_purchases.append({
                        'shares': transaction.amount,
                        'price': transaction.price,
                        'cost_per_share': cost_per_share,  # This includes transaction costs
                        'date': transaction.date,
                        'value': transaction.amount * cost_per_share  # Use cost_per_share for weighting
                    })
                elif transaction.type == 'sell':
                    # Remove sold shares using FIFO
                    shares_to_sell = transaction.amount
                    while shares_to_sell > 0 and active_purchases:
                        oldest_purchase = active_purchases[0]
                        if oldest_purchase['shares'] <= shares_to_sell:
                            # Sell entire oldest purchase
                            shares_to_sell -= oldest_purchase['shares']
                            active_purchases.pop(0)
                        else:
                            # Partially sell oldest purchase
                            old_shares = oldest_purchase['shares']
                            oldest_purchase['shares'] -= shares_to_sell
                            # Adjust value proportionally using cost_per_share
                            oldest_purchase['value'] = oldest_purchase['shares'] * oldest_purchase['cost_per_share']
                            shares_to_sell = 0
            
            if not active_purchases:
                return 0.0
            
            # Calculate weighted return for remaining purchases
            total_weighted_return = 0.0
            total_weight = 0.0
            
            for purchase in active_purchases:
                # Calculate years held for this specific purchase
                days_held = (target_date - purchase['date']).days
                years_held = max(days_held / 365.25, 1/365.25)  # Minimum 1 day to avoid division issues
                
                # Calculate total return multiple for this specific purchase (using cost_per_share that includes fees)
                price_multiple = current_price / purchase['cost_per_share']
                
                # Calculate CAGR (Compound Annual Growth Rate)
                if price_multiple > 0 and years_held > 0:
                    annualized_return = (price_multiple ** (1 / years_held)) - 1
                else:
                    annualized_return = 0.0
                
                # Weight by the remaining value of this purchase
                weighted_return = annualized_return * purchase['value']
                
                total_weighted_return += weighted_return
                total_weight += purchase['value']
            
            if total_weight == 0:
                return 0.0
            
            # Return weighted average annualized return as decimal (AG Grid will multiply by 100 for %)
            return (total_weighted_return / total_weight)

    def get_earliest_purchase_date(self, ticker: str) -> datetime.date | None:
        """
        Get the earliest purchase date for a ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Earliest purchase date or None if no purchases found
        """
        with self.session_scope() as session:
            earliest_transaction = (session.query(Transaction)
                                   .filter(Transaction.ticker == ticker)
                                   .filter(Transaction.type == 'buy')
                                   .order_by(Transaction.date)
                                   .first())
            
            return earliest_transaction.date if earliest_transaction else None

    def get_years_held(self, ticker: str, target_date: datetime.date) -> float:
        """
        Calculate the number of years a position has been held.
        
        Args:
            ticker: Stock ticker
            target_date: Date to calculate years held until
            
        Returns:
            Number of years held (as decimal)
        """
        earliest_purchase = self.get_earliest_purchase_date(ticker)
        if not earliest_purchase:
            return 0.0
        
        days_held = (target_date - earliest_purchase).days
        return max(days_held / 365.25, 0.1)  # Minimum 0.1 years to avoid division by zero

    def get_dividend_yield_for_ticker(self, ticker: str, target_date: datetime.date = None) -> float:
        """
        Calculate dividend yield for a ticker based on dividends over the past 12 months.
        
        Args:
            ticker: Stock ticker
            target_date: Date to calculate yield for (defaults to today)
            
        Returns:
            Dividend yield as a percentage
        """
        if target_date is None:
            target_date = datetime.date.today()
        
        # Calculate date 12 months ago
        twelve_months_ago = target_date - datetime.timedelta(days=365)
        
        with self.session_scope() as session:
            # Get dividends from the past 12 months
            dividends = (session.query(Dividend)
                        .filter(Dividend.ticker == ticker)
                        .filter(Dividend.date > twelve_months_ago)
                        .filter(Dividend.date <= target_date)
                        .all())
            
            if not dividends:
                return 0.0
            
            # Calculate total dividend per share over the past 12 months
            total_dividend_per_share = sum(dividend.amount_per_share for dividend in dividends)
            
            # Get the current stock price (use target_date price if available, otherwise latest)
            current_price = self.get_stock_price_at_date(ticker, target_date)
            if not current_price or current_price <= 0:
                # Fallback to latest price if no price available for target date
                latest_price_info = self.get_latest_stock_price(ticker)
                if not latest_price_info or latest_price_info.price <= 0:
                    return 0.0
                current_price = latest_price_info.price
            
            # Calculate dividend yield as decimal (AG Grid will multiply by 100 for %)
            dividend_yield = (total_dividend_per_share / current_price)
            
            return dividend_yield

# Singleton instance of the database manager
db_manager = DatabaseManager() 