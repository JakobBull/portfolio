import os
import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import json
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, Date, ForeignKey, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
import pandas as pd
from io import StringIO

# Set up logging
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'portfolio.db')
DB_URL = f"sqlite:///{DB_PATH}"

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Define SQLAlchemy models
class Portfolio(Base):
    """Model for portfolio positions"""
    __tablename__ = "portfolio"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)  # Company/stock name
    sector = Column(String)  # Industry sector
    country = Column(String)  # Country/market of the stock
    amount = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    purchase_currency = Column(String, nullable=False)
    purchase_date = Column(Date, nullable=False)
    last_value = Column(Float)
    last_value_currency = Column(String)
    last_value_date = Column(Date)
    unrealized_pl = Column(Float)
    return_percentage = Column(Float)
    total_dividends = Column(Float, default=0.0)
    cost_basis = Column(Float)
    
    def __repr__(self):
        return f"<Position(ticker='{self.ticker}', amount={self.amount})>"

class Transaction(Base):
    """Model for transactions"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)  # buy, sell, dividend
    ticker = Column(String, nullable=False, index=True)
    name = Column(String)  # Company/stock name
    sector = Column(String)  # Industry sector
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    currency = Column(String, nullable=False)
    transaction_date = Column(Date, nullable=False)
    cost = Column(Float, default=0.0)
    is_dividend = Column(Boolean, default=False)
    country = Column(String)  # Country/market of the stock
    
    def __repr__(self):
        return f"<Transaction(type='{self.type}', ticker='{self.ticker}', amount={self.amount})>"

class Watchlist(Base):
    """Model for watchlist items"""
    __tablename__ = "watchlist"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)  # Company/stock name
    sector = Column(String)  # Industry sector
    country = Column(String)  # Country/market of the stock
    strike_price = Column(Float)  # Target price for alerts
    date_added = Column(Date, nullable=False)
    notes = Column(Text)
    
    def __repr__(self):
        return f"<Watchlist(ticker='{self.ticker}')>"

class MarketData(Base):
    """Model for market data"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(Date, nullable=False, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float, nullable=False)
    volume = Column(Float)
    currency = Column(String, nullable=False)
    is_synthetic = Column(Boolean, default=False)
    
    __table_args__ = (
        # Composite unique constraint
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f"<MarketData(ticker='{self.ticker}', date='{self.date}', close={self.close_price}, synthetic={self.is_synthetic})>"

class ExchangeRate(Base):
    """Model for exchange rates"""
    __tablename__ = "exchange_rates"
    
    id = Column(Integer, primary_key=True)
    from_currency = Column(String, nullable=False)
    to_currency = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    rate = Column(Float, nullable=False)
    last_updated = Column(DateTime, nullable=False)
    
    __table_args__ = (
        # Composite unique constraint
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f"<ExchangeRate(from='{self.from_currency}', to='{self.to_currency}', rate={self.rate})>"

class APIRequest(Base):
    """Model for API requests"""
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True)
    api_name = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    
    def __repr__(self):
        return f"<APIRequest(api_name='{self.api_name}', timestamp='{self.timestamp}', success={self.success})>"

# Create all tables
Base.metadata.create_all(bind=engine)

class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self, db_url: str = DB_URL):
        """Initialize database manager"""
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def close_session(self, session: Session) -> None:
        """Close a database session"""
        session.close()
        
    # Portfolio operations
    def add_position(self, ticker: str, amount: float, purchase_price: float, 
                    purchase_currency: str, purchase_date: datetime.date,
                    cost_basis: float = None, name: str = None, sector: str = None,
                    country: str = None) -> bool:
        """
        Add or update a position in the portfolio
        
        Args:
            ticker: Stock ticker symbol
            amount: Number of shares
            purchase_price: Purchase price per share
            purchase_currency: Currency of the purchase
            purchase_date: Date of the purchase
            cost_basis: Total cost basis including fees
            name: Company/stock name
            sector: Industry sector
            country: Country/market of the stock
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Check if position already exists
            position = session.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            
            if position:
                # Update existing position
                position.amount = amount
                position.purchase_price = purchase_price
                position.purchase_currency = purchase_currency
                position.purchase_date = purchase_date
                if cost_basis is not None:
                    position.cost_basis = cost_basis
                if name is not None:
                    position.name = name
                if sector is not None:
                    position.sector = sector
                if country is not None:
                    position.country = country
            else:
                # Create new position
                position = Portfolio(
                    ticker=ticker,
                    amount=amount,
                    purchase_price=purchase_price,
                    purchase_currency=purchase_currency,
                    purchase_date=purchase_date,
                    cost_basis=cost_basis or (purchase_price * amount),
                    name=name,
                    sector=sector,
                    country=country
                )
                session.add(position)
                
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding position: {e}")
            return False
        finally:
            self.close_session(session)
            
    def update_position_value(self, ticker: str, last_value: float, 
                             value_currency: str, value_date: datetime.date,
                             unrealized_pl: float = None, 
                             return_percentage: float = None) -> bool:
        """
        Update the current value of a position
        
        Args:
            ticker: Stock ticker symbol
            last_value: Current value of the position
            value_currency: Currency of the value
            value_date: Date of the valuation
            unrealized_pl: Unrealized profit/loss
            return_percentage: Return percentage
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Get position
            position = session.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            
            if not position:
                logger.warning(f"Position {ticker} not found")
                return False
                
            # Update position
            position.last_value = last_value
            position.last_value_currency = value_currency
            position.last_value_date = value_date
            
            if unrealized_pl is not None:
                position.unrealized_pl = unrealized_pl
                
            if return_percentage is not None:
                position.return_percentage = return_percentage
                
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating position value: {e}")
            return False
        finally:
            self.close_session(session)
            
    def update_position_dividends(self, ticker: str, total_dividends: float) -> bool:
        """
        Update the total dividends for a position
        
        Args:
            ticker: Stock ticker symbol
            total_dividends: Total dividends received
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Get position
            position = session.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            
            if not position:
                logger.warning(f"Position {ticker} not found")
                return False
                
            # Update position
            position.total_dividends = total_dividends
                
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating position dividends: {e}")
            return False
        finally:
            self.close_session(session)
            
    def delete_position(self, ticker: str) -> bool:
        """
        Delete a position from the portfolio
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Get position
            position = session.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            
            if not position:
                logger.warning(f"Position {ticker} not found")
                return False
                
            # Delete position
            session.delete(position)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting position: {e}")
            return False
        finally:
            self.close_session(session)
            
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all positions in the portfolio
        
        Returns:
            List of dictionaries with position data
        """
        session = self.get_session()
        try:
            positions = session.query(Portfolio).all()
            
            result = []
            for position in positions:
                result.append({
                    'id': position.id,
                    'ticker': position.ticker,
                    'name': position.name,
                    'sector': position.sector,
                    'country': position.country,
                    'amount': position.amount,
                    'purchase_price': position.purchase_price,
                    'purchase_currency': position.purchase_currency,
                    'purchase_date': position.purchase_date,
                    'last_value': position.last_value,
                    'last_value_currency': position.last_value_currency,
                    'last_value_date': position.last_value_date,
                    'unrealized_pl': position.unrealized_pl,
                    'return_percentage': position.return_percentage,
                    'total_dividends': position.total_dividends,
                    'cost_basis': position.cost_basis
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
        finally:
            self.close_session(session)
            
    def get_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific position
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with position data or None if not found
        """
        session = self.get_session()
        try:
            position = session.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            
            if not position:
                return None
                
            return {
                'id': position.id,
                'ticker': position.ticker,
                'amount': position.amount,
                'purchase_price': position.purchase_price,
                'purchase_currency': position.purchase_currency,
                'purchase_date': position.purchase_date,
                'last_value': position.last_value,
                'last_value_currency': position.last_value_currency,
                'last_value_date': position.last_value_date,
                'unrealized_pl': position.unrealized_pl,
                'return_percentage': position.return_percentage,
                'total_dividends': position.total_dividends,
                'cost_basis': position.cost_basis,
                'name': position.name,
                'sector': position.sector,
                'country': position.country
            }
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None
        finally:
            self.close_session(session)
    
    # Transaction operations
    def add_transaction(self, transaction_type: str, ticker: str, amount: float, 
                       price: float, currency: str, transaction_date: datetime.date,
                       cost: float = 0.0, is_dividend: bool = False, 
                       name: str = None, sector: str = None, country: str = None) -> bool:
        """
        Add a transaction to the database
        
        Args:
            transaction_type: Type of transaction (buy, sell, dividend)
            ticker: Stock ticker symbol
            amount: Number of shares
            price: Price per share
            currency: Currency of the transaction
            transaction_date: Date of the transaction
            cost: Transaction cost
            is_dividend: Whether this is a dividend transaction
            name: Company/stock name
            sector: Industry sector
            country: Country/market of the stock
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Create new transaction
            transaction = Transaction(
                type=transaction_type,
                ticker=ticker,
                amount=amount,
                price=price,
                currency=currency,
                transaction_date=transaction_date,
                cost=cost,
                is_dividend=is_dividend,
                name=name,
                sector=sector,
                country=country
            )
            
            session.add(transaction)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding transaction: {e}")
            return False
        finally:
            self.close_session(session)
            
    def get_all_transactions(self, transaction_type: str = None) -> List[Dict[str, Any]]:
        """
        Get all transactions
        
        Args:
            transaction_type: Optional filter by transaction type
            
        Returns:
            List of dictionaries with transaction data
        """
        session = self.get_session()
        try:
            query = session.query(Transaction)
            
            if transaction_type:
                query = query.filter(Transaction.type == transaction_type)
                
            transactions = query.order_by(Transaction.transaction_date).all()
            
            result = []
            for transaction in transactions:
                result.append({
                    'id': transaction.id,
                    'type': transaction.type,
                    'ticker': transaction.ticker,
                    'name': transaction.name,
                    'sector': transaction.sector,
                    'amount': transaction.amount,
                    'price': transaction.price,
                    'currency': transaction.currency,
                    'transaction_date': transaction.transaction_date,
                    'cost': transaction.cost,
                    'is_dividend': transaction.is_dividend
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return []
        finally:
            self.close_session(session)
            
    def get_transactions_by_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get all transactions for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of dictionaries with transaction data
        """
        session = self.get_session()
        try:
            transactions = session.query(Transaction).filter(
                Transaction.ticker == ticker
            ).order_by(Transaction.transaction_date).all()
            
            result = []
            for transaction in transactions:
                result.append({
                    'id': transaction.id,
                    'type': transaction.type,
                    'ticker': transaction.ticker,
                    'name': transaction.name,
                    'sector': transaction.sector,
                    'amount': transaction.amount,
                    'price': transaction.price,
                    'currency': transaction.currency,
                    'transaction_date': transaction.transaction_date,
                    'cost': transaction.cost,
                    'is_dividend': transaction.is_dividend
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting transactions by ticker: {e}")
            return []
        finally:
            self.close_session(session)
            
    def get_dividend_transactions(self) -> List[Dict[str, Any]]:
        """
        Get all dividend transactions
        
        Returns:
            List of dictionaries with dividend transaction data
        """
        session = self.get_session()
        try:
            transactions = session.query(Transaction).filter(
                Transaction.is_dividend == True
            ).order_by(Transaction.transaction_date).all()
            
            result = []
            for transaction in transactions:
                result.append({
                    'id': transaction.id,
                    'ticker': transaction.ticker,
                    'name': transaction.name,
                    'sector': transaction.sector,
                    'amount': transaction.amount,
                    'dividend_per_share': transaction.price,
                    'currency': transaction.currency,
                    'transaction_date': transaction.transaction_date
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting dividend transactions: {e}")
            return []
        finally:
            self.close_session(session)
    
    # Watchlist operations
    def add_watchlist_item(self, ticker: str, strike_price: float = None, 
                          notes: str = None, name: str = None, sector: str = None,
                          country: str = None) -> bool:
        """
        Add or update an item in the watchlist
        
        Args:
            ticker: Stock ticker symbol
            strike_price: Target price for alerts
            notes: Additional notes
            name: Company/stock name
            sector: Industry sector
            country: Country/market of the stock
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Check if item already exists
            item = session.query(Watchlist).filter(Watchlist.ticker == ticker).first()
            
            if item:
                # Update existing item
                if strike_price is not None:
                    item.strike_price = strike_price
                if notes is not None:
                    item.notes = notes
                if name is not None:
                    item.name = name
                if sector is not None:
                    item.sector = sector
                if country is not None:
                    item.country = country
            else:
                # Create new item
                item = Watchlist(
                    ticker=ticker,
                    strike_price=strike_price,
                    date_added=datetime.date.today(),
                    notes=notes,
                    name=name,
                    sector=sector,
                    country=country
                )
                session.add(item)
                
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding watchlist item: {e}")
            return False
        finally:
            self.close_session(session)
            
    def delete_watchlist_item(self, ticker: str) -> bool:
        """
        Delete an item from the watchlist
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Get item
            item = session.query(Watchlist).filter(Watchlist.ticker == ticker).first()
            
            if not item:
                logger.warning(f"Watchlist item {ticker} not found")
                return False
                
            # Delete item
            session.delete(item)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting watchlist item: {e}")
            return False
        finally:
            self.close_session(session)
            
    def get_all_watchlist_items(self) -> List[Dict[str, Any]]:
        """
        Get all items in the watchlist
        
        Returns:
            List of dictionaries with watchlist item data
        """
        session = self.get_session()
        try:
            items = session.query(Watchlist).all()
            
            result = []
            for item in items:
                result.append({
                    'id': item.id,
                    'ticker': item.ticker,
                    'name': item.name,
                    'sector': item.sector,
                    'country': item.country,
                    'strike_price': item.strike_price,
                    'date_added': item.date_added,
                    'notes': item.notes
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting watchlist items: {e}")
            return []
        finally:
            self.close_session(session)
            
    def get_watchlist_item(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get a single watchlist item by ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with watchlist item data or None if not found
        """
        session = self.get_session()
        try:
            item = session.query(Watchlist).filter(Watchlist.ticker == ticker).first()
            
            if not item:
                return None
                
            return {
                'id': item.id,
                'ticker': item.ticker,
                'name': item.name,
                'sector': item.sector,
                'country': item.country,
                'strike_price': item.strike_price,
                'date_added': item.date_added,
                'notes': item.notes
            }
        except Exception as e:
            logger.error(f"Error getting watchlist item: {e}")
            return None
        finally:
            self.close_session(session)
    
    # Market data operations
    def add_market_data(self, ticker: str, date: datetime.date, close_price: float,
                       currency: str, open_price: float = None, high_price: float = None,
                       low_price: float = None, volume: float = None, is_synthetic: bool = False) -> bool:
        """
        Add or update market data
        
        Args:
            ticker: Stock ticker symbol
            date: Date of the market data
            close_price: Closing price
            currency: Currency of the prices
            open_price: Opening price
            high_price: High price
            low_price: Low price
            volume: Trading volume
            is_synthetic: Whether this data is synthetic (e.g., copied from last known data)
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Check if data already exists
            data = session.query(MarketData).filter(
                MarketData.ticker == ticker,
                MarketData.date == date
            ).first()
            
            if data:
                # Update existing data
                data.close_price = close_price
                data.currency = currency
                data.is_synthetic = is_synthetic
                if open_price is not None:
                    data.open_price = open_price
                if high_price is not None:
                    data.high_price = high_price
                if low_price is not None:
                    data.low_price = low_price
                if volume is not None:
                    data.volume = volume
            else:
                # Create new data
                data = MarketData(
                    ticker=ticker,
                    date=date,
                    close_price=close_price,
                    currency=currency,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    volume=volume,
                    is_synthetic=is_synthetic
                )
                session.add(data)
                
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding market data: {e}")
            return False
        finally:
            self.close_session(session)
            
    def get_market_data(self, ticker: str, date: datetime.date) -> Optional[Dict[str, Any]]:
        """
        Get market data for a specific ticker and date
        
        Args:
            ticker: Stock ticker symbol
            date: Date of the market data
            
        Returns:
            Dictionary with market data or None if not found
        """
        session = self.get_session()
        try:
            data = session.query(MarketData).filter(
                MarketData.ticker == ticker,
                MarketData.date == date
            ).first()
            
            if not data:
                return None
                
            return {
                'id': data.id,
                'ticker': data.ticker,
                'date': data.date,
                'open_price': data.open_price,
                'high_price': data.high_price,
                'low_price': data.low_price,
                'close_price': data.close_price,
                'volume': data.volume,
                'currency': data.currency
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
        finally:
            self.close_session(session)
            
    def get_latest_market_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest market data for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with market data or None if not found
        """
        session = self.get_session()
        try:
            data = session.query(MarketData).filter(
                MarketData.ticker == ticker
            ).order_by(MarketData.date.desc()).first()
            
            if not data:
                return None
                
            return {
                'id': data.id,
                'ticker': data.ticker,
                'date': data.date,
                'open_price': data.open_price,
                'high_price': data.high_price,
                'low_price': data.low_price,
                'close_price': data.close_price,
                'volume': data.volume,
                'currency': data.currency
            }
        except Exception as e:
            logger.error(f"Error getting latest market data: {e}")
            return None
        finally:
            self.close_session(session)
            
    def get_historical_market_data(self, ticker: str, start_date: datetime.date,
                                 end_date: datetime.date) -> List[Dict[str, Any]]:
        """
        Get historical market data for a specific ticker and date range
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dictionaries with market data
        """
        session = self.get_session()
        try:
            data = session.query(MarketData).filter(
                MarketData.ticker == ticker,
                MarketData.date >= start_date,
                MarketData.date <= end_date
            ).order_by(MarketData.date).all()
            
            result = []
            for item in data:
                result.append({
                    'id': item.id,
                    'ticker': item.ticker,
                    'date': item.date,
                    'open_price': item.open_price,
                    'high_price': item.high_price,
                    'low_price': item.low_price,
                    'close_price': item.close_price,
                    'volume': item.volume,
                    'currency': item.currency
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting historical market data: {e}")
            return []
        finally:
            self.close_session(session)
    
    # Exchange rate operations
    def add_exchange_rate(self, from_currency: str, to_currency: str, 
                         date: datetime.date, rate: float) -> bool:
        """
        Add or update an exchange rate
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Date of the exchange rate
            rate: Exchange rate value
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Check if rate already exists
            exchange_rate = session.query(ExchangeRate).filter(
                ExchangeRate.from_currency == from_currency,
                ExchangeRate.to_currency == to_currency,
                ExchangeRate.date == date
            ).first()
            
            if exchange_rate:
                # Update existing rate
                exchange_rate.rate = rate
                exchange_rate.last_updated = datetime.datetime.now()
            else:
                # Create new rate
                exchange_rate = ExchangeRate(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    date=date,
                    rate=rate,
                    last_updated=datetime.datetime.now()
                )
                session.add(exchange_rate)
                
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding exchange rate: {e}")
            return False
        finally:
            self.close_session(session)
            
    def get_exchange_rate(self, from_currency: str, to_currency: str, 
                         date: datetime.date) -> Optional[float]:
        """
        Get an exchange rate
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Date of the exchange rate
            
        Returns:
            Exchange rate if found, None otherwise
        """
        session = self.get_session()
        try:
            exchange_rate = session.query(ExchangeRate).filter(
                ExchangeRate.from_currency == from_currency,
                ExchangeRate.to_currency == to_currency,
                ExchangeRate.date == date
            ).first()
            
            if not exchange_rate:
                return None
                
            return exchange_rate.rate
        except Exception as e:
            logger.error(f"Error getting exchange rate: {e}")
            return None
        finally:
            self.close_session(session)
            
    # Utility methods
    def get_tickers_to_track(self) -> List[str]:
        """
        Get all tickers that should be tracked (portfolio + watchlist)
        
        Returns:
            List of ticker symbols
        """
        session = self.get_session()
        try:
            # Get portfolio tickers
            portfolio_tickers = [p.ticker for p in session.query(Portfolio.ticker).all()]
            
            # Get watchlist tickers
            watchlist_tickers = [w.ticker for w in session.query(Watchlist.ticker).all()]
            
            # Combine and remove duplicates
            all_tickers = list(set(portfolio_tickers + watchlist_tickers))
            
            return all_tickers
        except Exception as e:
            logger.error(f"Error getting tickers to track: {e}")
            return []
        finally:
            self.close_session(session)
            
    def clear_all_data(self) -> bool:
        """
        Clear all data from the database
        
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            session.query(Portfolio).delete()
            session.query(Transaction).delete()
            session.query(Watchlist).delete()
            session.query(MarketData).delete()
            session.query(ExchangeRate).delete()
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing database: {e}")
            return False
        finally:
            self.close_session(session)

    # Market data methods
    def get_stock_price(self, ticker: str, date: datetime.date) -> Optional[float]:
        """
        Get stock price for a specific ticker and date from the database
        This method should be used by the frontend to retrieve price data
        
        Args:
            ticker: Stock ticker symbol
            date: Date for the price
            
        Returns:
            The stock price or None if not found
        """
        session = self.get_session()
        try:
            data = session.query(MarketData).filter(
                MarketData.ticker == ticker,
                MarketData.date == date
            ).first()
            
            if not data:
                return None
                
            return data.close_price
        except Exception as e:
            logger.error(f"Error getting stock price: {e}")
            return None
        finally:
            self.close_session(session)
            
    def store_stock_price(self, ticker: str, date: datetime.date, price: float) -> bool:
        """
        Store stock price in the database
        
        Args:
            ticker: Stock ticker symbol
            date: Date for the price
            price: Stock price
            
        Returns:
            True if successful, False otherwise
        """
        return self.add_market_data(ticker, date, price, 'USD')
            
    def get_last_known_price(self, ticker: str) -> Optional[float]:
        """
        Get the last known price for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            The last known price or None if not found
        """
        session = self.get_session()
        try:
            data = session.query(MarketData).filter(
                MarketData.ticker == ticker
            ).order_by(MarketData.date.desc()).first()
            
            if not data:
                return None
                
            return data.close_price
        except Exception as e:
            logger.error(f"Error getting last known price: {e}")
            return None
        finally:
            self.close_session(session)
            
    def find_closest_date_data(self, table: str, identifier: str, target_date: datetime.date, 
                              max_days: int = 7) -> Optional[Tuple[datetime.date, float]]:
        """
        Find data for the closest date to the target date in the database
        This method should be used by the frontend to find approximate data when exact dates are not available
        
        Args:
            table: Table name ('market_data' for stock prices or 'exchange_rates' for currency rates)
            identifier: Ticker or currency pair identifier (e.g., 'AAPL' or 'USD_EUR')
            target_date: Target date to find data for
            max_days: Maximum number of days to look before/after target date
            
        Returns:
            Tuple of (date, value) or None if not found within the specified range
            
        Example:
            # Find the closest stock price for AAPL near 2023-01-15
            closest_data = db_manager.find_closest_date_data('market_data', 'AAPL', datetime.date(2023, 1, 15))
            if closest_data:
                date, price = closest_data
                print(f"Found price {price} on {date}")
        """
        session = self.get_session()
        try:
            if table == 'market_data':
                # For stock prices
                for delta in range(1, max_days + 1):
                    for offset in [delta, -delta]:
                        check_date = target_date + datetime.timedelta(days=offset)
                        data = session.query(MarketData).filter(
                            MarketData.ticker == identifier,
                            MarketData.date == check_date
                        ).first()
                        
                        if data:
                            return (data.date, data.close_price)
            elif table == 'exchange_rates':
                # For exchange rates
                try:
                    from_currency, to_currency = identifier.split('_')
                    for delta in range(1, max_days + 1):
                        for offset in [delta, -delta]:
                            check_date = target_date + datetime.timedelta(days=offset)
                            data = session.query(ExchangeRate).filter(
                                ExchangeRate.from_currency == from_currency,
                                ExchangeRate.to_currency == to_currency,
                                ExchangeRate.date == check_date
                            ).first()
                            
                            if data:
                                return (data.date, data.rate)
                except ValueError:
                    logger.error(f"Invalid identifier format for exchange rates: {identifier}. Expected format: 'FROM_TO'")
            else:
                logger.error(f"Invalid table name: {table}. Expected 'market_data' or 'exchange_rates'")
                            
            return None
        except Exception as e:
            logger.error(f"Error finding closest date data: {e}")
            return None
        finally:
            self.close_session(session)
            
    def get_similar_tickers(self, ticker: str) -> List[str]:
        """
        Get similar tickers that might be related to the given ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of similar ticker symbols
        """
        session = self.get_session()
        try:
            # Get all tickers that start with the same letter
            if len(ticker) > 0:
                first_letter = ticker[0]
                similar_tickers = session.query(MarketData.ticker).filter(
                    MarketData.ticker.like(f"{first_letter}%")
                ).distinct().all()
                
                return [t[0] for t in similar_tickers]
            return []
        except Exception as e:
            logger.error(f"Error getting similar tickers: {e}")
            return []
        finally:
            self.close_session(session)
            
    def get_historical_prices(self, ticker: str, start_date: datetime.date, 
                             end_date: datetime.date, partial_match: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical prices for a ticker over a date range from the database
        This method should be used by the frontend to retrieve historical price data
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            partial_match: Whether to return partial data if full range not available
            
        Returns:
            DataFrame with historical prices or None if not found
        """
        session = self.get_session()
        try:
            query = session.query(MarketData).filter(
                MarketData.ticker == ticker,
                MarketData.date >= start_date,
                MarketData.date <= end_date
            ).order_by(MarketData.date)
            
            data = query.all()
            
            if not data:
                return None
                
            # Check if we have data for the full range
            dates = [d.date for d in data]
            if not partial_match and (min(dates) > start_date or max(dates) < end_date):
                return None
                
            # Create DataFrame
            df = pd.DataFrame({
                'date': [d.date for d in data],
                'price': [d.close_price for d in data]
            })
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error getting historical prices: {e}")
            return None
        finally:
            self.close_session(session)
            
    def store_historical_prices(self, ticker: str, start_date: datetime.date, 
                               end_date: datetime.date, data: pd.DataFrame) -> bool:
        """
        Store historical prices in the database
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            data: DataFrame with historical prices
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # If the DataFrame has a DatetimeIndex, convert it to a column
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={'index': 'date'}, inplace=True)
            
            # Ensure the date column contains date objects, not datetime objects
            if 'date' in df.columns:
                # Check if the date column contains datetime objects
                if pd.api.types.is_datetime64_any_dtype(df['date']):
                    # Convert datetime to date
                    df['date'] = df['date'].dt.date
                # Check if the date column contains strings
                elif pd.api.types.is_string_dtype(df['date']):
                    # Convert string to date
                    df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Store each data point
            success_count = 0
            error_count = 0
            
            # Process the DataFrame row by row
            for idx in range(len(df)):
                try:
                    # Get the date and price
                    if 'date' in df.columns:
                        date_value = df.iloc[idx]['date']
                        # Ensure date is a date object, not a datetime object
                        if isinstance(date_value, datetime.datetime):
                            date = date_value.date()
                        # Ensure date is a date object, not a string
                        elif isinstance(date_value, str):
                            date = datetime.datetime.strptime(date_value, '%Y-%m-%d').date()
                        # If it's already a date object, use it directly
                        elif isinstance(date_value, datetime.date):
                            date = date_value
                        # Handle pandas Series objects
                        elif isinstance(date_value, pd.Series):
                            if len(date_value) > 0 and pd.notna(date_value.iloc[0]):
                                first_value = date_value.iloc[0]
                                if isinstance(first_value, datetime.datetime):
                                    date = first_value.date()
                                elif isinstance(first_value, datetime.date):
                                    date = first_value
                                elif isinstance(first_value, str):
                                    date = datetime.datetime.strptime(first_value, '%Y-%m-%d').date()
                                else:
                                    # Skip invalid date values
                                    logger.warning(f"Skipping row with invalid date value in Series: {type(first_value)}")
                                    error_count += 1
                                    continue
                            else:
                                # Skip empty or NA Series
                                logger.warning(f"Skipping row with empty date Series")
                                error_count += 1
                                continue
                        else:
                            # Skip invalid date values
                            logger.warning(f"Skipping row with invalid date type: {type(date_value)}")
                            error_count += 1
                            continue
                    else:
                        # If there's no date column, use the start_date + index
                        date = start_date + datetime.timedelta(days=int(idx))
                    
                    # Get the price
                    if 'price' in df.columns:
                        price_value = df.iloc[idx]['price']
                        # Handle pandas Series objects
                        if isinstance(price_value, pd.Series):
                            if len(price_value) > 0 and pd.notna(price_value.iloc[0]):
                                price = float(price_value.iloc[0])
                            else:
                                # Skip empty or NA Series
                                logger.warning(f"Skipping row with empty price Series")
                                error_count += 1
                                continue
                        else:
                            price = float(price_value)
                    else:
                        # Skip rows without price
                        logger.warning(f"Skipping row without price")
                        error_count += 1
                        continue
                    
                    # Check if data already exists
                    existing = session.query(MarketData).filter(
                        MarketData.ticker == ticker,
                        MarketData.date == date
                    ).first()
                    
                    if existing:
                        # Update existing data
                        existing.close_price = price
                    else:
                        # Create new data
                        market_data = MarketData(
                            ticker=ticker,
                            date=date,
                            close_price=price,
                            currency='USD',  # Default currency
                            is_synthetic=False
                        )
                        session.add(market_data)
                    
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error storing data point for {ticker} on {date if 'date' in locals() else 'unknown date'}: {e}")
                    # Continue with the next row instead of failing the entire operation
                    continue
            
            session.commit()
            logger.info(f"Stored {success_count} data points for {ticker} ({error_count} errors)")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing historical prices: {e}")
            return False
        finally:
            self.close_session(session)
            
    # Exchange rate methods
    def store_exchange_rate(self, from_currency: str, to_currency: str, 
                           date: datetime.date, rate: float) -> bool:
        """
        Store exchange rate in the database
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Date of the exchange rate
            rate: Exchange rate value
            
        Returns:
            True if successful, False otherwise
        """
        return self.add_exchange_rate(from_currency, to_currency, date, rate)
            
    # API status methods
    def get_api_status(self, api_name: str) -> Dict[str, Any]:
        """
        Get status of an API
        
        Args:
            api_name: Name of the API
            
        Returns:
            Dictionary with API status information
        """
        # For now, always return that the API is up
        return {
            'status': 'up',
            'is_in_cooldown': False,
            'cooldown_until': None,
            'last_success': datetime.datetime.now().isoformat(),
            'success_count': 100,
            'failure_count': 0
        }
        
    def log_api_request(self, api_name: str, success: bool, error_message: str = None) -> bool:
        """Log an API request to the database"""
        try:
            session = self.get_session()
            
            # Create a new API request log entry
            api_request = APIRequest(
                api_name=api_name,
                timestamp=datetime.datetime.now(),
                success=success,
                error_message=error_message
            )
            
            session.add(api_request)
            session.commit()
            
            if not success and error_message:
                logger.warning(f"API request to {api_name} failed: {error_message}")
            
            return True
        except Exception as e:
            logger.error(f"Error logging API request: {e}")
            return False
        finally:
            self.close_session(session)
    
    def get_recent_api_requests(self, minutes: int = 60) -> int:
        """Get the count of API requests in the last X minutes"""
        try:
            session = self.get_session()
            
            # Calculate the timestamp for X minutes ago
            cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
            
            # Count API requests since the cutoff time
            count = session.query(APIRequest).filter(
                APIRequest.timestamp >= cutoff_time
            ).count()
            
            return count
        except Exception as e:
            logger.error(f"Error getting recent API requests: {e}")
            return 0
        finally:
            self.close_session(session)

    def update_position_country(self, ticker: str, country: str) -> bool:
        """
        Update the country of a position
        
        Args:
            ticker: Stock ticker symbol
            country: Country/market of the stock
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            # Check if position exists
            position = session.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            
            if not position:
                logger.warning(f"Cannot update country for {ticker} - position does not exist")
                return False
                
            # Update country
            position.country = country
            
            # Commit changes
            session.commit()
            logger.info(f"Updated country for {ticker} to {country}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating country for {ticker}: {e}")
            return False
        finally:
            self.close_session(session)

    def get_market_data_at_date(self, ticker: str, date_val: datetime.date) -> Optional[Dict[str, Any]]:
        """
        Get market data for a ticker at a specific date
        
        Args:
            ticker: Stock ticker symbol
            date_val: The specific date to get market data for
            
        Returns:
            Dictionary with market data or None if not found
        """
        session = self.get_session()
        try:
            # Query for market data on the exact date
            market_data = session.query(MarketData).filter(
                MarketData.ticker == ticker,
                MarketData.date == date_val
            ).first()
            
            if market_data:
                return {
                    'ticker': market_data.ticker,
                    'date': market_data.date,
                    'open_price': market_data.open_price,
                    'high_price': market_data.high_price,
                    'low_price': market_data.low_price,
                    'close_price': market_data.close_price,
                    'volume': market_data.volume,
                    'currency': market_data.currency,
                    'is_synthetic': market_data.is_synthetic
                }
            return None
        except Exception as e:
            logger.error(f"Error getting market data at date: {e}")
            return None
        finally:
            self.close_session(session)
            
    def get_closest_market_data_before(self, ticker: str, date_val: datetime.date) -> Optional[Dict[str, Any]]:
        """
        Get the closest market data for a ticker before a specific date
        
        Args:
            ticker: Stock ticker symbol
            date_val: The date to find market data before
            
        Returns:
            Dictionary with market data or None if not found
        """
        session = self.get_session()
        try:
            # Query for the most recent market data before the given date
            market_data = session.query(MarketData).filter(
                MarketData.ticker == ticker,
                MarketData.date < date_val
            ).order_by(MarketData.date.desc()).first()
            
            if market_data:
                return {
                    'ticker': market_data.ticker,
                    'date': market_data.date,
                    'open_price': market_data.open_price,
                    'high_price': market_data.high_price,
                    'low_price': market_data.low_price,
                    'close_price': market_data.close_price,
                    'volume': market_data.volume,
                    'currency': market_data.currency,
                    'is_synthetic': market_data.is_synthetic
                }
            return None
        except Exception as e:
            logger.error(f"Error getting closest market data before date: {e}")
            return None
        finally:
            self.close_session(session)

# Create a global instance of the database manager
db_manager = DatabaseManager() 