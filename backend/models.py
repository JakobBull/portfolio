from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Enum, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()

class TransactionType(enum.Enum):
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"

class Stock(Base):
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(100), nullable=True)
    currency = Column(String(3), nullable=False)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock")
    portfolio_entries = relationship("Portfolio", back_populates="stock")
    transactions = relationship("Transaction", back_populates="stock")
    watchlist_entries = relationship("Watchlist", back_populates="stock")
    
    def __repr__(self):
        return f"<Stock(ticker='{self.ticker}', name='{self.name}')>"

class StockPrice(Base):
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), ForeignKey('stocks.ticker'), nullable=False)
    date = Column(Date, nullable=False)
    price = Column(Float, nullable=False)
    
    # Relationship
    stock = relationship("Stock", back_populates="prices")
    
    # Composite unique constraint on ticker and date
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uix_stock_price_date'),
    )
    
    def __repr__(self):
        return f"<StockPrice(ticker='{self.ticker}', date={self.date}, price={self.price})>"

class Portfolio(Base):
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), ForeignKey('stocks.ticker'), nullable=False)
    date_added = Column(Date, nullable=False)
    shares = Column(Float, nullable=False)
    
    # Relationship
    stock = relationship("Stock", back_populates="portfolio_entries")
    
    def __repr__(self):
        return f"<Portfolio(ticker='{self.ticker}', shares={self.shares})>"

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    type = Column(Enum(TransactionType), nullable=False)
    ticker = Column(String(20), ForeignKey('stocks.ticker'), nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    transaction_cost = Column(Float, nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # Relationship
    stock = relationship("Stock", back_populates="transactions")
    
    def __repr__(self):
        return f"<Transaction(type='{self.type}', ticker='{self.ticker}', amount={self.amount})>"

class Watchlist(Base):
    __tablename__ = 'watchlist'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), ForeignKey('stocks.ticker'), nullable=False, unique=True)
    date_added = Column(Date, nullable=False)
    
    # Relationship
    stock = relationship("Stock", back_populates="watchlist_entries")
    
    def __repr__(self):
        return f"<Watchlist(ticker='{self.ticker}')>"

# Create SQLite database
def init_db(db_path: str = 'sqlite:///portfolio.db'):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine 