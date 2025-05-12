from datetime import date
from typing import Optional, Dict, List
import logging
from backend.market_interface import MarketInterface
from backend.money_amount import MoneyAmount
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Stock:
    """Class representing a stock with pricing information"""
    
    # Map of exchange suffixes to currencies
    EXCHANGE_CURRENCY_MAP: Dict[str, str] = {
        '.L': 'GBP',    # London
        '.DE': 'EUR',   # Germany
        '.PA': 'EUR',   # Paris
        '.MI': 'EUR',   # Milan
        '.MC': 'EUR',   # Madrid
        '.AS': 'EUR',   # Amsterdam
        '.BR': 'EUR',   # Brussels
        '.ST': 'SEK',   # Stockholm
        '.CO': 'DKK',   # Copenhagen
        '.HE': 'EUR',   # Helsinki
        '.OL': 'NOK',   # Oslo
        '.SW': 'CHF',   # Switzerland
        '.TO': 'CAD',   # Toronto
        '.AX': 'AUD',   # Australia
        '.HK': 'HKD',   # Hong Kong
        '.T': 'JPY',    # Tokyo
        '.KS': 'KRW',   # Korea
        '.TW': 'TWD',   # Taiwan
        '.SZ': 'CNY',   # Shenzhen
        '.SS': 'CNY',   # Shanghai
    }
    
    def __init__(self, ticker: str) -> None:
        """
        Initialize a stock with a ticker symbol
        
        Args:
            ticker: The stock ticker symbol
            
        Raises:
            ValueError: If ticker is empty
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
            
        self.ticker: str = ticker
        self._market_interface: MarketInterface = MarketInterface()
        
    def get_price(self, currency: str, evaluation_date: date) -> float:
        """
        Get the stock price in specified currency at given date
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the price
            
        Returns:
            The stock price in the specified currency
            
        Raises:
            ValueError: If price cannot be retrieved
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        # First get price in stock's native currency
        native_currency = self._get_native_currency()
        
        try:
            price = self._market_interface.get_price(self.ticker, evaluation_date)
            
            if price is None:
                logger.error(f"Could not get price for {self.ticker} at {evaluation_date}")
                raise ValueError(f"Could not get price for {self.ticker} at {evaluation_date}")
                
            # Create MoneyAmount for the price in native currency
            price_money = MoneyAmount(price, native_currency, evaluation_date)
            
            # Return the amount in requested currency
            return price_money.get_money_amount(currency)
        except Exception as e:
            logger.error(f"Error getting price for {self.ticker} at {evaluation_date}: {e}")
            raise ValueError(f"Failed to get price for {self.ticker} at {evaluation_date}: {e}") from e
        
    def _get_native_currency(self) -> str:
        """
        Determine the native currency of the stock based on ticker
        
        Returns:
            The currency code for the stock's native currency
        """
        # Check for exchange suffixes
        for suffix, currency in self.EXCHANGE_CURRENCY_MAP.items():
            if self.ticker.endswith(suffix):
                return currency
                
        # Check for specific country ETFs or indices
        if self.ticker.startswith('^'):
            if self.ticker in ['^GDAXI', '^MDAXI', '^SDAXI', '^TECDAX']:
                return 'EUR'  # German indices
            elif self.ticker in ['^FTSE', '^FTMC', '^FTLC']:
                return 'GBP'  # UK indices
            elif self.ticker in ['^FCHI', '^SBF120']:
                return 'EUR'  # French indices
            elif self.ticker in ['^N225', '^TOPX']:
                return 'JPY'  # Japanese indices
                
        # Default to USD for NYSE/NASDAQ stocks
        return 'USD'
            
    def get_value(self, currency: str, evaluation_date: date, amount: float = 1.0) -> float:
        """
        Calculate value of specified amount of shares
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the valuation
            amount: Number of shares (default: 1.0)
            
        Returns:
            The total value of the shares in the specified currency
            
        Raises:
            ValueError: If value cannot be calculated
        """
        if amount < 0:
            raise ValueError("Amount cannot be negative")
            
        try:
            price = self.get_price(currency, evaluation_date)
            return price * amount
        except Exception as e:
            logger.error(f"Error calculating value for {amount} shares of {self.ticker}: {e}")
            raise ValueError(f"Failed to calculate value for {amount} shares of {self.ticker}: {e}") from e
            
    def __eq__(self, other: object) -> bool:
        """
        Check if two stocks are equal (same ticker)
        
        Args:
            other: Another Stock object
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(other, Stock):
            return False
        return self.ticker == other.ticker
        
    def __hash__(self) -> int:
        """
        Hash function for Stock objects
        
        Returns:
            Hash value based on ticker
        """
        return hash(self.ticker)
        
    def __str__(self) -> str:
        """String representation of stock"""
        return f"Stock({self.ticker})"
        
    def __repr__(self) -> str:
        """Detailed representation of stock"""
        return f"Stock(ticker='{self.ticker}')"
        
    def get_name(self) -> Optional[str]:
        """
        Get the name of the stock
        
        Returns:
            The stock name or None if not available
        """
        try:
            stock_info = self._market_interface.get_stock_info(self.ticker)
            return stock_info.get('name')
        except Exception as e:
            logger.error(f"Error getting name for {self.ticker}: {e}")
            return None
            
    def get_sector(self) -> Optional[str]:
        """
        Get the sector of the stock
        
        Returns:
            The stock sector or None if not available
        """
        try:
            stock_info = self._market_interface.get_stock_info(self.ticker)
            return stock_info.get('sector')
        except Exception as e:
            logger.error(f"Error getting sector for {self.ticker}: {e}")
            return None
            
    def get_country(self) -> Optional[str]:
        """
        Get the country/market of the stock
        
        Returns:
            The stock country/market or a best guess based on ticker and currency
        """
        try:
            # First try to get country from stock info
            stock_info = self._market_interface.get_stock_info(self.ticker)
            country = stock_info.get('country')
            
            # If country is available from stock info, return it
            if country:
                return country
                
            # If country is not available, determine from exchange suffix
            # Map exchange suffixes to countries
            exchange_country_map = {
                '.L': 'United Kingdom',    # London
                '.DE': 'Germany',          # Germany
                '.PA': 'France',           # Paris
                '.MI': 'Italy',            # Milan
                '.MC': 'Spain',            # Madrid
                '.AS': 'Netherlands',      # Amsterdam
                '.BR': 'Belgium',          # Brussels
                '.ST': 'Sweden',           # Stockholm
                '.CO': 'Denmark',          # Copenhagen
                '.HE': 'Finland',          # Helsinki
                '.OL': 'Norway',           # Oslo
                '.SW': 'Switzerland',      # Switzerland
                '.TO': 'Canada',           # Toronto
                '.AX': 'Australia',        # Australia
                '.HK': 'Hong Kong',        # Hong Kong
                '.T': 'Japan',             # Tokyo
                '.KS': 'South Korea',      # Korea
                '.TW': 'Taiwan',           # Taiwan
                '.SZ': 'China',            # Shenzhen
                '.SS': 'China',            # Shanghai
            }
            
            for suffix, country_name in exchange_country_map.items():
                if self.ticker.endswith(suffix):
                    return country_name
            
            # Check for US stock patterns (no suffix, typically 1-5 letters)
            if re.match(r'^[A-Z]{1,5}$', self.ticker):
                return 'United States'
                
            # Check for indices
            if self.ticker.startswith('^'):
                index_country_map = {
                    # US indices
                    '^DJI': 'United States',    # Dow Jones
                    '^GSPC': 'United States',   # S&P 500
                    '^IXIC': 'United States',   # NASDAQ
                    '^RUT': 'United States',    # Russell 2000
                    '^VIX': 'United States',    # Volatility Index
                    
                    # European indices
                    '^GDAXI': 'Germany',        # DAX
                    '^MDAXI': 'Germany',        # MDAX
                    '^SDAXI': 'Germany',        # SDAX
                    '^TECDAX': 'Germany',       # TecDAX
                    '^FTSE': 'United Kingdom',  # FTSE 100
                    '^FTMC': 'United Kingdom',  # FTSE 250
                    '^FTLC': 'United Kingdom',  # FTSE 350
                    '^FCHI': 'France',          # CAC 40
                    '^SBF120': 'France',        # SBF 120
                    '^STOXX50E': 'Eurozone',    # EURO STOXX 50
                    '^STOXX': 'Europe',         # STOXX Europe 600
                    
                    # Asian indices
                    '^N225': 'Japan',           # Nikkei 225
                    '^TOPX': 'Japan',           # TOPIX
                    '^HSI': 'Hong Kong',        # Hang Seng
                    '^SSEC': 'China',           # Shanghai Composite
                    '^SZSC': 'China',           # Shenzhen Component
                    '^KS11': 'South Korea',     # KOSPI
                    '^TWII': 'Taiwan',          # Taiwan Weighted
                    
                    # Other global indices
                    '^GSPTSE': 'Canada',        # S&P/TSX Composite
                    '^AXJO': 'Australia',       # S&P/ASX 200
                    '^BSESN': 'India',          # BSE SENSEX
                    '^NSEI': 'India',           # NIFTY 50
                    '^BVSP': 'Brazil',          # IBOVESPA
                    '^MXX': 'Mexico',           # IPC Mexico
                }
                
                if self.ticker in index_country_map:
                    return index_country_map[self.ticker]
                
                # Generic index pattern matching
                if self.ticker.startswith('^GDAXI') or self.ticker.startswith('^MDAXI') or self.ticker.startswith('^SDAXI'):
                    return 'Germany'
                elif self.ticker.startswith('^FTS'):
                    return 'United Kingdom'
                elif self.ticker.startswith('^FCH'):
                    return 'France'
                elif self.ticker.startswith('^N') or self.ticker.startswith('^TOP'):
                    return 'Japan'
            
            # Use currency as a fallback
            currency = self._get_native_currency()
            currency_country_map = {
                'USD': 'United States',
                'EUR': 'Eurozone',
                'GBP': 'United Kingdom',
                'JPY': 'Japan',
                'CAD': 'Canada',
                'AUD': 'Australia',
                'CHF': 'Switzerland',
                'CNY': 'China',
                'HKD': 'Hong Kong',
                'SEK': 'Sweden',
                'NOK': 'Norway',
                'DKK': 'Denmark',
                'KRW': 'South Korea',
                'TWD': 'Taiwan',
                'INR': 'India',
                'BRL': 'Brazil',
                'MXN': 'Mexico',
                'ZAR': 'South Africa',
                'SGD': 'Singapore',
                'NZD': 'New Zealand',
                'TRY': 'Turkey',
                'RUB': 'Russia'
            }
            
            country = currency_country_map.get(currency)
            if country:
                return country
                
            # If all else fails, return a default based on the first character of the ticker
            # This is just a fallback to avoid returning "Unknown"
            if self.ticker and len(self.ticker) > 0:
                first_char = self.ticker[0].upper()
                if first_char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    # Arbitrary but deterministic mapping to major markets
                    char_country_map = {
                        'A': 'United States', 'B': 'United Kingdom', 'C': 'Canada', 
                        'D': 'Germany', 'E': 'Spain', 'F': 'France', 
                        'G': 'United States', 'H': 'Hong Kong', 'I': 'Italy', 
                        'J': 'Japan', 'K': 'South Korea', 'L': 'United Kingdom', 
                        'M': 'United States', 'N': 'Netherlands', 'O': 'United States', 
                        'P': 'United States', 'Q': 'United States', 'R': 'Russia', 
                        'S': 'Switzerland', 'T': 'United States', 'U': 'United States', 
                        'V': 'United States', 'W': 'United States', 'X': 'China', 
                        'Y': 'United States', 'Z': 'South Africa'
                    }
                    return char_country_map.get(first_char, 'United States')
            
            # If we still can't determine the country, default to None
            return None
            
        except Exception as e:
            logger.error(f"Error getting country for {self.ticker}: {e}")
            # Default to None as a fallback
            return None