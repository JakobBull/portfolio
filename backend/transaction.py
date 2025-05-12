from datetime import date
from typing import List, Optional, Union, Literal, ClassVar
import logging
from backend.stock import Stock
from backend.market_interface import MarketInterface
from backend.money_amount import MoneyAmount

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TransactionType = Literal["buy", "sell", "dividend"]

class Transaction:
    """Class representing a stock transaction (buy, sell, dividend)"""
    
    VALID_TYPES: ClassVar[List[TransactionType]] = ["buy", "sell", "dividend"]
    
    def __init__(self, transaction_type: TransactionType, stock: Stock, amount: float, 
                price: float, currency: str, transaction_date: date,
                transaction_cost: Optional[float] = None) -> None:
        """
        Initialize a transaction
        
        Args:
            transaction_type: Type of transaction (buy, sell, dividend)
            stock: The stock involved in the transaction
            amount: Number of shares
            price: Price per share
            currency: Currency of the transaction
            transaction_date: Date of the transaction
            transaction_cost: Custom transaction cost (default: None, will be calculated automatically)
            
        Raises:
            ValueError: If transaction parameters are invalid
        """
        if transaction_type not in self.VALID_TYPES:
            raise ValueError(f"Transaction type must be one of {self.VALID_TYPES}")
            
        if not isinstance(stock, Stock):
            raise TypeError(f"Stock must be a Stock object, got {type(stock)}")
            
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError(f"Amount must be a positive number, got {amount}")
            
        if not isinstance(price, (int, float)) or (price <= 0 and transaction_type != "dividend"):
            # Allow zero price for dividends
            raise ValueError(f"Price must be a positive number, got {price}")
            
        if not currency or not isinstance(currency, str):
            raise ValueError(f"Currency must be a non-empty string, got {currency}")
            
        if not isinstance(transaction_date, date):
            raise TypeError(f"Transaction date must be a date object, got {type(transaction_date)}")
            
        self.type: TransactionType = transaction_type
        self.stock: Stock = stock
        self.amount: float = float(amount)
        self.currency: str = currency
        # Use MoneyAmount for price
        self.price: MoneyAmount = MoneyAmount(price, currency, transaction_date)
        self.date: date = transaction_date
        
        # Calculate transaction cost (simplified example - could be made more sophisticated)
        self.cost: MoneyAmount = self._calculate_transaction_cost() if transaction_cost is None else MoneyAmount(transaction_cost, currency, transaction_date)
        
    def _calculate_transaction_cost(self) -> MoneyAmount:
        """
        Calculate transaction cost based on amount and price
        
        Returns:
            A MoneyAmount representing the transaction cost
        """
        if self.type == "dividend":
            # No transaction costs for dividends
            return MoneyAmount(0.0, self.price.currency, self.date)
            
        base_fee = 5.0  # Base transaction fee
        variable_fee = self.amount * self.price.amount * 0.001  # 0.1% of transaction value
        total_fee = base_fee + variable_fee
        # Return as MoneyAmount in same currency as price
        return MoneyAmount(total_fee, self.price.currency, self.date)
        
    def get_market_value(self, currency: str, evaluation_date: Optional[date] = None) -> float:
        """
        Get the market value of this transaction at a specific date
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the valuation (default: today)
            
        Returns:
            The market value in the specified currency
            
        Raises:
            ValueError: If market value cannot be calculated
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        if evaluation_date is None:
            evaluation_date = date.today()
            
        try:
            if self.type == "dividend":
                # For dividends, market value is the dividend amount
                dividend_amount = MoneyAmount(
                    self.price.amount * self.amount,  # Total dividend (per share * shares)
                    self.price.currency,
                    self.date
                )
                return dividend_amount.get_money_amount(currency)
                
            price = self.stock.get_price(currency, evaluation_date)
            return price * self.amount
        except Exception as e:
            logger.error(f"Error calculating market value for transaction: {e}")
            raise ValueError(f"Failed to calculate market value for transaction: {e}") from e
        
    def get_transaction_value(self, currency: Optional[str] = None) -> float:
        """
        Get the actual transaction value (price paid/received) in specified currency
        
        Args:
            currency: Target currency code (default: transaction currency)
            
        Returns:
            The transaction value in the specified currency
            
        Raises:
            ValueError: If transaction value cannot be calculated
        """
        transaction_value = self.price.amount * self.amount
        
        if currency is None or currency == self.price.currency:
            return transaction_value
        
        try:
            # Convert to requested currency
            money_amount = MoneyAmount(transaction_value, self.price.currency, self.date)
            return money_amount.get_money_amount(currency)
        except Exception as e:
            logger.error(f"Error calculating transaction value: {e}")
            raise ValueError(f"Failed to calculate transaction value: {e}") from e
        
    def get_total_cost(self, currency: Optional[str] = None) -> float:
        """
        Get the total cost including transaction fees in specified currency
        
        Args:
            currency: Target currency code (default: transaction currency)
            
        Returns:
            The total cost in the specified currency
            
        Raises:
            ValueError: If total cost cannot be calculated
        """
        if currency is None or currency == self.cost.currency:
            return self.cost.amount
        
        try:
            # Convert to requested currency
            return self.cost.get_money_amount(currency)
        except Exception as e:
            logger.error(f"Error calculating total cost: {e}")
            raise ValueError(f"Failed to calculate total cost: {e}") from e
        
    @property
    def is_dividend(self) -> bool:
        """Check if this is a dividend transaction"""
        return self.type == "dividend"
        
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy transaction"""
        return self.type == "buy"
        
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell transaction"""
        return self.type == "sell"
        
    def __str__(self) -> str:
        """String representation of transaction"""
        return f"{self.type.capitalize()} {self.amount} {self.stock.ticker} @ {self.price} on {self.date}"
        
    def __repr__(self) -> str:
        """Detailed representation of transaction"""
        return (f"Transaction(type='{self.type}', stock={self.stock}, amount={self.amount}, "
                f"price={self.price}, date={self.date})")


class Dividend(Transaction):
    """Specialized class for dividend transactions"""
    
    def __init__(self, stock: Stock, shares: float, dividend_per_share: float, 
                currency: str, transaction_date: date, transaction_cost: Optional[float] = None) -> None:
        """
        Initialize a dividend transaction
        
        Args:
            stock: The stock paying the dividend
            shares: Number of shares owned
            dividend_per_share: Dividend amount per share
            currency: Currency of the dividend
            transaction_date: Date of the dividend payment
            transaction_cost: Custom transaction cost (default: None, will be calculated automatically)
            
        Raises:
            ValueError: If dividend parameters are invalid
        """
        if not isinstance(shares, (int, float)) or shares <= 0:
            raise ValueError(f"Shares must be a positive number, got {shares}")
            
        if not isinstance(dividend_per_share, (int, float)) or dividend_per_share < 0:
            raise ValueError(f"Dividend per share must be a non-negative number, got {dividend_per_share}")
            
        super().__init__("dividend", stock, shares, dividend_per_share, currency, transaction_date, transaction_cost)
        
    @property
    def dividend_per_share(self) -> float:
        """Get dividend amount per share"""
        return self.price.amount
        
    @property
    def total_dividend(self) -> float:
        """Get total dividend amount"""
        return self.price.amount * self.amount
        
    def get_dividend_yield(self, currency: Optional[str] = None) -> float:
        """
        Calculate dividend yield based on current stock price
        
        Args:
            currency: Target currency code (default: dividend currency)
            
        Returns:
            The dividend yield as a decimal value
            
        Raises:
            ValueError: If dividend yield cannot be calculated
        """
        try:
            # Get current stock price
            current_price = self.stock.get_price(
                self.price.currency if currency is None else currency,
                self.date
            )
            
            if current_price <= 0:
                logger.warning(f"Stock price for {self.stock.ticker} is zero or negative, cannot calculate yield")
                return 0.0
                
            # Calculate annual dividend (assuming this is a quarterly dividend)
            annual_dividend = self.dividend_per_share * 4
            
            # Calculate yield
            return (annual_dividend / current_price)  # Return as decimal for .2% format
        except Exception as e:
            logger.error(f"Error calculating dividend yield: {e}")
            return 0.0  # Return 0 instead of raising an error
            
    def __str__(self) -> str:
        """String representation of dividend"""
        return f"Dividend {self.total_dividend} {self.price.currency} for {self.amount} shares of {self.stock.ticker} on {self.date}"
        
    def __repr__(self) -> str:
        """Detailed representation of dividend"""
        return (f"Dividend(stock={self.stock}, shares={self.amount}, "
                f"dividend_per_share={self.dividend_per_share}, currency='{self.currency}', "
                f"date={self.date})")


class Buy(Transaction):
    """Specialized class for buy transactions"""
    
    def __init__(self, stock: Stock, shares: float, price_per_share: float,
                currency: str, transaction_date: date, transaction_cost: Optional[float] = None) -> None:
        """
        Initialize a buy transaction
        
        Args:
            stock: The stock being purchased
            shares: Number of shares purchased
            price_per_share: Price per share
            currency: Currency of the transaction
            transaction_date: Date of the transaction
            transaction_cost: Custom transaction cost (default: None, will be calculated automatically)
        """
        super().__init__("buy", stock, shares, price_per_share, currency, transaction_date, transaction_cost)


class Sell(Transaction):
    """Specialized class for sell transactions"""
    
    def __init__(self, stock: Stock, shares: float, price_per_share: float,
                currency: str, transaction_date: date, transaction_cost: Optional[float] = None) -> None:
        """
        Initialize a sell transaction
        
        Args:
            stock: The stock being sold
            shares: Number of shares sold
            price_per_share: Price per share
            currency: Currency of the transaction
            transaction_date: Date of the transaction
            transaction_cost: Custom transaction cost (default: None, will be calculated automatically)
        """
        super().__init__("sell", stock, shares, price_per_share, currency, transaction_date, transaction_cost)