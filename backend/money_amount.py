import datetime
from typing import Union, Optional, cast
from backend.market_interface import MarketInterface
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default currency for the system
DEFAULT_CURRENCY = 'EUR'

class MoneyAmount:
    """Class representing a monetary amount with currency and date information"""
    
    def __init__(self, amount: float, currency: str = DEFAULT_CURRENCY, date: Optional[datetime.date] = None) -> None:
        """
        Initialize a money amount with currency and date
        
        Args:
            amount: The monetary amount
            currency: The currency code (default: EUR)
            date: The date of the amount (default: today)
            
        Raises:
            ValueError: If amount is negative or currency is invalid
            TypeError: If amount is not a number
        """
        if not isinstance(amount, (int, float)):
            raise TypeError(f"Amount must be a number, got {type(amount)}")
            
        if amount < 0:
            raise ValueError("Amount cannot be negative")
            
        if not currency or not isinstance(currency, str):
            raise ValueError(f"Currency must be a non-empty string, got {currency}")
            
        if date is None:
            date = datetime.date.today()
            
        self.amount: float = float(amount)  # Ensure it's a float
        self.currency: str = currency
        self.date: datetime.date = date
        self._market_interface: Optional[MarketInterface] = None  # Lazy initialization
        
    def get_money_amount(self, currency: str = DEFAULT_CURRENCY) -> float:
        """
        Get the amount in the specified currency
        
        Args:
            currency: Target currency code
            
        Returns:
            The amount converted to the target currency
            
        Raises:
            ValueError: If currency conversion fails
        """
        if not currency or not isinstance(currency, str):
            raise ValueError(f"Currency must be a non-empty string, got {currency}")
            
        if currency == self.currency:
            return self.amount
            
        # Initialize market interface if needed
        if self._market_interface is None:
            self._market_interface = MarketInterface()
            
        try:
            # Convert to requested currency
            return self._market_interface.convert_currency(
                self.amount,
                self.currency,
                currency,
                self.date
            )
        except Exception as e:
            logger.error(f"Error converting {self.amount} {self.currency} to {currency}: {e}")
            # Re-raise with more context
            raise ValueError(f"Failed to convert {self.amount} {self.currency} to {currency}: {e}") from e
        
    @property
    def euro_value(self) -> float:
        """Get the amount in euros"""
        return self.get_money_amount('EUR')
        
    def __add__(self, other: Union['MoneyAmount', int, float]) -> 'MoneyAmount':
        """
        Add two money amounts or a money amount and a number
        
        Args:
            other: Another MoneyAmount or a number
            
        Returns:
            A new MoneyAmount with the sum
            
        Raises:
            TypeError: If other is not a MoneyAmount or a number
        """
        if isinstance(other, MoneyAmount):
            # Convert other to this currency
            try:
                other_amount = other.get_money_amount(self.currency)
                return MoneyAmount(self.amount + other_amount, self.currency, self.date)
            except Exception as e:
                logger.error(f"Error adding money amounts: {e}")
                raise ValueError(f"Failed to add money amounts: {e}") from e
        elif isinstance(other, (int, float)):
            # Add a raw number
            return MoneyAmount(self.amount + other, self.currency, self.date)
        else:
            raise TypeError(f"Cannot add {type(other)} to MoneyAmount")
            
    def __sub__(self, other: Union['MoneyAmount', int, float]) -> 'MoneyAmount':
        """
        Subtract two money amounts or a number from a money amount
        
        Args:
            other: Another MoneyAmount or a number
            
        Returns:
            A new MoneyAmount with the difference
            
        Raises:
            TypeError: If other is not a MoneyAmount or a number
        """
        if isinstance(other, MoneyAmount):
            # Convert other to this currency
            try:
                other_amount = other.get_money_amount(self.currency)
                return MoneyAmount(self.amount - other_amount, self.currency, self.date)
            except Exception as e:
                logger.error(f"Error subtracting money amounts: {e}")
                raise ValueError(f"Failed to subtract money amounts: {e}") from e
        elif isinstance(other, (int, float)):
            # Subtract a raw number
            return MoneyAmount(self.amount - other, self.currency, self.date)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from MoneyAmount")
            
    def __mul__(self, other: Union[int, float]) -> 'MoneyAmount':
        """
        Multiply money amount by a scalar
        
        Args:
            other: A number to multiply by
            
        Returns:
            A new MoneyAmount with the product
            
        Raises:
            TypeError: If other is not a number
        """
        if isinstance(other, (int, float)):
            return MoneyAmount(self.amount * other, self.currency, self.date)
        else:
            raise TypeError(f"Cannot multiply MoneyAmount by {type(other)}")
            
    def __truediv__(self, other: Union[int, float]) -> 'MoneyAmount':
        """
        Divide money amount by a scalar
        
        Args:
            other: A number to divide by
            
        Returns:
            A new MoneyAmount with the quotient
            
        Raises:
            TypeError: If other is not a number
            ZeroDivisionError: If other is zero
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return MoneyAmount(self.amount / other, self.currency, self.date)
        else:
            raise TypeError(f"Cannot divide MoneyAmount by {type(other)}")
            
    def __eq__(self, other: object) -> bool:
        """
        Check if two money amounts are equal
        
        Args:
            other: Another MoneyAmount or a number
            
        Returns:
            True if equal, False otherwise
        """
        if isinstance(other, MoneyAmount):
            # Convert both to a common currency (EUR) for comparison
            try:
                return abs(self.euro_value - other.euro_value) < 1e-6  # Use small epsilon for float comparison
            except Exception as e:
                logger.error(f"Error comparing money amounts: {e}")
                return False
        elif isinstance(other, (int, float)):
            return abs(self.amount - other) < 1e-6  # Use small epsilon for float comparison
        else:
            return False
            
    def __lt__(self, other: Union['MoneyAmount', int, float]) -> bool:
        """
        Check if this money amount is less than another
        
        Args:
            other: Another MoneyAmount or a number
            
        Returns:
            True if less than, False otherwise
            
        Raises:
            TypeError: If other is not a MoneyAmount or a number
        """
        if isinstance(other, MoneyAmount):
            try:
                return self.euro_value < other.euro_value
            except Exception as e:
                logger.error(f"Error comparing money amounts: {e}")
                raise ValueError(f"Failed to compare money amounts: {e}") from e
        elif isinstance(other, (int, float)):
            return self.amount < other
        else:
            raise TypeError(f"Cannot compare MoneyAmount with {type(other)}")
            
    def __gt__(self, other: Union['MoneyAmount', int, float]) -> bool:
        """
        Check if this money amount is greater than another
        
        Args:
            other: Another MoneyAmount or a number
            
        Returns:
            True if greater than, False otherwise
            
        Raises:
            TypeError: If other is not a MoneyAmount or a number
        """
        if isinstance(other, MoneyAmount):
            try:
                return self.euro_value > other.euro_value
            except Exception as e:
                logger.error(f"Error comparing money amounts: {e}")
                raise ValueError(f"Failed to compare money amounts: {e}") from e
        elif isinstance(other, (int, float)):
            return self.amount > other
        else:
            raise TypeError(f"Cannot compare MoneyAmount with {type(other)}")
            
    def __str__(self) -> str:
        """String representation of money amount"""
        return f"{self.amount:.2f} {self.currency} ({self.date})"
        
    def __repr__(self) -> str:
        """Detailed representation of money amount"""
        return f"MoneyAmount({self.amount}, '{self.currency}', {self.date})"

if __name__ == "__main__":
    money = MoneyAmount(100, "USD", datetime.date(2025, 2, 21))
    print(money.euro_value)  # 90.0