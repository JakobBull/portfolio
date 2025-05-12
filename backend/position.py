from datetime import date
from typing import Optional, Union, cast
import logging
from backend.market_interface import MarketInterface
from backend.money_amount import MoneyAmount
from backend.transaction import Transaction

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Position:
    """Class representing a position in a specific stock"""

    def __init__(self, transaction: Transaction) -> None:
        """
        Initialize a position from a buy transaction
        
        Args:
            transaction: A buy transaction to open the position
            
        Raises:
            ValueError: If transaction is not a buy transaction
            TypeError: If transaction is not a Transaction object
        """
        if not isinstance(transaction, Transaction):
            raise TypeError(f"Expected Transaction object, got {type(transaction)}")
            
        if transaction.type != "buy":
            raise ValueError(f"Transaction type {transaction.type} is not supported. New positions can only be opened with a 'buy'.")
            
        self.amount: float = transaction.amount
        self.stock = transaction.stock
        self.ticker: str = transaction.stock.ticker
        self.purchase_price_net: MoneyAmount = transaction.price
        self.purchase_cost: MoneyAmount = transaction.cost
        self.purchase_date: date = transaction.date
        
        # Calculate cost per share
        if transaction.amount <= 0:
            raise ValueError("Transaction amount must be positive")
            
        cost_per_share = transaction.cost.amount / transaction.amount
        
        # Calculate gross purchase price (price + cost per share)
        self.purchase_price_gross: MoneyAmount = MoneyAmount(
            transaction.price.amount + cost_per_share,
            transaction.price.currency,
            transaction.price.date
        )
        
        self.currency: str = transaction.currency
        self._market_interface: MarketInterface = MarketInterface()

    def update(self, transaction: Transaction) -> None:
        """
        Update position based on new transaction
        
        Args:
            transaction: A buy or sell transaction to update the position
            
        Raises:
            ValueError: If transaction type is not supported or if trying to sell more than owned
            TypeError: If transaction is not a Transaction object
        """
        if not isinstance(transaction, Transaction):
            raise TypeError(f"Expected Transaction object, got {type(transaction)}")
            
        if transaction.stock.ticker != self.ticker:
            raise ValueError(f"Transaction ticker {transaction.stock.ticker} does not match position ticker {self.ticker}")
            
        if transaction.type == "buy":
            # Calculate new weighted average purchase price
            total_amount = self.amount + transaction.amount
            
            if total_amount <= 0:
                raise ValueError("Total position amount must be positive after update")
                
            try:
                # Convert transaction price to same currency as position if needed
                trans_price_in_pos_currency = transaction.price.get_money_amount(
                    self.purchase_price_net.currency
                )
                trans_cost_in_pos_currency = transaction.cost.get_money_amount(
                    self.purchase_cost.currency
                )
                
                # Calculate new weighted average price
                new_price_amount = (
                    (self.purchase_price_net.amount * self.amount + 
                     trans_price_in_pos_currency * transaction.amount) / total_amount
                )
                
                # Update net price as MoneyAmount
                self.purchase_price_net = MoneyAmount(
                    new_price_amount,
                    self.purchase_price_net.currency,
                    date.today()
                )
                
                # Calculate new weighted average cost
                new_cost_amount = (
                    (self.purchase_cost.amount * self.amount + 
                     trans_cost_in_pos_currency * transaction.amount) / total_amount
                )
                
                # Update cost as MoneyAmount
                self.purchase_cost = MoneyAmount(
                    new_cost_amount,
                    self.purchase_cost.currency,
                    date.today()
                )
                
                # Update gross price
                self.purchase_price_gross = MoneyAmount(
                    self.purchase_price_net.amount + (self.purchase_cost.amount / total_amount),
                    self.purchase_price_net.currency,
                    date.today()
                )
                
                self.amount = total_amount
            except Exception as e:
                logger.error(f"Error updating position with buy transaction: {e}")
                raise ValueError(f"Failed to update position with buy transaction: {e}") from e
            
        elif transaction.type == "sell":
            if transaction.amount > self.amount:
                raise ValueError(f"Cannot sell {transaction.amount} shares when only {self.amount} are owned")
                
            self.amount -= transaction.amount
            # Note: We don't adjust the cost basis when selling - using FIFO accounting
        else:
            raise ValueError(f"Unsupported transaction type: {transaction.type}")

    def get_value(self, currency: str, evaluation_date: Optional[date] = None) -> float:
        """
        Get current market value of position in specified currency
        
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
            return self.stock.get_value(currency, evaluation_date, self.amount)
        except Exception as e:
            logger.error(f"Error getting position value: {e}")
            raise ValueError(f"Failed to get position value: {e}") from e
        
    def get_cost_basis(self, currency: Optional[str] = None) -> float:
        """
        Get total cost basis (purchase price + fees) in specified currency
        
        Args:
            currency: Target currency code (default: position currency)
            
        Returns:
            The cost basis in the specified currency
            
        Raises:
            ValueError: If cost basis cannot be calculated
        """
        # If no currency specified, use the position's currency
        if currency is None:
            currency = self.purchase_price_gross.currency
            
        try:
            # Calculate total cost
            total_cost = self.purchase_price_gross.amount * self.amount
            
            # Convert if needed
            if currency == self.purchase_price_gross.currency:
                return total_cost
                
            # Create MoneyAmount for conversion
            money_amount = MoneyAmount(
                total_cost, 
                self.purchase_price_gross.currency,
                self.purchase_price_gross.date
            )
            return money_amount.get_money_amount(currency)
        except Exception as e:
            logger.error(f"Error calculating cost basis: {e}")
            raise ValueError(f"Failed to calculate cost basis: {e}") from e
        
    def get_unrealized_pl(self, currency: str, evaluation_date: Optional[date] = None) -> float:
        """
        Calculate unrealized profit/loss
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the valuation (default: today)
            
        Returns:
            The unrealized profit/loss in the specified currency
            
        Raises:
            ValueError: If unrealized P/L cannot be calculated
        """
        try:
            current_value = self.get_value(currency, evaluation_date)
            cost_basis = self.get_cost_basis(currency)
            return current_value - cost_basis
        except Exception as e:
            logger.error(f"Error calculating unrealized P/L: {e}")
            raise ValueError(f"Failed to calculate unrealized P/L: {e}") from e
            
    def get_return_percentage(self, currency: str, evaluation_date: Optional[date] = None) -> float:
        """
        Calculate return percentage
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the valuation (default: today)
            
        Returns:
            The return percentage
            
        Raises:
            ValueError: If return percentage cannot be calculated
        """
        try:
            cost_basis = self.get_cost_basis(currency)
            
            if cost_basis <= 0:
                return 0.0
                
            unrealized_pl = self.get_unrealized_pl(currency, evaluation_date)
            return (unrealized_pl / cost_basis) * 100
        except Exception as e:
            logger.error(f"Error calculating return percentage: {e}")
            return 0.0  # Return 0 instead of raising an error

    def get_annualized_return_percentage(self, currency: str, evaluation_date: Optional[date] = None) -> float:
        """
        Calculate annualized return percentage
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the valuation (default: today)
            
        Returns:
            The annualized return percentage
            
        Raises:
            ValueError: If evaluation_date is before purchase_date or if annualized return cannot be calculated
        """
        try:
            if evaluation_date is None:
                evaluation_date = date.today()
                
            # Calculate total return percentage
            total_return = self.get_return_percentage(currency, evaluation_date) / 100
            
            # Calculate number of years held
            years_held = (evaluation_date - self.purchase_date).days / 365.25
            
            if years_held < 0:
                raise ValueError(f"Evaluation date {evaluation_date} is before purchase date {self.purchase_date}")
                
            # For same-day trades, return the total return directly
            if years_held == 0:
                return total_return * 100
                
            # Calculate annualized return
            annualized_return = ((1 + total_return) ** (1 / years_held) - 1) * 100
            return annualized_return
        except Exception as e:
            logger.error(f"Error calculating annualized return percentage: {e}")
            raise ValueError(f"Failed to calculate annualized return percentage: {e}") from e
            
    def __str__(self) -> str:
        """String representation of position"""
        return f"Position: {self.amount} shares of {self.ticker} @ {self.purchase_price_net}"
        
    def __repr__(self) -> str:
        """Detailed representation of position"""
        return (f"Position(ticker='{self.ticker}', amount={self.amount}, "
                f"purchase_price_net={self.purchase_price_net}, "
                f"purchase_cost={self.purchase_cost})")
