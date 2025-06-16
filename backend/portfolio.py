from backend.money_amount import MoneyAmount
from datetime import date, timedelta
from typing import List, Dict, Optional, Union, cast, Any
import logging
import pandas as pd
from backend.database import Stock, Transaction, TransactionType, Dividend

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Portfolio:
    """Represents a portfolio based on transactions."""

    def __init__(self, positions: list, transactions: list[Transaction]):
        """Initializes the portfolio with transactions."""
        # positions parameter is kept for backward compatibility but ignored
        self.transactions = transactions
        self.dividends: list[Dividend] = [
            t for t in transactions if isinstance(t, Dividend) or t.type == TransactionType.DIVIDEND
        ]

    def get_positions(self) -> dict[str, float]:
        """Get current positions calculated from transactions."""
        from backend.database import db_manager
        return db_manager.get_portfolio_positions_at_date(date.today())

    def get_total_value(self, value_date: date | None = None) -> float:
        """Calculates the total value of the portfolio on a given date."""
        from backend.database import db_manager
        if value_date is None:
            value_date = date.today()
        return db_manager.get_portfolio_value_at_date(value_date)

    def get_performance(self, start_date: date, end_date: date | None = None) -> dict[str, float]:
        """Calculates portfolio performance over a period."""
        end_date = end_date or date.today()
        initial_value = self.get_total_value(start_date)
        final_value = self.get_total_value(end_date)

        if initial_value == 0:
            return {"absolute_return": 0, "percentage_return": 0}

        absolute_return = final_value - initial_value
        percentage_return = (absolute_return / initial_value) * 100
        return {
            "absolute_return": absolute_return,
            "percentage_return": percentage_return,
        }

    def get_position_breakdown(self) -> dict[str, float]:
        """Returns a breakdown of portfolio value by position."""
        from backend.database import db_manager
        positions = self.get_positions()
        total_value = self.get_total_value()
        
        if total_value == 0:
            return {ticker: 0.0 for ticker in positions.keys()}

        breakdown = {}
        for ticker, shares in positions.items():
            latest_price = db_manager.get_latest_stock_price(ticker)
            if latest_price:
                position_value = shares * latest_price.price
                breakdown[ticker] = (position_value / total_value) * 100
            else:
                breakdown[ticker] = 0.0
        return breakdown

    def get_transaction_history(self, ticker: str | None = None, transaction_type: str | None = None) -> list[Transaction]:
        """Returns the transaction history, optionally filtered by ticker and/or transaction type."""
        filtered_transactions = self.transactions
        
        if ticker:
            filtered_transactions = [t for t in filtered_transactions if t.ticker == ticker]
            
        if transaction_type:
            filtered_transactions = [t for t in filtered_transactions if t.type == transaction_type]
            
        return filtered_transactions

    def get_value(self, currency: str, evaluation_date: Optional[date] = None) -> float:
        """
        Calculate total portfolio value in specified currency at given date
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the valuation (default: today)
            
        Returns:
            The total portfolio value in the specified currency
            
        Raises:
            ValueError: If portfolio value cannot be calculated
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        if evaluation_date is None:
            evaluation_date = date.today()
        
        try:
            # TODO: Add currency conversion when implemented
            return self.get_total_value(evaluation_date)
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            raise ValueError(f"Failed to calculate portfolio value: {e}") from e

    @property
    def euro_value(self) -> float:
        """Get total portfolio value in EUR"""
        try:
            return self.get_value('EUR')
        except Exception as e:
            logger.error(f"Error calculating portfolio euro value: {e}")
            return 0.0  # Return 0 instead of raising an error

    def get_gross_purchase_price(self, currency: str) -> float:
        """
        Calculate total purchase price including transaction costs
        
        Args:
            currency: Target currency code
            
        Returns:
            The total purchase price in the specified currency
            
        Raises:
            ValueError: If purchase price cannot be calculated
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        try:
            from backend.database import db_manager
            positions = self.get_positions()
            total_cost = 0.0
            
            for ticker in positions.keys():
                cost_basis = db_manager.get_position_cost_basis(ticker, date.today(), currency)
                total_cost += cost_basis
                
            return total_cost
        except Exception as e:
            logger.error(f"Error calculating gross purchase price: {e}")
            raise ValueError(f"Failed to calculate gross purchase price: {e}") from e

    @property
    def euro_gross_purchase_price(self) -> float:
        """Get total purchase price in EUR"""
        try:
            return self.get_gross_purchase_price('EUR')
        except Exception as e:
            logger.error(f"Error calculating euro gross purchase price: {e}")
            return 0.0  # Return 0 instead of raising an error

    def get_unrealized_pl(self, currency: str, evaluation_date: Optional[date] = None) -> float:
        """
        Calculate total return profit/loss for the portfolio (market value + dividends - cost basis).
        
        Args:
            currency: Target currency code
            evaluation_date: Date for the valuation (default: today)
            
        Returns:
            The total return P/L in the specified currency (includes dividends received)
            
        Raises:
            ValueError: If total return P/L cannot be calculated
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        if evaluation_date is None:
            evaluation_date = date.today()
            
        try:
            from backend.database import db_manager
            
            # Current market value of positions
            current_value = self.get_value(currency, evaluation_date)
            # Original cost basis
            cost_basis = self.get_gross_purchase_price(currency)
            # Dividend income received up to evaluation date
            dividend_income = self.get_dividend_income(currency, end_date=evaluation_date)
            
            # Total Return P/L = (Current Value + Dividends) - Cost Basis
            return current_value + dividend_income - cost_basis
        except Exception as e:
            logger.error(f"Error calculating unrealized P/L: {e}")
            raise ValueError(f"Failed to calculate unrealized P/L: {e}") from e

    def get_dividend_income(self, currency: str, start_date: Optional[date] = None, 
                           end_date: Optional[date] = None) -> float:
        """
        Calculate total dividend income over a period using the dividends table
        
        Args:
            currency: Target currency code
            start_date: Start date for calculation (default: beginning of time)
            end_date: End date for calculation (default: today)
            
        Returns:
            Total dividend income in the specified currency
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        if end_date is None:
            end_date = date.today()
            
        try:
            from backend.database import db_manager
            
            # Use the database manager method to calculate dividend income
            if start_date:
                # Calculate dividend income for the specific period
                total_dividends = 0.0
                with db_manager.session_scope() as session:
                    dividends = (session.query(Dividend)
                               .filter(Dividend.date >= start_date)
                               .filter(Dividend.date <= end_date)
                               .all())
                    
                    for dividend in dividends:
                        # Calculate shares held at dividend date to get total dividend received
                        shares_held = db_manager.get_shares_held_at_date(dividend.ticker, dividend.date)
                        if shares_held > 0:
                            dividend_received = dividend.amount_per_share * shares_held
                            # TODO: Add currency conversion if needed
                            total_dividends += dividend_received
                
                return total_dividends
            else:
                # Get all dividend income up to end_date
                return db_manager.get_dividend_income_up_to_date(end_date, currency)
                
        except Exception as e:
            logger.warning(f"Error getting dividend income from dividends table: {e}")
            # Fallback to transaction-based dividend calculation
            total_dividends = 0.0
            for transaction in self.transactions:
                if (transaction.type == TransactionType.DIVIDEND and
                    (start_date is None or transaction.date >= start_date) and
                    transaction.date <= end_date):
                    # TODO: Add currency conversion when implemented
                    dividend_amount = transaction.amount * transaction.price
                    total_dividends += dividend_amount
                    
            return total_dividends

    def get_dividend_yield(self, currency: str = 'EUR') -> float:
        """
        Calculate current dividend yield for the portfolio
        
        Args:
            currency: Target currency code (default: EUR)
            
        Returns:
            The dividend yield as a decimal value
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        try:
            # Get annual dividend income (last 12 months)
            today = date.today()
            one_year_ago = date(today.year - 1, today.month, today.day)
            annual_dividend = self.get_dividend_income(currency, one_year_ago, today)
            
            # If no dividends in the last year, try using all historical dividends
            if annual_dividend <= 0:
                # Use a fixed start date that includes historical data
                fixed_start_date = date(2022, 1, 1)
                annual_dividend = self.get_dividend_income(currency, fixed_start_date, today)
                logger.info(f"No dividends in last year, using all dividends since {fixed_start_date}")
                
                # If still no dividends, return 0
                if annual_dividend <= 0:
                    return 0.0
                
                # Annualize the dividend income if we're using more than a year of data
                days_in_period = (today - fixed_start_date).days
                if days_in_period > 365:
                    annual_dividend = annual_dividend * 365 / days_in_period
            
            # Get current portfolio value
            current_value = self.get_value(currency)
            
            # Calculate yield
            if current_value > 0:
                return (annual_dividend / current_value)  # Return as decimal
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating dividend yield: {e}")
            return 0.0

    def get_historical_value(self, start_date: date, end_date: date) -> dict[date, float]:
        """
        Calculates the historical value of the portfolio over a date range.
        """
        from backend.database import db_manager
        values_series = db_manager.get_portfolio_values_over_time(start_date, end_date)
        return {date.date(): value for date, value in values_series.items()}
        
    def __str__(self) -> str:
        """String representation of portfolio"""
        positions = self.get_positions()
        return f"Portfolio with {len(positions)} positions, {len(self.transactions)} transactions"
        
    def __repr__(self) -> str:
        """Detailed representation of portfolio"""
        positions = self.get_positions()
        return f"Portfolio(positions={len(positions)}, transactions={len(self.transactions)})"

    def get_dividend_history(self, start_date: Optional[date] = None, 
                            end_date: Optional[date] = None, 
                            ticker: Optional[str] = None) -> list[Transaction]:
        """
        Get filtered dividend history
        
        Args:
            start_date: Start date for filtering (default: None)
            end_date: End date for filtering (default: None)
            ticker: Stock ticker to filter by (default: None)
            
        Returns:
            List of dividend transactions matching the filters
        """
        dividend_transactions = [t for t in self.transactions if t.type == TransactionType.DIVIDEND]
        
        filtered_dividends = []
        for dividend in dividend_transactions:
            if start_date and dividend.date < start_date:
                continue
            if end_date and dividend.date > end_date:
                continue
            if ticker and dividend.ticker != ticker:
                continue
            filtered_dividends.append(dividend)
        return filtered_dividends

    def get_realized_pl_transactions(self, currency: str) -> list[dict]:
        """
        Get realized profit/loss transactions in the specified currency
        
        Args:
            currency: Target currency code
            
        Returns:
            List of dictionaries with realized P/L transaction data
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        try:
            # Get all sell transactions
            sell_transactions = [t for t in self.transactions if t.type == 'sell']
            
            if not sell_transactions:
                return []
                
            # Group transactions by ticker
            ticker_transactions: Dict[str, List[Dict[str, Any]]] = {}
            
            # First, process all buy transactions
            buy_transactions = [t for t in self.transactions if t.type == 'buy']
            
            for transaction in buy_transactions:
                ticker = transaction.ticker
                
                if ticker not in ticker_transactions:
                    ticker_transactions[ticker] = []
                    
                ticker_transactions[ticker].append({
                    'type': 'buy',
                    'date': transaction.date,
                    'amount': transaction.amount,
                    'price': transaction.price,  # Already a float
                    'cost': transaction.cost     # Already a float
                })
                
            # Sort buy transactions by date (FIFO)
            for ticker in ticker_transactions:
                ticker_transactions[ticker].sort(key=lambda x: x['date'])
                
            # Process sell transactions
            realized_pl_transactions = []
            
            for transaction in sell_transactions:
                ticker = transaction.ticker
                
                if ticker not in ticker_transactions:
                    logger.warning(f"No buy transactions found for {ticker}")
                    continue
                    
                sell_amount = transaction.amount
                sell_price = transaction.price
                sell_cost = transaction.cost
                sell_date = transaction.date
                
                # Match sell with buy transactions (FIFO)
                remaining_sell = sell_amount
                buy_index = 0
                
                while remaining_sell > 0 and buy_index < len(ticker_transactions[ticker]):
                    buy_transaction = ticker_transactions[ticker][buy_index]
                    
                    if buy_transaction['type'] != 'buy':
                        buy_index += 1
                        continue
                        
                    buy_amount = buy_transaction['amount']
                    buy_price = buy_transaction['price']
                    buy_cost = buy_transaction['cost']
                    buy_date = buy_transaction['date']
                    
                    if buy_amount <= 0:
                        buy_index += 1
                        continue
                        
                    # Calculate amount to match
                    match_amount = min(remaining_sell, buy_amount)
                    
                    # Calculate P/L for this match
                    buy_value = match_amount * buy_price
                    sell_value = match_amount * sell_price
                    
                    # Adjust for costs (proportional)
                    buy_cost_portion = (match_amount / buy_amount) * buy_cost
                    sell_cost_portion = (match_amount / sell_amount) * sell_cost
                    
                    # Calculate P/L
                    match_pl = sell_value - buy_value - buy_cost_portion - sell_cost_portion
                    
                    # Add to realized P/L transactions
                    realized_pl_transactions.append({
                        'ticker': ticker,
                        'buy_date': buy_date,
                        'sell_date': sell_date,
                        'amount': match_amount,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'pl': match_pl,
                        'year': sell_date.year,
                        'currency': currency
                    })
                    
                    # Update remaining amounts
                    remaining_sell -= match_amount
                    ticker_transactions[ticker][buy_index]['amount'] -= match_amount
                    
                    if ticker_transactions[ticker][buy_index]['amount'] <= 0:
                        buy_index += 1
                        
            return realized_pl_transactions
        except Exception as e:
            logger.error(f"Error calculating realized P/L transactions: {e}")
            return []

    def get_dividend_transactions(self, currency: str) -> list[dict]:
        """
        Get dividend transactions in the specified currency
        
        Args:
            currency: Target currency code
            
        Returns:
            List of dictionaries with dividend transaction data
        """
        if not currency or not isinstance(currency, str):
            raise ValueError("Currency must be a non-empty string")
            
        try:
            # Get all dividend transactions
            dividends = self.get_dividend_history()
            
            if not dividends:
                return []
                
            # Convert to dictionary format
            dividend_transactions = []
            
            for dividend in dividends:
                dividend_value = dividend.amount * dividend.price
                
                dividend_transactions.append({
                    'ticker': dividend.ticker,
                    'date': dividend.date,
                    'shares': dividend.amount,
                    'dividend_per_share': dividend.price,
                    'amount': dividend_value,
                    'year': dividend.date.year,
                    'currency': currency
                })
                
            return dividend_transactions
        except Exception as e:
            logger.error(f"Error calculating dividend transactions: {e}")
            return []

