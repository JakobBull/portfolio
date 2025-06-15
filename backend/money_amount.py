from __future__ import annotations
from typing import Dict

class MoneyAmount:
    """Represents an amount of money in a specific currency."""

    def __init__(self, amount: float, currency: str):
        """
        Initializes a MoneyAmount object.

        Args:
            amount: The amount of money.
            currency: The currency code (e.g., 'USD', 'EUR').
        """
        if not isinstance(amount, (int, float)):
            raise TypeError("Amount must be a numeric value.")
        if not isinstance(currency, str) or len(currency) != 3:
            raise ValueError("Currency must be a 3-letter string code.")

        self.amount = amount
        self.currency = currency.upper()

    def __repr__(self) -> str:
        """Return a string representation of the MoneyAmount."""
        return f"MoneyAmount({self.amount}, '{self.currency}')"

    def __str__(self) -> str:
        """Return a user-friendly string representation."""
        return f"{self.amount:,.2f} {self.currency}"

    def get_money_amount(self, currency: str) -> float:
        """
        Returns the amount in the specified currency.
        
        Note: This is a simplified version and does not perform currency conversion.
        """
        if self.currency == currency:
            return self.amount
        # In a real application, you would have exchange rates here.
        # For now, we'll assume the caller handles conversion.
        raise ValueError(f"Currency mismatch: expected {currency}, have {self.currency}") 