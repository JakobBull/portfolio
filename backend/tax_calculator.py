from __future__ import annotations
from backend.database import Transaction
from typing import List, Dict, Any, Optional
from datetime import date

class GermanTaxCalculator:
    """Calculates German capital gains taxes."""

    CAPITAL_GAINS_TAX_RATE = 0.25
    SOLIDARITY_SURCHARGE_RATE = 0.055
    CHURCH_TAX_RATE = 0.08
    ANNUAL_EXEMPTION_SINGLE = 1000.0  # Updated for 2023
    ANNUAL_EXEMPTION_MARRIED = 2000.0

    def __init__(
        self,
        is_married: bool = False,
        church_tax: bool = False,
        partial_exemption_rate: float = 0.0,
    ):
        """Initializes the tax calculator."""
        self.is_married = is_married
        self.church_tax = church_tax
        self.partial_exemption_rate = partial_exemption_rate
        self.annual_exemption = (
            self.ANNUAL_EXEMPTION_MARRIED
            if is_married
            else self.ANNUAL_EXEMPTION_SINGLE
        )
        self.tax_year = date.today().year
        self.realized_gains = 0.0
        self.realized_losses = 0.0
        self.dividend_income = 0.0

    def reset_for_year(self, year: int):
        """Resets the calculator for a new tax year."""
        self.tax_year = year
        self.realized_gains = 0.0
        self.realized_losses = 0.0
        self.dividend_income = 0.0

    def get_tax_report(self) -> dict[str, float]:
        """Calculates total tax liability based on the stored financial data."""
        # Net gains and losses
        net_gains = self.realized_gains - self.realized_losses
        
        # Total capital income
        total_income = net_gains + self.dividend_income
        
        # Apply partial exemption if applicable (e.g., for equity funds)
        taxable_income = total_income * (1 - self.partial_exemption_rate)
        
        # Deduct annual exemption
        taxable_amount = max(0, taxable_income - self.annual_exemption)

        if taxable_amount == 0:
            return {
                "total_tax": 0.0,
                "capital_gains_tax": 0.0,
                "solidarity_surcharge": 0.0,
                "church_tax": 0.0,
            }

        # Calculate taxes
        capital_gains_tax = taxable_amount * self.CAPITAL_GAINS_TAX_RATE
        solidarity_surcharge = capital_gains_tax * self.SOLIDARITY_SURCHARGE_RATE
        
        # Church tax is applied on the capital gains tax itself
        church_tax_amount = capital_gains_tax * self.CHURCH_TAX_RATE if self.church_tax else 0.0
        
        total_tax = capital_gains_tax + solidarity_surcharge + church_tax_amount

        return {
            "total_tax": total_tax,
            "capital_gains_tax": capital_gains_tax,
            "solidarity_surcharge": solidarity_surcharge,
            "church_tax": church_tax_amount,
        } 