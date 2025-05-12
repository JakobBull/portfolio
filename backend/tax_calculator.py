from datetime import date
from typing import Dict, List, Optional
from backend.money_amount import MoneyAmount
from backend.transaction import Transaction

class GermanTaxCalculator:
    """Tax calculator implementing German tax rules for capital gains and dividends"""
    
    # German tax constants
    CAPITAL_GAINS_TAX_RATE = 0.25  # 25% capital gains tax
    SOLIDARITY_SURCHARGE_RATE = 0.055  # 5.5% solidarity surcharge on tax amount
    CHURCH_TAX_RATE = 0.08  # 8-9% church tax (using 8% as default)
    ANNUAL_EXEMPTION_SINGLE = 801.0  # Annual exemption for single person (EUR)
    ANNUAL_EXEMPTION_MARRIED = 1602.0  # Annual exemption for married couple (EUR)
    
    def __init__(self, is_married: bool = False, church_tax: bool = False, 
                 partial_exemption: bool = False):
        """Initialize tax calculator with personal settings
        
        Args:
            is_married: Whether the taxpayer is married (affects exemption amount)
            church_tax: Whether church tax applies
            partial_exemption: Whether partial exemption for funds applies
        """
        self.is_married = is_married
        self.church_tax = church_tax
        self.partial_exemption = partial_exemption
        self.annual_exemption = self.ANNUAL_EXEMPTION_MARRIED if is_married else self.ANNUAL_EXEMPTION_SINGLE
        self.used_exemption = 0.0
        self.tax_year = date.today().year
        
        # Track realized gains and losses
        self.realized_gains = 0.0
        self.realized_losses = 0.0
        self.dividend_income = 0.0
        
    def reset_for_year(self, year: int):
        """Reset tax calculations for a new tax year"""
        if self.tax_year != year:
            self.tax_year = year
            self.used_exemption = 0.0
            self.realized_gains = 0.0
            self.realized_losses = 0.0
            self.dividend_income = 0.0
            
    def calculate_transaction_tax(self, buy_transaction: Transaction, 
                                 sell_transaction: Transaction) -> MoneyAmount:
        """Calculate tax for a buy-sell transaction pair"""
        # Reset for year if needed
        self.reset_for_year(sell_transaction.date.year)
        
        # Calculate gain/loss in EUR
        buy_value = buy_transaction.get_transaction_value('EUR')
        sell_value = sell_transaction.get_transaction_value('EUR')
        gain = sell_value - buy_value
        
        # Apply partial exemption if applicable (e.g., for equity funds)
        if self.partial_exemption:
            gain *= 0.7  # 30% exemption for equity funds
            
        # Track gains/losses
        if gain > 0:
            self.realized_gains += gain
        else:
            self.realized_losses += abs(gain)
            
        # Calculate taxable amount after exemption
        taxable_gain = max(0, gain)
        
        # Apply annual exemption
        remaining_exemption = max(0, self.annual_exemption - self.used_exemption)
        if remaining_exemption > 0:
            exemption_used = min(remaining_exemption, taxable_gain)
            taxable_gain -= exemption_used
            self.used_exemption += exemption_used
            
        # Calculate base tax
        base_tax = taxable_gain * self.CAPITAL_GAINS_TAX_RATE
        
        # Add solidarity surcharge
        solidarity_surcharge = base_tax * self.SOLIDARITY_SURCHARGE_RATE
        
        # Add church tax if applicable
        church_tax = base_tax * self.CHURCH_TAX_RATE if self.church_tax else 0.0
        
        # Total tax
        total_tax = base_tax + solidarity_surcharge + church_tax
        
        # Return as MoneyAmount
        return MoneyAmount(total_tax, 'EUR', sell_transaction.date)
        
    def calculate_dividend_tax(self, dividend_amount: MoneyAmount) -> MoneyAmount:
        """Calculate tax for dividend income"""
        # Reset for year if needed
        self.reset_for_year(dividend_amount.date.year)
        
        # Convert to EUR if needed
        amount_eur = dividend_amount.get_money_amount('EUR')
        
        # Track dividend income
        self.dividend_income += amount_eur
        
        # Apply partial exemption if applicable (e.g., for equity funds)
        taxable_amount = amount_eur
        if self.partial_exemption:
            taxable_amount *= 0.7  # 30% exemption for equity funds
            
        # Apply annual exemption
        remaining_exemption = max(0, self.annual_exemption - self.used_exemption)
        if remaining_exemption > 0:
            exemption_used = min(remaining_exemption, taxable_amount)
            taxable_amount -= exemption_used
            self.used_exemption += exemption_used
            
        # Calculate base tax
        base_tax = taxable_amount * self.CAPITAL_GAINS_TAX_RATE
        
        # Add solidarity surcharge
        solidarity_surcharge = base_tax * self.SOLIDARITY_SURCHARGE_RATE
        
        # Add church tax if applicable
        church_tax = base_tax * self.CHURCH_TAX_RATE if self.church_tax else 0.0
        
        # Total tax
        total_tax = base_tax + solidarity_surcharge + church_tax
        
        # Return as MoneyAmount
        return MoneyAmount(total_tax, 'EUR', dividend_amount.date)
        
    def get_tax_report(self) -> Dict:
        """Generate a tax report for the current tax year"""
        # Calculate net taxable gains (after offsetting losses)
        net_taxable_gains = max(0, self.realized_gains - self.realized_losses)
        
        # Calculate total taxable income
        total_taxable_income = net_taxable_gains + self.dividend_income
        
        # Apply remaining exemption
        remaining_exemption = max(0, self.annual_exemption - self.used_exemption)
        taxable_after_exemption = max(0, total_taxable_income - remaining_exemption)
        
        # Calculate base tax
        base_tax = taxable_after_exemption * self.CAPITAL_GAINS_TAX_RATE
        
        # Add solidarity surcharge
        solidarity_surcharge = base_tax * self.SOLIDARITY_SURCHARGE_RATE
        
        # Add church tax if applicable
        church_tax = base_tax * self.CHURCH_TAX_RATE if self.church_tax else 0.0
        
        # Total tax
        total_tax = base_tax + solidarity_surcharge + church_tax
        
        return {
            'tax_year': self.tax_year,
            'realized_gains': self.realized_gains,
            'realized_losses': self.realized_losses,
            'dividend_income': self.dividend_income,
            'net_taxable_gains': net_taxable_gains,
            'total_taxable_income': total_taxable_income,
            'annual_exemption': self.annual_exemption,
            'used_exemption': self.used_exemption,
            'remaining_exemption': remaining_exemption,
            'taxable_after_exemption': taxable_after_exemption,
            'base_tax': base_tax,
            'solidarity_surcharge': solidarity_surcharge,
            'church_tax': church_tax,
            'total_tax': total_tax
        } 