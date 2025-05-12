import pytest
from datetime import date

from backend.money_amount import MoneyAmount
from backend.market_interface import MarketInterface

@pytest.fixture
def amount_date():
    """Create a sample date for testing"""
    return date(2023, 1, 15)

@pytest.fixture
def money_amount(amount_date):
    """Create a sample money amount for testing"""
    return MoneyAmount(100.0, "USD", amount_date)

class TestMoneyAmount:
    """Test cases for the MoneyAmount class"""
    
    def test_money_amount_initialization(self, money_amount, amount_date):
        """Test that money amounts are initialized correctly"""
        assert money_amount.amount == 100.0
        assert money_amount.currency == "USD"
        assert money_amount.date == amount_date
        
        # Test default date
        money_amount = MoneyAmount(100.0, "USD")
        assert money_amount.date == date.today()
        
        # Test default currency
        money_amount = MoneyAmount(100.0)
        assert money_amount.currency == "EUR"  # Default currency
    
    def test_negative_amount(self):
        """Test that negative amounts raise ValueError"""
        with pytest.raises(ValueError, match="Amount cannot be negative"):
            money_amount = MoneyAmount(-100.0, "USD")
    
    def test_get_money_amount_same_currency(self, money_amount):
        """Test getting money amount in the same currency"""
        amount = money_amount.get_money_amount("USD")
        assert amount == 100.0
    
    def test_get_money_amount_different_currency(self, money_amount, mocker):
        """Test getting money amount in a different currency"""
        # Mock the market interface's convert_currency method
        mocker.patch('backend.market_interface.MarketInterface.convert_currency', return_value=85.0)
        
        # Get amount in EUR
        amount = money_amount.get_money_amount("EUR")
        assert amount == 85.0
    
    def test_euro_value(self, money_amount, mocker):
        """Test euro_value property"""
        # Mock the get_money_amount method
        mocker.patch.object(money_amount, 'get_money_amount', return_value=85.0)
        
        # Get euro value
        euro_value = money_amount.euro_value
        assert euro_value == 85.0
    
    def test_add_money_amounts(self, money_amount, amount_date):
        """Test adding two money amounts"""
        # Create another money amount
        other_amount = MoneyAmount(50.0, "USD", amount_date)
        
        # Add money amounts
        result = money_amount + other_amount
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 150.0
        assert result.currency == "USD"
        assert result.date == amount_date
    
    def test_add_money_amounts_different_currencies(self, money_amount, amount_date, mocker):
        """Test adding money amounts with different currencies"""
        # Create another money amount in EUR
        other_amount = MoneyAmount(50.0, "EUR", amount_date)
        
        # Mock the other amount's get_money_amount method
        mocker.patch.object(other_amount, 'get_money_amount', return_value=60.0)
        
        # Add money amounts
        result = money_amount + other_amount
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 160.0  # 100 + 60
        assert result.currency == "USD"
        assert result.date == amount_date
    
    def test_add_number(self, money_amount):
        """Test adding a number to a money amount"""
        # Add a number
        result = money_amount + 50.0
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 150.0
        assert result.currency == "USD"
    
    def test_subtract_money_amounts(self, money_amount, amount_date):
        """Test subtracting two money amounts"""
        # Create another money amount
        other_amount = MoneyAmount(50.0, "USD", amount_date)
        
        # Subtract money amounts
        result = money_amount - other_amount
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 50.0
        assert result.currency == "USD"
    
    def test_subtract_money_amounts_different_currencies(self, money_amount, amount_date, mocker):
        """Test subtracting money amounts with different currencies"""
        # Create another money amount in EUR
        other_amount = MoneyAmount(50.0, "EUR", amount_date)
        
        # Mock the other amount's get_money_amount method
        mocker.patch.object(other_amount, 'get_money_amount', return_value=60.0)
        
        # Subtract money amounts
        result = money_amount - other_amount
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 40.0  # 100 - 60
        assert result.currency == "USD"
        assert result.date == amount_date
    
    def test_subtract_number(self, money_amount):
        """Test subtracting a number from a money amount"""
        # Subtract a number
        result = money_amount - 50.0
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 50.0
        assert result.currency == "USD"
    
    def test_multiply(self, money_amount):
        """Test multiplying a money amount by a scalar"""
        # Multiply by a scalar
        result = money_amount * 2
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 200.0
        assert result.currency == "USD"
    
    def test_divide(self, money_amount):
        """Test dividing a money amount by a scalar"""
        # Divide by a scalar
        result = money_amount / 2
        
        # Check result
        assert isinstance(result, MoneyAmount)
        assert result.amount == 50.0
        assert result.currency == "USD"
    
    def test_divide_by_zero(self, money_amount):
        """Test that dividing by zero raises ZeroDivisionError"""
        with pytest.raises(ZeroDivisionError):
            money_amount / 0
    
    def test_equality(self, money_amount, amount_date):
        """Test equality comparison"""
        # Create another money amount with the same value
        other_amount = MoneyAmount(100.0, "USD", amount_date)
        
        # Check equality
        assert money_amount == other_amount
        
        # Check equality with a number
        assert money_amount == 100.0
    
    def test_equality_different_currencies(self, money_amount, amount_date, mocker):
        """Test equality comparison with different currencies"""
        # Create another money amount in EUR
        other_amount = MoneyAmount(85.0, "EUR", amount_date)
        
        # Mock the get_money_amount method instead of the euro_value property
        mocker.patch.object(money_amount, 'get_money_amount', return_value=85.0)
        mocker.patch.object(other_amount, 'get_money_amount', return_value=85.0)
        
        # Check equality
        assert money_amount == other_amount
    
    def test_less_than(self, money_amount, amount_date):
        """Test less than comparison"""
        # Create another money amount with a higher value
        other_amount = MoneyAmount(150.0, "USD", amount_date)
        
        # Check less than
        assert money_amount < other_amount
        
        # Check less than with a number
        assert money_amount < 150.0
    
    def test_less_than_different_currencies(self, money_amount, amount_date, mocker):
        """Test less than comparison with different currencies"""
        # Create another money amount in EUR
        other_amount = MoneyAmount(100.0, "EUR", amount_date)
        
        # Mock the get_money_amount method instead of the euro_value property
        mocker.patch.object(money_amount, 'get_money_amount', return_value=85.0)
        mocker.patch.object(other_amount, 'get_money_amount', return_value=100.0)
        
        # Check less than
        assert money_amount < other_amount
    
    def test_greater_than(self, money_amount, amount_date):
        """Test greater than comparison"""
        # Create another money amount with a lower value
        other_amount = MoneyAmount(50.0, "USD", amount_date)
        
        # Check greater than
        assert money_amount > other_amount
        
        # Check greater than with a number
        assert money_amount > 50.0
    
    def test_greater_than_different_currencies(self, money_amount, amount_date, mocker):
        """Test greater than comparison with different currencies"""
        # Create another money amount in EUR
        other_amount = MoneyAmount(70.0, "EUR", amount_date)
        
        # Mock the get_money_amount method instead of the euro_value property
        mocker.patch.object(money_amount, 'get_money_amount', return_value=85.0)
        mocker.patch.object(other_amount, 'get_money_amount', return_value=70.0)
        
        # Check greater than
        assert money_amount > other_amount
    
    def test_string_representation(self, money_amount):
        """Test string representation"""
        # Check string representation
        assert str(money_amount) == "100.00 USD (2023-01-15)"
    
    def test_repr(self, money_amount, amount_date):
        """Test repr representation"""
        # Check repr representation
        assert repr(money_amount) == f"MoneyAmount(100.0, 'USD', {amount_date})" 