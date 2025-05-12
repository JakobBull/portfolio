import pytest
from datetime import date

from backend.transaction import Transaction, Dividend
from backend.stock import Stock
from backend.money_amount import MoneyAmount

@pytest.fixture
def mock_stock(mocker):
    """Create a mock Stock object for testing"""
    stock = mocker.Mock(spec=Stock)
    stock.ticker = "AAPL"
    stock.get_price.return_value = 150.0
    return stock

@pytest.fixture
def transaction_date():
    """Create a sample transaction date"""
    return date(2023, 1, 15)

@pytest.fixture
def buy_transaction(mock_stock, transaction_date):
    """Create a sample buy transaction"""
    return Transaction(
        "buy", 
        mock_stock, 
        10, 
        150.0, 
        "USD", 
        transaction_date,
    )

@pytest.fixture
def sell_transaction(mock_stock, transaction_date):
    """Create a sample sell transaction"""
    return Transaction(
        "sell", 
        mock_stock, 
        5, 
        160.0, 
        "USD", 
        transaction_date
    )

@pytest.fixture
def dividend_transaction(mock_stock, transaction_date):
    """Create a sample dividend transaction"""
    return Transaction(
        "dividend", 
        mock_stock, 
        10, 
        0.5, 
        "USD", 
        transaction_date
    )

@pytest.fixture
def dividend(mock_stock, transaction_date):
    """Create a sample dividend"""
    return Dividend(
        mock_stock, 
        10, 
        0.5, 
        "USD", 
        transaction_date
    )

class TestTransaction:
    """Test cases for the Transaction class"""
    
    def test_transaction_initialization(self, buy_transaction, mock_stock, transaction_date):
        """Test that transactions are initialized correctly"""
        # Test buy transaction
        assert buy_transaction.type == "buy"
        assert buy_transaction.stock == mock_stock
        assert buy_transaction.amount == 10
        assert buy_transaction.price.amount == 150.0
        assert buy_transaction.price.currency == "USD"
        assert buy_transaction.date == transaction_date
    
    def test_sell_transaction_initialization(self, sell_transaction):
        """Test sell transaction initialization"""
        assert sell_transaction.type == "sell"
        assert sell_transaction.amount == 5
    
    def test_dividend_transaction_initialization(self, dividend_transaction):
        """Test dividend transaction initialization"""
        assert dividend_transaction.type == "dividend"
        assert dividend_transaction.price.amount == 0.5
    
    def test_invalid_transaction_type(self, mock_stock, transaction_date):
        """Test that invalid transaction types raise ValueError"""
        with pytest.raises(ValueError):
            Transaction("invalid", mock_stock, 10, 150.0, "USD", transaction_date)
    
    def test_negative_amount(self, mock_stock, transaction_date):
        """Test that negative amounts raise ValueError"""
        with pytest.raises(ValueError):
            Transaction("buy", mock_stock, -10, 150.0, "USD", transaction_date)
    
    def test_negative_price(self, mock_stock, transaction_date):
        """Test that negative prices raise ValueError"""
        with pytest.raises(ValueError):
            Transaction("buy", mock_stock, 10, -150.0, "USD", transaction_date)
    
    def test_transaction_cost(self, buy_transaction, dividend_transaction):
        """Test transaction cost calculation"""
        # Buy transaction should have a cost
        assert buy_transaction.cost.amount > 0
        
        # Dividend transaction should have zero cost
        assert dividend_transaction.cost.amount == 0
    
    def test_get_market_value(self, buy_transaction, dividend_transaction, mock_stock):
        """Test market value calculation"""
        # Mock the stock's get_price method
        mock_stock.get_price.return_value = 160.0
        
        # Test buy transaction market value
        market_value = buy_transaction.get_market_value("USD")
        assert market_value == 1600.0  # 10 shares * $160
        
        # Test dividend transaction market value
        dividend_value = dividend_transaction.get_market_value("USD")
        assert dividend_value == 5.0  # 10 shares * $0.5 dividend
    
    def test_get_transaction_value(self, buy_transaction, mocker):
        """Test transaction value calculation"""
        # Test buy transaction value
        transaction_value = buy_transaction.get_transaction_value("USD")
        assert transaction_value == 1500.0  # 10 shares * $150
        
        # Test with currency conversion (mocked)
        mocker.patch('backend.money_amount.MoneyAmount.get_money_amount', return_value=1275.0)
        eur_value = buy_transaction.get_transaction_value("EUR")
        assert eur_value == 1275.0  # Mocked conversion
    
    def test_transaction_properties(self, buy_transaction, sell_transaction, dividend_transaction):
        """Test transaction type properties"""
        assert buy_transaction.is_buy is True
        assert buy_transaction.is_sell is False
        assert buy_transaction.is_dividend is False
        
        assert sell_transaction.is_sell is True
        assert sell_transaction.is_buy is False
        assert sell_transaction.is_dividend is False
        
        assert dividend_transaction.is_dividend is True
        assert dividend_transaction.is_buy is False
        assert dividend_transaction.is_sell is False


class TestDividend:
    """Test cases for the Dividend class"""
    
    def test_dividend_initialization(self, dividend, mock_stock, transaction_date):
        """Test that dividends are initialized correctly"""
        assert dividend.type == "dividend"
        assert dividend.stock == mock_stock
        assert dividend.amount == 10  # shares
        assert dividend.price.amount == 0.5  # dividend per share
        assert dividend.price.currency == "USD"
        assert dividend.date == transaction_date
    
    def test_dividend_properties(self, dividend):
        """Test dividend-specific properties"""
        assert dividend.dividend_per_share == 0.5
        assert dividend.total_dividend == 5.0  # 10 shares * $0.5
    
    def test_get_dividend_yield(self, dividend, mock_stock):
        """Test dividend yield calculation"""
        # Mock the stock's get_price method
        mock_stock.get_price.return_value = 100.0
        
        # Test yield calculation (assuming quarterly dividend)
        # Annual dividend = 0.5 * 4 = 2.0
        # Yield = 2.0 / 100.0 = 0.02 (2%)
        yield_value = dividend.get_dividend_yield("USD")
        assert yield_value == 0.02