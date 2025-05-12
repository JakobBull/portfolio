import pytest
from datetime import date
from unittest.mock import ANY

from backend.position import Position
from backend.transaction import Transaction
from backend.stock import Stock
from backend.money_amount import MoneyAmount

@pytest.fixture
def mock_stock(mocker):
    """Create a mock Stock object for testing"""
    stock = mocker.Mock(spec=Stock)
    stock.ticker = "AAPL"
    stock.get_price.return_value = 150.0
    stock.get_value.return_value = 1500.0  # 10 shares * $150
    return stock

@pytest.fixture
def transaction_date():
    """Create a sample transaction date"""
    return date(2023, 1, 15)

@pytest.fixture
def buy_transaction(mock_stock, transaction_date, mocker):
    """Create a sample buy transaction"""
    transaction = mocker.Mock(spec=Transaction)
    transaction.type = "buy"
    transaction.stock = mock_stock
    transaction.amount = 10
    transaction.price = MoneyAmount(150.0, "USD", transaction_date)
    transaction.cost = MoneyAmount(10.0, "USD", transaction_date)
    transaction.date = transaction_date
    transaction.currency = "USD"
    transaction.ticker = "AAPL"
    return transaction

@pytest.fixture
def sell_transaction(mock_stock, transaction_date, mocker):
    """Create a sample sell transaction"""
    transaction = mocker.Mock(spec=Transaction)
    transaction.type = "sell"
    transaction.stock = mock_stock
    transaction.amount = 5
    transaction.price = MoneyAmount(160.0, "USD", transaction_date)
    transaction.cost = MoneyAmount(5.0, "USD", transaction_date)
    transaction.date = transaction_date
    transaction.currency = "USD"
    transaction.ticker = "AAPL"
    return transaction

@pytest.fixture
def position(buy_transaction):
    """Create a sample position"""
    return Position(buy_transaction)

class TestPosition:
    """Test cases for the Position class"""
    
    def test_position_initialization(self, position, mock_stock, buy_transaction):
        """Test that positions are initialized correctly"""
        assert position.amount == 10
        assert position.stock == mock_stock
        assert position.ticker == "AAPL"
        assert position.purchase_price_net.amount == 150.0
        assert position.purchase_cost.amount == 10.0
        assert position.purchase_price_gross.amount > 150.0
    
    def test_position_update_buy(self, position, mock_stock, transaction_date, mocker):
        """Test updating position with a buy transaction"""
        # Create another buy transaction
        buy_transaction2 = mocker.Mock(spec=Transaction)
        buy_transaction2.type = "buy"
        buy_transaction2.stock = mock_stock
        buy_transaction2.amount = 5
        buy_transaction2.price = MoneyAmount(170.0, "USD", transaction_date)
        buy_transaction2.cost = MoneyAmount(5.0, "USD", transaction_date)
        buy_transaction2.date = transaction_date
        buy_transaction2.currency = "USD"
        buy_transaction2.ticker = "AAPL"
        
        # Update position with second buy transaction
        position.update(buy_transaction2)
        
        # Check updated position
        assert position.amount == 15  # 10 + 5
        assert position.purchase_price_net.amount > 150.0  # Average price should increase
        assert position.purchase_price_gross.amount > 150.0
    
    def test_position_update_sell(self, position, sell_transaction):
        """Test updating position with a sell transaction"""
        # Update position with sell transaction
        position.update(sell_transaction)
        
        # Check updated position
        assert position.amount == 5  # 10 - 5
        
        # Price should remain the same after selling
        assert position.purchase_price_net.amount == 150.0
    
    def test_position_update_sell_too_much(self, position, mock_stock, mocker):
        """Test that selling more than owned raises ValueError"""
        # Create a sell transaction for more than owned
        sell_too_much = mocker.Mock(spec=Transaction)
        sell_too_much.type = "sell"
        sell_too_much.stock = mock_stock
        sell_too_much.amount = 15  # More than the 10 owned
        sell_too_much.ticker = "AAPL"
        
        # Attempt to update position (should raise ValueError)
        with pytest.raises(ValueError, match="Cannot sell 15 shares"):
            position.update(sell_too_much)
    
    def test_position_update_invalid_type(self, position, mocker):
        """Test that invalid transaction types raise ValueError"""
        # Create an invalid transaction
        invalid_transaction = mocker.Mock(spec=Transaction)
        invalid_transaction.type = "invalid"
        invalid_transaction.ticker = "AAPL"
        
        # Mock the stock attribute
        mock_stock = mocker.Mock()
        mock_stock.ticker = "AAPL"
        invalid_transaction.stock = mock_stock
        
        # Attempt to update position (should raise ValueError)
        with pytest.raises(ValueError, match="Unsupported transaction type"):
            position.update(invalid_transaction)
    
    def test_get_value(self, position, mock_stock):
        """Test position value calculation"""
        # Mock the stock's get_value method
        mock_stock.get_value.return_value = 1600.0  # 10 shares * $160
        
        # Test position value
        position_value = position.get_value("USD")
        assert position_value == 1600.0
        
        # Verify the stock's get_value method was called correctly
        mock_stock.get_value.assert_called_with("USD", ANY, 10)
    
    def test_get_cost_basis(self, position, mocker):
        """Test cost basis calculation"""
        # Test with same currency
        cost_basis = position.get_cost_basis("USD")
        expected_cost = position.purchase_price_gross.amount * position.amount
        assert cost_basis == expected_cost
        
        # Test with currency conversion (mocked)
        with mocker.patch('backend.money_amount.MoneyAmount.get_money_amount', return_value=1300.0):
            eur_cost = position.get_cost_basis("EUR")
            assert eur_cost == 1300.0  # Mocked conversion
    
    def test_get_unrealized_pl(self, position, mocker):
        """Test unrealized profit/loss calculation"""
        # Mock the position's get_value and get_cost_basis methods
        with mocker.patch.object(position, 'get_value', return_value=1600.0):
            with mocker.patch.object(position, 'get_cost_basis', return_value=1500.0):
                # Test unrealized P/L
                unrealized_pl = position.get_unrealized_pl("USD")
                assert unrealized_pl == 100.0  # 1600 - 1500 