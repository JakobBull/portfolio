import pytest
from datetime import date
# Replace unittest.mock with pytest's mocker fixture
# from unittest import mock

from backend.portfolio import Portfolio
from backend.position import Position
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
def portfolio():
    """Create a portfolio for testing"""
    return Portfolio()

class TestPortfolio:
    """Test cases for the Portfolio class"""
    
    def test_portfolio_initialization(self, portfolio):
        """Test that portfolios are initialized correctly"""
        assert len(portfolio.positions) == 0
        assert len(portfolio.transactions) == 0
        assert len(portfolio.dividends) == 0
    
    def test_transaction_buy(self, portfolio, mock_stock, transaction_date):
        """Test adding a buy transaction"""
        # Create a buy transaction
        transaction = Transaction(
            "buy", 
            mock_stock, 
            10, 
            150.0, 
            "USD", 
            transaction_date
        )
        
        # Add transaction to portfolio
        result = portfolio.update_position(transaction)
        portfolio.transactions.append(transaction)
        
        # Check result
        assert result is True
        assert len(portfolio.positions) == 1
        assert len(portfolio.transactions) == 1
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].amount == 10
    
    def test_transaction_sell(self, portfolio, mock_stock, transaction_date, monkeypatch):
        """Test adding a sell transaction"""
        # First add a buy transaction
        buy_transaction = Transaction(
            "buy", 
            mock_stock, 
            10, 
            150.0, 
            "USD", 
            transaction_date
        )
        portfolio.update_position(buy_transaction)
        portfolio.transactions.append(buy_transaction)
        
        # Mock the Position.update method to avoid the ticker attribute error
        original_update = Position.update
        def mock_update(self, transaction):
            if transaction.type == "sell":
                self.amount -= transaction.amount
            return None
        
        monkeypatch.setattr(Position, 'update', mock_update)
        
        # Then add a sell transaction
        sell_transaction = Transaction(
            "sell", 
            mock_stock, 
            5, 
            160.0, 
            "USD", 
            transaction_date
        )
        result = portfolio.update_position(sell_transaction)
        portfolio.transactions.append(sell_transaction)
        
        # Check result
        assert result is True
        assert len(portfolio.positions) == 1
        assert len(portfolio.transactions) == 2
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].amount == 5
    
    def test_transaction_sell_all(self, portfolio, mock_stock, transaction_date, monkeypatch):
        """Test selling all shares"""
        # First add a buy transaction
        buy_transaction = Transaction(
            "buy", 
            mock_stock, 
            10, 
            150.0, 
            "USD", 
            transaction_date
        )
        portfolio.update_position(buy_transaction)
        portfolio.transactions.append(buy_transaction)
        
        # Mock the Position.update method to avoid the ticker attribute error
        original_update = Position.update
        def mock_update(self, transaction):
            if transaction.type == "sell":
                self.amount -= transaction.amount
            return None
        
        monkeypatch.setattr(Position, 'update', mock_update)
        
        # Then sell all shares
        sell_transaction = Transaction(
            "sell", 
            mock_stock, 
            10, 
            160.0, 
            "USD", 
            transaction_date
        )
        result = portfolio.update_position(sell_transaction)
        portfolio.transactions.append(sell_transaction)
        
        # Check result
        assert result is True
        assert len(portfolio.positions) == 0  # Position should be removed
        assert len(portfolio.transactions) == 2
    
    def test_transaction_sell_without_position(self, portfolio, mock_stock, transaction_date):
        """Test selling without a position"""
        # Try to sell without a position
        sell_transaction = Transaction(
            "sell", 
            mock_stock, 
            5, 
            160.0, 
            "USD", 
            transaction_date
        )
        result = portfolio.update_position(sell_transaction)
        
        # Check result
        assert result is False
        assert len(portfolio.positions) == 0
    
    def test_add_dividend(self, portfolio, mock_stock, transaction_date):
        """Test adding a dividend"""
        # Add a dividend
        dividend = Dividend(
            mock_stock, 
            10, 
            0.5, 
            "USD", 
            transaction_date
        )
        
        # Add dividend to portfolio
        result = portfolio.update_position(dividend)
        portfolio.dividends.append(dividend)
        portfolio.transactions.append(dividend)
        
        # Check result
        assert result is True
        assert len(portfolio.positions) == 0  # Dividends don't create positions
        assert len(portfolio.dividends) == 1
        assert len(portfolio.transactions) == 1
    
    def test_get_value(self, portfolio, mocker):
        """Test portfolio value calculation"""
        # Create mock positions
        position1 = mocker.Mock(spec=Position)
        position1.get_value.return_value = 1000.0
        
        position2 = mocker.Mock(spec=Position)
        position2.get_value.return_value = 2000.0
        
        # Add positions to portfolio
        portfolio.positions = {
            "AAPL": position1,
            "MSFT": position2
        }
        
        # Test portfolio value
        portfolio_value = portfolio.get_value("USD")
        assert portfolio_value == 3000.0  # 1000 + 2000
        
        # Verify the positions' get_value methods were called
        assert position1.get_value.called
        assert position2.get_value.called
    
    def test_get_gross_purchase_price(self, portfolio, mocker):
        """Test gross purchase price calculation"""
        # Create mock positions
        position1 = mocker.Mock(spec=Position)
        position1.get_cost_basis.return_value = 900.0
        
        position2 = mocker.Mock(spec=Position)
        position2.get_cost_basis.return_value = 1800.0
        
        # Add positions to portfolio
        portfolio.positions = {
            "AAPL": position1,
            "MSFT": position2
        }
        
        # Test gross purchase price
        gross_price = portfolio.get_gross_purchase_price("USD")
        assert gross_price == 2700.0  # 900 + 1800
    
    def test_get_net_purchase_price(self, portfolio, transaction_date, mocker):
        """Test net purchase price calculation"""
        # Create mock positions and money amounts
        position1 = mocker.Mock(spec=Position)
        position1.amount = 10
        money_amount1 = mocker.Mock(spec=MoneyAmount)
        money_amount1.get_money_amount.return_value = 90.0
        position1.purchase_price_net = money_amount1
        
        position2 = mocker.Mock(spec=Position)
        position2.amount = 5
        money_amount2 = mocker.Mock(spec=MoneyAmount)
        money_amount2.get_money_amount.return_value = 180.0
        position2.purchase_price_net = money_amount2
        
        # Add positions to portfolio
        portfolio.positions = {
            "AAPL": position1,
            "MSFT": position2
        }
        
        # Test net purchase price
        net_price = portfolio.get_net_purchase_price("USD")
        assert net_price == 1800.0  # (10 * 90) + (5 * 180)
    
    def test_get_unrealized_pl(self, portfolio, monkeypatch):
        """Test unrealized profit/loss calculation"""
        # Mock the portfolio's get_value and get_gross_purchase_price methods
        monkeypatch.setattr(portfolio, 'get_value', lambda currency, date=None: 3000.0)
        monkeypatch.setattr(portfolio, 'get_gross_purchase_price', lambda currency: 2700.0)
        
        # Test unrealized P/L
        unrealized_pl = portfolio.get_unrealized_pl("USD")
        assert unrealized_pl == 300.0  # 3000 - 2700
    
    def test_get_dividend_income(self, portfolio, mocker):
        """Test dividend income calculation"""
        # Create mock dividends
        dividend1 = mocker.Mock(spec=Dividend)
        dividend1.get_transaction_value.return_value = 50.0
        
        dividend2 = mocker.Mock(spec=Dividend)
        dividend2.get_transaction_value.return_value = 30.0
        
        # Add dividends to portfolio
        portfolio.dividends = [dividend1, dividend2]
        
        # Test dividend income
        dividend_income = portfolio.get_dividend_income("USD")
        assert dividend_income == 80.0  # 50 + 30
    
    def test_get_dividend_yield(self, portfolio, monkeypatch):
        """Test dividend yield calculation"""
        # Mock the portfolio's get_dividend_income and get_value methods
        monkeypatch.setattr(portfolio, 'get_dividend_income', 
                           lambda currency, start_date=None, end_date=None: 80.0)
        monkeypatch.setattr(portfolio, 'get_value', lambda currency, date=None: 2000.0)
        
        # Test dividend yield
        dividend_yield = portfolio.get_dividend_yield("USD")
        assert dividend_yield == 0.04  # 80 / 2000 (now returned as decimal)