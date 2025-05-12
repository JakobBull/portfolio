import pytest
from datetime import date, timedelta
import pandas as pd

from backend.controller import Controller
from backend.portfolio import Portfolio
from backend.tax_calculator import GermanTaxCalculator
from backend.benchmark import BenchmarkComparison

@pytest.fixture
def mock_portfolio(mocker):
    """Create a mock Portfolio for testing"""
    return mocker.Mock(spec=Portfolio)

@pytest.fixture
def controller(mock_portfolio, mocker):
    """Create a controller with a mock portfolio"""
    controller = Controller(mock_portfolio)
    controller.tax_calculator = mocker.Mock(spec=GermanTaxCalculator)
    return controller

class TestController:
    """Test cases for the Controller class"""
    
    def test_controller_initialization(self, mock_portfolio, controller):
        """Test that controllers are initialized correctly"""
        # Test with default initialization
        default_controller = Controller()
        assert isinstance(default_controller.portfolio, Portfolio)
        
        # Test with provided portfolio
        assert controller.portfolio == mock_portfolio
        
        # Test tax calculator initialization
        assert hasattr(controller, 'tax_calculator')
    
    def test_configure_tax_calculator(self, controller):
        """Test tax calculator configuration"""
        # Configure tax calculator
        controller.configure_tax_calculator(
            is_married=True,
            church_tax=True,
            partial_exemption=True
        )
        
        # Check that a new tax calculator was created with the correct settings
        assert isinstance(controller.tax_calculator, GermanTaxCalculator)
    
    def test_add_transaction(self, controller, mock_portfolio, mocker):
        """Test adding a transaction"""
        # Mock the portfolio's transaction method
        mock_portfolio.transaction.return_value = True
        
        # Mock the transactions property
        mock_transactions = []
        type(mock_portfolio).transactions = mocker.PropertyMock(return_value=mock_transactions)
        
        # Mock the get_position method
        mock_portfolio.get_position.return_value = None
        
        # Add a transaction
        result = controller.add_transaction(
            "buy", "AAPL", 10, 150.0, "USD", date(2023, 1, 15)
        )
        
        # Check result
        assert result is True
        
        # Verify the portfolio's transaction method was called correctly
        mock_portfolio.transaction.assert_called_once()
        args = mock_portfolio.transaction.call_args[0]
        assert args[0] == "buy"
        assert args[2] == 10
        assert args[3] == 150.0
        assert args[4] == "USD"
        assert args[5] == date(2023, 1, 15)
    
    def test_add_dividend(self, controller, mock_portfolio, mocker):
        """Test adding a dividend"""
        # Mock the portfolio's add_dividend method
        mock_portfolio.add_dividend.return_value = True
        
        # Mock the get_position method
        mock_portfolio.get_position.return_value = None
        
        # Mock the get_dividend_history method
        mock_portfolio.get_dividend_history.return_value = []
        
        # Add a dividend
        result = controller.add_dividend(
            "AAPL", 10, 0.5, "USD", date(2023, 1, 15)
        )
        
        # Check result
        assert result is True
        
        # Verify the portfolio's add_dividend method was called correctly
        mock_portfolio.add_dividend.assert_called_once()
        args = mock_portfolio.add_dividend.call_args[0]
        assert args[1] == 10
        assert args[2] == 0.5
        assert args[3] == "USD"
        assert args[4] == date(2023, 1, 15)
    
    def test_get_portfolio_summary(self, controller, mock_portfolio, mocker):
        """Test portfolio summary generation"""
        # Mock the portfolio's methods
        mock_portfolio.get_value.return_value = 3000.0
        mock_portfolio.get_gross_purchase_price.return_value = 2700.0
        mock_portfolio.get_unrealized_pl.return_value = 300.0
        mock_portfolio.get_dividend_income.return_value = 80.0
        mock_portfolio.get_dividend_yield.return_value = 0.0267
        
        # Create mock positions
        mock_position1 = mocker.Mock()
        mock_position2 = mocker.Mock()
        mock_portfolio.positions = {"AAPL": mock_position1, "MSFT": mock_position2}
        
        # Get portfolio summary
        summary = controller.get_portfolio_summary()
        
        # Check summary
        assert summary["total_value"] == 3000.0
        assert summary["total_cost"] == 2700.0
        assert summary["unrealized_pl"] == 300.0
        assert pytest.approx(summary["percentage_return"], 0.01) == 11.11
        assert summary["position_count"] == 2
        assert summary["dividend_income"] == 80.0
        assert summary["dividend_yield"] == 0.0267
    
    def test_get_position_breakdown(self, controller, mock_portfolio, mocker):
        """Test position breakdown calculation"""
        # Create a mock DataFrame for the portfolio's get_position_breakdown method
        mock_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'name': ['Apple Inc.', 'Microsoft Corporation'],
            'amount': [10, 5],
            'value': [1500.0, 1000.0],
            'cost_basis': [1400.0, 900.0],
            'unrealized_pl': [100.0, 100.0],
            'unrealized_pl_percent': [7.14, 11.11],
            'dividend_income': [50.0, 30.0]
        })
        
        # Mock the controller's get_position_breakdown method
        mocker.patch.object(controller, '_calculate_position_dividend_yield', return_value=1.5)
        mocker.patch.object(mock_portfolio, 'get_position_breakdown', return_value=mock_df)
        
        # Get the position breakdown
        result = controller.get_position_breakdown()
        
        # Check the result
        assert 'dividend_yield' in result.columns
        assert result['dividend_yield'].iloc[0] == 1.5
        assert result['dividend_yield'].iloc[1] == 1.5
    
    def test_get_historical_value(self, controller, mock_portfolio):
        """Test historical value calculation"""
        # Mock the portfolio's get_value, get_gross_purchase_price, and get_unrealized_pl methods
        mock_portfolio.get_value.return_value = 3000.0
        mock_portfolio.get_gross_purchase_price.return_value = 2700.0
        mock_portfolio.get_unrealized_pl.return_value = 300.0
        
        # Get the historical value
        result = controller.get_historical_value(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            interval='daily'
        )
        
        # Check the result
        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'value' in result.columns
    
    def test_get_performance_metrics(self, controller, mock_portfolio, mocker):
        """Test performance metrics calculation"""
        # Mock the portfolio's methods
        mock_portfolio.get_value.return_value = 3000.0
        mock_portfolio.get_gross_purchase_price.return_value = 2700.0
        mock_portfolio.get_unrealized_pl.return_value = 300.0
        mock_portfolio.get_dividend_income.return_value = 80.0
        mock_portfolio.get_dividend_yield.return_value = 0.0267
        
        # Mock the portfolio's transactions property
        mock_transactions = []
        type(mock_portfolio).transactions = mocker.PropertyMock(return_value=mock_transactions)
        
        # Mock the get_realized_pl_transactions and get_dividend_transactions methods
        mock_portfolio.get_realized_pl_transactions.return_value = []
        mock_portfolio.get_dividend_transactions.return_value = []
        
        # Mock the controller's _calculate_realized_pl method
        mocker.patch.object(controller, '_calculate_realized_pl', return_value=50.0)
        
        # Mock the tax calculator's get_tax_report method
        controller.tax_calculator.get_tax_report.return_value = {
            'total_tax': 20.0,
            'tax_breakdown': {
                'dividend_tax': 10.0,
                'capital_gains_tax': 10.0
            }
        }
        
        # Get the performance metrics
        metrics = controller.get_performance_metrics()
        
        # Check the metrics
        assert metrics['total_value'] == 3000.0
        assert metrics['cost_basis'] == 2700.0
        assert metrics['unrealized_pl'] == 300.0
        assert metrics['unrealized_pl_percent'] == pytest.approx(11.11, 0.01)
        assert metrics['realized_pl'] == 50.0
        assert metrics['dividend_income'] == 80.0
        assert metrics['dividend_yield'] == 0.0267
        assert metrics['total_pl'] == 350.0
        assert metrics['total_pl_percent'] == pytest.approx(12.96, 0.01)
        assert metrics['total_tax'] == 20.0
        assert metrics['after_tax_pl'] == 330.0
        assert metrics['after_tax_pl_percent'] == pytest.approx(12.22, 0.01)
    
    def test_initialize_benchmark_comparison(self, controller, mocker):
        """Test benchmark comparison initialization"""
        # Mock the controller's get_historical_value method
        mock_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
            'value': [3000.0, 3010.0, 3020.0, 3030.0, 3040.0, 3050.0, 3060.0, 3070.0, 3080.0, 3090.0]
        })
        
        mocker.patch.object(controller, 'get_historical_value', return_value=mock_df)
        
        # Initialize the benchmark comparison
        controller.initialize_benchmark_comparison(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10)
        )
        
        # Check that the benchmark comparison was created
        assert hasattr(controller, 'benchmark_comparison')
        assert controller.benchmark_comparison is not None
    
    def test_add_custom_benchmark(self, controller, mocker):
        """Test adding a custom benchmark"""
        # Mock the controller's initialize_benchmark_comparison method
        mocker.patch.object(controller, 'initialize_benchmark_comparison')
        # Mock the benchmark comparison's add_custom_benchmark method
        controller.benchmark_comparison = mocker.Mock(spec=BenchmarkComparison)
        
        # Add a custom benchmark
        controller.add_custom_benchmark('Custom', [100.0, 101.0, 102.0])
        
        # Check that the benchmark was added
        controller.benchmark_comparison.add_custom_benchmark.assert_called_once_with('Custom', [100.0, 101.0, 102.0])
    
    def test_get_benchmark_comparison(self, controller, mocker):
        """Test getting benchmark comparison"""
        # Mock the controller's initialize_benchmark_comparison method
        mocker.patch.object(controller, 'initialize_benchmark_comparison')
        # Mock the benchmark comparison's get_comparison method
        controller.benchmark_comparison = mocker.Mock(spec=BenchmarkComparison)
        mock_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
            'portfolio': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'S&P 500': [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]
        })
        controller.benchmark_comparison.get_comparison.return_value = mock_df
        
        # Get the benchmark comparison
        result = controller.get_benchmark_comparison()
        
        # Check the result
        assert result is mock_df 