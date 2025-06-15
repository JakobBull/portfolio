import pytest
from backend.database import DatabaseManager
from backend.controller import Controller
from backend.portfolio import Portfolio
from unittest.mock import MagicMock

@pytest.fixture(scope="function")
def test_db():
    """Fixture for an in-memory database"""
    db_manager = DatabaseManager(db_url="sqlite:///:memory:")
    yield db_manager
    db_manager.close_session()

@pytest.fixture
def mock_db_manager():
    """Fixture for a mocked database manager"""
    return MagicMock()

@pytest.fixture
def controller(test_db):
    """Fixture for the controller"""
    market_interface = MagicMock()
    # Since we use an in-memory db, we need to bypass the portfolio loading from a persistent db
    # by initializing the portfolio directly.
    portfolio = Portfolio(positions=[], transactions=[])
    controller = Controller(db_manager=test_db, market_interface=market_interface)
    controller.portfolio = portfolio
    return controller 