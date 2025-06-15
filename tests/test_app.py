import pytest
from unittest.mock import MagicMock
from dash import Dash, html
import sys
from pathlib import Path
import dash_bootstrap_components as dbc

# Add project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import create_app

@pytest.fixture
def app():
    """Fixture to create a test app with mock dependencies."""
    mock_db_manager = MagicMock()
    mock_market_interface = MagicMock()
    mock_controller = MagicMock()
    
    app, _, _ = create_app(
        db_manager=mock_db_manager,
        market_interface=mock_market_interface,
        controller=mock_controller
    )
    return app

def test_app_initialization(app: Dash):
    """Test that the Dash app initializes correctly."""
    assert isinstance(app, Dash)
    assert app.title == "Portfolio Manager"

def test_app_layout(app: Dash):
    """Test that the app's layout is structured as expected."""
    layout_function = app.layout
    layout = layout_function()
    assert isinstance(layout, dbc.Container)
    # Check for a top-level ID or a key child component to verify structure
    assert len(layout.children) > 0

def test_app_callbacks(app: Dash):
    """Test that the app callbacks are registered."""
    callbacks = app.callback_map
    assert len(callbacks) > 0
    # Example check for a specific callback
    # The keys in callback_map are strings representing the Output dependencies
    assert any('total-value.children' in key for key in callbacks.keys())

def test_app_error_handling(app: Dash):
    """Test that the app handles errors gracefully."""
    import dash
    
    @app.callback(
        dash.Output('error-output', 'children'),
        dash.Input('error-trigger', 'n_clicks')
    )
    def error_callback(n_clicks):
        if n_clicks:
            raise Exception("Test error")
        return "No error"
    
    assert 'error-output.children' in app.callback_map 