import os
import sys
import json
import logging
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from backend.database import DatabaseManager
from backend.market_interface import MarketInterface
from backend.controller import Controller
from frontend.layout import create_layout
from frontend.callbacks import register_callbacks

# Reset all loggers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging with a stream handler to ensure console output
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

# Reduce logging verbosity for specific modules
logging.getLogger('backend.database').setLevel(logging.WARNING)
logging.getLogger('backend.market_interface').setLevel(logging.WARNING)
# Ensure Dash and Flask logs are visible
logging.getLogger('dash').setLevel(logging.INFO)
logging.getLogger('flask').setLevel(logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.INFO)  # Flask development server

# Import backend components
from backend.controller import Controller
from backend.constants import SUPPORTED_CURRENCIES
from backend.database import db_manager

# Import frontend components
from frontend.layout import create_layout
from frontend.callbacks import register_callbacks

def create_app(db_manager=None, market_interface=None, controller=None):
    """Create and configure the Dash app."""
    if db_manager is None:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend", "portfolio.db"))
        db_manager = DatabaseManager(f"sqlite:///{db_path}") 
    
    if market_interface is None:
        market_interface = MarketInterface()

    if controller is None:
        controller = Controller(db_manager, market_interface)

    # --- 2. Create Dash App ---
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True  # Allow for dynamic callbacks
    )
    app.title = "Portfolio Manager"
    server = app.server  # Expose server for gunicorn

    # Directly configure Flask app's logger
    server.logger.handlers = []
    server.logger.propagate = False
    server_handler = logging.StreamHandler(sys.stdout)
    server_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    server_handler.setFormatter(formatter)
    server.logger.addHandler(server_handler)
    server.logger.setLevel(logging.INFO)

    # Set the app layout with a function to ensure fresh initialization on page load
    def serve_layout():
        """Create a fresh layout on each page load to ensure callbacks are triggered"""
        layout = create_layout()
        logging.info("Creating fresh layout to trigger callbacks")
        return layout

    app.layout = serve_layout

    # --- 4. Register Callbacks ---
    register_callbacks(app, controller)

    return app, controller, db_manager


# Add some sample data for demonstration if the database is empty
def add_sample_data(controller, db_manager):
    """Add sample data to the portfolio for demonstration if no data exists"""
    try:
        # Check if there are any transactions in the database
        transactions = db_manager.get_all_transactions()
        if transactions:
            logging.info("Database already has transactions, skipping sample data addition")
            return
        
        # Check if there are any positions calculated from transactions
        positions = controller.portfolio.get_positions()
        if positions:
            logging.info("Portfolio already has calculated positions, skipping sample data addition")
            return
            
        logging.info("No existing data found, but sample data addition not implemented")
    except Exception as e:
        logging.error(f"Error adding sample data: {str(e)}")
        # Continue without sample data


# --- 5. Run Server ---
if __name__ == "__main__":
    # --- App Initialization ---
    app, controller, db_manager = create_app()
    server = app.server
    
    # Add sample data when starting the app if the portfolio is empty
    add_sample_data(controller, db_manager)

    # Add a custom log message to ensure the URL is visible
    logging.info("Starting Portfolio Manager app at http://0.0.0.0:8050/")
    # Add explicit print statements that will always show in the console
    print("\n" + "="*50)
    print("Dash is running on http://0.0.0.0:8050/")
    print("="*50 + "\n")
    
    # Force Flask to use its default logging
    import flask.cli
    flask.cli.show_server_banner = lambda *args, **kwargs: print("Flask server is starting...")
    
    # Run the server with explicit settings
    app.run_server(
        host='0.0.0.0', 
        port=8050, 
        debug=True, 
        use_reloader=True,
        dev_tools_ui=True,
        dev_tools_props_check=True
    )