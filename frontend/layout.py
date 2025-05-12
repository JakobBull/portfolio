import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from datetime import date, timedelta
import logging

def create_layout():
    """Create the main app layout"""
    logging.info("Creating application layout")
    
    # Set default dates for date range picker
    default_start_date = date.today() - timedelta(days=365)
    default_end_date = date.today()
    logging.info(f"Setting default date range: {default_start_date} to {default_end_date}")
    
    return dbc.Container([
        # Add a loading component to show when data is being fetched
        dcc.Loading(
            id="loading-indicator",
            type="default",
            children=[html.Div(id="loading-output")]
        ),
        
        # Add an error display area
        html.Div(id="global-error-display", className="alert alert-danger", style={"display": "none"}),
        
        # Add Store components to cache different types of data
        dcc.Store(id='portfolio-data-store'),
        dcc.Store(id='positions-data-store'),
        dcc.Store(id='performance-data-store'),
        dcc.Store(id='summary-data-store'),
        dcc.Store(id='search-results-store'),
        dcc.Store(id='transaction-result-store'),
        dcc.Store(id='tax-result-store'),
        
        # URL Location - this triggers the initial data loading
        dcc.Location(id='url', refresh=False),
        
        # Interval component for periodic refresh (every 15 minutes instead of 5)
        dcc.Interval(
            id='interval-component',
            interval=900*1000,  # in milliseconds (15 minutes)
            n_intervals=0
        ),
        
        dbc.Row([
            dbc.Col([
                html.H1("Portfolio Manager", className="text-center my-4"),
                html.Hr(),
            ], width=12)
        ]),
        
        # Wrap the main content in a loading component
        dcc.Loading(
            id="loading-main-content",
            type="default",
            children=[
                # Portfolio Summary Card
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Portfolio Summary")),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Total Value:"),
                                        html.H3(id="total-value", children="€0.00"),
                                    ], width=4),
                                    dbc.Col([
                                        html.H5("Unrealized P/L:"),
                                        html.H3(id="unrealized-pl", children="€0.00"),
                                    ], width=4),
                                    dbc.Col([
                                        html.H5("Dividend Yield:"),
                                        html.H3(id="dividend-yield", children="0.00%"),
                                    ], width=4),
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Total Cost:"),
                                        html.H3(id="total-cost", children="€0.00"),
                                    ], width=4),
                                    dbc.Col([
                                        html.H5("Return:"),
                                        html.H3(id="return-pct", children="0.00%"),
                                    ], width=4),
                                    dbc.Col([
                                        html.H5("Dividend Income (YTD):"),
                                        html.H3(id="dividend-income", children="€0.00"),
                                    ], width=4),
                                ]),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Portfolio Performance Chart
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Portfolio Performance vs Benchmarks")),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dcc.DatePickerRange(
                                            id='date-range',
                                            min_date_allowed=date.today() - timedelta(days=365*5),
                                            max_date_allowed=date.today(),
                                            start_date=default_start_date,
                                            end_date=default_end_date,
                                            display_format='YYYY-MM-DD'
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Checklist(
                                            id='benchmark-selector',
                                            options=[
                                                {'label': 'NASDAQ', 'value': 'NASDAQ'},
                                                {'label': 'S&P 500', 'value': 'S&P 500'},
                                                {'label': 'DAX 30', 'value': 'DAX 30'},
                                            ],
                                            value=['NASDAQ', 'S&P 500'],
                                            inline=True,
                                        ),
                                    ], width=6),
                                ], className="mb-4"),
                                dcc.Graph(id='performance-chart'),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Positions Table
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Portfolio Positions")),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id='positions-table',
                                    columns=[
                                        {'name': 'Ticker', 'id': 'ticker'},
                                        {'name': 'Name', 'id': 'name'},
                                        {'name': 'Shares', 'id': 'shares', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'Price/Sh', 'id': 'current_price', 'type': 'numeric', 'format': {'specifier': '$.2f', 'locale': {'symbol': ['€', '']}}},
                                        {'name': 'Market Value', 'id': 'market_value', 'type': 'numeric', 'format': {'specifier': '$.2f', 'locale': {'symbol': ['€', '']}}},
                                        {'name': 'Cost Basis', 'id': 'cost_basis', 'type': 'numeric', 'format': {'specifier': '$.2f', 'locale': {'symbol': ['€', '']}}},
                                        {'name': 'Unrealized P/L', 'id': 'unrealized_pl', 'type': 'numeric', 'format': {'specifier': '$.2f', 'locale': {'symbol': ['€', '']}}},
                                        {'name': 'Return %', 'id': 'return_pct', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                        {'name': 'Weight %', 'id': 'weight_pct', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                        {'name': 'Target', 'id': 'target_price', 'type': 'numeric', 'format': {'specifier': '$.2f', 'locale': {'symbol': ['€', '']}}},
                                        {'name': 'Dividend Yield', 'id': 'dividend_yield', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                    ],
                                    data=[],
                                    style_cell={'textAlign': 'center'},
                                    style_data_conditional=[
                                        {
                                            'if': {
                                                'filter_query': '{unrealized_pl} > 0',
                                                'column_id': 'unrealized_pl'
                                            },
                                            'color': 'green'
                                        },
                                        {
                                            'if': {
                                                'filter_query': '{unrealized_pl} < 0',
                                                'column_id': 'unrealized_pl'
                                            },
                                            'color': 'red'
                                        },
                                        {
                                            'if': {
                                                'filter_query': '{return_pct} > 0',
                                                'column_id': 'return_pct'
                                            },
                                            'color': 'green'
                                        },
                                        {
                                            'if': {
                                                'filter_query': '{return_pct} < 0',
                                                'column_id': 'return_pct'
                                            },
                                            'color': 'red'
                                        }
                                    ],
                                    sort_action='native',
                                    filter_action='native',
                                    page_size=10,
                                ),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Portfolio Breakdown Visualizations
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Portfolio Breakdown")),
                            dbc.CardBody([
                                # Store for breakdown data
                                dcc.Store(id='breakdown-data-store'),
                                
                                # Tabs for different visualizations
                                dbc.Tabs([
                                    dbc.Tab([
                                        dcc.Graph(id='sector-breakdown-chart')
                                    ], label="Sector Breakdown"),
                                    dbc.Tab([
                                        dcc.Graph(id='country-breakdown-chart')
                                    ], label="Country Breakdown"),
                                    dbc.Tab([
                                        dcc.Graph(id='currency-breakdown-chart')
                                    ], label="Currency Exposure"),
                                ]),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Add New Position
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Add New Position")),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Stock Search"),
                                        dbc.InputGroup([
                                            dbc.Input(id="stock-search", placeholder="Enter ticker or name..."),
                                            dbc.Button("Search", id="search-button", color="primary"),
                                        ]),
                                        html.Div(id="search-results", className="mt-2"),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Form([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Ticker"),
                                                    dbc.Input(id="ticker-input", placeholder="e.g., AAPL"),
                                                ], width=6),
                                                dbc.Col([
                                                    dbc.Label("Transaction Type"),
                                                    dbc.Select(
                                                        id="transaction-type",
                                                        options=[
                                                            {"label": "Buy", "value": "buy"},
                                                            {"label": "Sell", "value": "sell"},
                                                            {"label": "Dividend", "value": "dividend"},
                                                        ],
                                                        value="buy",
                                                    ),
                                                ], width=6),
                                            ], className="mb-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Shares"),
                                                    dbc.Input(id="shares-input", type="number", min=0, step=0.01),
                                                ], width=6),
                                                dbc.Col([
                                                    dbc.Label("Price per Share"),
                                                    dbc.Input(id="price-input", type="number", min=0, step=0.01),
                                                ], width=6),
                                            ], className="mb-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Currency"),
                                                    dbc.Select(
                                                        id="currency-input",
                                                        options=[
                                                            {"label": "EUR", "value": "EUR"},
                                                            {"label": "USD", "value": "USD"},
                                                            {"label": "GBP", "value": "GBP"},
                                                        ],
                                                        value="EUR",
                                                    ),
                                                ], width=6),
                                                dbc.Col([
                                                    dbc.Label("Transaction Cost"),
                                                    dbc.Input(id="transaction-cost-input", type="number", min=0, step=0.01, value=7.5, placeholder="Transaction fee"),
                                                ], width=6),
                                            ], className="mb-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Sell Target Price"),
                                                    dbc.Input(id="sell-target-input", type="number", min=0, step=0.01, placeholder="Optional"),
                                                ], width=6),
                                                dbc.Col([
                                                    dbc.Label("Date"),
                                                    html.Div([
                                                        dcc.DatePickerSingle(
                                                            id="date-input",
                                                            date=date.today(),
                                                            display_format='YYYY-MM-DD',
                                                            clearable=False,
                                                            style={
                                                                'width': '100%',
                                                            },
                                                        ),
                                                    ], style={
                                                        'width': '100%',
                                                        'margin-bottom': '0',
                                                    }),
                                                ], width=6),
                                            ], className="mb-3"),
                                            dbc.Button("Add Transaction", id="add-transaction-button", color="success", className="mt-3"),
                                            html.Div(id="transaction-result", className="mt-2"),
                                        ]),
                                    ], width=6),
                                ]),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Watchlist Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Watchlist")),
                            dbc.CardBody([
                                # Store for watchlist data
                                dcc.Store(id='watchlist-data-store'),
                                
                                # Add to watchlist form
                                dbc.Row([
                                    dbc.Col([
                                        dbc.InputGroup([
                                            dbc.Input(id="watchlist-ticker-input", placeholder="Enter ticker..."),
                                            dbc.Input(id="watchlist-name-input", placeholder="Enter name (optional)..."),
                                            dbc.Input(id="watchlist-target-price", type="number", min=0, step=0.01, placeholder="Target price..."),
                                            dbc.Button("Add to Watchlist", id="add-to-watchlist-button", color="primary"),
                                        ]),
                                    ], width=12),
                                ], className="mb-3"),
                                
                                # Watchlist table
                                dash_table.DataTable(
                                    id='watchlist-table',
                                    columns=[
                                        {'name': 'Ticker', 'id': 'ticker'},
                                        {'name': 'Name', 'id': 'name'},
                                        {'name': 'Sector', 'id': 'sector'},
                                        {'name': 'Country', 'id': 'country'},
                                        {'name': 'Current Price', 'id': 'current_price', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                                        {'name': 'Target Price', 'id': 'strike_price', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                                        {'name': 'Date Added', 'id': 'date_added'},
                                        {'name': 'Actions', 'id': 'actions', 'presentation': 'markdown'},
                                    ],
                                    data=[],
                                    style_cell={
                                        'textAlign': 'left',
                                        'padding': '10px',
                                    },
                                    style_header={
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                        'fontWeight': 'bold'
                                    },
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        }
                                    ],
                                    page_size=10,
                                ),
                                
                                html.Div(id="watchlist-result", className="mt-2"),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),

                # Tax Settings
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Tax Settings")),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Checklist(
                                            id='tax-settings',
                                            options=[
                                                {'label': 'Married (Joint Filing)', 'value': 'married'},
                                                {'label': 'Church Tax', 'value': 'church_tax'},
                                                {'label': 'Partial Exemption (Funds)', 'value': 'partial_exemption'},
                                            ],
                                            value=[],
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Button("Update Tax Settings", id="update-tax-button", color="primary"),
                                        html.Div(id="tax-update-result", className="mt-2"),
                                    ], width=6),
                                ]),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Footer
                dbc.Row([
                    dbc.Col([
                        html.Hr(),
                        html.P("Portfolio Manager - Developed with Dash and Python", className="text-center"),
                    ], width=12),
                ]),
            ]
        ),
    ], fluid=True) 