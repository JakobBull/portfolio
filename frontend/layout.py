import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
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
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Individual Stocks (Optional)", className="mb-2"),
                                        dbc.Checklist(
                                            id='stock-selector',
                                            options=[],  # Will be populated dynamically
                                            value=[],    # Default: show none
                                            inline=True,
                                        ),
                                    ], width=12),
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
                                        {'name': 'Price/Sh', 'id': 'current_price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                                        {'name': 'Market Value', 'id': 'market_value', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                                        {'name': 'Cost Basis', 'id': 'cost_basis', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                                        {'name': 'Unrealized P/L', 'id': 'unrealized_pl', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                                        {'name': 'Return %', 'id': 'return_pct', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                        {'name': 'Weight %', 'id': 'weight_pct', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                        {'name': 'Target', 'id': 'target_price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}, 'editable': True},
                                        {'name': 'Div. Yield', 'id': 'dividend_yield', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                        {'name': 'Sector', 'id': 'sector'},
                                        {'name': 'Country', 'id': 'country'},
                                        {'name': 'Currency', 'id': 'currency'},
                                    ],
                                    data=[],
                                    style_cell={'textAlign': 'center'},
                                    style_data_conditional=[
                                        {'if': {'filter_query': '{unrealized_pl} > 0', 'column_id': 'unrealized_pl'}, 'color': 'green'},
                                        {'if': {'filter_query': '{unrealized_pl} < 0', 'column_id': 'unrealized_pl'}, 'color': 'red'},
                                        {'if': {'filter_query': '{return_pct} > 0', 'column_id': 'return_pct'}, 'color': 'green'},
                                        {'if': {'filter_query': '{return_pct} < 0', 'column_id': 'return_pct'}, 'color': 'red'}
                                    ],
                                    sort_action='native',
                                    filter_action='native',
                                    page_size=15,
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
                                            dbc.Button("Search", id="search-button", n_clicks=0, color="primary"),
                                        ]),
                                        html.Div(id="search-results", className="mt-2"),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Form([
                                            dbc.Row([
                                                dbc.Col(dbc.Label("Ticker"), width=4),
                                                dbc.Col(dbc.Input(id="ticker-input", placeholder="e.g. AAPL"), width=8),
                                            ], className="mb-2"),
                                            dbc.Row([
                                                dbc.Col(dbc.Label("Type"), width=4),
                                                dbc.Col(dcc.Dropdown(
                                                    id="transaction-type-input",
                                                    options=[
                                                        {'label': 'Buy', 'value': 'buy'},
                                                        {'label': 'Sell', 'value': 'sell'}
                                                    ],
                                                    value='buy'
                                                ), width=8),
                                            ], className="mb-2"),
                                            dbc.Row([
                                                dbc.Col(dbc.Label("Shares"), width=4),
                                                dbc.Col(dbc.Input(id="shares-input", type="number", placeholder="e.g. 10"), width=8),
                                            ], className="mb-2"),
                                            dbc.Row([
                                                dbc.Col(dbc.Label("Price"), width=4),
                                                dbc.Col(dbc.Input(id="price-input", type="number", placeholder="e.g. 150.00"), width=8),
                                            ], className="mb-2"),
                                            dbc.Row([
                                                dbc.Col(dbc.Label("Date"), width=4),
                                                dbc.Col(dcc.DatePickerSingle(
                                                    id='transaction-date-input',
                                                    date=date.today(),
                                                    display_format='YYYY-MM-DD'
                                                ), width=8),
                                            ], className="mb-2"),
                                            dbc.Row([
                                                dbc.Col(dbc.Label("Cost"), width=4),
                                                dbc.Col(dbc.Input(id="cost-input", type="number", value=0, placeholder="e.g. 5.00"), width=8),
                                            ], className="mb-2"),
                                            dbc.Button("Add Transaction", id="add-transaction-button", color="success", className="mt-3"),
                                        ]),
                                        html.Div(id="transaction-status", className="mt-2"),
                                    ], width=6),
                                ]),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Data Management Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Data Management")),
                            dbc.CardBody([
                                # Store components for each table
                                dcc.Store(id='watchlist-data-store'),
                                dcc.Store(id='transactions-data-store'),
                                dcc.Store(id='dividends-data-store'),
                                
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
                                
                                # Tabs for different data tables
                                dbc.Tabs([
                                    dbc.Tab([
                                        dag.AgGrid(
                                            id='watchlist-aggrid',
                                            columnDefs=[
                                                {"field": "id", "hide": True},
                                                {"field": "ticker", "headerName": "Ticker", "width": 100, "editable": False},
                                                {"field": "name", "headerName": "Name", "width": 200, "editable": False},
                                                {"field": "sector", "headerName": "Sector", "width": 150, "editable": False},
                                                {"field": "country", "headerName": "Country", "width": 120, "editable": False},
                                                {"field": "current_price", "headerName": "Current Price", "width": 120, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.2f')(params.value || 0) : d3.format('$.2f')(params.value || 0)"}, "editable": False},
                                                {"field": "strike_price", "headerName": "Target Price", "width": 120, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.2f')(params.value || 0) : d3.format('$.2f')(params.value || 0)"}, "editable": True},
                                                {"field": "notes", "headerName": "Notes", "width": 200, "editable": True},
                                                {"field": "date_added", "headerName": "Date Added", "width": 120, "editable": False},
                                            ],
                                            rowData=[],
                                            defaultColDef={"resizable": True, "sortable": True, "filter": True},
                                            dashGridOptions={
                                                "pagination": True,
                                                "paginationPageSize": 10,
                                                "rowSelection": "single",
                                                "suppressRowClickSelection": True,
                                                "animateRows": True,
                                            },
                                            style={"height": "400px"}
                                        )
                                    ], label="Watchlist"),
                                    dbc.Tab([
                                        dag.AgGrid(
                                            id='transactions-aggrid',
                                            columnDefs=[
                                                {"field": "id", "hide": True},
                                                {"field": "type", "headerName": "Type", "width": 100, "editable": True, "cellEditor": "agSelectCellEditor", "cellEditorParams": {"values": ["buy", "sell", "dividend"]}},
                                                {"field": "ticker", "headerName": "Ticker", "width": 100, "editable": True},
                                                {"field": "name", "headerName": "Name", "width": 200, "editable": False},
                                                {"field": "amount", "headerName": "Shares", "width": 100, "type": "numericColumn", "valueFormatter": {"function": "d3.format('.2f')(params.value || 0)"}, "editable": True},
                                                {"field": "price", "headerName": "Price", "width": 100, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.2f')(params.value || 0) : d3.format('$.2f')(params.value || 0)"}, "editable": True},
                                                {"field": "currency", "headerName": "Currency", "width": 100, "editable": True, "cellEditor": "agSelectCellEditor", "cellEditorParams": {"values": ["EUR", "USD", "GBP"]}},
                                                {"field": "date", "headerName": "Date", "width": 120, "editable": True, "cellEditor": "agDateCellEditor"},
                                                {"field": "cost", "headerName": "Cost", "width": 100, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.2f')(params.value || 0) : d3.format('$.2f')(params.value || 0)"}, "editable": True},
                                                {"field": "total_value", "headerName": "Total Value", "width": 120, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.2f')(params.value || 0) : d3.format('$.2f')(params.value || 0)"}, "editable": False},
                                            ],
                                            rowData=[],
                                            defaultColDef={"resizable": True, "sortable": True, "filter": True},
                                            dashGridOptions={
                                                "pagination": True,
                                                "paginationPageSize": 10,
                                                "rowSelection": "single",
                                                "suppressRowClickSelection": True,
                                                "animateRows": True,
                                            },
                                            style={"height": "400px"}
                                        )
                                    ], label="Transactions"),
                                    dbc.Tab([
                                        dag.AgGrid(
                                            id='dividends-aggrid',
                                            columnDefs=[
                                                {"field": "id", "hide": True},
                                                {"field": "ticker", "headerName": "Ticker", "width": 100, "editable": True},
                                                {"field": "name", "headerName": "Name", "width": 200, "editable": False},
                                                {"field": "date", "headerName": "Date", "width": 120, "editable": True, "cellEditor": "agDateCellEditor"},
                                                {"field": "amount_per_share", "headerName": "Amount per Share", "width": 150, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.4f')(params.value || 0) : d3.format('$.4f')(params.value || 0)"}, "editable": True},
                                                {"field": "shares_held", "headerName": "Shares Held", "width": 120, "type": "numericColumn", "valueFormatter": {"function": "d3.format('.2f')(params.value || 0)"}, "editable": False},
                                                {"field": "total_dividend", "headerName": "Total Dividend", "width": 130, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.2f')(params.value || 0) : d3.format('$.2f')(params.value || 0)"}, "editable": False},
                                                {"field": "tax_withheld", "headerName": "Tax Withheld", "width": 120, "type": "numericColumn", "valueFormatter": {"function": "params.data && params.data.currency ? (params.data.currency === 'EUR' ? '€' : params.data.currency === 'GBP' ? '£' : '$') + d3.format('.2f')(params.value || 0) : d3.format('$.2f')(params.value || 0)"}, "editable": True},
                                                {"field": "currency", "headerName": "Currency", "width": 100, "editable": True, "cellEditor": "agSelectCellEditor", "cellEditorParams": {"values": ["EUR", "USD", "GBP"]}},
                                            ],
                                            rowData=[],
                                            defaultColDef={"resizable": True, "sortable": True, "filter": True},
                                            dashGridOptions={
                                                "pagination": True,
                                                "paginationPageSize": 10,
                                                "rowSelection": "single",
                                                "suppressRowClickSelection": True,
                                                "animateRows": True,
                                            },
                                            style={"height": "400px"}
                                        )
                                    ], label="Dividends"),
                                ]),
                                
                                html.Div(id="data-management-result", className="mt-2"),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Tax Settings
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Tax Settings (Germany)")),
                            dbc.CardBody([
                                dbc.Checklist(
                                    options=[
                                        {'label': 'Married', 'value': 'is_married'},
                                        {'label': 'Church Tax', 'value': 'church_tax'},
                                    ],
                                    value=[],
                                    id="tax-settings-checklist",
                                    inline=True,
                                ),
                                html.Div(id="tax-settings-status", className="mt-2"),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                # Individual Stock Price Chart
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Individual Stock Price Chart")),
                            dbc.CardBody([
                                dcc.Store(id='earning-edit-store'),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Select Stock"),
                                        dcc.Dropdown(id="single-stock-selector", options=[], placeholder="Select a stock..."),
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Select Date Range"),
                                        dcc.DatePickerRange(
                                            id='stock-chart-date-range',
                                            min_date_allowed=date.today() - timedelta(days=365*10),
                                            max_date_allowed=date.today(),
                                            start_date=default_start_date,
                                            end_date=default_end_date,
                                            display_format='YYYY-MM-DD'
                                        ),
                                    ], width=6),
                                ]),
                                dcc.Graph(id='single-stock-chart'),
                                html.Div(id='earnings-table-container', style={'display': 'none'}, children=[
                                    html.Hr(),
                                    html.Div([
                                        html.H5("Earnings Data (EPS)", className="mt-4 d-inline-block"),
                                        dbc.Button("Add Earning", id="add-earning-button", size="sm", className="ms-2 mb-2", n_clicks=0)
                                    ], className="d-flex align-items-center"),
                                    dag.AgGrid(
                                        id="earnings-table",
                                        columnDefs=[
                                            {'headerName': 'ID', 'field': 'id', 'hide': True},
                                            {'headerName': 'Date', 'field': 'date', 'filter': 'agDateColumnFilter', 'editable': True},
                                            {'headerName': 'EPS', 'field': 'eps', "sortable": True, 'editable': True, 'valueFormatter': {"function": "d3.format('.2f')(params.value)"}},
                                            {'headerName': 'Type', 'field': 'type', 'editable': True, 'cellEditor': 'agSelectCellEditor', 'cellEditorParams': {'values': ['quarterly', 'annual']}},
                                            {'headerName': 'Currency', 'field': 'currency', 'editable': True, 'cellEditor': 'agSelectCellEditor', 'cellEditorParams': {'values': ['USD', 'EUR']}},
                                        ],
                                        rowData=[],
                                        columnSize="sizeToFit",
                                        defaultColDef={
                                            "sortable": True,
                                            "filter": True,
                                            "floatingFilter": True
                                        },
                                        dashGridOptions={
                                            "rowSelection": "single",
                                            "getRowId": {"function": "params.data.id"},
                                            "cellClicked": {"function": "params.api.callbacks.cellClicked(params)"}
                                        },
                                    ),
                                    html.Div(id="earnings-table-status", className="mt-2 text-success"),
                                ]),
                            ]),
                        ], className="mb-4"),
                    ], width=12),
                ]),
                
                # Earnings Add/Edit Modal
                dbc.Modal([
                    dbc.ModalHeader(id="earning-modal-header"),
                    dbc.ModalBody([
                        # Store to hold data for add/edit mode
                        dcc.Store(id='earning-store'),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col(dbc.Label("Date"), width=2),
                                dbc.Col(dcc.DatePickerSingle(id='earning-date-picker'), width=10),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Label("EPS"), width=2),
                                dbc.Col(dbc.Input(id='earning-eps-input', type='number', placeholder="Enter EPS"), width=10),
                            ]),
                            dbc.Form([
                                dbc.Label("Type"),
                                dbc.RadioItems(
                                    options=[
                                        {'label': 'Quarterly', 'value': 'quarterly'},
                                        {'label': 'Annual', 'value': 'annual'},
                                    ],
                                    value='quarterly',
                                    id="earning-type-radios",
                                    inline=True,
                                ),
                            ]),
                            dbc.Form(
                                [
                                    dbc.Label("Currency"),
                                    dcc.Dropdown(
                                        id='earning-currency-dropdown',
                                        options=[
                                            {'label': 'USD', 'value': 'USD'},
                                            {'label': 'EUR', 'value': 'EUR'},
                                        ],
                                        value='USD'
                                    ),
                                ]
                            ),
                        ])
                    ]),
                    dbc.ModalFooter([
                        dbc.Button("Save", id="save-earning-button", color="primary", n_clicks=0),
                        dbc.Button("Cancel", id="cancel-earning-button", color="secondary", n_clicks=0),
                    ]),
                ], id="earning-modal", is_open=False),
                
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