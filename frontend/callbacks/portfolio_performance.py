import logging
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update, ClientsideFunction
from datetime import date, timedelta
from backend.stock import Stock
# Remove yfinance import as we should only use database data
# import yfinance as yf

def register_callbacks(app, controller):
    """Register callbacks for portfolio performance section"""
    
    # Register a client-side callback to show a loading message while data is being fetched
    app.clientside_callback(
        """
        function(data) {
            if (!data) {
                return {
                    'data': [{
                        x: [new Date().toISOString()],
                        y: [0],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Loading...'
                    }],
                    'layout': {
                        title: 'Loading Performance Data...',
                        xaxis: {title: 'Date'},
                        yaxis: {title: 'Value'},
                        template: 'plotly_white'
                    }
                };
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('performance-chart', 'figure'),
        [Input('performance-data-store', 'data')],
        prevent_initial_call=False
    )
    
    # Initial data load callback
    @app.callback(
        Output('performance-data-store', 'data'),
        [Input('url', 'pathname'),
         Input('interval-component', 'n_intervals')],
        [State('date-range', 'start_date'),
         State('date-range', 'end_date'),
         State('benchmark-selector', 'value')]
    )
    def initialize_performance_data(pathname, n_intervals, start_date, end_date, selected_benchmarks):
        """Initialize the performance data store when the app loads or on periodic refresh"""
        if n_intervals is not None and n_intervals > 0:
            logging.info(f"Refreshing performance data (interval {n_intervals})")
        else:
            logging.info("Initial load of performance data")
        return load_performance_data(start_date, end_date, selected_benchmarks)
    
    # User-triggered update callback
    @app.callback(
        Output('performance-data-store', 'data', allow_duplicate=True),
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('benchmark-selector', 'value')],
        prevent_initial_call=True
    )
    def update_performance_data(start_date, end_date, selected_benchmarks):
        """Update the performance data store when user changes date range or benchmarks"""
        logging.info("Updating performance data based on user selection")
        
        # Validate inputs
        if not start_date or not end_date:
            logging.warning("Missing date range values, using defaults")
            if not start_date:
                start_date = (date.today() - timedelta(days=365)).isoformat()
            if not end_date:
                end_date = date.today().isoformat()
            
        return load_performance_data(start_date, end_date, selected_benchmarks)
    
    def load_performance_data(start_date, end_date, selected_benchmarks):
        """Helper function to load performance data"""
        try:
            # Convert string dates to datetime
            if start_date is None:
                start_date = (date.today() - timedelta(days=365)).isoformat()
                logging.info(f"Using default start date: {start_date}")
            if end_date is None:
                end_date = date.today().isoformat()
                logging.info(f"Using default end date: {end_date}")
                
            start_date_obj = date.fromisoformat(start_date)
            end_date_obj = date.fromisoformat(end_date)
            
            # Get portfolio performance data
            logging.info(f"Fetching portfolio performance data from {start_date} to {end_date}")
            portfolio_df = controller.get_portfolio_performance(start_date, end_date)
            logging.info("Successfully retrieved portfolio performance data")
            
            # If no data, return None
            if portfolio_df is None or portfolio_df.empty:
                logging.warning("No portfolio performance data available")
                return None
                
            # Ensure date is a column, not an index for proper records serialization
            if isinstance(portfolio_df.index, pd.DatetimeIndex):
                portfolio_df = portfolio_df.reset_index()
                portfolio_df.rename(columns={'index': 'date'}, inplace=True)
            
            # If date is already a column but not the index
            if 'date' in portfolio_df.columns and not isinstance(portfolio_df.index, pd.DatetimeIndex):
                # Make sure it's sorted by date
                portfolio_df = portfolio_df.sort_values('date')
                
            # Convert date objects to strings for JSON serialization
            if 'date' in portfolio_df.columns:
                portfolio_df['date'] = portfolio_df['date'].apply(lambda d: d.isoformat() if isinstance(d, date) else d)
            
            # Check for and handle any NaN values in normalized_value
            if 'normalized_value' in portfolio_df.columns:
                if portfolio_df['normalized_value'].isna().any():
                    logging.warning(f"Found {portfolio_df['normalized_value'].isna().sum()} NaN values in normalized_value")
                    # Interpolate NaN values
                    portfolio_df['normalized_value'] = portfolio_df['normalized_value'].interpolate(method='linear')
                    logging.info("Interpolated NaN values in normalized_value")
                    
                # Check for any remaining NaN values after interpolation
                if portfolio_df['normalized_value'].isna().any():
                    # Forward fill any remaining NaNs at the beginning
                    portfolio_df['normalized_value'] = portfolio_df['normalized_value'].fillna(method='ffill')
                    # Backward fill any remaining NaNs at the end
                    portfolio_df['normalized_value'] = portfolio_df['normalized_value'].fillna(method='bfill')
                    # If still NaN (empty df), set to 100
                    portfolio_df['normalized_value'] = portfolio_df['normalized_value'].fillna(100)
            
            benchmark_dict = {'NASDAQ': '^IXIC', 'S&P 500': '^GSPC', 'DAX 30': '^GDAXI'}

            # Create an empty DataFrame for benchmark data with dates as index
            benchmark_data = pd.DataFrame(index=pd.date_range(start_date_obj, end_date_obj))
            
            # Only fetch benchmark data if we have selected benchmarks
            if selected_benchmarks and len(selected_benchmarks) > 0:
                logging.info(f"Fetching benchmark data for: {selected_benchmarks}")
                
                # Process each selected benchmark
                for benchmark in selected_benchmarks:
                    if benchmark in benchmark_dict:
                        ticker = benchmark_dict[benchmark]
                        try:
                            # Get historical data from database using get_historical_market_data for more complete data
                            historical_data = controller.db_manager.get_historical_market_data(
                                ticker, 
                                start_date_obj, 
                                end_date_obj
                            )

                            if not historical_data:
                                historical_data = Stock(ticker).get_historical_data(start_date_obj, end_date_obj)
                            
                            if historical_data:
                                # Convert list of dictionaries to DataFrame
                                benchmark_df = pd.DataFrame(historical_data)
                                
                                if not benchmark_df.empty:
                                    # Set date as index
                                    benchmark_df.set_index('date', inplace=True)
                                    
                                    # Use close_price for the benchmark value
                                    benchmark_data[benchmark] = benchmark_df['close_price']
                                    
                                    # Interpolate any missing values
                                    benchmark_data[benchmark] = benchmark_data[benchmark].interpolate(method='linear')
                                    
                                    logging.info(f"Successfully retrieved benchmark data for {benchmark} from database")
                                else:
                                    logging.warning(f"Empty benchmark data for {benchmark}")
                            else:
                                logging.warning(f"No historical data found in database for benchmark {benchmark}")
                        except Exception as e:
                            logging.error(f"Error retrieving benchmark data for {benchmark}: {e}")
            
            # Reset index to make date a column and convert to date objects without timestamps
            benchmark_data = benchmark_data.reset_index()
            benchmark_data.rename(columns={'index': 'date'}, inplace=True)
            benchmark_data['date'] = benchmark_data['date'].apply(lambda d: d.date().isoformat())
            
            # Prepare data for the store
            result = {
                'portfolio': portfolio_df.to_dict('records'),
                'benchmarks': benchmark_data.to_dict('records'),
                'start_date': start_date,
                'end_date': end_date
            }
            
            logging.info("Performance data prepared successfully")
            return result
        except Exception as e:
            logging.error(f"Error updating performance data: {str(e)}")
            return None

    @app.callback(
        Output('performance-chart', 'figure', allow_duplicate=True),
        [Input('performance-data-store', 'data')],
        prevent_initial_call=True
    )
    def update_performance_chart(data):
        """Update the performance chart with data from the store"""
        logging.info("Updating performance chart")
        
        # Create a default figure in case of error
        start_date = (date.today() - timedelta(days=365))
        end_date = date.today()
        
        default_fig = go.Figure()
        default_fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[100, 100],
            mode='lines',
            name='Portfolio (No Data)',
        ))
        default_fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Value (Normalized to 100)',
            template='plotly_white'
        )
        
        if not data:
            logging.warning("No performance data available for chart")
            return default_fig
            
        try:
            # Extract data from the store
            if 'portfolio' not in data or 'benchmarks' not in data or 'start_date' not in data or 'end_date' not in data:
                logging.warning("Performance data is missing required fields")
                return default_fig
                
            portfolio_df = pd.DataFrame(data['portfolio'])
            if portfolio_df.empty:
                logging.warning("Portfolio data is empty")
                return default_fig
                
            start_date = date.fromisoformat(data['start_date'])
            end_date = date.fromisoformat(data['end_date'])
            
            logging.info(f"Creating chart with data from {start_date} to {end_date}")
            
            # Ensure date column is datetime and set as index if needed
            if 'date' not in portfolio_df.columns:
                logging.warning("No date column found in portfolio data")
                return default_fig
                
            # Convert date strings to date objects without timestamps
            portfolio_df['date'] = portfolio_df['date'].apply(lambda d: date.fromisoformat(d) if isinstance(d, str) else d)
            portfolio_df.set_index('date', inplace=True)
            
            # Sort by date to ensure chronological order
            portfolio_df = portfolio_df.sort_index()
            
            # Create figure
            fig = go.Figure()
            
            # Add portfolio trace if normalized_value exists
            if 'normalized_value' in portfolio_df.columns:
                # Check for NaN values
                if portfolio_df['normalized_value'].isna().all():
                    logging.warning("All normalized_value entries are NaN")
                    return default_fig
                    
                # Filter out NaN values
                valid_df = portfolio_df.dropna(subset=['normalized_value'])
                
                if not valid_df.empty:
                    # Apply a small amount of smoothing to reduce spikes
                    # Use a 3-day rolling average if we have enough data points
                    if len(valid_df) > 5:
                        smoothed_values = valid_df['normalized_value'].rolling(window=3, min_periods=1).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=valid_df.index,
                            y=smoothed_values,
                            mode='lines',
                            name='Portfolio',
                            line=dict(color='blue', width=2)
                        ))
                    else:
                        # Not enough data for smoothing
                        fig.add_trace(go.Scatter(
                            x=valid_df.index,
                            y=valid_df['normalized_value'],
                            mode='lines',
                            name='Portfolio',
                            line=dict(color='blue', width=2)
                        ))
                    logging.info("Added portfolio trace to chart")
                else:
                    logging.warning("No valid normalized_value data after filtering NaNs")
            else:
                logging.warning("No normalized_value column found in portfolio data")
            
            # Add benchmark traces
            benchmark_df = pd.DataFrame(data['benchmarks'])
            
            # Check if we have data and date column
            if not benchmark_df.empty and 'date' in benchmark_df.columns:
                # Convert date strings to date objects without timestamps
                benchmark_df['date'] = benchmark_df['date'].apply(lambda d: date.fromisoformat(d) if isinstance(d, str) else d)
                benchmark_df.set_index('date', inplace=True)
                
                # Sort by date to ensure chronological order
                benchmark_df = benchmark_df.sort_index()
                
                # Add a trace for each benchmark column (excluding date)
                for column in benchmark_df.columns:
                    # Skip columns with all NaN values
                    if benchmark_df[column].isna().all():
                        logging.warning(f"Skipping benchmark {column} - all values are NaN")
                        continue
                        
                    # Filter out NaN values
                    valid_benchmark = benchmark_df[column].dropna()
                    
                    if not valid_benchmark.empty and pd.notna(valid_benchmark.iloc[0]):
                        first_value = valid_benchmark.iloc[0]
                        if first_value > 0:  # Avoid division by zero
                            normalized_values = valid_benchmark / first_value * 100
                            
                            # Apply a small amount of smoothing to reduce spikes
                            if len(normalized_values) > 5:
                                smoothed_benchmark = normalized_values.rolling(window=3, min_periods=1).mean()
                                
                                fig.add_trace(go.Scatter(
                                    x=smoothed_benchmark.index,
                                    y=smoothed_benchmark.values,
                                    mode='lines',
                                    name=column,
                                    line=dict(dash='dash')
                                ))
                            else:
                                # Not enough data for smoothing
                                fig.add_trace(go.Scatter(
                                    x=normalized_values.index,
                                    y=normalized_values.values,
                                    mode='lines',
                                    name=column,
                                    line=dict(dash='dash')
                                ))
                            logging.info(f"Added benchmark trace for {column}")
                        else:
                            logging.warning(f"First value for {column} is zero or negative: {first_value}")
                    else:
                        logging.warning(f"No valid data for benchmark {column} after filtering NaNs")
            
            # Update layout
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Value (Normalized to 100)',
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            logging.info("Performance chart created successfully")
            return fig
        except Exception as e:
            logging.error(f"Error rendering performance chart: {str(e)}")
            return default_fig 