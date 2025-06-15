import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from backend.database import DatabaseManager

class Benchmark:
    """Base class for benchmarks to compare portfolio performance"""
    
    def __init__(self, name: str, ticker: str, db_manager: DatabaseManager):
        """Initialize benchmark with name and ticker
        
        Args:
            name: Human-readable name of the benchmark
            ticker: Yahoo Finance ticker symbol
            db_manager: The database manager instance.
        """
        self.name = name
        self.ticker = ticker
        self.db_manager = db_manager
        self._ensure_stock_in_db()
        
    def _ensure_stock_in_db(self):
        """Ensures the benchmark's ticker exists in the stocks table."""
        stock = self.db_manager.get_stock(self.ticker)
        if not stock:
            # Benchmarks might not have a standard currency, sector, or country
            self.db_manager.add_stock(
                ticker=self.ticker,
                name=self.name,
                currency='N/A',
                sector='Benchmark',
                country='N/A'
            )

    def get_returns(self, start_date: datetime.date, end_date: datetime.date = None,
                   interval: str = 'daily') -> pd.DataFrame:
        """Get benchmark returns over a period
        
        Args:
            start_date: Start date for returns calculation
            end_date: End date for returns calculation (defaults to today)
            interval: Return interval ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with dates and returns
        """
        if end_date is None:
            end_date = datetime.date.today()
            
        # Get historical prices from the database
        price_data = self.db_manager.get_historical_stock_prices(
            self.ticker, start_date, end_date
        )
        
        if not price_data:
            return pd.DataFrame(columns=['date', 'price', 'return'])
            
        dates = [p.date for p in price_data]
        prices_values = [p.price for p in price_data]
        prices = pd.DataFrame({'price': prices_values}, index=pd.to_datetime(dates))
        prices.index.name = 'date'

        # Calculate returns
        prices['return'] = prices['price'].pct_change()
        
        # Resample if needed
        if interval == 'weekly':
            prices = prices.resample('W').last()
        elif interval == 'monthly':
            prices = prices.resample('M').last()
            
        # Reset index to make date a column
        prices = prices.reset_index()
        
        return prices
        
    def get_cumulative_returns(self, start_date: datetime.date, 
                              end_date: datetime.date = None) -> pd.DataFrame:
        """Get cumulative returns over a period"""
        daily_returns = self.get_returns(start_date, end_date)
        
        if daily_returns.empty:
            return pd.DataFrame(columns=['date', 'cumulative_return'])
            
        # Calculate cumulative returns
        daily_returns['cumulative_return'] = (1 + daily_returns['return']).cumprod() - 1
        
        return daily_returns[['date', 'cumulative_return']]
        
    def get_annualized_return(self, start_date: datetime.date, 
                             end_date: datetime.date = None) -> float:
        """Calculate annualized return over a period"""
        if end_date is None:
            end_date = datetime.date.today()
            
        # Get cumulative return
        cumulative_returns = self.get_cumulative_returns(start_date, end_date)
        
        if cumulative_returns.empty:
            return 0.0
            
        total_return = cumulative_returns['cumulative_return'].iloc[-1]
        
        # Calculate years
        days = (end_date - start_date).days
        years = days / 365.25
        
        # Calculate annualized return
        if years > 0:
            return ((1 + total_return) ** (1 / years)) - 1
        return 0.0
        
    def get_volatility(self, start_date: datetime.date, 
                      end_date: datetime.date = None) -> float:
        """Calculate volatility (standard deviation of returns)"""
        daily_returns = self.get_returns(start_date, end_date)
        
        if daily_returns.empty or len(daily_returns) < 2:
            return 0.0
            
        # Calculate annualized volatility (standard deviation * sqrt(252))
        return daily_returns['return'].std() * np.sqrt(252)


class NasdaqBenchmark(Benchmark):
    """NASDAQ Composite Index benchmark"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__("NASDAQ Composite", "^IXIC", db_manager)


class SP500Benchmark(Benchmark):
    """S&P 500 Index benchmark"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__("S&P 500", "^GSPC", db_manager)


class DAX30Benchmark(Benchmark):
    """DAX 30 Index benchmark"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__("DAX 30", "^GDAXI", db_manager)


class BenchmarkComparison:
    """Class to compare portfolio performance against benchmarks"""
    
    def __init__(self, portfolio_returns: pd.DataFrame, db_manager: DatabaseManager):
        """Initialize with portfolio returns
        
        Args:
            portfolio_returns: DataFrame with dates and portfolio returns
            db_manager: An instance of DatabaseManager
        """
        self.portfolio_returns = portfolio_returns
        self.benchmarks = {}
        self.db_manager = db_manager
        
        # Add default benchmarks
        self.add_benchmark(NasdaqBenchmark(db_manager))
        self.add_benchmark(SP500Benchmark(db_manager))
        self.add_benchmark(DAX30Benchmark(db_manager))
        
    def add_benchmark(self, benchmark: Benchmark):
        """Add a benchmark for comparison"""
        self.benchmarks[benchmark.name] = benchmark
        
    def add_custom_benchmark(self, name: str, ticker: str):
        """Add a custom benchmark by ticker"""
        self.add_benchmark(Benchmark(name, ticker, self.db_manager))
        
    def get_comparison(self, start_date: datetime.date, 
                      end_date: datetime.date = None) -> pd.DataFrame:
        """Compare portfolio returns against benchmarks
        
        Returns:
            DataFrame with dates and cumulative returns for portfolio and benchmarks
        """
        if end_date is None:
            end_date = datetime.date.today()
            
        # Filter portfolio returns to date range
        portfolio_data = self.portfolio_returns[
            (self.portfolio_returns['date'] >= start_date) & 
            (self.portfolio_returns['date'] <= end_date)
        ].copy()
        
        if portfolio_data.empty:
            return pd.DataFrame()
            
        # Calculate cumulative portfolio returns
        portfolio_data['cumulative_return'] = (
            (1 + portfolio_data['return']).cumprod() - 1
        )
        
        # Create result DataFrame with dates
        result = portfolio_data[['date', 'cumulative_return']].rename(
            columns={'cumulative_return': 'Portfolio'}
        )
        
        # Add benchmark returns
        for name, benchmark in self.benchmarks.items():
            benchmark_returns = benchmark.get_cumulative_returns(start_date, end_date)
            if not benchmark_returns.empty:
                # Merge on date
                result = pd.merge(
                    result, 
                    benchmark_returns.rename(
                        columns={'cumulative_return': name}
                    ),
                    on='date',
                    how='outer'
                )
                
        # Sort by date
        result = result.sort_values('date')
        
        return result
        
    def get_performance_metrics(self, start_date: datetime.date,
                               end_date: datetime.date = None) -> pd.DataFrame:
        """Calculate performance metrics for portfolio and benchmarks
        
        Returns:
            DataFrame with performance metrics for portfolio and benchmarks
        """
        if end_date is None:
            end_date = datetime.date.today()
            
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(start_date, end_date)
        
        # Calculate benchmark metrics
        benchmark_metrics = []
        for name, benchmark in self.benchmarks.items():
            metrics = {
                'Name': name,
                'Total Return': benchmark.get_cumulative_returns(
                    start_date, end_date
                )['cumulative_return'].iloc[-1] if not benchmark.get_cumulative_returns(
                    start_date, end_date
                ).empty else 0.0,
                'Annualized Return': benchmark.get_annualized_return(start_date, end_date),
                'Volatility': benchmark.get_volatility(start_date, end_date)
            }
            benchmark_metrics.append(metrics)
            
        # Combine metrics
        all_metrics = [portfolio_metrics] + benchmark_metrics
        
        return pd.DataFrame(all_metrics)
        
    def _calculate_portfolio_metrics(self, start_date: datetime.date,
                                   end_date: datetime.date) -> Dict:
        """Calculate performance metrics for portfolio"""
        # Filter portfolio returns to date range
        portfolio_data = self.portfolio_returns[
            (self.portfolio_returns['date'] >= start_date) & 
            (self.portfolio_returns['date'] <= end_date)
        ].copy()
        
        if portfolio_data.empty:
            return {
                'Name': 'Portfolio',
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Volatility': 0.0
            }
            
        # Calculate total return
        portfolio_data['cumulative_return'] = (
            (1 + portfolio_data['return']).cumprod() - 1
        )
        total_return = portfolio_data['cumulative_return'].iloc[-1]
        
        # Calculate annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0.0
        
        # Calculate volatility
        volatility = portfolio_data['return'].std() * np.sqrt(252)
        
        return {
            'Name': 'Portfolio',
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility
        }