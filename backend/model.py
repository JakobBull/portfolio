import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from backend.database import DatabaseManager

# Set up logging
logger = logging.getLogger(__name__)

class FundamentalValuationModel:
    """
    Fundamental valuation model based on:
    P = PE * d * E + γ * δE + c
    
    Where:
    - P: Stock price (target)
    - PE: Exponentially smoothed price-earnings ratio (constrained 3-90)
    - d: Earnings scaling parameter (tunable)
    - E: Trailing 12 months earnings (constrained >= 0)
    - γ: Earnings trend coefficient (tunable)
    - δE: Exponentially smoothed earnings trend (tunable)
    - c: Constant term (constrained >= 0)
    """
    
    def __init__(self, db_manager: DatabaseManager, alpha: float = 0.25, beta: float = 0.15, 
                 ridge_alpha: float = 1.0):
        """
        Initialize the fundamental valuation model.
        
        Args:
            db_manager: Database manager instance
            alpha: Smoothing parameter for PE ratio (0 < alpha <= 1)
            beta: Smoothing parameter for earnings trend (0 < beta <= 1)
            ridge_alpha: Ridge regression regularization parameter
        """
        self.db_manager = db_manager
        self.alpha = alpha  # PE smoothing parameter
        self.beta = beta    # Earnings trend smoothing parameter
        self.ridge_alpha = ridge_alpha
        
        # Model parameters (fitted values)
        self.pe_d_coeff = None    # Coefficient for PE * d * E term
        self.gamma = None         # Coefficient for earnings trend
        self.constant = None      # Constant term
        
        # Model components
        self.scaler = StandardScaler()
        self.ridge_model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        
        # Fitted model info
        self.is_fitted = False
        self.feature_names = ['pe_d_earnings', 'earnings_trend', 'constant']
        
    def _get_earnings_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Retrieve earnings data from database.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with earnings data
        """
        try:
            with self.db_manager.session_scope() as session:
                # Use SQLAlchemy query approach instead of raw SQL with pandas
                from backend.database import Base
                from sqlalchemy import text
                
                query = text("""
                SELECT date, amount, currency
                FROM earnings 
                WHERE ticker = :ticker AND date BETWEEN :start_date AND :end_date
                ORDER BY date
                """)
                
                result = session.execute(query, {
                    'ticker': ticker,
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                rows = result.fetchall()
                
                if not rows:
                    logger.warning(f"No earnings data found for {ticker}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                earnings_data = pd.DataFrame([
                    {'date': row[0], 'amount': row[1], 'currency': row[2]}
                    for row in rows
                ])
                
                earnings_data['date'] = pd.to_datetime(earnings_data['date'])
                
                return earnings_data
                
        except Exception as e:
            logger.error(f"Error retrieving earnings data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _get_price_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Retrieve stock price data from database.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with price data
        """
        try:
            price_records = self.db_manager.get_historical_stock_prices(ticker, start_date, end_date)
            
            if not price_records:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()
            
            price_data = pd.DataFrame([
                {'date': record.date, 'price': record.price}
                for record in price_records
            ])
            
            price_data['date'] = pd.to_datetime(price_data['date'])
            price_data = price_data.sort_values('date').reset_index(drop=True)
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error retrieving price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _calculate_trailing_12m_earnings(self, earnings_df: pd.DataFrame, 
                                       reference_date: pd.Timestamp) -> float:
        """
        Calculate trailing 12 months earnings from a reference date.
        
        Args:
            earnings_df: DataFrame with earnings data
            reference_date: Reference date for calculation
            
        Returns:
            Trailing 12 months earnings (constrained >= 0)
        """
        if earnings_df.empty:
            return 0.0
        
        # Get earnings from the last 12 months
        start_date = reference_date - timedelta(days=365)
        trailing_earnings = earnings_df[
            (earnings_df['date'] >= start_date) & 
            (earnings_df['date'] <= reference_date)
        ]
        
        if trailing_earnings.empty:
            return 0.0
        
        total_earnings = trailing_earnings['amount'].sum()
        return max(0.0, total_earnings)  # Constrain >= 0
    
    def _calculate_exponential_pe_ratio(self, price_series: pd.Series, 
                                      earnings_series: pd.Series) -> pd.Series:
        """
        Calculate exponentially smoothed PE ratio.
        
        Args:
            price_series: Stock price series
            earnings_series: Earnings series (aligned with prices)
            
        Returns:
            Exponentially smoothed PE ratio series (constrained 3-90)
        """
        # Calculate instantaneous PE ratios
        pe_ratios = price_series / earnings_series.replace(0, np.nan)
        pe_ratios = pe_ratios.bfill().ffill()
        
        # Apply exponential smoothing
        smoothed_pe = pe_ratios.ewm(alpha=self.alpha, adjust=False).mean()
        
        # Constrain PE ratio between 3 and 90
        smoothed_pe = np.clip(smoothed_pe, 3.0, 90.0)
        
        return smoothed_pe
    
    def _calculate_earnings_trend(self, earnings_df: pd.DataFrame, 
                                reference_dates: pd.Series) -> pd.Series:
        """
        Calculate exponentially smoothed earnings trend (δE).
        
        Args:
            earnings_df: DataFrame with earnings data
            reference_dates: Series of reference dates
            
        Returns:
            Exponentially smoothed earnings trend series
        """
        earnings_trends = []
        
        for ref_date in reference_dates:
            # Get earnings changes over the last 2 years
            start_date = ref_date - timedelta(days=730)  # ~2 years
            historical_earnings = earnings_df[
                (earnings_df['date'] >= start_date) & 
                (earnings_df['date'] <= ref_date)
            ].copy()
            
            if len(historical_earnings) < 2:
                earnings_trends.append(0.0)
                continue
            
            # Calculate quarter-to-quarter changes
            historical_earnings = historical_earnings.sort_values('date')
            earnings_changes = historical_earnings['amount'].diff().dropna()
            
            if earnings_changes.empty:
                earnings_trends.append(0.0)
                continue
            
            # Apply exponential smoothing to earnings changes
            smoothed_trend = earnings_changes.ewm(alpha=self.beta, adjust=False).mean().iloc[-1]
            earnings_trends.append(smoothed_trend)
        
        return pd.Series(earnings_trends, index=reference_dates.index)
    
    def prepare_training_data(self, ticker: str, start_date: date, 
                            end_date: date) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for the model.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Get price and earnings data
        price_data = self._get_price_data(ticker, start_date, end_date)
        earnings_data = self._get_earnings_data(ticker, start_date, end_date)
        
        if price_data.empty or earnings_data.empty:
            logger.error(f"Insufficient data for {ticker}")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Prepare features DataFrame
        features_list = []
        targets = []

        # Use monthly sampling to reduce overfitting
        price_data_monthly = price_data.set_index('date').resample('ME').last().reset_index()
        
        for _, row in price_data_monthly.iterrows():
            ref_date = row['date']
            target_price = row['price']
            
            # Calculate trailing 12M earnings
            trailing_earnings = self._calculate_trailing_12m_earnings(earnings_data, ref_date)
            
            if trailing_earnings <= 0:
                continue  # Skip if no positive earnings
            
            # Calculate PE ratio (we'll smooth it later across all data points)
            current_pe = target_price / trailing_earnings if trailing_earnings > 0 else np.nan
            if pd.isna(current_pe):
                continue
            
            # Calculate earnings trend
            earnings_trend = self._calculate_earnings_trend(
                earnings_data, 
                pd.Series([ref_date])
            ).iloc[0]
            
            features_list.append({
                'date': ref_date,
                'price': target_price,
                'trailing_earnings': trailing_earnings,
                'raw_pe': current_pe,
                'earnings_trend': earnings_trend
            })
            targets.append(target_price)
        
        if not features_list:
            logger.error(f"No valid features generated for {ticker}")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        features_df = pd.DataFrame(features_list)
        
        # Calculate exponentially smoothed PE ratio
        pe_series = pd.Series(features_df['raw_pe'].values, index=features_df.index)
        smoothed_pe = pe_series.ewm(alpha=self.alpha, adjust=False).mean()
        smoothed_pe = np.clip(smoothed_pe, 3.0, 90.0)  # Constrain PE
        
        # Create final feature matrix
        # Feature 1: PE * d * E (where d will be learned as part of the coefficient)
        features_df['pe_d_earnings'] = smoothed_pe * features_df['trailing_earnings']
        
        # Feature 2: Earnings trend
        features_df['earnings_trend'] = features_df['earnings_trend']
        
        # Feature 3: Constant term
        features_df['constant'] = 1.0
        
        # Select only the model features
        model_features = features_df[self.feature_names]
        target_series = pd.Series(targets, index=features_df.index)
        
        return model_features, target_series
    
    def fit(self, ticker: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Fit the fundamental valuation model.
        
        Args:
            ticker: Stock ticker to fit model on
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Dictionary with fitting results and metrics
        """
        logger.info(f"Fitting fundamental valuation model for {ticker}")
        
        # Prepare training data
        X, y = self.prepare_training_data(ticker, start_date, end_date)
        
        if X.empty or y.empty:
            logger.error(f"No training data available for {ticker}")
            return {'success': False, 'error': 'No training data available'}
        
        try:
            # Fit the model with Ridge regression
            self.ridge_model.fit(X, y)
            
            # Extract coefficients
            coefficients = self.ridge_model.coef_
            
            # Map coefficients to model parameters
            self.pe_d_coeff = coefficients[0]  # This represents PE * d coefficient
            self.gamma = coefficients[1]       # Earnings trend coefficient
            self.constant = max(0.0, coefficients[2])  # Constant (constrained >= 0)
            
            self.is_fitted = True
            
            # Calculate performance metrics
            y_pred = self.ridge_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            results = {
                'success': True,
                'ticker': ticker,
                'pe_d_coefficient': self.pe_d_coeff,
                'gamma': self.gamma,
                'constant': self.constant,
                'mse': mse,
                'r2_score': r2,
                'n_samples': len(X),
                'training_period': f"{start_date} to {end_date}"
            }
            
            logger.info(f"Model fitted successfully for {ticker}. R² = {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting model for {ticker}: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, ticker: str, prediction_date: date) -> Optional[float]:
        """
        Predict stock price using the fitted model.
        
        Args:
            ticker: Stock ticker
            prediction_date: Date for prediction
            
        Returns:
            Predicted stock price or None if prediction fails
        """
        if not self.is_fitted:
            logger.error("Model must be fitted before making predictions")
            return None
        
        try:
            # Get required data for prediction
            start_date = prediction_date - timedelta(days=730)  # 2 years for trend
            
            earnings_data = self._get_earnings_data(ticker, start_date, prediction_date)
            if earnings_data.empty:
                logger.error(f"No earnings data available for {ticker} prediction")
                return None
            
            # Calculate features
            trailing_earnings = self._calculate_trailing_12m_earnings(
                earnings_data, pd.Timestamp(prediction_date)
            )
            
            if trailing_earnings <= 0:
                logger.warning(f"No positive earnings for {ticker}, cannot predict")
                return None
            
            earnings_trend = self._calculate_earnings_trend(
                earnings_data, pd.Series([pd.Timestamp(prediction_date)])
            ).iloc[0]
            
            # Estimate current PE (use historical average or market PE)
            # For simplicity, we'll use a reasonable market PE
            estimated_pe = 15.0
            estimated_pe = np.clip(estimated_pe, 3.0, 90.0)
            
            # Create feature vector
            features = np.array([
                estimated_pe * trailing_earnings,  # pe_d_earnings
                earnings_trend,                    # earnings_trend
                1.0                               # constant
            ]).reshape(1, -1)
            
            # Make prediction
            predicted_price = self.ridge_model.predict(features)[0]
            
            # Ensure positive price prediction
            predicted_price = max(0.0, predicted_price)
            
            logger.info(f"Predicted price for {ticker} on {prediction_date}: ${predicted_price:.2f}")
            return predicted_price
            
        except Exception as e:
            logger.error(f"Error predicting price for {ticker}: {e}")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of the fitted model.
        
        Returns:
            Dictionary with model parameters and information
        """
        if not self.is_fitted:
            return {'fitted': False, 'message': 'Model not fitted yet'}
        
        return {
            'fitted': True,
            'model_formula': 'P = PE * d * E + γ * δE + c',
            'parameters': {
                'pe_d_coefficient': self.pe_d_coeff,
                'gamma (earnings_trend_coeff)': self.gamma,
                'constant': self.constant
            },
            'hyperparameters': {
                'alpha (PE_smoothing)': self.alpha,
                'beta (trend_smoothing)': self.beta,
                'ridge_alpha': self.ridge_alpha
            },
            'constraints': {
                'PE_ratio': '3 <= PE <= 90',
                'earnings': 'E >= 0',
                'constant': 'c >= 0'
            }
        }
    
    def validate_model(self, ticker: str, start_date: date, end_date: date, 
                      n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform time series cross-validation on the model.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for validation data
            end_date: End date for validation data
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with validation results
        """
        X, y = self.prepare_training_data(ticker, start_date, end_date)
        
        if X.empty or y.empty:
            return {'success': False, 'error': 'No validation data available'}
        
        try:
            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_scores = []
            cv_mse = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Fit model on train set
                temp_model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
                temp_model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = temp_model.predict(X_test)
                
                # Calculate metrics
                cv_scores.append(r2_score(y_test, y_pred))
                cv_mse.append(mean_squared_error(y_test, y_pred))
            
            return {
                'success': True,
                'mean_r2': np.mean(cv_scores),
                'std_r2': np.std(cv_scores),
                'mean_mse': np.mean(cv_mse),
                'std_mse': np.std(cv_mse),
                'cv_scores': cv_scores,
                'n_splits': n_splits
            }
            
        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            return {'success': False, 'error': str(e)}

    def get_fundamental_value_series(self, ticker: str, start_date: date, 
                                   end_date: date) -> Optional[pd.Series]:
        """
        Get a time series of fundamental values for plotting against actual prices.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for the series
            end_date: End date for the series
            
        Returns:
            Series with fundamental values indexed by date, or None if fails
        """
        if not self.is_fitted:
            logger.error("Model must be fitted before generating fundamental values")
            return None
        
        try:
            # Get price and earnings data
            price_data = self._get_price_data(ticker, start_date, end_date)
            earnings_data = self._get_earnings_data(ticker, start_date, end_date)
            
            if price_data.empty or earnings_data.empty:
                logger.error(f"Insufficient data for {ticker} fundamental value series")
                return None
            
            # Generate fundamental values for each date
            fundamental_values = []
            dates = []
            
            # Use weekly sampling for smoother visualization
            price_data_weekly = price_data.set_index('date').resample('W').last().reset_index()
            
            for _, row in price_data_weekly.iterrows():
                ref_date = row['date']
                
                # Calculate trailing 12M earnings
                trailing_earnings = self._calculate_trailing_12m_earnings(earnings_data, ref_date)
                
                if trailing_earnings <= 0:
                    continue  # Skip if no positive earnings
                
                # Calculate earnings trend
                earnings_trend = self._calculate_earnings_trend(
                    earnings_data, pd.Series([ref_date])
                ).iloc[0]
                
                # Estimate PE ratio (for prediction, we use a market average)
                estimated_pe = 15.0
                estimated_pe = np.clip(estimated_pe, 3.0, 90.0)
                
                # Create feature vector
                features = np.array([
                    estimated_pe * trailing_earnings,  # pe_d_earnings
                    earnings_trend,                    # earnings_trend
                    1.0                               # constant
                ]).reshape(1, -1)
                
                # Predict fundamental value
                fundamental_value = self.ridge_model.predict(features)[0]
                fundamental_value = max(0.0, fundamental_value)
                
                fundamental_values.append(fundamental_value)
                dates.append(ref_date.date())
            
            if not fundamental_values:
                logger.error(f"No fundamental values generated for {ticker}")
                return None
            
            # Create pandas Series
            fundamental_series = pd.Series(
                fundamental_values, 
                index=pd.to_datetime(dates),
                name=f'{ticker}_fundamental_value'
            )
            
            return fundamental_series
            
        except Exception as e:
            logger.error(f"Error generating fundamental value series for {ticker}: {e}")
            return None 