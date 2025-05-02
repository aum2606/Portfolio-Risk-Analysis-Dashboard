"""
Module for preprocessing portfolio and price data.

This module provides functions to clean, validate, and prepare data for portfolio analysis.
It handles missing values, calculates additional metrics, and prepares time series data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing portfolio and price data."""

    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the DataPreprocessor.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        self._configure_logging(log_level)
        logger.info("DataPreprocessor initialized")

    def _configure_logging(self, log_level: int) -> None:
        """
        Configure the logger.

        Args:
            log_level: Logging level to use
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger.setLevel(log_level)

    def preprocess_portfolio(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess portfolio data.

        Args:
            portfolio_df: DataFrame containing raw portfolio data

        Returns:
            Preprocessed portfolio DataFrame
        """
        if portfolio_df is None or portfolio_df.empty:
            logger.error("Cannot preprocess empty portfolio data")
            return pd.DataFrame()

        logger.info("Preprocessing portfolio data")
        df = portfolio_df.copy()

        try:
            # Ensure proper datatypes
            df['Ticker'] = df['Ticker'].astype(str)
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

            # Fill NaN values
            if df['Quantity'].isna().any():
                logger.warning("Missing quantity values found, filling with 0")
                df['Quantity'] = df['Quantity'].fillna(0)

            if df['Close'].isna().any():
                logger.warning("Missing close price values found, filling with 0")
                df['Close'] = df['Close'].fillna(0)

            if df['Weight'].isna().any():
                logger.warning("Missing weight values found, recalculating weights")
                # Recalculate weights if missing
                total_value = (df['Quantity'] * df['Close']).sum()
                df['Weight'] = df['Quantity'] * df['Close'] / total_value * 100

            # Add value column
            df['Value'] = df['Quantity'] * df['Close']
            
            # Check and normalize weights
            weight_sum = df['Weight'].sum()
            if abs(weight_sum - 100) > 1:  # If weights don't sum to approximately 100
                logger.warning(f"Portfolio weights sum to {weight_sum}, normalizing to 100%")
                df['Weight'] = df['Weight'] / weight_sum * 100

            # Remove any completely invalid rows
            initial_rows = len(df)
            df = df.dropna(subset=['Ticker', 'Quantity', 'Close'])
            if len(df) < initial_rows:
                logger.warning(f"Removed {initial_rows - len(df)} invalid rows from portfolio data")

            logger.info(f"Portfolio preprocessing complete: {len(df)} assets")
            return df

        except Exception as e:
            logger.error(f"Error preprocessing portfolio data: {str(e)}")
            return portfolio_df  # Return original data if preprocessing fails

    def preprocess_price_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess historical price data.

        Args:
            price_df: DataFrame containing raw price data

        Returns:
            Preprocessed price DataFrame
        """
        if price_df is None or price_df.empty:
            logger.error("Cannot preprocess empty price data")
            return pd.DataFrame()

        logger.info("Preprocessing price data")
        df = price_df.copy()

        try:
            # Ensure proper datatypes
            df['Ticker'] = df['Ticker'].astype(str)
            
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adjusted', 'Returns', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure Date is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Sort data for time series analysis
            df = df.sort_values(['Ticker', 'Date'])
            
            # Handle missing values
            for col in ['Open', 'High', 'Low', 'Close', 'Adjusted']:
                if col in df.columns and df[col].isna().any():
                    # Forward fill prices for each ticker separately
                    df[col] = df.groupby('Ticker')[col].ffill()
                    # If still missing (at the beginning), backward fill
                    df[col] = df.groupby('Ticker')[col].bfill()

            # If Returns column has missing values, calculate from Adjusted prices
            if 'Returns' in df.columns and df['Returns'].isna().any():
                logger.info("Recalculating missing Returns values")
                # Calculate returns for each ticker
                df['Returns'] = df.groupby('Ticker')['Adjusted'].pct_change()
            
            # Handle missing Volume with median values per ticker
            if 'Volume' in df.columns and df['Volume'].isna().any():
                logger.info("Filling missing Volume data with median values")
                df['Volume'] = df.groupby('Ticker')['Volume'].transform(
                    lambda x: x.fillna(x.median())
                )
            
            # Remove rows with still missing critical data
            critical_cols = ['Date', 'Ticker', 'Close']
            initial_rows = len(df)
            df = df.dropna(subset=critical_cols)
            if len(df) < initial_rows:
                logger.warning(f"Removed {initial_rows - len(df)} rows with missing critical data")
            
            # Flag and handle extreme outliers in Returns
            if 'Returns' in df.columns:
                returns_std = df['Returns'].std()
                outlier_threshold = 5 * returns_std  # 5 standard deviations
                outliers = df[abs(df['Returns']) > outlier_threshold]
                
                if len(outliers) > 0:
                    logger.warning(f"Found {len(outliers)} extreme return outliers")
                    # Winsorize extreme values (clamp to threshold)
                    df.loc[df['Returns'] > outlier_threshold, 'Returns'] = outlier_threshold
                    df.loc[df['Returns'] < -outlier_threshold, 'Returns'] = -outlier_threshold
            
            logger.info(f"Price data preprocessing complete: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error preprocessing price data: {str(e)}")
            return price_df  # Return original data if preprocessing fails

    def create_analysis_dataset(
        self, 
        portfolio_df: pd.DataFrame, 
        price_df: pd.DataFrame,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Create a dataset ready for portfolio analysis by merging portfolio and price data.

        Args:
            portfolio_df: Preprocessed portfolio data
            price_df: Preprocessed price data
            start_date: Optional start date for analysis period
            end_date: Optional end date for analysis period

        Returns:
            Analysis-ready DataFrame
        """
        if portfolio_df is None or portfolio_df.empty or price_df is None or price_df.empty:
            logger.error("Cannot create analysis dataset: missing data")
            return pd.DataFrame()

        logger.info("Creating analysis dataset")
        
        try:
            # Apply date filters if provided
            filtered_price_df = price_df.copy()
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                filtered_price_df = filtered_price_df[filtered_price_df['Date'] >= start_date]
                logger.info(f"Filtering data from {start_date}")
            
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                filtered_price_df = filtered_price_df[filtered_price_df['Date'] <= end_date]
                logger.info(f"Filtering data to {end_date}")
            
            # Filter price data to only include tickers in the portfolio
            portfolio_tickers = set(portfolio_df['Ticker'])
            filtered_price_df = filtered_price_df[filtered_price_df['Ticker'].isin(portfolio_tickers)]
            
            logger.info(f"Filtered price data contains {len(filtered_price_df)} records for {len(portfolio_tickers)} tickers")
            
            # Merge portfolio info with price data
            # Use left join to keep all price data, even if a ticker is not in the portfolio
            analysis_df = pd.merge(
                filtered_price_df,
                portfolio_df[['Ticker', 'Quantity', 'Sector', 'Weight']],
                on='Ticker',
                how='left'
            )
            
            # Calculate weighted returns
            analysis_df['WeightedReturn'] = analysis_df['Returns'] * analysis_df['Weight'] / 100
            
            # Calculate position value over time
            analysis_df['PositionValue'] = analysis_df['Adjusted'] * analysis_df['Quantity']
            
            # Create a pivot table for portfolio performance over time
            # This will have dates as index and tickers as columns
            pivot_df = analysis_df.pivot_table(
                index='Date',
                columns='Ticker',
                values=['Adjusted', 'Returns', 'WeightedReturn', 'PositionValue']
            )
            
            # Calculate daily portfolio returns
            daily_portfolio_returns = analysis_df.groupby('Date')['WeightedReturn'].sum()
            daily_portfolio_value = analysis_df.groupby('Date')['PositionValue'].sum()
            
            # Create a summary DataFrame with portfolio-level metrics
            portfolio_summary = pd.DataFrame({
                'PortfolioReturn': daily_portfolio_returns,
                'PortfolioValue': daily_portfolio_value
            })
            
            # Calculate cumulative returns
            portfolio_summary['CumulativeReturn'] = (1 + portfolio_summary['PortfolioReturn']).cumprod() - 1
            
            # Generate additional features
            portfolio_summary['RollingVolatility'] = portfolio_summary['PortfolioReturn'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            portfolio_summary['RollingSharpe'] = (portfolio_summary['PortfolioReturn'].rolling(window=20).mean() * 252) / (portfolio_summary['PortfolioReturn'].rolling(window=20).std() * np.sqrt(252))
            
            logger.info(f"Analysis dataset created with {len(portfolio_summary)} daily records")
            
            # Return both the detailed and summary datasets
            return {
                'analysis_df': analysis_df,
                'pivot_df': pivot_df,
                'portfolio_summary': portfolio_summary
            }
            
        except Exception as e:
            logger.error(f"Error creating analysis dataset: {str(e)}")
            return {
                'analysis_df': pd.DataFrame(),
                'pivot_df': pd.DataFrame(),
                'portfolio_summary': pd.DataFrame()
            }


# Demo usage if run as script
if __name__ == "__main__":
    # Example usage with dummy data
    from pathlib import Path
    from data_loader import DataLoader
    
    # Load data
    data_dir = Path("../../data")
    loader = DataLoader(data_dir)
    portfolio_df, price_df = loader.load_all_data()
    
    if portfolio_df is not None and price_df is not None:
        # Preprocess data
        preprocessor = DataPreprocessor()
        cleaned_portfolio_df = preprocessor.preprocess_portfolio(portfolio_df)
        cleaned_price_df = preprocessor.preprocess_price_data(price_df)
        
        # Create analysis dataset for last year
        end_date = price_df['Date'].max()
        start_date = end_date - timedelta(days=365)
        
        analysis_data = preprocessor.create_analysis_dataset(
            cleaned_portfolio_df,
            cleaned_price_df,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print summary
        if 'portfolio_summary' in analysis_data and not analysis_data['portfolio_summary'].empty:
            summary_df = analysis_data['portfolio_summary']
            print("\n--- Portfolio Performance Summary ---")
            print(f"Analysis Period: {start_date.date()} to {end_date.date()}")
            print(f"Total Return: {summary_df['CumulativeReturn'].iloc[-1]:.2%}")
            print(f"Latest Portfolio Value: ${summary_df['PortfolioValue'].iloc[-1]:,.2f}")
            print(f"Current Volatility (annualized): {summary_df['RollingVolatility'].iloc[-1]:.2%}")
            print(f"Current Sharpe Ratio: {summary_df['RollingSharpe'].iloc[-1]:.2f}")
        else:
            print("Failed to create analysis dataset")