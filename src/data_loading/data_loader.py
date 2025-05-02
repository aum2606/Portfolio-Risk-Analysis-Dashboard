"""
Module for loading portfolio and price data from CSV files.

This module provides functions to load portfolio holdings and historical price data
from CSV files, with error handling and logging.
"""

import os
import logging
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading portfolio and price data."""

    def __init__(self, data_dir: Union[str, Path], log_level: int = logging.INFO):
        """
        Initialize the DataLoader with paths to data files.

        Args:
            data_dir: Directory containing the data files
            log_level: Logging level (default: logging.INFO)
        """
        self.data_dir = Path(data_dir)
        self._configure_logging(log_level)
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")

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

    def load_portfolio(self, file_name: str = "Portfolio.csv") -> Optional[pd.DataFrame]:
        """
        Load portfolio data from a CSV file.

        Args:
            file_name: Name of the portfolio CSV file (default: "Portfolio.csv")

        Returns:
            DataFrame containing portfolio data, or None if loading fails
        """
        try:
            file_path = self.data_dir / file_name
            logger.info(f"Loading portfolio data from {file_path}")
            
            if not file_path.exists():
                logger.error(f"Portfolio file not found: {file_path}")
                return None
            
            portfolio_df = pd.read_csv(file_path)
            
            # Basic validation
            required_columns = ["Ticker", "Quantity", "Sector", "Close", "Weight"]
            missing_columns = [col for col in required_columns if col not in portfolio_df.columns]
            
            if missing_columns:
                logger.error(f"Portfolio data missing required columns: {missing_columns}")
                return None
            
            logger.info(f"Successfully loaded portfolio with {len(portfolio_df)} assets")
            return portfolio_df
            
        except Exception as e:
            logger.error(f"Error loading portfolio data: {str(e)}")
            return None

    def load_price_data(self, file_name: str = "Portfolio_prices.csv") -> Optional[pd.DataFrame]:
        """
        Load historical price data from a CSV file.

        Args:
            file_name: Name of the price data CSV file (default: "Portfolio_prices.csv")

        Returns:
            DataFrame containing historical price data, or None if loading fails
        """
        try:
            file_path = self.data_dir / file_name
            logger.info(f"Loading price data from {file_path}")
            
            if not file_path.exists():
                logger.error(f"Price data file not found: {file_path}")
                return None
            
            # Using a chunked approach for potentially large files
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=100000):
                chunks.append(chunk)
            
            price_df = pd.concat(chunks, ignore_index=True)
            
            # Basic validation
            required_columns = ["Date", "Ticker", "Close", "Adjusted", "Returns"]
            missing_columns = [col for col in required_columns if col not in price_df.columns]
            
            if missing_columns:
                logger.error(f"Price data missing required columns: {missing_columns}")
                return None
            
            # Convert Date to datetime
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            
            logger.info(f"Successfully loaded price data with {len(price_df)} records")
            logger.info(f"Date range: {price_df['Date'].min()} to {price_df['Date'].max()}")
            
            return price_df
            
        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}")
            return None

    def load_all_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load both portfolio and price data.

        Returns:
            Tuple of (portfolio_df, price_df)
        """
        portfolio_df = self.load_portfolio()
        price_df = self.load_price_data()
        
        if portfolio_df is not None and price_df is not None:
            # Check if all portfolio tickers exist in price data
            portfolio_tickers = set(portfolio_df['Ticker'])
            price_tickers = set(price_df['Ticker'])
            missing_tickers = portfolio_tickers - price_tickers
            
            if missing_tickers:
                logger.warning(f"Some portfolio tickers have no price data: {missing_tickers}")
        
        return portfolio_df, price_df


# Demo usage if run as script
if __name__ == "__main__":
    # Example usage
    data_dir = Path("../../data")
    loader = DataLoader(data_dir)
    
    portfolio_df, price_df = loader.load_all_data()
    
    if portfolio_df is not None:
        print("\n--- Portfolio Summary ---")
        print(f"Total Assets: {len(portfolio_df)}")
        print("\nSector Distribution:")
        print(portfolio_df['Sector'].value_counts())
        print("\nTop 5 Holdings by Weight:")
        print(portfolio_df.sort_values('Weight', ascending=False).head(5)[['Ticker', 'Weight']])
    
    if price_df is not None:
        print("\n--- Price Data Summary ---")
        print(f"Total Records: {len(price_df)}")
        print(f"Date Range: {price_df['Date'].min()} to {price_df['Date'].max()}")
        print(f"Unique Tickers: {len(price_df['Ticker'].unique())}")