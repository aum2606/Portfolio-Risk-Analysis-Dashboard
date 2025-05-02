"""
Main application for Portfolio Risk Analysis Dashboard.

This is the entry point for the application that loads data, initializes models,
and sets up the interactive dashboard.
"""

import os
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from data_loading.data_loader import DataLoader
from data_loading.data_preprocessing import DataPreprocessor
from models.portfolio import Portfolio
from models.risk_metrics import RiskMetrics
from visualization.dashboard import PortfolioDashboard

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Portfolio Risk Analysis Dashboard')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing data files (default: ./data)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run the dashboard on (default: 8050)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    return parser.parse_args()

def load_data(data_dir):
    """
    Load portfolio and price data.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (portfolio_df, price_df)
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Initialize data loader
    loader = DataLoader(data_dir)
    
    # Load data
    portfolio_df, price_df = loader.load_all_data()
    
    if portfolio_df is None:
        logger.error("Failed to load portfolio data")
        raise FileNotFoundError(f"Portfolio data not found in {data_dir}")
    
    if price_df is None:
        logger.error("Failed to load price data")
        raise FileNotFoundError(f"Price data not found in {data_dir}")
    
    logger.info(f"Successfully loaded portfolio ({len(portfolio_df)} assets) and price data ({len(price_df)} records)")
    
    return portfolio_df, price_df


def preprocess_data(portfolio_df, price_df):
    """
    Preprocess the loaded data.
    
    Args:
        portfolio_df: Portfolio data DataFrame
        price_df: Price data DataFrame
        
    Returns:
        Tuple of (cleaned_portfolio_df, cleaned_price_df)
    """
    logger.info("Preprocessing data")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    cleaned_portfolio = preprocessor.preprocess_portfolio(portfolio_df)
    cleaned_prices = preprocessor.preprocess_price_data(price_df)
    
    logger.info(f"Data preprocessing complete: {len(cleaned_portfolio)} assets, {len(cleaned_prices)} price records")
    
    return cleaned_portfolio, cleaned_prices


def initialize_models(portfolio_df, price_df):
    """
    Initialize portfolio and risk metrics models.
    
    Args:
        portfolio_df: Processed portfolio data
        price_df: Processed price data
        
    Returns:
        Tuple of (portfolio, risk_metrics)
    """
    logger.info("Initializing models")
    
    # Create portfolio
    portfolio = Portfolio(portfolio_df, price_df)
    
    # Initialize risk metrics calculator
    risk_metrics = RiskMetrics()
    
    # Set benchmark data (using portfolio average return as proxy)
    benchmark_returns = price_df.groupby('Date')['Returns'].mean()
    risk_metrics.set_benchmark_data('PORTFOLIO_AVG', benchmark_returns)
    
    logger.info("Models initialized successfully")
    
    return portfolio, risk_metrics


def main():
    """Main application function."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Set up data directory
        data_dir = Path(args.data_dir)
        
        # Load data
        portfolio_df, price_df = load_data(data_dir)
        
        # Preprocess data
        portfolio_df, price_df = preprocess_data(portfolio_df, price_df)
        
        # Initialize models
        portfolio, risk_metrics = initialize_models(portfolio_df, price_df)
        
        # Create dashboard
        dashboard = PortfolioDashboard(portfolio, risk_metrics)
        
        # Create and run dash app
        app = dashboard.create_dash_app()
        
        logger.info(f"Starting dashboard on port {args.port}")
        app.run_server(
            debug=args.debug,
            port=args.port,
            host='0.0.0.0'  # Make accessible externally
        )
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        raise


if __name__ == "__main__":
    main()
