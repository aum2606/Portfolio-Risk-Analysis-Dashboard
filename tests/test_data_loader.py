"""
Unit tests for the data_loader module.
"""

import os
import unittest
import pandas as pd
import tempfile
from pathlib import Path
from src.data_loading.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test environment with temporary directory and sample data files."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Create sample portfolio data
        self.portfolio_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOG'],
            'Quantity': [10, 5, 2],
            'Sector': ['Technology', 'Technology', 'Technology'],
            'Close': [150.0, 300.0, 2000.0],
            'Weight': [30.0, 30.0, 40.0]
        })
        
        # Create sample price data
        self.price_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-01', '2023-01-01', 
                     '2023-01-02', '2023-01-02', '2023-01-02'],
            'Ticker': ['AAPL', 'MSFT', 'GOOG', 'AAPL', 'MSFT', 'GOOG'],
            'Open': [145.0, 295.0, 1950.0, 150.0, 300.0, 2000.0],
            'High': [155.0, 305.0, 2050.0, 160.0, 310.0, 2100.0],
            'Low': [140.0, 290.0, 1900.0, 145.0, 295.0, 1950.0],
            'Close': [150.0, 300.0, 2000.0, 155.0, 305.0, 2050.0],
            'Adjusted': [150.0, 300.0, 2000.0, 155.0, 305.0, 2050.0],
            'Returns': [0.01, 0.015, 0.02, 0.033, 0.016, 0.025],
            'Volume': [1000000, 500000, 200000, 1100000, 550000, 210000]
        })
        
        # Save sample data to temporary directory
        self.portfolio_data.to_csv(self.data_dir / 'Portfolio.csv', index=False)
        self.price_data.to_csv(self.data_dir / 'Portfolio_prices.csv', index=False)
        
        # Initialize data loader
        self.loader = DataLoader(self.data_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test DataLoader initialization."""
        self.assertEqual(self.loader.data_dir, self.data_dir)
        self.assertIsNotNone(self.loader)
    
    def test_load_portfolio(self):
        """Test loading portfolio data."""
        portfolio_df = self.loader.load_portfolio()
        
        # Check that data was loaded correctly
        self.assertIsNotNone(portfolio_df)
        self.assertEqual(len(portfolio_df), 3)
        self.assertListEqual(list(portfolio_df.columns), 
                            ['Ticker', 'Quantity', 'Sector', 'Close', 'Weight'])
        
        # Check specific values
        self.assertEqual(portfolio_df.loc[0, 'Ticker'], 'AAPL')
        self.assertEqual(portfolio_df.loc[0, 'Quantity'], 10)
    
    def test_load_portfolio_missing_file(self):
        """Test loading portfolio data with missing file."""
        result = self.loader.load_portfolio('NonExistentFile.csv')
        self.assertIsNone(result)
    
    def test_load_price_data(self):
        """Test loading price data."""
        price_df = self.loader.load_price_data()
        
        # Check that data was loaded correctly
        self.assertIsNotNone(price_df)
        self.assertEqual(len(price_df), 6)
        
        # Check that Date column was converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(price_df['Date']))
        
        # Check specific values
        self.assertEqual(price_df.loc[0, 'Ticker'], 'AAPL')
        self.assertEqual(price_df.loc[0, 'Close'], 150.0)
    
    def test_load_price_data_missing_file(self):
        """Test loading price data with missing file."""
        result = self.loader.load_price_data('NonExistentFile.csv')
        self.assertIsNone(result)
    
    def test_load_all_data(self):
        """Test loading both portfolio and price data."""
        portfolio_df, price_df = self.loader.load_all_data()
        
        # Check that both datasets were loaded correctly
        self.assertIsNotNone(portfolio_df)
        self.assertIsNotNone(price_df)
        self.assertEqual(len(portfolio_df), 3)
        self.assertEqual(len(price_df), 6)
    
    def test_load_all_data_missing_portfolio(self):
        """Test loading all data with missing portfolio file."""
        # Rename portfolio file to simulate missing file
        os.rename(
            self.data_dir / 'Portfolio.csv',
            self.data_dir / 'Portfolio_backup.csv'
        )
        
        portfolio_df, price_df = self.loader.load_all_data()
        
        # Check that portfolio is None but price data is loaded
        self.assertIsNone(portfolio_df)
        self.assertIsNotNone(price_df)
    
    def test_load_all_data_missing_price_data(self):
        """Test loading all data with missing price data file."""
        # Rename price data file to simulate missing file
        os.rename(
            self.data_dir / 'Portfolio_prices.csv',
            self.data_dir / 'Portfolio_prices_backup.csv'
        )
        
        portfolio_df, price_df = self.loader.load_all_data()
        
        # Check that portfolio is loaded but price data is None
        self.assertIsNotNone(portfolio_df)
        self.assertIsNone(price_df)


if __name__ == '__main__':
    unittest.main()