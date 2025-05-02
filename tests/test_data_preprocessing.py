"""
Unit tests for the data_preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_loading.data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test environment with sample data."""
        # Initialize the preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Create sample portfolio data
        self.portfolio_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB'],
            'Quantity': [10, 5, 2, 3, 8],
            'Sector': ['Technology', 'Technology', 'Technology', 'Consumer', 'Technology'],
            'Close': [150.0, 300.0, 2000.0, 3000.0, 200.0],
            'Weight': [15.0, 15.0, 40.0, 20.0, 10.0]  # Intentionally doesn't sum to 100
        })
        
        # Create sample price data with some issues (missing values, outliers)
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']
        
        data = []
        for date in dates:
            for ticker in tickers:
                # Add some missing values and outliers
                if ticker == 'AAPL' and date.day == 3:
                    # Missing values
                    row = {
                        'Date': date,
                        'Ticker': ticker,
                        'Open': np.nan,
                        'High': np.nan,
                        'Low': np.nan,
                        'Close': np.nan,
                        'Adjusted': np.nan,
                        'Returns': np.nan,
                        'Volume': None
                    }
                elif ticker == 'MSFT' and date.day == 4:
                    # Outlier
                    row = {
                        'Date': date,
                        'Ticker': ticker,
                        'Open': 300.0,
                        'High': 350.0,
                        'Low': 290.0,
                        'Close': 340.0,
                        'Adjusted': 340.0,
                        'Returns': 0.3,  # Extreme outlier
                        'Volume': 1000000
                    }
                else:
                    # Normal data
                    base_price = 150.0 if ticker == 'AAPL' else \
                                300.0 if ticker == 'MSFT' else \
                                2000.0 if ticker == 'GOOG' else \
                                3000.0 if ticker == 'AMZN' else 200.0
                    
                    daily_change = np.random.normal(0.001, 0.02)
                    price = base_price * (1 + daily_change)
                    
                    row = {
                        'Date': date,
                        'Ticker': ticker,
                        'Open': price * 0.99,
                        'High': price * 1.02,
                        'Low': price * 0.98,
                        'Close': price,
                        'Adjusted': price,
                        'Returns': daily_change,
                        'Volume': int(np.random.normal(500000, 100000))
                    }
                
                data.append(row)
        
        self.price_data = pd.DataFrame(data)
    
    def test_preprocess_portfolio(self):
        """Test preprocessing portfolio data."""
        processed_df = self.preprocessor.preprocess_portfolio(self.portfolio_data)
        
        # Check that all required columns exist
        self.assertIn('Value', processed_df.columns)
        
        # Check that weights are normalized to sum to 100
        self.assertAlmostEqual(processed_df['Weight'].sum(), 100.0, places=1)
        
        # Check that Value column is correctly calculated
        expected_value = self.portfolio_data['Quantity'] * self.portfolio_data['Close']
        pd.testing.assert_series_equal(
            processed_df['Value'], 
            expected_value,
            check_names=False
        )
    
    def test_preprocess_portfolio_missing_values(self):
        """Test preprocessing portfolio data with missing values."""
        # Create portfolio data with missing values
        portfolio_with_missing = self.portfolio_data.copy()
        portfolio_with_missing.loc[0, 'Quantity'] = None
        portfolio_with_missing.loc[1, 'Close'] = None
        portfolio_with_missing.loc[2, 'Weight'] = None
        
        processed_df = self.preprocessor.preprocess_portfolio(portfolio_with_missing)
        
        # Check that missing Quantity and Close are filled with 0
        self.assertEqual(processed_df.loc[0, 'Quantity'], 0)
        self.assertEqual(processed_df.loc[1, 'Close'], 0)
        
        # Check that weights are recalculated and normalized
        self.assertAlmostEqual(processed_df['Weight'].sum(), 100.0, places=1)
    
    def test_preprocess_price_data(self):
        """Test preprocessing price data."""
        processed_df = self.preprocessor.preprocess_price_data(self.price_data)
        
        # Check that data is sorted correctly
        self.assertTrue(processed_df.sort_values(['Ticker', 'Date']).equals(processed_df))
        
        # Check that missing values are filled
        self.assertEqual(processed_df['Close'].isna().sum(), 0)
        self.assertEqual(processed_df['Returns'].isna().sum(), 0)
        self.assertEqual(processed_df['Volume'].isna().sum(), 0)
        
        # Check that outliers are handled
        max_returns = processed_df['Returns'].max()
        min_returns = processed_df['Returns'].min()
        returns_std = processed_df['Returns'].std()
        
        # Extreme outliers should be capped at 5 standard deviations
        self.assertLessEqual(max_returns, 5 * returns_std)
        self.assertGreaterEqual(min_returns, -5 * returns_std)
    
    def test_create_analysis_dataset(self):
        """Test creating analysis dataset."""
        # First preprocess the data
        processed_portfolio = self.preprocessor.preprocess_portfolio(self.portfolio_data)
        processed_prices = self.preprocessor.preprocess_price_data(self.price_data)
        
        # Create analysis dataset
        result = self.preprocessor.create_analysis_dataset(
            processed_portfolio,
            processed_prices
        )
        
        # Check that all expected components are returned
        self.assertIn('analysis_df', result)
        self.assertIn('pivot_df', result)
        self.assertIn('portfolio_summary', result)
        
        analysis_df = result['analysis_df']
        portfolio_summary = result['portfolio_summary']
        
        # Check that analysis_df has all required columns
        required_columns = [
            'Date', 'Ticker', 'Close', 'Returns', 'Quantity', 
            'Weight', 'WeightedReturn', 'PositionValue'
        ]
        for col in required_columns:
            self.assertIn(col, analysis_df.columns)
        
        # Check that portfolio_summary has all required columns
        required_summary_columns = [
            'PortfolioReturn', 'PortfolioValue', 'CumulativeReturn', 
            'RollingVolatility', 'RollingSharpe'
        ]
        for col in required_summary_columns:
            self.assertIn(col, portfolio_summary.columns)
        
        # Check that portfolio returns are calculated correctly
        # The sum of weighted returns should equal portfolio return for each date
        for date in analysis_df['Date'].unique():
            date_data = analysis_df[analysis_df['Date'] == date]
            weighted_sum = date_data['WeightedReturn'].sum()
            portfolio_return = portfolio_summary.loc[date, 'PortfolioReturn']
            self.assertAlmostEqual(weighted_sum, portfolio_return, places=6)
    
    def test_create_analysis_dataset_with_date_range(self):
        """Test creating analysis dataset with date range filters."""
        # First preprocess the data
        processed_portfolio = self.preprocessor.preprocess_portfolio(self.portfolio_data)
        processed_prices = self.preprocessor.preprocess_price_data(self.price_data)
        
        # Set date range
        start_date = datetime(2023, 1, 2)
        end_date = datetime(2023, 1, 4)
        
        # Create analysis dataset with date range
        result = self.preprocessor.create_analysis_dataset(
            processed_portfolio,
            processed_prices,
            start_date=start_date,
            end_date=end_date
        )
        
        analysis_df = result['analysis_df']
        
        # Check that only dates within the range are included
        self.assertTrue((analysis_df['Date'] >= start_date).all())
        self.assertTrue((analysis_df['Date'] <= end_date).all())
        
        # Check that the number of unique dates is correct (3 days)
        self.assertEqual(len(analysis_df['Date'].unique()), 3)
    
    def test_empty_input(self):
        """Test behavior with empty input data."""
        # Test with empty portfolio
        empty_portfolio_result = self.preprocessor.preprocess_portfolio(pd.DataFrame())
        self.assertTrue(empty_portfolio_result.empty)
        
        # Test with empty price data
        empty_price_result = self.preprocessor.preprocess_price_data(pd.DataFrame())
        self.assertTrue(empty_price_result.empty)
        
        # Test create_analysis_dataset with empty inputs
        empty_analysis_result = self.preprocessor.create_analysis_dataset(
            pd.DataFrame(),
            self.price_data
        )
        self.assertTrue(empty_analysis_result['analysis_df'].empty)
        self.assertTrue(empty_analysis_result['portfolio_summary'].empty)


if __name__ == '__main__':
    unittest.main()