"""
Unit tests for the portfolio module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.portfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    """Test cases for Portfolio class."""
    
    def setUp(self):
        """Set up test environment with sample data."""
        # Create sample portfolio data
        self.portfolio_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB'],
            'Quantity': [10, 5, 2, 3, 8],
            'Sector': ['Technology', 'Technology', 'Technology', 'Consumer', 'Technology'],
            'Close': [150.0, 300.0, 2000.0, 3000.0, 200.0],
            'Weight': [15.0, 15.0, 40.0, 20.0, 10.0]  # Intentionally doesn't sum to 100
        })
        
        # Create sample price data
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']
        
        price_data = []
        for date in dates:
            for ticker in tickers:
                # Generate some price data
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
                
                price_data.append(row)
        
        self.price_data = pd.DataFrame(price_data)
        
        # Initialize portfolio with data
        self.portfolio = Portfolio(self.portfolio_data, self.price_data)
    
    def test_init(self):
        """Test Portfolio initialization."""
        self.assertIsNotNone(self.portfolio)
        self.assertIsNotNone(self.portfolio.holdings)
        self.assertIsNotNone(self.portfolio.price_data)
        self.assertIsNotNone(self.portfolio.analysis_data)
        
        # Test initialization with no data
        empty_portfolio = Portfolio()
        self.assertIsNone(empty_portfolio.holdings)
        self.assertIsNone(empty_portfolio.price_data)
        self.assertIsNone(empty_portfolio.analysis_data)
    
    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation."""
        # Value should already be calculated during initialization
        self.assertTrue('Value' in self.portfolio.holdings.columns)
        
        # Check if total value is correct
        expected_value = sum(self.portfolio_data['Quantity'] * self.portfolio_data['Close'])
        self.assertAlmostEqual(self.portfolio.total_value, expected_value, places=2)
    
    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 100%."""
        # Original weights don't sum to 100
        original_weight_sum = self.portfolio_data['Weight'].sum()
        self.assertNotAlmostEqual(original_weight_sum, 100.0, places=1)
        
        # Portfolio should have normalized the weights
        normalized_weight_sum = self.portfolio.holdings['Weight'].sum()
        self.assertAlmostEqual(normalized_weight_sum, 100.0, places=1)
    
    def test_set_holdings(self):
        """Test setting new holdings."""
        # Create new holdings
        new_holdings = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOG'],
            'Quantity': [20, 10, 5],
            'Sector': ['Technology', 'Technology', 'Technology'],
            'Close': [160.0, 310.0, 2100.0],
            'Weight': [33.33, 33.33, 33.34]
        })
        
        # Set new holdings
        self.portfolio.set_holdings(new_holdings)
        
        # Check if holdings were updated
        self.assertEqual(len(self.portfolio.holdings), 3)
        self.assertEqual(self.portfolio.holdings['Ticker'].tolist(), ['AAPL', 'MSFT', 'GOOG'])
        
        # Check if analysis data was updated
        self.assertIsNotNone(self.portfolio.analysis_data)
        unique_tickers = self.portfolio.analysis_data['Ticker'].unique()
        self.assertEqual(set(unique_tickers), set(['AAPL', 'MSFT', 'GOOG']))
    
    def test_set_price_data(self):
        """Test setting new price data."""
        # Create new price data with only 3 tickers
        new_price_data = self.price_data[self.price_data['Ticker'].isin(['AAPL', 'MSFT', 'GOOG'])]
        
        # Set new price data
        self.portfolio.set_price_data(new_price_data)
        
        # Check if price data was updated
        self.assertEqual(len(self.portfolio.price_data), len(new_price_data))
        
        # Check if analysis data was updated
        unique_tickers = self.portfolio.analysis_data['Ticker'].unique()
        self.assertEqual(set(unique_tickers), set(['AAPL', 'MSFT', 'GOOG']))
    
    def test_rebalance_target_weights(self):
        """Test portfolio rebalancing with target weights."""
        # Define target weights
        target_weights = {
            'AAPL': 20.0,
            'MSFT': 20.0,
            'GOOG': 30.0,
            'AMZN': 20.0,
            'FB': 10.0
        }
        
        # Rebalance portfolio
        new_holdings = self.portfolio.rebalance(target_weights=target_weights)
        
        # Check if weights are updated correctly
        for ticker, weight in target_weights.items():
            idx = new_holdings[new_holdings['Ticker'] == ticker].index[0]
            self.assertAlmostEqual(new_holdings.loc[idx, 'Weight'], weight, places=1)
    
    def test_rebalance_target_allocation(self):
        """Test portfolio rebalancing with target sector allocation."""
        # Define target sector allocation
        target_allocation = {
            'Technology': 80.0,
            'Consumer': 20.0
        }
        
        # Rebalance portfolio
        new_holdings = self.portfolio.rebalance(target_allocation=target_allocation)
        
        # Check if sector allocation is updated correctly
        sector_weights = new_holdings.groupby('Sector')['Weight'].sum()
        self.assertAlmostEqual(sector_weights['Technology'], 80.0, places=1)
        self.assertAlmostEqual(sector_weights['Consumer'], 20.0, places=1)
    
    def test_calculate_historical_performance(self):
        """Test historical performance calculation."""
        # Calculate performance
        performance = self.portfolio.calculate_historical_performance()
        
        # Check if performance data is created
        self.assertIsNotNone(performance)
        self.assertTrue('Portfolio Value' in performance.columns)
        self.assertTrue('Daily Return' in performance.columns)
        self.assertTrue('Cumulative Return' in performance.columns)
        
        # Check if performance data covers the correct date range
        min_date = self.price_data['Date'].min()
        max_date = self.price_data['Date'].max()
        self.assertEqual(performance['Date'].min(), min_date)
        self.assertEqual(performance['Date'].max(), max_date)
    
    def test_calculate_historical_performance_date_range(self):
        """Test historical performance calculation with date range."""
        # Define date range
        start_date = self.price_data['Date'].min() + timedelta(days=2)
        end_date = self.price_data['Date'].max() - timedelta(days=2)
        
        # Calculate performance with date range
        performance = self.portfolio.calculate_historical_performance(
            start_date=start_date,
            end_date=end_date
        )
        
        # Check if performance data covers the correct date range
        self.assertEqual(performance['Date'].min(), start_date)
        self.assertEqual(performance['Date'].max(), end_date)
    
    def test_get_asset_allocation(self):
        """Test asset allocation calculation."""
        # Get asset allocation
        allocation = self.portfolio.get_asset_allocation()
        
        # Check if allocation data is created
        self.assertIsNotNone(allocation)
        self.assertTrue('by_sector' in allocation)
        self.assertTrue('by_ticker' in allocation)
        
        # Check sector allocation
        sector_allocation = allocation['by_sector']
        self.assertEqual(set(sector_allocation.index), set(['Technology', 'Consumer']))
        self.assertAlmostEqual(sector_allocation['Weight'].sum(), 100.0, places=1)
        
        # Check ticker allocation
        ticker_allocation = allocation['by_ticker']
        self.assertEqual(set(ticker_allocation['Ticker']), set(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']))
        self.assertAlmostEqual(ticker_allocation['Weight'].sum(), 100.0, places=1)
    
    def test_get_portfolio_stats(self):
        """Test portfolio statistics calculation."""
        # Get portfolio stats
        stats = self.portfolio.get_portfolio_stats()
        
        # Check if stats are created
        self.assertIsNotNone(stats)
        self.assertTrue('total_value' in stats)
        self.assertTrue('num_holdings' in stats)
        self.assertTrue('largest_holding' in stats)
        self.assertTrue('top_5_concentration' in stats)
        
        # Check specific stats
        self.assertEqual(stats['num_holdings'], 5)
        self.assertGreater(stats['top_5_concentration'], 0)
        self.assertTrue(stats['largest_holding'] in ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB'])
    
    def test_get_what_if_analysis(self):
        """Test what-if scenario analysis."""
        # Run market crash scenario
        scenario = self.portfolio.get_what_if_analysis(scenario='market_crash')
        
        # Check if scenario analysis is created
        self.assertIsNotNone(scenario)
        self.assertTrue('Impact' in scenario.columns)
        self.assertTrue('New Value' in scenario.columns)
        self.assertTrue('Value Change' in scenario.columns)
        
        # Check if impacts are negative (market crash)
        self.assertTrue((scenario['Impact'] < 0).all())
        
        # Check if total row is included
        self.assertTrue('TOTAL' in scenario['Ticker'].values)
    
    def test_get_what_if_analysis_custom(self):
        """Test custom what-if scenario analysis."""
        # Define custom returns
        custom_returns = {
            'Technology': 0.10,
            'Consumer': -0.05
        }
        
        # Run custom scenario
        scenario = self.portfolio.get_what_if_analysis(
            scenario='custom',
            custom_returns=custom_returns
        )
        
        # Check if impacts match custom returns
        tech_impact = scenario[scenario['Sector'] == 'Technology']['Impact'].iloc[0]
        consumer_impact = scenario[scenario['Sector'] == 'Consumer']['Impact'].iloc[0]
        
        self.assertEqual(tech_impact, 0.10)
        self.assertEqual(consumer_impact, -0.05)


if __name__ == '__main__':
    unittest.main()