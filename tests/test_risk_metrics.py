"""
Unit tests for the risk_metrics module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.risk_metrics import RiskMetrics


class TestRiskMetrics(unittest.TestCase):
    """Test cases for RiskMetrics class."""
    
    def setUp(self):
        """Set up test environment with sample data."""
        # Initialize risk metrics calculator
        self.risk_calc = RiskMetrics()
        
        # Create sample returns data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        
        # Create asset returns with different characteristics
        np.random.seed(42)  # For reproducibility
        
        # Asset with low volatility
        low_vol = np.random.normal(0.0005, 0.01, len(dates))
        
        # Asset with high volatility
        high_vol = np.random.normal(0.001, 0.025, len(dates))
        
        # Asset with negative returns
        negative = np.random.normal(-0.0005, 0.015, len(dates))
        
        # Create Series objects
        self.low_vol_returns = pd.Series(low_vol, index=dates)
        self.high_vol_returns = pd.Series(high_vol, index=dates)
        self.negative_returns = pd.Series(negative, index=dates)
        
        # Create benchmark returns
        benchmark = np.random.normal(0.0007, 0.012, len(dates))
        self.benchmark_returns = pd.Series(benchmark, index=dates)
        
        # Set benchmark data
        self.risk_calc.set_benchmark_data('BENCHMARK', self.benchmark_returns)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Calculate volatility
        low_vol = self.risk_calc.calculate_volatility(self.low_vol_returns)
        high_vol = self.risk_calc.calculate_volatility(self.high_vol_returns)
        
        # Check if high volatility asset has higher value
        self.assertGreater(high_vol, low_vol)
        
        # Check if annualized volatility is higher than non-annualized
        non_annualized = self.risk_calc.calculate_volatility(self.low_vol_returns, annualized=False)
        self.assertGreater(low_vol, non_annualized)
        
        # Test rolling volatility
        rolling_vol = self.risk_calc.calculate_volatility(self.low_vol_returns, rolling_window=20)
        self.assertEqual(len(rolling_vol), len(self.low_vol_returns))
        self.assertTrue(isinstance(rolling_vol, pd.Series))
    
    def test_calculate_beta(self):
        """Test beta calculation."""
        # Calculate beta
        low_vol_beta = self.risk_calc.calculate_beta(self.low_vol_returns, 'BENCHMARK')
        high_vol_beta = self.risk_calc.calculate_beta(self.high_vol_returns, 'BENCHMARK')
        
        # Check if values are reasonable
        self.assertTrue(0 < low_vol_beta < 2)
        self.assertTrue(0 < high_vol_beta < 2)
        
        # Test using Series directly
        direct_beta = self.risk_calc.calculate_beta(self.low_vol_returns, self.benchmark_returns)
        self.assertAlmostEqual(direct_beta, low_vol_beta, places=6)
        
        # Test rolling beta
        rolling_beta = self.risk_calc.calculate_beta(
            self.low_vol_returns, 'BENCHMARK', rolling_window=20
        )
        self.assertEqual(len(rolling_beta), len(self.low_vol_returns))
        self.assertTrue(isinstance(rolling_beta, pd.Series))
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Calculate Sharpe ratio
        low_vol_sharpe = self.risk_calc.calculate_sharpe_ratio(self.low_vol_returns)
        high_vol_sharpe = self.risk_calc.calculate_sharpe_ratio(self.high_vol_returns)
        neg_sharpe = self.risk_calc.calculate_sharpe_ratio(self.negative_returns)
        
        # Check if negative returns have lower Sharpe
        self.assertLess(neg_sharpe, low_vol_sharpe)
        
        # Test with custom risk-free rate
        custom_rf_sharpe = self.risk_calc.calculate_sharpe_ratio(self.low_vol_returns, risk_free_rate=0.05)
        self.assertLess(custom_rf_sharpe, low_vol_sharpe)  # Higher rf should give lower Sharpe
        
        # Test rolling Sharpe
        rolling_sharpe = self.risk_calc.calculate_sharpe_ratio(
            self.low_vol_returns, rolling_window=20
        )
        self.assertEqual(len(rolling_sharpe), len(self.low_vol_returns))
        self.assertTrue(isinstance(rolling_sharpe, pd.Series))
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        # Calculate Sortino ratio
        low_vol_sortino = self.risk_calc.calculate_sortino_ratio(self.low_vol_returns)
        high_vol_sortino = self.risk_calc.calculate_sortino_ratio(self.high_vol_returns)
        neg_sortino = self.risk_calc.calculate_sortino_ratio(self.negative_returns)
        
        # Check if negative returns have lower Sortino
        self.assertLess(neg_sortino, low_vol_sortino)
        
        # Test rolling Sortino
        rolling_sortino = self.risk_calc.calculate_sortino_ratio(
            self.low_vol_returns, rolling_window=20
        )
        self.assertEqual(len(rolling_sortino), len(self.low_vol_returns))
        self.assertTrue(isinstance(rolling_sortino, pd.Series))
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Calculate maximum drawdown
        low_vol_dd = self.risk_calc.calculate_max_drawdown(self.low_vol_returns)
        high_vol_dd = self.risk_calc.calculate_max_drawdown(self.high_vol_returns)
        
        # Check if high volatility has larger drawdown (more negative)
        self.assertLess(high_vol_dd, low_vol_dd)
        
        # Check if drawdown is negative
        self.assertLess(low_vol_dd, 0)
        self.assertLess(high_vol_dd, 0)
        
        # Test rolling drawdown
        rolling_dd = self.risk_calc.calculate_max_drawdown(
            self.low_vol_returns, rolling_window=20
        )
        self.assertEqual(len(rolling_dd), len(self.low_vol_returns))
        self.assertTrue(isinstance(rolling_dd, pd.Series))
    
    def test_calculate_var(self):
        """Test Value at Risk (VaR) calculation."""
        # Calculate VaR with different confidence levels
        var_95 = self.risk_calc.calculate_var(self.high_vol_returns, confidence_level=0.95)
        var_99 = self.risk_calc.calculate_var(self.high_vol_returns, confidence_level=0.99)
        
        # 99% VaR should be more extreme (more negative) than 95% VaR
        self.assertLess(var_99, var_95)
        
        # Test with parametric method
        parametric_var = self.risk_calc.calculate_var(
            self.high_vol_returns, method='parametric'
        )
        self.assertTrue(isinstance(parametric_var, float))
        
        # Test rolling VaR
        rolling_var = self.risk_calc.calculate_var(
            self.high_vol_returns, rolling_window=20
        )
        self.assertEqual(len(rolling_var), len(self.high_vol_returns))
        self.assertTrue(isinstance(rolling_var, pd.Series))
    
    def test_calculate_cvar(self):
        """Test Conditional Value at Risk (CVaR) calculation."""
        # Calculate CVaR
        cvar_95 = self.risk_calc.calculate_cvar(self.high_vol_returns, confidence_level=0.95)
        var_95 = self.risk_calc.calculate_var(self.high_vol_returns, confidence_level=0.95)
        
        # CVaR should be more extreme (more negative) than VaR
        self.assertLess(cvar_95, var_95)
        
        # Test rolling CVaR
        rolling_cvar = self.risk_calc.calculate_cvar(
            self.high_vol_returns, rolling_window=20
        )
        self.assertEqual(len(rolling_cvar), len(self.high_vol_returns))
        self.assertTrue(isinstance(rolling_cvar, pd.Series))
    
    def test_calculate_tracking_error(self):
        """Test tracking error calculation."""
        # Calculate tracking error
        low_vol_te = self.risk_calc.calculate_tracking_error(self.low_vol_returns, 'BENCHMARK')
        high_vol_te = self.risk_calc.calculate_tracking_error(self.high_vol_returns, 'BENCHMARK')
        
        # Higher volatility should have higher tracking error
        self.assertGreater(high_vol_te, low_vol_te)
        
        # Test rolling tracking error
        rolling_te = self.risk_calc.calculate_tracking_error(
            self.low_vol_returns, 'BENCHMARK', rolling_window=20
        )
        self.assertEqual(len(rolling_te), len(self.low_vol_returns))
        self.assertTrue(isinstance(rolling_te, pd.Series))
    
    def test_calculate_information_ratio(self):
        """Test information ratio calculation."""
        # Create returns with different excess returns
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        benchmark = np.random.normal(0.0007, 0.012, len(dates))
        benchmark_returns = pd.Series(benchmark, index=dates)
        
        # Asset with positive excess return
        pos_excess = benchmark + 0.0005 + np.random.normal(0, 0.005, len(dates))
        pos_excess_returns = pd.Series(pos_excess, index=dates)
        
        # Asset with negative excess return
        neg_excess = benchmark - 0.0005 + np.random.normal(0, 0.005, len(dates))
        neg_excess_returns = pd.Series(neg_excess, index=dates)
        
        # Set new benchmark
        self.risk_calc.set_benchmark_data('TEST_BENCHMARK', benchmark_returns)
        
        # Calculate information ratio
        pos_ir = self.risk_calc.calculate_information_ratio(pos_excess_returns, 'TEST_BENCHMARK')
        neg_ir = self.risk_calc.calculate_information_ratio(neg_excess_returns, 'TEST_BENCHMARK')
        
        # Positive excess return should have positive IR
        self.assertGreater(pos_ir, 0)
        
        # Negative excess return should have negative IR
        self.assertLess(neg_ir, 0)
        
        # Test rolling information ratio
        rolling_ir = self.risk_calc.calculate_information_ratio(
            pos_excess_returns, 'TEST_BENCHMARK', rolling_window=20
        )
        self.assertEqual(len(rolling_ir), len(pos_excess_returns))
        self.assertTrue(isinstance(rolling_ir, pd.Series))
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        # Calculate all metrics
        metrics = self.risk_calc.calculate_all_metrics(
            self.high_vol_returns, benchmark='BENCHMARK'
        )
        
        # Check if all metrics are present
        expected_metrics = [
            'volatility', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'var_95', 'cvar_95', 'beta', 'tracking_error', 'information_ratio',
            'mean_return', 'min_return', 'max_return', 'median_return',
            'skewness', 'kurtosis', 'annualized_return'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Test rolling metrics
        rolling_metrics = self.risk_calc.calculate_all_metrics(
            self.high_vol_returns, benchmark='BENCHMARK', rolling_window=20
        )
        
        # Check if all rolling metrics are present
        expected_rolling_metrics = [
            'volatility', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'var_95', 'cvar_95', 'beta', 'tracking_error', 'information_ratio',
            'rolling_mean', 'rolling_min', 'rolling_max', 'rolling_median',
            'rolling_skewness', 'rolling_kurtosis', 'rolling_annualized_return'
        ]
        
        for metric in expected_rolling_metrics:
            self.assertIn(metric, rolling_metrics)
            self.assertEqual(len(rolling_metrics[metric]), len(self.high_vol_returns))
    
    def test_calculate_portfolio_risk_contribution(self):
        """Test portfolio risk contribution calculation."""
        # Create a weight series
        weights = pd.Series({
            'Asset1': 0.2,
            'Asset2': 0.3,
            'Asset3': 0.5
        })
        
        # Create a covariance matrix
        cov_matrix = pd.DataFrame({
            'Asset1': [0.04, 0.02, 0.01],
            'Asset2': [0.02, 0.09, 0.03],
            'Asset3': [0.01, 0.03, 0.16]
        }, index=['Asset1', 'Asset2', 'Asset3'])
        
        # Calculate risk contribution
        risk_contrib = self.risk_calc.calculate_portfolio_risk_contribution(weights, cov_matrix)
        
        # Check if contributions sum to 100%
        self.assertAlmostEqual(risk_contrib.sum(), 100.0, places=6)
        
        # Check if all assets have positive contribution
        self.assertTrue((risk_contrib > 0).all())
    
    def test_calculate_portfolio_diversification_ratio(self):
        """Test portfolio diversification ratio calculation."""
        # Create weights for a diversified portfolio
        div_weights = pd.Series({
            'Asset1': 0.33,
            'Asset2': 0.33,
            'Asset3': 0.34
        })
        
        # Create weights for a concentrated portfolio
        conc_weights = pd.Series({
            'Asset1': 0.9,
            'Asset2': 0.05,
            'Asset3': 0.05
        })
        
        # Create a covariance matrix with low correlation
        low_corr_cov = pd.DataFrame({
            'Asset1': [0.04, 0.01, 0.005],
            'Asset2': [0.01, 0.09, 0.01],
            'Asset3': [0.005, 0.01, 0.16]
        }, index=['Asset1', 'Asset2', 'Asset3'])
        
        # Create a covariance matrix with high correlation
        high_corr_cov = pd.DataFrame({
            'Asset1': [0.04, 0.03, 0.03],
            'Asset2': [0.03, 0.09, 0.06],
            'Asset3': [0.03, 0.06, 0.16]
        }, index=['Asset1', 'Asset2', 'Asset3'])
        
        # Calculate diversification ratios
        div_ratio_1 = self.risk_calc.calculate_portfolio_diversification_ratio(div_weights, low_corr_cov)
        div_ratio_2 = self.risk_calc.calculate_portfolio_diversification_ratio(conc_weights, low_corr_cov)
        div_ratio_3 = self.risk_calc.calculate_portfolio_diversification_ratio(div_weights, high_corr_cov)
        
        # Diversified weights with low correlation should have the highest ratio
        self.assertGreater(div_ratio_1, div_ratio_2)  # More diversified weights
        self.assertGreater(div_ratio_1, div_ratio_3)  # Lower correlation


if __name__ == '__main__':
    unittest.main()