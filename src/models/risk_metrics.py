"""
Risk metrics calculation utilities for portfolio analysis.

This module provides functions to calculate various risk and performance metrics
for portfolio analysis, including volatility, beta, Sharpe ratio, etc.
"""

import logging
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class RiskMetrics:
    """Class for calculating portfolio risk metrics."""

    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the RiskMetrics calculator.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        self._configure_logging(log_level)
        logger.info("RiskMetrics calculator initialized")
        
        # Defined constants
        self.TRADING_DAYS_PER_YEAR = 252
        self.DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
        
        # Cache for benchmark data
        self.benchmark_data = {}
    
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
    
    def set_benchmark_data(self, ticker: str, returns: pd.Series) -> None:
        """
        Set benchmark return data for beta calculations.

        Args:
            ticker: Benchmark ticker symbol (e.g., 'SPY', 'MSCI_World')
            returns: Series of benchmark returns with datetime index
        """
        if not isinstance(returns, pd.Series):
            logger.error("Benchmark returns must be provided as a pandas Series")
            return
            
        if not isinstance(returns.index, pd.DatetimeIndex):
            logger.error("Benchmark returns must have a DatetimeIndex")
            return
            
        self.benchmark_data[ticker] = returns
        logger.info(f"Benchmark data for {ticker} set with {len(returns)} observations")
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        annualized: bool = True,
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate volatility (standard deviation of returns).

        Args:
            returns: Series of asset returns
            annualized: Whether to annualize the volatility
            rolling_window: Window size for rolling volatility calculation

        Returns:
            Volatility value or Series of rolling volatility values
        """
        try:
            if rolling_window is not None:
                # Calculate rolling volatility
                vol = returns.rolling(window=rolling_window).std()
                
                # Annualize if requested
                if annualized:
                    vol = vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return vol
            else:
                # Calculate single volatility value
                vol = returns.std()
                
                # Annualize if requested
                if annualized:
                    vol = vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return vol
                
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_beta(
        self,
        returns: pd.Series,
        benchmark: Union[str, pd.Series] = 'SPY',
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate beta relative to a benchmark.

        Args:
            returns: Series of asset returns
            benchmark: Benchmark ticker or returns Series
            rolling_window: Window size for rolling beta calculation

        Returns:
            Beta value or Series of rolling beta values
        """
        try:
            # Get benchmark returns
            if isinstance(benchmark, str):
                if benchmark not in self.benchmark_data:
                    logger.error(f"Benchmark data for {benchmark} not found. Use set_benchmark_data first.")
                    if rolling_window is not None:
                        return pd.Series(index=returns.index)
                    else:
                        return np.nan
                
                benchmark_returns = self.benchmark_data[benchmark]
            else:
                benchmark_returns = benchmark
                
            # Align data
            merged_data = pd.merge(
                pd.DataFrame({'asset': returns}),
                pd.DataFrame({'benchmark': benchmark_returns}),
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if merged_data.empty:
                logger.warning("No overlapping data between asset and benchmark returns")
                if rolling_window is not None:
                    return pd.Series(index=returns.index)
                else:
                    return np.nan
            
            if rolling_window is not None:
                # Calculate rolling beta
                merged_data['covariance'] = (
                    merged_data['asset'].rolling(window=rolling_window)
                    .cov(merged_data['benchmark'])
                )
                merged_data['benchmark_var'] = (
                    merged_data['benchmark'].rolling(window=rolling_window)
                    .var()
                )
                
                # Calculate beta as covariance / variance
                beta = merged_data['covariance'] / merged_data['benchmark_var']
                return beta
            else:
                # Calculate single beta value
                covariance = merged_data['asset'].cov(merged_data['benchmark'])
                benchmark_var = merged_data['benchmark'].var()
                
                beta = covariance / benchmark_var
                return beta
                
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None,
        annualized: bool = True,
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of asset returns
            risk_free_rate: Annual risk-free rate (default: from class setting)
            annualized: Whether to annualize the Sharpe ratio
            rolling_window: Window size for rolling Sharpe calculation

        Returns:
            Sharpe ratio value or Series of rolling Sharpe values
        """
        try:
            # Use default risk-free rate if not provided
            if risk_free_rate is None:
                risk_free_rate = self.DEFAULT_RISK_FREE_RATE
                
            # Convert annual risk-free rate to daily
            daily_rf = (1 + risk_free_rate) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
            
            if rolling_window is not None:
                # Calculate rolling mean returns
                rolling_mean = returns.rolling(window=rolling_window).mean()
                
                # Calculate rolling standard deviation
                rolling_std = returns.rolling(window=rolling_window).std()
                
                # Calculate rolling Sharpe ratio
                sharpe = (rolling_mean - daily_rf) / rolling_std
                
                # Annualize if requested
                if annualized:
                    sharpe = sharpe * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return sharpe
            else:
                # Calculate mean return
                mean_return = returns.mean()
                
                # Calculate standard deviation
                std_dev = returns.std()
                
                # Calculate Sharpe ratio
                sharpe = (mean_return - daily_rf) / std_dev
                
                # Annualize if requested
                if annualized:
                    sharpe = sharpe * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return sharpe
                
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None,
        annualized: bool = True,
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate Sortino ratio, using downside deviation instead of standard deviation.

        Args:
            returns: Series of asset returns
            risk_free_rate: Annual risk-free rate (default: from class setting)
            annualized: Whether to annualize the Sortino ratio
            rolling_window: Window size for rolling Sortino calculation

        Returns:
            Sortino ratio value or Series of rolling Sortino values
        """
        try:
            # Use default risk-free rate if not provided
            if risk_free_rate is None:
                risk_free_rate = self.DEFAULT_RISK_FREE_RATE
                
            # Convert annual risk-free rate to daily
            daily_rf = (1 + risk_free_rate) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
            
            if rolling_window is not None:
                # Function to calculate downside deviation
                def downside_dev(x):
                    return np.sqrt(np.mean(np.minimum(x - daily_rf, 0) ** 2))
                
                # Calculate rolling mean returns
                rolling_mean = returns.rolling(window=rolling_window).mean()
                
                # Calculate rolling downside deviation
                rolling_downside = returns.rolling(window=rolling_window).apply(
                    downside_dev, raw=True
                )
                
                # Calculate rolling Sortino ratio (handle division by zero)
                sortino = pd.Series(index=rolling_mean.index, dtype=float)
                mask = rolling_downside > 0
                sortino[mask] = (rolling_mean[mask] - daily_rf) / rolling_downside[mask]
                sortino[~mask] = np.nan
                
                # Annualize if requested
                if annualized:
                    sortino = sortino * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return sortino
            else:
                # Calculate mean return
                mean_return = returns.mean()
                
                # Calculate downside deviation
                downside_returns = np.minimum(returns - daily_rf, 0)
                downside_dev = np.sqrt(np.mean(downside_returns ** 2))
                
                # Calculate Sortino ratio
                if downside_dev > 0:
                    sortino = (mean_return - daily_rf) / downside_dev
                    
                    # Annualize if requested
                    if annualized:
                        sortino = sortino * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                        
                    return sortino
                else:
                    return np.nan
                
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_max_drawdown(
        self, 
        returns: pd.Series,
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate maximum drawdown.

        Args:
            returns: Series of asset returns
            rolling_window: Window size for rolling max drawdown calculation

        Returns:
            Max drawdown value or Series of rolling max drawdown values
        """
        try:
            # Convert returns to cumulative performance series
            cumulative = (1 + returns).cumprod()
            
            if rolling_window is not None:
                # Calculate rolling maximum drawdown
                rolling_max_dd = pd.Series(index=returns.index, dtype=float)
                
                for i in range(rolling_window, len(cumulative)):
                    window = cumulative.iloc[i - rolling_window:i + 1]
                    peak = window.expanding().max()
                    drawdown = (window / peak) - 1
                    rolling_max_dd.iloc[i] = drawdown.min()
                
                return rolling_max_dd
            else:
                # Calculate the running maximum
                running_max = cumulative.expanding().max()
                
                # Calculate drawdown at each point
                drawdown = (cumulative / running_max) - 1
                
                # Find the minimum drawdown (maximum loss)
                max_drawdown = drawdown.min()
                
                return max_drawdown
                
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical',
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series of asset returns
            confidence_level: Confidence level for VaR (default: 0.95)
            method: Method for VaR calculation ('historical', 'parametric')
            rolling_window: Window size for rolling VaR calculation

        Returns:
            VaR value or Series of rolling VaR values
        """
        try:
            alpha = 1 - confidence_level
            
            if rolling_window is not None:
                # Calculate rolling VaR
                if method == 'historical':
                    # Historical method: use empirical quantile
                    var = returns.rolling(window=rolling_window).quantile(alpha)
                elif method == 'parametric':
                    # Parametric method: assume normal distribution
                    rolling_mean = returns.rolling(window=rolling_window).mean()
                    rolling_std = returns.rolling(window=rolling_window).std()
                    var = rolling_mean + stats.norm.ppf(alpha) * rolling_std
                else:
                    logger.error(f"Unknown VaR method: {method}")
                    return pd.Series(index=returns.index)
                
                return var
            else:
                # Calculate single VaR value
                if method == 'historical':
                    # Historical method: use empirical quantile
                    var = returns.quantile(alpha)
                elif method == 'parametric':
                    # Parametric method: assume normal distribution
                    mean_return = returns.mean()
                    std_dev = returns.std()
                    var = mean_return + stats.norm.ppf(alpha) * std_dev
                else:
                    logger.error(f"Unknown VaR method: {method}")
                    return np.nan
                
                return var
                
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical',
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

        Args:
            returns: Series of asset returns
            confidence_level: Confidence level for CVaR (default: 0.95)
            method: Method for CVaR calculation ('historical', 'parametric')
            rolling_window: Window size for rolling CVaR calculation

        Returns:
            CVaR value or Series of rolling CVaR values
        """
        try:
            alpha = 1 - confidence_level
            
            if rolling_window is not None:
                # Calculate rolling CVaR
                cvar = pd.Series(index=returns.index, dtype=float)
                
                for i in range(rolling_window, len(returns)):
                    window = returns.iloc[i - rolling_window:i]
                    
                    if method == 'historical':
                        # Historical method: average of returns below VaR
                        var = window.quantile(alpha)
                        cvar.iloc[i] = window[window <= var].mean()
                    elif method == 'parametric':
                        # Parametric method: closed-form solution under normality
                        mean = window.mean()
                        std = window.std()
                        var = mean + stats.norm.ppf(alpha) * std
                        cvar.iloc[i] = mean - std * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
                    else:
                        logger.error(f"Unknown CVaR method: {method}")
                        return pd.Series(index=returns.index)
                
                return cvar
            else:
                # Calculate single CVaR value
                if method == 'historical':
                    # Historical method: average of returns below VaR
                    var = returns.quantile(alpha)
                    cvar = returns[returns <= var].mean()
                elif method == 'parametric':
                    # Parametric method: closed-form solution under normality
                    mean = returns.mean()
                    std = returns.std()
                    cvar = mean - std * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
                else:
                    logger.error(f"Unknown CVaR method: {method}")
                    return np.nan
                
                return cvar
                
        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_tracking_error(
        self,
        returns: pd.Series,
        benchmark: Union[str, pd.Series],
        annualized: bool = True,
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate tracking error relative to a benchmark.

        Args:
            returns: Series of asset returns
            benchmark: Benchmark ticker or returns Series
            annualized: Whether to annualize the tracking error
            rolling_window: Window size for rolling tracking error calculation

        Returns:
            Tracking error value or Series of rolling tracking error values
        """
        try:
            # Get benchmark returns
            if isinstance(benchmark, str):
                if benchmark not in self.benchmark_data:
                    logger.error(f"Benchmark data for {benchmark} not found. Use set_benchmark_data first.")
                    if rolling_window is not None:
                        return pd.Series(index=returns.index)
                    else:
                        return np.nan
                
                benchmark_returns = self.benchmark_data[benchmark]
            else:
                benchmark_returns = benchmark
                
            # Align data
            merged_data = pd.merge(
                pd.DataFrame({'asset': returns}),
                pd.DataFrame({'benchmark': benchmark_returns}),
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if merged_data.empty:
                logger.warning("No overlapping data between asset and benchmark returns")
                if rolling_window is not None:
                    return pd.Series(index=returns.index)
                else:
                    return np.nan
            
            # Calculate return differences
            merged_data['diff'] = merged_data['asset'] - merged_data['benchmark']
            
            if rolling_window is not None:
                # Calculate rolling tracking error
                tracking_error = merged_data['diff'].rolling(window=rolling_window).std()
                
                # Annualize if requested
                if annualized:
                    tracking_error = tracking_error * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return tracking_error
            else:
                # Calculate standard deviation of return differences
                tracking_error = merged_data['diff'].std()
                
                # Annualize if requested
                if annualized:
                    tracking_error = tracking_error * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return tracking_error
                
        except Exception as e:
            logger.error(f"Error calculating tracking error: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark: Union[str, pd.Series],
        annualized: bool = True,
        rolling_window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate information ratio (excess return over tracking error).

        Args:
            returns: Series of asset returns
            benchmark: Benchmark ticker or returns Series
            annualized: Whether to annualize the information ratio
            rolling_window: Window size for rolling information ratio calculation

        Returns:
            Information ratio value or Series of rolling information ratio values
        """
        try:
            # Get benchmark returns
            if isinstance(benchmark, str):
                if benchmark not in self.benchmark_data:
                    logger.error(f"Benchmark data for {benchmark} not found. Use set_benchmark_data first.")
                    if rolling_window is not None:
                        return pd.Series(index=returns.index)
                    else:
                        return np.nan
                
                benchmark_returns = self.benchmark_data[benchmark]
            else:
                benchmark_returns = benchmark
                
            # Align data
            merged_data = pd.merge(
                pd.DataFrame({'asset': returns}),
                pd.DataFrame({'benchmark': benchmark_returns}),
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if merged_data.empty:
                logger.warning("No overlapping data between asset and benchmark returns")
                if rolling_window is not None:
                    return pd.Series(index=returns.index)
                else:
                    return np.nan
            
            # Calculate return differences
            merged_data['diff'] = merged_data['asset'] - merged_data['benchmark']
            
            if rolling_window is not None:
                # Calculate rolling excess return
                rolling_excess = merged_data['diff'].rolling(window=rolling_window).mean()
                
                # Calculate rolling tracking error
                rolling_te = merged_data['diff'].rolling(window=rolling_window).std()
                
                # Calculate rolling information ratio
                ir = rolling_excess / rolling_te
                
                # Annualize if requested
                if annualized:
                    ir = ir * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                    
                return ir
            else:
                # Calculate average excess return
                excess_return = merged_data['diff'].mean()
                
                # Calculate tracking error
                tracking_error = merged_data['diff'].std()
                
                # Calculate information ratio
                if tracking_error > 0:
                    ir = excess_return / tracking_error
                    
                    # Annualize if requested
                    if annualized:
                        ir = ir * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                        
                    return ir
                else:
                    return np.nan
                
        except Exception as e:
            logger.error(f"Error calculating information ratio: {str(e)}")
            if rolling_window is not None:
                return pd.Series(index=returns.index)
            else:
                return np.nan
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        benchmark: Optional[Union[str, pd.Series]] = None,
        risk_free_rate: float = None,
        rolling_window: Optional[int] = None
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate all risk metrics in one call.

        Args:
            returns: Series of asset returns
            benchmark: Optional benchmark ticker or returns Series
            risk_free_rate: Annual risk-free rate (default: from class setting)
            rolling_window: Window size for rolling metrics calculation

        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        try:
            # Use default risk-free rate if not provided
            if risk_free_rate is None:
                risk_free_rate = self.DEFAULT_RISK_FREE_RATE
            
            # Basic metrics
            metrics['volatility'] = self.calculate_volatility(
                returns, annualized=True, rolling_window=rolling_window
            )
            
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(
                returns, risk_free_rate=risk_free_rate, annualized=True, rolling_window=rolling_window
            )
            
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(
                returns, risk_free_rate=risk_free_rate, annualized=True, rolling_window=rolling_window
            )
            
            metrics['max_drawdown'] = self.calculate_max_drawdown(
                returns, rolling_window=rolling_window
            )
            
            metrics['var_95'] = self.calculate_var(
                returns, confidence_level=0.95, method='historical', rolling_window=rolling_window
            )
            
            metrics['cvar_95'] = self.calculate_cvar(
                returns, confidence_level=0.95, method='historical', rolling_window=rolling_window
            )
            
            # Benchmark-relative metrics (if benchmark is provided)
            if benchmark is not None:
                metrics['beta'] = self.calculate_beta(
                    returns, benchmark=benchmark, rolling_window=rolling_window
                )
                
                metrics['tracking_error'] = self.calculate_tracking_error(
                    returns, benchmark=benchmark, annualized=True, rolling_window=rolling_window
                )
                
                metrics['information_ratio'] = self.calculate_information_ratio(
                    returns, benchmark=benchmark, annualized=True, rolling_window=rolling_window
                )
            
            # Calculate returns statistics
            if rolling_window is None:
                metrics['mean_return'] = returns.mean()
                metrics['min_return'] = returns.min()
                metrics['max_return'] = returns.max()
                metrics['median_return'] = returns.median()
                metrics['skewness'] = returns.skew()
                metrics['kurtosis'] = returns.kurtosis()
                
                # Calculate annualized return
                metrics['annualized_return'] = (1 + returns.mean()) ** self.TRADING_DAYS_PER_YEAR - 1
            else:
                metrics['rolling_mean'] = returns.rolling(window=rolling_window).mean()
                metrics['rolling_min'] = returns.rolling(window=rolling_window).min()
                metrics['rolling_max'] = returns.rolling(window=rolling_window).max()
                metrics['rolling_median'] = returns.rolling(window=rolling_window).median()
                metrics['rolling_skewness'] = returns.rolling(window=rolling_window).skew()
                metrics['rolling_kurtosis'] = returns.rolling(window=rolling_window).kurt()
                
                # Calculate rolling annualized return
                metrics['rolling_annualized_return'] = (
                    (1 + metrics['rolling_mean']) ** self.TRADING_DAYS_PER_YEAR - 1
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return metrics
    
    def calculate_portfolio_risk_contribution(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk contribution of each asset in a portfolio.

        Args:
            weights: Series of portfolio weights (indexed by ticker)
            covariance_matrix: Covariance matrix of asset returns

        Returns:
            Series of risk contributions (in percentage)
        """
        try:
            # Validate inputs
            if not isinstance(weights, pd.Series) or not isinstance(covariance_matrix, pd.DataFrame):
                logger.error("Weights must be a Series and covariance_matrix must be a DataFrame")
                return pd.Series()
                
            # Align indices
            common_tickers = list(set(weights.index) & set(covariance_matrix.index))
            if not common_tickers:
                logger.error("No common tickers between weights and covariance matrix")
                return pd.Series()
                
            # Filter to common tickers
            weights = weights[common_tickers]
            covariance_matrix = covariance_matrix.loc[common_tickers, common_tickers]
            
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            
            # Calculate portfolio variance
            portfolio_variance = weights.dot(covariance_matrix).dot(weights)
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate marginal risk contribution
            marginal_contrib = covariance_matrix.dot(weights)
            
            # Calculate risk contribution
            risk_contrib = weights * marginal_contrib / portfolio_volatility
            
            # Calculate percentage contribution
            risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100
            
            return risk_contrib_pct
            
        except Exception as e:
            logger.error(f"Error calculating risk contribution: {str(e)}")
            return pd.Series()
    
    def calculate_portfolio_diversification_ratio(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio diversification ratio.

        Args:
            weights: Series of portfolio weights (indexed by ticker)
            covariance_matrix: Covariance matrix of asset returns

        Returns:
            Diversification ratio (higher is better)
        """
        try:
            # Validate inputs
            if not isinstance(weights, pd.Series) or not isinstance(covariance_matrix, pd.DataFrame):
                logger.error("Weights must be a Series and covariance_matrix must be a DataFrame")
                return np.nan
                
            # Align indices
            common_tickers = list(set(weights.index) & set(covariance_matrix.index))
            if not common_tickers:
                logger.error("No common tickers between weights and covariance matrix")
                return np.nan
                
            # Filter to common tickers
            weights = weights[common_tickers]
            covariance_matrix = covariance_matrix.loc[common_tickers, common_tickers]
            
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            
            # Extract volatilities (standard deviations)
            volatilities = np.sqrt(np.diag(covariance_matrix))
            
            # Calculate weighted average of individual volatilities
            weighted_avg_vol = np.sum(weights * volatilities)
            
            # Calculate portfolio volatility
            portfolio_variance = weights.dot(covariance_matrix).dot(weights)
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate diversification ratio
            div_ratio = weighted_avg_vol / portfolio_volatility
            
            return div_ratio
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {str(e)}")
            return np.nan


# Example usage when run as script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    
    from src.data_loading.data_loader import DataLoader
    from src.data_loading.data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    data_dir = Path("../../data")
    loader = DataLoader(data_dir)
    portfolio_df, price_df = loader.load_all_data()
    
    if portfolio_df is not None and price_df is not None:
        preprocessor = DataPreprocessor()
        cleaned_portfolio = preprocessor.preprocess_portfolio(portfolio_df)
        cleaned_prices = preprocessor.preprocess_price_data(price_df)
        
        # Filter to one year of data for demonstration
        end_date = cleaned_prices['Date'].max()
        start_date = end_date - timedelta(days=365)
        recent_prices = cleaned_prices[cleaned_prices['Date'] >= start_date]
        
        # Initialize risk metrics calculator
        risk_calc = RiskMetrics()
        
        # Create a benchmark series (using portfolio average return as proxy)
        benchmark_returns = recent_prices.groupby('Date')['Returns'].mean()
        risk_calc.set_benchmark_data('PORTFOLIO_AVG', benchmark_returns)
        
        # Calculate metrics for a few stocks
        print("\n--- Risk Metrics Examples ---")
        for ticker in cleaned_portfolio['Ticker'].head(3):
            # Get stock returns
            ticker_data = recent_prices[recent_prices['Ticker'] == ticker]
            stock_returns = pd.Series(ticker_data['Returns'].values, index=ticker_data['Date'])
            
            # Calculate various metrics
            volatility = risk_calc.calculate_volatility(stock_returns)
            sharpe = risk_calc.calculate_sharpe_ratio(stock_returns)
            beta = risk_calc.calculate_beta(stock_returns, 'PORTFOLIO_AVG')
            max_dd = risk_calc.calculate_max_drawdown(stock_returns)
            
            print(f"\nMetrics for {ticker}:")
            print(f"Annualized Volatility: {volatility:.4f}")
            print(f"Sharpe Ratio: {sharpe:.4f}")
            print(f"Beta: {beta:.4f}")
            print(f"Maximum Drawdown: {max_dd:.4f}")
            
        # Calculate portfolio risk contribution
        print("\n--- Portfolio Risk Contribution ---")
        # Create weight series
        weights = pd.Series(
            dict(zip(cleaned_portfolio['Ticker'], cleaned_portfolio['Weight'] / 100))
        )
        
        # Create return series for correlation calculation
        returns_by_ticker = {}
        for ticker in cleaned_portfolio['Ticker']:
            ticker_returns = recent_prices[recent_prices['Ticker'] == ticker]['Returns']
            if not ticker_returns.empty:
                returns_by_ticker[ticker] = ticker_returns.reset_index(drop=True)
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_by_ticker)
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        # Calculate risk contribution
        risk_contrib = risk_calc.calculate_portfolio_risk_contribution(weights, cov_matrix)
        
        print("Top 5 risk contributors:")
        print(risk_contrib.sort_values(ascending=False).head(5))
        
        # Calculate diversification ratio
        div_ratio = risk_calc.calculate_portfolio_diversification_ratio(weights, cov_matrix)
        print(f"\nPortfolio Diversification Ratio: {div_ratio:.4f}")