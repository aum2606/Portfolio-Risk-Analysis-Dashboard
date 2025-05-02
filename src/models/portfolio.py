"""
Portfolio class and related utilities for portfolio risk analysis.

This module provides a Portfolio class that handles portfolio operations such as
asset management, rebalancing, and historical performance calculation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class Portfolio:
    """Class representing an investement portfolio with various analysis capabilities"""
    def __init__(
        self,
        holdings_df:Optional[pd.DataFrame] = None,
        price_df:Optional[pd.DataFrame]=None,
        log_level:int = logging.INFO
    ):
        """
        Initialize a Portfolio instance.

        Args:
            holdings_df: DataFrame containing portfolio holdings
            price_df: DataFrame containing historical price data
            log_level: Logging level (default: logging.INFO)
        """
        self._configure_logging(log_level)
        logger.info("Initializing Portfolio")
        
        self.holdings = holdings_df
        self.price_data = price_df
        self.analysis_data = None
        self.performance_data = None
        
        # Initialize with current holdings data if available
        if holdings_df is not None:
            self._validate_holdings()
            self._calculate_portfolio_value()
            
        # Initialize with price data if available
        if price_df is not None:
            self._validate_price_data()
            
        # Additional initialization if both are available
        if holdings_df is not None and price_df is not None:
            self._create_analysis_data()

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

    def _validate_holdings(self) -> None:
        """Validate the portfolio holdings data structure."""
        if self.holdings is None:
            logger.warning("No holdings data available")
            return
        required_columns = ["Ticker","Quantity","Close","Weight"]
        missing_columns = [col for col in required_columns if col not in self.holdings.columns]
        
        if missing_columns:
            logger.error(f"Holdings data missing required columns: {missing_columns}")
            return
        
        #check for missing values
        if self.holdings[required_columns].isna().any().any():
            logger.warning("Holdings data contains missing values")
            
    def _validate_price_data(self)->None:
        """Validate the historical price data structure"""
        if self.price_data is None:
            logger.warning("No price data available")
            return
        required_columns = ["Date", "Ticker", "Close", "Adjusted", "Returns"]
        missing_columns = [col for col in required_columns if col not in self.price_data.columns]
        
        if missing_columns:
            logger.error(f"Price data missing required columns: {missing_columns}")
            return

        #ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.price_data['Date']):
            logger.warning("converting date column to datetime")
            self.price_data['Date']=pd.to_datetime(self.price_data['Date'])
            
        #check data range
        start_date = self.price_data['Date'].min()
        end_date = self.price_data['Date'].max()
        logger.info(f"Price data covers period: {start_date} to {end_date}")
        
    def _calculate_portfolio_value(self)->None:
        """Calculate portfolio value and update stats"""
        if self.holdings is None:
            logger.warning("Cannot calculate portfolio value - no holdings data")
            return
        
        if 'Value' not in self.holdings.columns:
            self.holdings['Value'] = self.holdings['Quantity']  * self.holdings['Close']
            
        self.total_value = self.holdings['Value'].sum()
        
        #recalculate weights if needed
        if abs(self.holdings['Weight'].sum()-100) > 1:
            logger.warning("Portfolio weights do not sum to 100%, recalculating")
            self.holdings['Weight'] = self.holdings['Value']/self.total_value * 100
            
        logger.info(f"Porfolio total value: {self.total_value:.2f}")
        
    def _create_analysis_data(self) -> None:
        """Create analysis data combining holdings and price history."""
        if self.holdings is None or self.price_data is None:
            logger.warning("Cannot create analysis data - missing holdings or price data")
            return
            
        # Get tickers from holdings
        portfolio_tickers = set(self.holdings['Ticker'])
        
        # Filter price data to only include portfolio tickers
        filtered_prices = self.price_data[self.price_data['Ticker'].isin(portfolio_tickers)].copy()
        
        if filtered_prices.empty:
            logger.error("No price data found for portfolio tickers")
            return
            
        # Create a ticker -> quantity mapping
        quantity_dict = dict(zip(self.holdings['Ticker'], self.holdings['Quantity']))
        
        # Add quantity column to price data
        filtered_prices['Quantity'] = filtered_prices['Ticker'].map(quantity_dict)
        
        # Calculate position values
        filtered_prices['Position Value'] = filtered_prices['Adjusted'] * filtered_prices['Quantity']
        
        # Create analysis data
        self.analysis_data = filtered_prices
        
        logger.info(f"Created analysis data with {len(self.analysis_data)} records")

    def set_holdings(self, holdings_df: pd.DataFrame) -> None:
        """
        Set or update portfolio holdings.

        Args:
            holdings_df: DataFrame containing portfolio holdings
        """
        self.holdings = holdings_df
        self._validate_holdings()
        self._calculate_portfolio_value()
        
        # Update analysis data if price data is available
        if self.price_data is not None:
            self._create_analysis_data()
            
        logger.info("Portfolio holdings updated")

    def set_price_data(self, price_df: pd.DataFrame) -> None:
        """
        Set or update historical price data.

        Args:
            price_df: DataFrame containing historical price data
        """
        self.price_data = price_df
        self._validate_price_data()
        
        # Update analysis data if holdings are available
        if self.holdings is not None:
            self._create_analysis_data()
            
        logger.info("Price data updated")

    def rebalance(
        self, 
        target_weights: Optional[Dict[str, float]] = None,
        target_allocation: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Rebalance the portfolio to target weights or allocation.

        Args:
            target_weights: Dictionary of {ticker: weight} for target weights
            target_allocation: Dictionary of {sector: allocation} for sector allocation

        Returns:
            DataFrame with updated holdings after rebalancing
        """
        if self.holdings is None:
            logger.error("Cannot rebalance - no holdings data")
            return pd.DataFrame()
            
        if target_weights is None and target_allocation is None:
            logger.error("Must provide either target_weights or target_allocation")
            return self.holdings
            
        # Create a copy of holdings for rebalancing
        new_holdings = self.holdings.copy()
        
        try:
            if target_weights is not None:
                # Rebalance based on ticker weights
                logger.info("Rebalancing based on target ticker weights")
                
                # Validate target weights
                total_weight = sum(target_weights.values())
                if abs(total_weight - 100) > 1:
                    logger.warning(f"Target weights sum to {total_weight}, normalizing to 100%")
                    target_weights = {k: v/total_weight*100 for k, v in target_weights.items()}
                
                # Update weights for existing holdings
                for ticker, weight in target_weights.items():
                    if ticker in new_holdings['Ticker'].values:
                        idx = new_holdings[new_holdings['Ticker'] == ticker].index[0]
                        new_holdings.loc[idx, 'Weight'] = weight
                
                # Calculate new position values
                new_holdings['Value'] = new_holdings['Weight'] / 100 * self.total_value
                
                # Update quantities based on new values
                new_holdings['Quantity'] = new_holdings['Value'] / new_holdings['Close']
                
                # Round quantities to whole shares
                new_holdings['Quantity'] = new_holdings['Quantity'].round()
                
                # Recalculate based on rounded quantities
                new_holdings['Value'] = new_holdings['Quantity'] * new_holdings['Close']
                new_holdings['Weight'] = new_holdings['Value'] / new_holdings['Value'].sum() * 100
                
            elif target_allocation is not None:
                # Rebalance based on sector allocation
                logger.info("Rebalancing based on target sector allocation")
                
                # Validate target allocation
                total_allocation = sum(target_allocation.values())
                if abs(total_allocation - 100) > 1:
                    logger.warning(f"Target allocation sums to {total_allocation}, normalizing to 100%")
                    target_allocation = {k: v/total_allocation*100 for k, v in target_allocation.items()}
                
                # Calculate current sector allocations
                current_sectors = new_holdings.groupby('Sector')['Value'].sum()
                
                # Calculate scaling factor for each sector
                sector_scaling = {}
                for sector, target in target_allocation.items():
                    if sector in current_sectors.index:
                        current_pct = current_sectors[sector] / self.total_value * 100
                        sector_scaling[sector] = target / current_pct
                    else:
                        logger.warning(f"Sector {sector} not found in current holdings")
                
                # Apply scaling to each holding based on its sector
                for idx, row in new_holdings.iterrows():
                    sector = row['Sector']
                    if sector in sector_scaling:
                        # Scale the weight
                        new_holdings.loc[idx, 'Weight'] = row['Weight'] * sector_scaling[sector]
                
                # Normalize weights to ensure they sum to 100%
                new_holdings['Weight'] = new_holdings['Weight'] / new_holdings['Weight'].sum() * 100
                
                # Calculate new values and quantities
                new_holdings['Value'] = new_holdings['Weight'] / 100 * self.total_value
                new_holdings['Quantity'] = new_holdings['Value'] / new_holdings['Close']
                
                # Round quantities to whole shares
                new_holdings['Quantity'] = new_holdings['Quantity'].round()
                
                # Recalculate based on rounded quantities
                new_holdings['Value'] = new_holdings['Quantity'] * new_holdings['Close']
                new_holdings['Weight'] = new_holdings['Value'] / new_holdings['Value'].sum() * 100
            
            logger.info("Portfolio rebalancing completed")
            return new_holdings
            
        except Exception as e:
            logger.error(f"Error during portfolio rebalancing: {str(e)}")
            return self.holdings
        
    def calculate_historical_performance(
        self,
        start_date: Optional[Union[str,datetime]] = None,
        end_date: Optional[Union[str,datetime]] = None
    )->pd.DataFrame:
        """ 
        Calculate historical portfolio performance.

        Args:
            start_date: Optional start date for analysis period
            end_date: Optional end date for analysis period

        Returns:
            DataFrame with historical performance metrics
        """
        if self.analysis_data is None:
            logger.error("Cannot calculate performance - no analysis data available")
            return pd.DataFrame()
        
        logger.info("Calculating historical portfolio performance")
        
        try:
            #filter data by data range if provided
            data = self.analysis_data.copy()
            
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                data = data[data['Date']>=start_date]
                
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                data = data[data['Date']<=end_date]
                
            if data.empty:
                logger.warning("No data available for specified date range")
                return pd.DataFrame()
            
            #calculate daily portfolio values
            daily_values = data.groupby('Date')['Position Value'].sum().reset_index()
            daily_values.columns = ['Date','Portfolio Value']
            
            #calculate daily returns
            daily_values = data.groupby('Date')['Position Value'].sum().reset_index()
            daily_values.columns = ['Date','Portfolio Value']
            
            #calculate daily returns
            daily_values['Daily Return'] = daily_values['Portfolio Value'].pct_change()
            
            # Calculate cumulative return
            daily_values['Cumulative Return'] = (1 + daily_values['Daily Return'].fillna(0)).cumprod() - 1
            
            # Calculate additional metrics
            # Rolling volatility (annualized)
            daily_values['Rolling Volatility'] = daily_values['Daily Return'].rolling(window=20).std() * np.sqrt(252)
            
            # Rolling Sharpe ratio (annualized, assuming 0% risk-free rate for simplicity)
            daily_values['Rolling Sharpe'] = (daily_values['Daily Return'].rolling(window=20).mean() * 252) / \
                                            (daily_values['Daily Return'].rolling(window=20).std() * np.sqrt(252))
            
            #dropdown calculation
            portfolio_value = daily_values['Portfolio Value']
            peak = portfolio_value.expanding().max()
            daily_values['Drawdown'] = (portfolio_value/peak)-1
            
            #store results
            self.performance_data = daily_values
            logger.info(f"Calculated performance data for {len(daily_values)} trading days")
            return daily_values
                        
        except Exception as e:
            logger.error(f"Error calculating historical performance: {str(e)}")
            return pd.DataFrame()
        
    def get_asset_allocation(self) -> Dict[str, pd.DataFrame]:
        """
        Get current asset allocation by different dimensions.

        Returns:
            Dictionary containing different allocation views
        """
        if self.holdings is None:
            logger.error("Cannot get asset allocation - no holdings data")
            return {}
        
        try:
            #by sector 
            sector_allocation = self.holdings.groupby('Sector').agg({
                'Value':'sum',
                'Weight':'sum'
            }).sort_values('Weight',ascending=False)
            
            #by ticker
            ticker_allocation = self.holdings[['Ticker','Value','Weight']].sort_values('Weight',ascending=False)
            
            return {
                'by_sector':sector_allocation,
                'by_ticker':ticker_allocation
            }
            
        except Exception as e:
            logger.error(f"Error getting asset allocation: {str(e)}")
            return {}

    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        Get current portfolio statistics.

        Returns:
            Dictionary of portfolio statistics
        """
        stats = {}
        
        if self.holdings is None:
            logger.error("Cannot calculate stats - no holdings data")
            return stats

        try:
            #basic portfolio stats
            stats['total_value'] = self.total_value
            stats['num_holdings'] = len(self.holdings)
            
            #get largest holding
            largest = self.holdings.loc[self.holdings['Weight'].idxmax()]
            stats['largest_holding'] = largest['Ticker']
            stats['largets_weight'] = largest['Weight']
            
            #calculate concentration metrics
            top_5_weight = self.holdings.nlargest(5,'Weight')['Weight'].sum()
            stats['top_5_concentration'] = top_5_weight
            
            #calculate sector metrics
            sector_weights = self.holdings.groupby('Sector')['Weight'].sum()
            stats['num_sectors'] = len(sector_weights)
            stats['largest_sector'] = sector_weights.idxmax()
            stats['largest_sector_weight'] = sector_weights.max()

            #add historical performance if available
            if self.performance_data is not None:
                #latest performance values
                latest = self.performance_data.iloc[-1]
                stats['current_value'] = latest['Portfolio Value']
                stats['ytd_return'] = self.performance_data.iloc[-1]['Cumulative Return']
                stats['current_drawdown'] = latest['Drawdown']
                
                # Volatility and Sharpe ratio metrics
                if len(self.performance_data) >= 20:  # Need at least 20 days for rolling metrics
                    stats['current_volatility'] = latest['Rolling Volatility']
                    stats['current_sharpe'] = latest['Rolling Sharpe']
            
            return stats

        except Exception as e:
            logger.error(f"Error calculating portfolio stats: {str(e)}")
            return {}

    def get_what_if_analysis(
        self, 
        scenario: str = 'market_crash',
        custom_returns: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Run what-if scenario analysis on the portfolio.

        Args:
            scenario: Predefined scenario type ('market_crash', 'recession', 'recovery', etc.)
            custom_returns: Custom return impacts by ticker or sector

        Returns:
            DataFrame with scenario impact analysis
        """
        if self.holdings is None:
            logger.error("Cannot run what-if analysis - no holdings data")
            return pd.DataFrame()
        
        try:
            #define scenario impacts
            scenario_impacts = {
                'market_crash': {
                    'default': -0.30,  # 30% drop by default
                    'sector_impacts': {
                        'Technology': -0.40,
                        'Healthcare': -0.20,
                        'Utilities': -0.15,
                        'Consumer Staples': -0.15
                    }
                },
                'recovery': {
                    'default': 0.15,  # 15% gain by default
                    'sector_impacts': {
                        'Technology': 0.25,
                        'Consumer Discretionary': 0.20,
                        'Financials': 0.20,
                        'Utilities': 0.05
                    }
                },
                'inflation': {
                    'default': -0.10,  # 10% drop by default
                    'sector_impacts': {
                        'Real Estate': 0.05,
                        'Energy': 0.10,
                        'Materials': 0.05,
                        'Technology': -0.20,
                        'Consumer Discretionary': -0.15
                    }
                },
                'custom': {
                    'default': 0.0,  # No impact by default
                    'sector_impacts': {}
                }
            }
            #select scenario or use custom
            impact = scenario_impacts.get(scenario,scenario_impacts['custom'])
            #apply custom returns if provided
            if custom_returns and scenario == 'custom':
                for key,value in custom_returns.items():
                    impact['sector_impacts'][key] = value
                    
            #create analysis dataframe
            what_if_df = self.holdings.copy()
            
            #apply impacts
            what_if_df['Impact'] = what_if_df['Sector'].map(
                lambda x:impact['sector_impacts'].get(x,impact['default'])
            )
            
            #calculate new values
            what_if_df['New Value'] = what_if_df['Value'] * (1 + what_if_df['Impact'])
            what_if_df['Value Change'] = what_if_df['New Value'] - what_if_df['Value']
            what_if_df['New Weight'] = what_if_df['New Value'] / what_if_df['New Value'].sum() * 100
            what_if_df['Weight Change'] = what_if_df['New Weight'] - what_if_df['Weight']
            
            # Calculate totals
            total_value = what_if_df['Value'].sum()
            new_total_value = what_if_df['New Value'].sum()
            total_impact = (new_total_value / total_value) - 1

            # Add summary row
            summary = pd.DataFrame({
                'Ticker': ['TOTAL'],
                'Sector': [''],
                'Value': [total_value],
                'Weight': [100.0],
                'Impact': [total_impact],
                'New Value': [new_total_value],
                'Value Change': [new_total_value - total_value],
                'New Weight': [100.0],
                'Weight Change': [0.0]
            })
            
            # Combine with main data
            result = pd.concat([what_if_df, summary], ignore_index=True)
            
            logger.info(f"Completed {scenario} what-if analysis: overall impact {total_impact:.2%}")
            
            return result
            
            
        except Exception as e:
            logger.error(f"Error in what - if analysis: {str(e)}")
            return pd.DataFrame()
        
# Example usage when run as script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

    from src.data_loading.data_loader import DataLoader
    from src.data_loading.data_preprocessing import DataPreprocessor
    
    #load and preprocess data
    data_dir = Path('../../data')
    loader = DataLoader(data_dir)
    portfolio_df,price_df = loader.load_all_data()
    if portfolio_df is not None and price_df is not None:
        preprocessor = DataPreprocessor()
        cleaned_portfolio = preprocessor.preprocess_portfolio(portfolio_df)
        cleaned_prices = preprocessor.preprocess_price_data(price_df)
        
        # Create portfolio
        portfolio = Portfolio(cleaned_portfolio, cleaned_prices)
        
        # Calculate historical performance
        performance = portfolio.calculate_historical_performance()
        
        # Print portfolio stats
        stats = portfolio.get_portfolio_stats()
        print("\n--- Portfolio Statistics ---")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        # Print asset allocation
        allocation = portfolio.get_asset_allocation()
        print("\n--- Sector Allocation ---")
        print(allocation['by_sector'])
        
        # What-if analysis
        print("\n--- Market Crash Scenario ---")
        scenario = portfolio.get_what_if_analysis('market_crash')
        print(scenario[['Ticker', 'Sector', 'Value', 'New Value', 'Impact']].head())
