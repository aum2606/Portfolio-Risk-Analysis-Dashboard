"""
Chart generation functions for portfolio risk dashboard.

This module provides functions to create visualizations for portfolio analysis,
including performance charts, risk metrics, and allocation breakdowns.
"""

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


def configure_logging(log_level: int = logging.INFO) -> None:
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


def create_performance_chart(
    performance_data: pd.DataFrame,
    title: str = "Portfolio Performance",
    height: int = 600
) -> go.Figure:
    """
    Create a performance chart showing portfolio value, returns, and drawdowns.

    Args:
        performance_data: DataFrame with historical performance data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if required columns exist
        required_cols = ['Date', 'Portfolio Value', 'Cumulative Return', 'Drawdown']
        missing_cols = [col for col in required_cols if col not in performance_data.columns]
        
        if missing_cols:
            logger.error(f"Performance data missing required columns: {missing_cols}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: Missing columns - {', '.join(missing_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Portfolio Value", "Cumulative Return", "Drawdown"),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Add portfolio value line
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'], 
                y=performance_data['Portfolio Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='royalblue', width=2)
            ),
            row=1, col=1
        )
        
        # Add cumulative return line
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'], 
                y=performance_data['Cumulative Return'] * 100,  # Convert to percentage
                mode='lines',
                name='Cumulative Return',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Add zero line for reference
        fig.add_trace(
            go.Scatter(
                x=[performance_data['Date'].min(), performance_data['Date'].max()],
                y=[0, 0],
                mode='lines',
                name='Break-even',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        # Add drawdown area
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'], 
                y=performance_data['Drawdown'] * 100,  # Convert to percentage
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        # Update x-axis for bottom plot
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_risk_metrics_chart(
    performance_data: pd.DataFrame,
    title: str = "Risk Metrics Over Time",
    height: int = 500
) -> go.Figure:
    """
    Create a chart showing risk metrics over time.

    Args:
        performance_data: DataFrame with historical performance and risk data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if required columns exist
        required_cols = ['Date', 'Rolling Volatility', 'Rolling Sharpe']
        missing_cols = [col for col in required_cols if col not in performance_data.columns]
        
        if missing_cols:
            logger.error(f"Performance data missing required columns: {missing_cols}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: Missing columns - {', '.join(missing_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Rolling Volatility (Annualized)", "Rolling Sharpe Ratio"),
            row_heights=[0.5, 0.5]
        )
        
        # Add rolling volatility line
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'], 
                y=performance_data['Rolling Volatility'] * 100,  # Convert to percentage
                mode='lines',
                name='Volatility',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        # Add rolling Sharpe ratio line
        fig.add_trace(
            go.Scatter(
                x=performance_data['Date'], 
                y=performance_data['Rolling Sharpe'],
                mode='lines',
                name='Sharpe Ratio',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Add zero line for Sharpe
        fig.add_trace(
            go.Scatter(
                x=[performance_data['Date'].min(), performance_data['Date'].max()],
                y=[0, 0],
                mode='lines',
                name='Zero',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        
        # Update x-axis for bottom plot
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk metrics chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_asset_allocation_chart(
    holdings_data: pd.DataFrame,
    group_by: str = 'Sector',
    title: str = "Asset Allocation",
    height: int = 500
) -> go.Figure:
    """
    Create a pie chart showing asset allocation.

    Args:
        holdings_data: DataFrame with portfolio holdings
        group_by: Column to group by ('Sector' or 'Ticker')
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if required columns exist
        required_cols = [group_by, 'Value', 'Weight']
        missing_cols = [col for col in required_cols if col not in holdings_data.columns]
        
        if missing_cols:
            logger.error(f"Holdings data missing required columns: {missing_cols}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: Missing columns - {', '.join(missing_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Group data
        if group_by == 'Ticker':
            # Already at ticker level, just need to select columns
            grouped_data = holdings_data[['Ticker', 'Value', 'Weight']].copy()
            labels = grouped_data['Ticker']
        else:
            # Group by the specified column (e.g., Sector)
            grouped_data = holdings_data.groupby(group_by).agg({
                'Value': 'sum',
                'Weight': 'sum'
            }).reset_index()
            labels = grouped_data[group_by]
        
        # Create pie chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=grouped_data['Weight'],
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>Weight: %{value:.2f}%<br>Value: $%{customdata:,.2f}<extra></extra>',
                    customdata=grouped_data['Value'],
                    hole=0.4
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating asset allocation chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_risk_contribution_chart(
    risk_contribution_data: pd.DataFrame,
    title: str = "Risk Contribution",
    height: int = 500
) -> go.Figure:
    """
    Create a chart showing risk contribution by asset.

    Args:
        risk_contribution_data: DataFrame with risk contribution data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if required columns exist
        required_cols = ['Ticker', 'RiskContribution']
        missing_cols = [col for col in required_cols if col not in risk_contribution_data.columns]
        
        if missing_cols:
            logger.error(f"Risk contribution data missing required columns: {missing_cols}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: Missing columns - {', '.join(missing_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort data by risk contribution (descending)
        sorted_data = risk_contribution_data.sort_values('RiskContribution', ascending=True)
        
        # Create bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_data['RiskContribution'],
                    y=sorted_data['Ticker'],
                    orientation='h',
                    marker=dict(
                        color=sorted_data['RiskContribution'],
                        colorscale='Reds'
                    )
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Risk Contribution (%)",
            yaxis_title="",
            xaxis=dict(ticksuffix="%")
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk contribution chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_correlation_heatmap(
    returns_data: pd.DataFrame,
    title: str = "Asset Correlation Heatmap",
    height: int = 700
) -> go.Figure:
    """
    Create a correlation heatmap for asset returns.

    Args:
        returns_data: DataFrame with returns data (time series, columns=tickers)
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        # Create heatmap
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=np.around(corr_matrix.values, decimals=2),
                    texttemplate="%{text:.2f}",
                    hovertemplate='%{x} & %{y}<br>Correlation: %{z:.4f}<extra></extra>'
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed')
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_risk_return_scatter(
    risk_return_data: pd.DataFrame,
    title: str = "Risk-Return Profile",
    height: int = 600
) -> go.Figure:
    """
    Create a risk-return scatter plot.

    Args:
        risk_return_data: DataFrame with risk and return metrics by asset
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if required columns exist
        required_cols = ['Ticker', 'Return', 'Volatility', 'Sharpe']
        missing_cols = [col for col in required_cols if col not in risk_return_data.columns]
        
        if missing_cols:
            logger.error(f"Risk-return data missing required columns: {missing_cols}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: Missing columns - {', '.join(missing_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create scatter plot
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=risk_return_data['Volatility'] * 100,  # Convert to percentage
                    y=risk_return_data['Return'] * 100,      # Convert to percentage
                    mode='markers+text',
                    text=risk_return_data['Ticker'],
                    textposition="top center",
                    marker=dict(
                        size=risk_return_data['Sharpe'] * 5 + 10,  # Scale bubble size
                        color=risk_return_data['Sharpe'],
                        colorscale='Viridis',
                        colorbar=dict(title="Sharpe Ratio"),
                        showscale=True
                    ),
                    hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>'
                )
            ]
        )
        
        # Add zero return line
        fig.add_hline(
            y=0, 
            line_width=1, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Zero Return",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Volatility (%)",
            yaxis_title="Return (%)",
            xaxis=dict(ticksuffix="%"),
            yaxis=dict(ticksuffix="%")
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk-return scatter: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating scatter plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_rolling_beta_chart(
    beta_data: pd.DataFrame,
    benchmark_name: str = "Benchmark",
    title: str = "Rolling Beta",
    height: int = 500
) -> go.Figure:
    """
    Create a chart showing rolling beta for portfolio assets.

    Args:
        beta_data: DataFrame with Date index and ticker columns containing beta values
        benchmark_name: Name of the benchmark
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add beta line for each ticker
        for column in beta_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=beta_data.index,
                    y=beta_data[column],
                    mode='lines',
                    name=column
                )
            )
        
        # Add beta=1 reference line
        fig.add_trace(
            go.Scatter(
                x=[beta_data.index.min(), beta_data.index.max()],
                y=[1, 1],
                mode='lines',
                name=f'Same as {benchmark_name}',
                line=dict(color='gray', width=1, dash='dash')
            )
        )
        
        # Add beta=0 reference line
        fig.add_trace(
            go.Scatter(
                x=[beta_data.index.min(), beta_data.index.max()],
                y=[0, 0],
                mode='lines',
                name='No correlation',
                line=dict(color='lightgray', width=1, dash='dot')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Date",
            yaxis_title=f"Beta (vs {benchmark_name})",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating rolling beta chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_comparative_performance_chart(
    performance_data: pd.DataFrame,
    reference_cols: List[str],
    title: str = "Comparative Performance",
    height: int = 500
) -> go.Figure:
    """
    Create a chart comparing multiple performance series.

    Args:
        performance_data: DataFrame with Date index and performance columns
        reference_cols: List of column names to include in chart
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if required columns exist
        missing_cols = [col for col in reference_cols if col not in performance_data.columns]
        
        if missing_cols:
            logger.error(f"Performance data missing required columns: {missing_cols}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: Missing columns - {', '.join(missing_cols)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Add line for each performance series
        for column in reference_cols:
            fig.add_trace(
                go.Scatter(
                    x=performance_data.index if isinstance(performance_data.index, pd.DatetimeIndex) else performance_data['Date'],
                    y=performance_data[column] * 100,  # Convert to percentage
                    mode='lines',
                    name=column
                )
            )
        
        # Add zero reference line
        fig.add_trace(
            go.Scatter(
                x=[performance_data.index.min(), performance_data.index.max()] 
                  if isinstance(performance_data.index, pd.DatetimeIndex) 
                  else [performance_data['Date'].min(), performance_data['Date'].max()],
                y=[0, 0],
                mode='lines',
                name='Break-even',
                line=dict(color='gray', width=1, dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            yaxis=dict(ticksuffix="%"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comparative performance chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_what_if_chart(
    base_holdings: pd.DataFrame,
    scenario_holdings: pd.DataFrame,
    scenario_name: str = "Scenario",
    sort_by: str = 'Value Change',
    title: str = "What-If Analysis",
    height: int = 600
) -> go.Figure:
    """
    Create a chart comparing base portfolio holdings with scenario results.

    Args:
        base_holdings: DataFrame with base portfolio holdings
        scenario_holdings: DataFrame with scenario portfolio holdings
        scenario_name: Name of the scenario
        sort_by: Column to sort by ('Value Change' or 'Weight Change')
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Check if required columns exist
        base_required_cols = ['Ticker', 'Value', 'Weight']
        scenario_required_cols = ['Ticker', 'New Value', 'New Weight', 'Value Change', 'Weight Change']
        
        base_missing_cols = [col for col in base_required_cols if col not in base_holdings.columns]
        scenario_missing_cols = [col for col in scenario_required_cols if col not in scenario_holdings.columns]
        
        if base_missing_cols or scenario_missing_cols:
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: Missing columns in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Merge data
        merged_data = pd.merge(
            base_holdings[['Ticker', 'Value', 'Weight']],
            scenario_holdings[['Ticker', 'New Value', 'New Weight', 'Value Change', 'Weight Change']],
            on='Ticker',
            how='inner'
        )
        
        # Calculate percentage changes
        merged_data['Value Change %'] = merged_data['Value Change'] / merged_data['Value'] * 100
        
        # Filter out TOTAL row if it exists
        merged_data = merged_data[merged_data['Ticker'] != 'TOTAL']
        
        # Sort by specified column
        if sort_by == 'Value Change':
            merged_data = merged_data.sort_values('Value Change', ascending=True)
        else:
            merged_data = merged_data.sort_values('Weight Change', ascending=True)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"Value Change in {scenario_name} Scenario", f"Weight Change in {scenario_name} Scenario"),
            row_heights=[0.5, 0.5]
        )
        
        # Add Value Change bar chart
        fig.add_trace(
            go.Bar(
                x=merged_data['Ticker'],
                y=merged_data['Value Change'],
                name='Value Change ($)',
                marker_color=np.where(merged_data['Value Change'] >= 0, 'green', 'red')
            ),
            row=1, col=1
        )
        
        # Add Value Change % as text
        for i, row in merged_data.iterrows():
            fig.add_annotation(
                x=row['Ticker'],
                y=row['Value Change'],
                text=f"{row['Value Change %']:.1f}%",
                showarrow=False,
                yshift=10 if row['Value Change'] >= 0 else -10,
                row=1, col=1
            )
        
        # Add Weight Change bar chart
        fig.add_trace(
            go.Bar(
                x=merged_data['Ticker'],
                y=merged_data['Weight Change'],
                name='Weight Change (%)',
                marker_color=np.where(merged_data['Weight Change'] >= 0, 'blue', 'orange')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Value Change ($)", row=1, col=1)
        fig.update_yaxes(title_text="Weight Change (%)", row=2, col=1)
        
        # Update x-axis for bottom plot
        fig.update_xaxes(title_text="Ticker", row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating what-if chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_risk_metrics_comparison_chart(
    risk_metrics: Dict[str, float],
    benchmark_metrics: Optional[Dict[str, float]] = None,
    title: str = "Risk Metrics Comparison",
    height: int = 500
) -> go.Figure:
    """
    Create a radar chart comparing portfolio risk metrics.

    Args:
        risk_metrics: Dictionary of risk metrics for the portfolio
        benchmark_metrics: Optional dictionary of risk metrics for a benchmark
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Define metrics to include in the radar chart
        radar_metrics = [
            'sharpe_ratio',
            'sortino_ratio',
            'information_ratio',
            'diversification_ratio',
            'return_to_risk'
        ]
        
        # Prepare data
        metrics_data = {}
        
        # Check which metrics are available
        available_metrics = [m for m in radar_metrics if m in risk_metrics]
        
        if not available_metrics:
            logger.error("No valid metrics available for radar chart")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text="Error: No valid metrics available for radar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Normalize metrics to 0-1 scale
        max_values = {}
        for metric in available_metrics:
            portfolio_value = risk_metrics.get(metric, 0)
            benchmark_value = benchmark_metrics.get(metric, 0) if benchmark_metrics else 0
            max_values[metric] = max(abs(portfolio_value), abs(benchmark_value), 0.0001)
        
        # Prepare portfolio data
        portfolio_values = []
        for metric in available_metrics:
            value = risk_metrics.get(metric, 0)
            # Normalize to 0-1 range, ensuring higher is better
            normalized = value / max_values[metric]
            # Handle negative values (e.g., negative Sharpe)
            if value < 0:
                normalized = 0
            portfolio_values.append(normalized)
        
        # Prepare benchmark data if available
        benchmark_values = []
        if benchmark_metrics:
            for metric in available_metrics:
                value = benchmark_metrics.get(metric, 0)
                # Normalize to 0-1 range, ensuring higher is better
                normalized = value / max_values[metric]
                # Handle negative values
                if value < 0:
                    normalized = 0
                benchmark_values.append(normalized)
        
        # Create figure
        fig = go.Figure()
        
        # Add portfolio trace
        fig.add_trace(
            go.Scatterpolar(
                r=portfolio_values,
                theta=available_metrics,
                fill='toself',
                name='Portfolio'
            )
        )
        
        # Add benchmark trace if available
        if benchmark_metrics:
            fig.add_trace(
                go.Scatterpolar(
                    r=benchmark_values,
                    theta=available_metrics,
                    fill='toself',
                    name='Benchmark'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk metrics comparison chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_returns_distribution_chart(
    returns_data: pd.DataFrame,
    title: str = "Returns Distribution",
    height: int = 500
) -> go.Figure:
    """
    Create a histogram showing the distribution of returns.

    Args:
        returns_data: DataFrame or Series of return values
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    try:
        # Extract returns data
        if isinstance(returns_data, pd.DataFrame):
            if 'Returns' in returns_data.columns:
                returns = returns_data['Returns']
            elif 'Daily Return' in returns_data.columns:
                returns = returns_data['Daily Return']
            else:
                # Use first column if specific return column not found
                returns = returns_data.iloc[:, 0]
        else:
            returns = returns_data
        
        # Calculate statistics
        mean_return = returns.mean()
        median_return = returns.median()
        std_dev = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Create histogram with normal distribution overlay
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=returns * 100,  # Convert to percentage
                histnorm='probability density',
                name='Returns',
                marker_color='blue',
                opacity=0.7
            )
        )
        
        # Add normal distribution curve
        x_range = np.linspace(
            min(returns * 100), 
            max(returns * 100), 
            100
        )
        from scipy.stats import norm
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=norm.pdf(x_range, mean_return * 100, std_dev * 100),
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            )
        )
        
        # Add mean and median lines
        fig.add_vline(
            x=mean_return * 100,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text="Mean",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=median_return * 100,
            line_width=2,
            line_dash="dot",
            line_color="orange",
            annotation_text="Median",
            annotation_position="top left"
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_title="Return (%)",
            yaxis_title="Density",
            xaxis=dict(ticksuffix="%"),
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Mean: {mean_return * 100:.2f}%<br>Std Dev: {std_dev * 100:.2f}%<br>Skew: {skewness:.2f}<br>Kurtosis: {kurtosis:.2f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating returns distribution chart: {str(e)}")
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


# Example usage when run as script
if __name__ == "__main__":
    print("This module provides chart generation functions for the portfolio dashboard.")
    print("Import and use these functions in your dashboard application.")