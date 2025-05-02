"""
Dashboard layout and callbacks for portfolio risk analysis.

This module defines the layout and interactive functionality for the portfolio
risk dashboard, using Dash components and callbacks.
"""

import logging
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from .charts import (
    create_performance_chart,
    create_risk_metrics_chart,
    create_asset_allocation_chart,
    create_risk_contribution_chart,
    create_correlation_heatmap,
    create_risk_return_scatter,
    create_rolling_beta_chart,
    create_comparative_performance_chart,
    create_what_if_chart,
    create_risk_metrics_comparison_chart,
    create_returns_distribution_chart
)

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


class PortfolioDashboard:
    """Dashboard for portfolio risk analysis."""

    def __init__(self, portfolio, risk_metrics):
        """
        Initialize the dashboard with portfolio and risk metrics.

        Args:
            portfolio: Portfolio instance with holdings and price data
            risk_metrics: RiskMetrics instance for calculating risk metrics
        """
        self.portfolio = portfolio
        self.risk_metrics = risk_metrics
        self.app = None
        self.layout = None
        
        # Set up logging
        configure_logging()
        logger.info("Initializing Portfolio Dashboard")

    def create_layout(self) -> html.Div:
        """
        Create the dashboard layout.

        Returns:
            Dash layout component
        """
        try:
            # Portfolio overview section
            overview_card = dbc.Card([
                dbc.CardHeader(html.H4("Portfolio Overview")),
                dbc.CardBody([
                    html.Div(id="portfolio-stats-container", className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Portfolio Value Over Time"),
                                dbc.CardBody([
                                    dcc.Graph(id="performance-chart")
                                ])
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
            
            # Asset allocation section
            allocation_card = dbc.Card([
                dbc.CardHeader(html.H4("Asset Allocation")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Sector Allocation"),
                                dbc.CardBody([
                                    dcc.Graph(id="sector-allocation-chart")
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Top Holdings"),
                                dbc.CardBody([
                                    dcc.Graph(id="holdings-allocation-chart")
                                ])
                            ])
                        ], width=6)
                    ])
                ])
            ], className="mb-4")
            
            # Risk metrics section
            risk_card = dbc.Card([
                dbc.CardHeader(html.H4("Risk Analysis")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Volatility & Sharpe Ratio"),
                                dbc.CardBody([
                                    dcc.Graph(id="risk-metrics-chart")
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Risk Contribution by Asset"),
                                dbc.CardBody([
                                    dcc.Graph(id="risk-contribution-chart")
                                ])
                            ])
                        ], width=6)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Risk-Return Profile"),
                                dbc.CardBody([
                                    dcc.Graph(id="risk-return-chart")
                                ])
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
            
            # Correlation analysis section
            correlation_card = dbc.Card([
                dbc.CardHeader(html.H4("Correlation Analysis")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Asset Correlation Heatmap"),
                                dbc.CardBody([
                                    dcc.Graph(id="correlation-heatmap")
                                ])
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
            
            # What-if analysis section
            whatif_card = dbc.Card([
                dbc.CardHeader(html.H4("What-If Analysis")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Scenario:"),
                            dcc.Dropdown(
                                id="scenario-dropdown",
                                options=[
                                    {"label": "Market Crash (-30%)", "value": "market_crash"},
                                    {"label": "Recession (-20%)", "value": "recession"},
                                    {"label": "Recovery (+15%)", "value": "recovery"},
                                    {"label": "Inflation Spike", "value": "inflation"},
                                    {"label": "Custom", "value": "custom"}
                                ],
                                value="market_crash"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Div(id="custom-scenario-container", className="d-none")
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Scenario Impact"),
                                dbc.CardBody([
                                    dcc.Graph(id="what-if-chart")
                                ])
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
            
            # Portfolio rebalancing section
            rebalance_card = dbc.Card([
                dbc.CardHeader(html.H4("Portfolio Rebalancing")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Rebalance Method:"),
                            dcc.RadioItems(
                                id="rebalance-method",
                                options=[
                                    {"label": "By Asset Weights", "value": "weights"},
                                    {"label": "By Sector Allocation", "value": "sectors"}
                                ],
                                value="weights",
                                inline=True
                            )
                        ], width=12)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="rebalance-weights-container")
                        ], width=12)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                "Rebalance Portfolio", 
                                id="rebalance-button", 
                                className="btn btn-primary"
                            )
                        ], width=12, className="text-center")
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Rebalancing Impact"),
                                dbc.CardBody([
                                    html.Div(id="rebalance-impact-container")
                                ])
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
            
            # Time period selection
            time_period_card = dbc.Card([
                dbc.CardHeader("Select Time Period"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Preset Periods:"),
                            dcc.Dropdown(
                                id="preset-periods-dropdown",
                                options=[
                                    {"label": "Max Available", "value": "max"},
                                    {"label": "Year to Date", "value": "ytd"},
                                    {"label": "Last 1 Year", "value": "1y"},
                                    {"label": "Last 3 Years", "value": "3y"},
                                    {"label": "Last 5 Years", "value": "5y"},
                                    {"label": "Last 10 Years", "value": "10y"},
                                    {"label": "Custom Range", "value": "custom"}
                                ],
                                value="1y"
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Start Date:"),
                            dcc.DatePickerSingle(
                                id="start-date-picker",
                                min_date_allowed=datetime(2010, 1, 1),
                                max_date_allowed=datetime.now(),
                                date=datetime.now() - timedelta(days=365),
                                disabled=True
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("End Date:"),
                            dcc.DatePickerSingle(
                                id="end-date-picker",
                                min_date_allowed=datetime(2010, 1, 1),
                                max_date_allowed=datetime.now(),
                                date=datetime.now(),
                                disabled=True
                            )
                        ], width=4)
                    ])
                ])
            ], className="mb-4")
            
            # Create the main layout
            layout = html.Div([
                dbc.Container([
                    html.H1("Portfolio Risk Analysis Dashboard", className="mt-4 mb-4"),
                    html.P("Interactive dashboard for analyzing portfolio risk metrics and performance."),
                    
                    dbc.Row([
                        dbc.Col([
                            time_period_card
                        ], width=12)
                    ]),
                    
                    dbc.Tabs([
                        dbc.Tab(overview_card, label="Overview"),
                        dbc.Tab(allocation_card, label="Allocation"),
                        dbc.Tab(risk_card, label="Risk Analysis"),
                        dbc.Tab(correlation_card, label="Correlation"),
                        dbc.Tab(whatif_card, label="What-If Analysis"),
                        dbc.Tab(rebalance_card, label="Rebalancing")
                    ]),
                    
                    html.Hr(),
                    html.Footer([
                        html.P("Portfolio Risk Analysis Dashboard © 2025", className="text-center")
                    ])
                ], fluid=True)
            ])
            
            self.layout = layout
            return layout
            
        except Exception as e:
            logger.error(f"Error creating dashboard layout: {str(e)}")
            # Return minimal layout with error message
            return html.Div([
                html.H1("Error Creating Dashboard"),
                html.P(f"An error occurred: {str(e)}")
            ])

    def register_callbacks(self, app: dash.Dash) -> None:
        """
        Register all dashboard callbacks.

        Args:
            app: Dash application instance
        """
        try:
            # Store the app instance
            self.app = app
            
            # Date range callback
            @app.callback(
                [
                    Output("start-date-picker", "date"),
                    Output("end-date-picker", "date"),
                    Output("start-date-picker", "disabled"),
                    Output("end-date-picker", "disabled")
                ],
                [Input("preset-periods-dropdown", "value")]
            )
            def update_date_pickers(preset_period):
                end_date = datetime.now()
                
                if preset_period == "max":
                    # Use earliest available date in price data
                    start_date = self.portfolio.price_data['Date'].min()
                    return start_date, end_date, True, True
                
                elif preset_period == "ytd":
                    # Year to date
                    start_date = datetime(end_date.year, 1, 1)
                    return start_date, end_date, True, True
                
                elif preset_period == "1y":
                    # Last 1 year
                    start_date = end_date - timedelta(days=365)
                    return start_date, end_date, True, True
                
                elif preset_period == "3y":
                    # Last 3 years
                    start_date = end_date - timedelta(days=365 * 3)
                    return start_date, end_date, True, True
                
                elif preset_period == "5y":
                    # Last 5 years
                    start_date = end_date - timedelta(days=365 * 5)
                    return start_date, end_date, True, True
                
                elif preset_period == "10y":
                    # Last 10 years
                    start_date = end_date - timedelta(days=365 * 10)
                    return start_date, end_date, True, True
                
                elif preset_period == "custom":
                    # Custom range (enable pickers)
                    start_date = end_date - timedelta(days=365)
                    return start_date, end_date, False, False
                
                # Default to 1 year
                start_date = end_date - timedelta(days=365)
                return start_date, end_date, True, True
            
            # Portfolio stats callback
            @app.callback(
                Output("portfolio-stats-container", "children"),
                [
                    Input("start-date-picker", "date"),
                    Input("end-date-picker", "date")
                ]
            )
            def update_portfolio_stats(start_date, end_date):
                # Calculate performance for the selected period
                performance = self.portfolio.calculate_historical_performance(
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Get portfolio stats
                stats = self.portfolio.get_portfolio_stats()
                
                # Add performance metrics
                if not performance.empty:
                    latest = performance.iloc[-1]
                    stats['period_return'] = latest['Cumulative Return'] * 100  # Convert to percentage
                    
                    # Calculate annualized return
                    days = (performance['Date'].max() - performance['Date'].min()).days
                    if days > 0:
                        annualized_return = ((1 + latest['Cumulative Return']) ** (365 / days) - 1) * 100
                        stats['annualized_return'] = annualized_return
                
                # Create stats display
                stats_cards = []
                
                # Portfolio value card
                stats_cards.append(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Total Value", className="card-title"),
                                html.H3(f"${stats.get('total_value', 0):,.2f}", className="card-text text-primary")
                            ])
                        ]),
                        width=3
                    )
                )
                
                # Period return card
                stats_cards.append(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Period Return", className="card-title"),
                                html.H3(
                                    f"{stats.get('period_return', 0):.2f}%", 
                                    className=f"card-text {'text-success' if stats.get('period_return', 0) >= 0 else 'text-danger'}"
                                )
                            ])
                        ]),
                        width=3
                    )
                )
                
                # Current volatility card
                stats_cards.append(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Volatility", className="card-title"),
                                html.H3(
                                    f"{stats.get('current_volatility', 0) * 100:.2f}%", 
                                    className="card-text text-secondary"
                                )
                            ])
                        ]),
                        width=3
                    )
                )
                
                # Sharpe ratio card
                stats_cards.append(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Sharpe Ratio", className="card-title"),
                                html.H3(
                                    f"{stats.get('current_sharpe', 0):.2f}", 
                                    className=f"card-text {'text-success' if stats.get('current_sharpe', 0) >= 1 else 'text-warning'}"
                                )
                            ])
                        ]),
                        width=3
                    )
                )
                
                return dbc.Row(stats_cards)
            
            # Performance chart callback
            @app.callback(
                Output("performance-chart", "figure"),
                [
                    Input("start-date-picker", "date"),
                    Input("end-date-picker", "date")
                ]
            )
            def update_performance_chart(start_date, end_date):
                # Calculate performance for the selected period
                performance = self.portfolio.calculate_historical_performance(
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Create chart
                if not performance.empty:
                    chart = create_performance_chart(
                        performance_data=performance,
                        title="Portfolio Performance"
                    )
                    return chart
                else:
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No performance data available for the selected period",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Sector allocation chart callback
            @app.callback(
                Output("sector-allocation-chart", "figure"),
                [Input("start-date-picker", "date")]  # Dummy input to trigger callback
            )
            def update_sector_allocation_chart(_):
                # Get current holdings
                holdings = self.portfolio.holdings
                
                # Create chart
                if holdings is not None and not holdings.empty:
                    chart = create_asset_allocation_chart(
                        holdings_data=holdings,
                        group_by='Sector',
                        title="Sector Allocation"
                    )
                    return chart
                else:
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No holdings data available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Holdings allocation chart callback
            @app.callback(
                Output("holdings-allocation-chart", "figure"),
                [Input("start-date-picker", "date")]  # Dummy input to trigger callback
            )
            def update_holdings_allocation_chart(_):
                # Get current holdings
                holdings = self.portfolio.holdings
                
                # Create chart
                if holdings is not None and not holdings.empty:
                    # Filter to top 10 holdings by value
                    top_holdings = holdings.nlargest(10, 'Value')
                    
                    chart = create_asset_allocation_chart(
                        holdings_data=top_holdings,
                        group_by='Ticker',
                        title="Top 10 Holdings"
                    )
                    return chart
                else:
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No holdings data available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Risk metrics chart callback
            @app.callback(
                Output("risk-metrics-chart", "figure"),
                [
                    Input("start-date-picker", "date"),
                    Input("end-date-picker", "date")
                ]
            )
            def update_risk_metrics_chart(start_date, end_date):
                # Calculate performance for the selected period
                performance = self.portfolio.calculate_historical_performance(
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Create chart
                if not performance.empty:
                    chart = create_risk_metrics_chart(
                        performance_data=performance,
                        title="Portfolio Risk Metrics"
                    )
                    return chart
                else:
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No risk data available for the selected period",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Risk contribution chart callback
            @app.callback(
                Output("risk-contribution-chart", "figure"),
                [
                    Input("start-date-picker", "date"),
                    Input("end-date-picker", "date")
                ]
            )
            def update_risk_contribution_chart(start_date, end_date):
                try:
                    # Get performance data for the period
                    performance = self.portfolio.calculate_historical_performance(
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if performance.empty:
                        raise ValueError("No performance data available")
                    
                    # Get holdings
                    holdings = self.portfolio.holdings
                    
                    if holdings is None or holdings.empty:
                        raise ValueError("No holdings data available")
                    
                    # Get returns by ticker
                    returns_by_ticker = {}
                    price_data = self.portfolio.price_data
                    
                    # Filter price data to the selected period
                    filtered_price_data = price_data[
                        (price_data['Date'] >= pd.to_datetime(start_date)) &
                        (price_data['Date'] <= pd.to_datetime(end_date))
                    ]
                    
                    # Get returns for each ticker in portfolio
                    for ticker in holdings['Ticker']:
                        ticker_data = filtered_price_data[filtered_price_data['Ticker'] == ticker]
                        if not ticker_data.empty:
                            returns_by_ticker[ticker] = ticker_data['Returns'].values
                    
                    # Create returns DataFrame
                    returns_df = pd.DataFrame(returns_by_ticker)
                    
                    # Calculate covariance matrix
                    cov_matrix = returns_df.cov() * 252  # Annualize
                    
                    # Create weights series
                    weights = pd.Series(
                        dict(zip(holdings['Ticker'], holdings['Weight'] / 100))
                    )
                    
                    # Calculate risk contribution
                    risk_contrib = self.risk_metrics.calculate_portfolio_risk_contribution(weights, cov_matrix)
                    
                    # Create DataFrame for chart
                    risk_contrib_df = pd.DataFrame({
                        'Ticker': risk_contrib.index,
                        'RiskContribution': risk_contrib.values
                    })
                    
                    # Create chart
                    chart = create_risk_contribution_chart(
                        risk_contribution_data=risk_contrib_df,
                        title="Risk Contribution by Asset"
                    )
                    return chart
                    
                except Exception as e:
                    logger.error(f"Error calculating risk contribution: {str(e)}")
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Error calculating risk contribution: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Risk-return chart callback
            @app.callback(
                Output("risk-return-chart", "figure"),
                [
                    Input("start-date-picker", "date"),
                    Input("end-date-picker", "date")
                ]
            )
            def update_risk_return_chart(start_date, end_date):
                try:
                    # Get holdings
                    holdings = self.portfolio.holdings
                    
                    if holdings is None or holdings.empty:
                        raise ValueError("No holdings data available")
                    
                    # Get price data
                    price_data = self.portfolio.price_data
                    
                    # Filter price data to the selected period
                    filtered_price_data = price_data[
                        (price_data['Date'] >= pd.to_datetime(start_date)) &
                        (price_data['Date'] <= pd.to_datetime(end_date))
                    ]
                    
                    # Calculate risk and return metrics for each ticker
                    risk_return_data = []
                    
                    for ticker in holdings['Ticker']:
                        ticker_data = filtered_price_data[filtered_price_data['Ticker'] == ticker]
                        
                        if len(ticker_data) > 20:  # Ensure enough data points
                            returns = ticker_data['Returns']
                            
                            # Annualized return
                            ann_return = (1 + returns.mean()) ** 252 - 1
                            
                            # Annualized volatility
                            ann_vol = returns.std() * np.sqrt(252)
                            
                            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
                            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                            
                            risk_return_data.append({
                                'Ticker': ticker,
                                'Return': ann_return,
                                'Volatility': ann_vol,
                                'Sharpe': sharpe
                            })
                    
                    # Create DataFrame
                    risk_return_df = pd.DataFrame(risk_return_data)
                    
                    # Create chart
                    if not risk_return_df.empty:
                        chart = create_risk_return_scatter(
                            risk_return_data=risk_return_df,
                            title="Risk-Return Profile"
                        )
                        return chart
                    else:
                        raise ValueError("No risk-return data calculated")
                    
                except Exception as e:
                    logger.error(f"Error creating risk-return chart: {str(e)}")
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Error creating risk-return chart: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Correlation heatmap callback
            @app.callback(
                Output("correlation-heatmap", "figure"),
                [
                    Input("start-date-picker", "date"),
                    Input("end-date-picker", "date")
                ]
            )
            def update_correlation_heatmap(start_date, end_date):
                try:
                    # Get holdings
                    holdings = self.portfolio.holdings
                    
                    if holdings is None or holdings.empty:
                        raise ValueError("No holdings data available")
                    
                    # Get price data
                    price_data = self.portfolio.price_data
                    
                    # Filter price data to the selected period
                    filtered_price_data = price_data[
                        (price_data['Date'] >= pd.to_datetime(start_date)) &
                        (price_data['Date'] <= pd.to_datetime(end_date))
                    ]
                    
                    # Get returns by ticker
                    returns_by_ticker = {}
                    
                    for ticker in holdings['Ticker']:
                        ticker_data = filtered_price_data[filtered_price_data['Ticker'] == ticker]
                        if not ticker_data.empty:
                            # Convert to time series
                            ticker_returns = ticker_data.set_index('Date')['Returns']
                            returns_by_ticker[ticker] = ticker_returns
                    
                    # Create returns DataFrame
                    returns_df = pd.DataFrame(returns_by_ticker)
                    
                    # Fill missing values (ensure aligned dates)
                    returns_df = returns_df.fillna(method='ffill')
                    
                    # Create chart
                    if not returns_df.empty:
                        chart = create_correlation_heatmap(
                            returns_data=returns_df,
                            title="Asset Correlation Matrix"
                        )
                        return chart
                    else:
                        raise ValueError("No correlation data calculated")
                    
                except Exception as e:
                    logger.error(f"Error creating correlation heatmap: {str(e)}")
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Error creating correlation heatmap: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Display/hide custom scenario inputs
            @app.callback(
                Output("custom-scenario-container", "className"),
                [Input("scenario-dropdown", "value")]
            )
            def toggle_custom_scenario(scenario):
                if scenario == "custom":
                    return ""  # Show container
                else:
                    return "d-none"  # Hide container
            
            # What-if analysis chart callback
            @app.callback(
                Output("what-if-chart", "figure"),
                [Input("scenario-dropdown", "value")]
            )
            def update_what_if_chart(scenario):
                try:
                    # Get current holdings
                    holdings = self.portfolio.holdings
                    
                    if holdings is None or holdings.empty:
                        raise ValueError("No holdings data available")
                    
                    # Run what-if analysis
                    scenario_results = self.portfolio.get_what_if_analysis(scenario=scenario)
                    
                    # Create chart
                    if not scenario_results.empty:
                        chart = create_what_if_chart(
                            base_holdings=holdings,
                            scenario_holdings=scenario_results,
                            scenario_name=scenario.replace("_", " ").title(),
                            title=f"What-If Analysis: {scenario.replace('_', ' ').title()} Scenario"
                        )
                        return chart
                    else:
                        raise ValueError("No scenario results calculated")
                    
                except Exception as e:
                    logger.error(f"Error creating what-if chart: {str(e)}")
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Error creating what-if chart: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
            
            # Rebalance weights container callback
            @app.callback(
                Output("rebalance-weights-container", "children"),
                [Input("rebalance-method", "value")]
            )
            def update_rebalance_weights_container(method):
                try:
                    # Get current holdings
                    holdings = self.portfolio.holdings
                    
                    if holdings is None or holdings.empty:
                        return html.Div("No holdings data available")
                    
                    if method == "weights":
                        # Create slider for each ticker
                        sliders = []
                        
                        for _, row in holdings.iterrows():
                            ticker = row['Ticker']
                            current_weight = row['Weight']
                            
                            slider_row = dbc.Row([
                                dbc.Col(html.Label(f"{ticker}: {current_weight:.2f}%"), width=2),
                                dbc.Col(
                                    dcc.Slider(
                                        id=f"weight-slider-{ticker}",
                                        min=0,
                                        max=100,
                                        step=1,
                                        value=current_weight,
                                        marks={i: f"{i}%" for i in range(0, 101, 20)}
                                    ),
                                    width=10
                                )
                            ], className="mb-2")
                            
                            sliders.append(slider_row)
                        
                        return html.Div(sliders)
                        
                    elif method == "sectors":
                        # Group by sector
                        sector_weights = holdings.groupby('Sector')['Weight'].sum()
                        
                        # Create slider for each sector
                        sliders = []
                        
                        for sector, weight in sector_weights.items():
                            slider_row = dbc.Row([
                                dbc.Col(html.Label(f"{sector}: {weight:.2f}%"), width=2),
                                dbc.Col(
                                    dcc.Slider(
                                        id=f"sector-slider-{sector}",
                                        min=0,
                                        max=100,
                                        step=1,
                                        value=weight,
                                        marks={i: f"{i}%" for i in range(0, 101, 20)}
                                    ),
                                    width=10
                                )
                            ], className="mb-2")
                            
                            sliders.append(slider_row)
                        
                        return html.Div(sliders)
                    
                    return html.Div("Select a rebalance method")
                    
                except Exception as e:
                    logger.error(f"Error creating rebalance weights container: {str(e)}")
                    return html.Div(f"Error: {str(e)}")
            
            # More callbacks can be added for rebalancing, etc.
            
            logger.info("Dashboard callbacks registered successfully")
            
        except Exception as e:
            logger.error(f"Error registering callbacks: {str(e)}")
    
    def create_dash_app(self, server=None) -> dash.Dash:
        """
        Create and configure the Dash application.

        Args:
            server: Optional Flask server instance

        Returns:
            Configured Dash application
        """
        try:
            # Create Dash app
            app = dash.Dash(
                __name__,
                server=server if server is not None else True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                ],
                suppress_callback_exceptions=True
            )
            
            # Set layout
            app.layout = self.create_layout()
            
            # Register callbacks
            self.register_callbacks(app)
            
            return app
            
        except Exception as e:
            logger.error(f"Error creating Dash app: {str(e)}")
            raise


# Example usage when run as script
if __name__ == "__main__":
    print("This module provides dashboard layout and callbacks.")
    print("Import and use these components in your main application.")