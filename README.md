# Portfolio Risk Analysis Dashboard

An interactive dashboard for visualizing and analyzing portfolio risk metrics and performance over time.

## Project Overview

This dashboard application allows users to:

- Visualize key risk metrics and performance of a portfolio
- Calculate metrics such as volatility, beta, Sharpe ratio for the overall portfolio and individual holdings
- Adjust portfolio weights and see the impact on risk
- Backtest portfolio performance using historical data

## Project Structure

```
portfolio_risk_dashboard/
├── data/
│   ├── Portfolio.csv              # Portfolio holdings data
│   └── Portfolio_prices.csv       # Historical price data
├── notebooks/
│   └── data_exploration.ipynb     # Exploratory data analysis notebook
├── src/
│   ├── data_loading/              # Data loading modules
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Functions to load data from CSV files
|   |   └── data_preprocessing.py  # Functions to clean and prepare data
│   ├── models/                    # Portfolio and risk model modules
│   │   ├── __init__.py
│   │   ├── portfolio.py           # Portfolio class and related functions
│   │   └── risk_metrics.py        # Risk metric calculations
│   ├── visualization/             # Visualization modules
│   │   ├── __init__.py
│   │   ├── charts.py              # Chart generation functions
│   │   └── dashboard.py           # Dashboard layout and callbacks
│   └── app.py                     # Main application entry point
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py        # Tests for data loading
│   ├── test_data_preprocessing.py # Tests for data Preprocessing
│   ├── test_portfolio.py          # Tests for portfolio models
│   └── test_risk_metrics.py       # Tests for risk metrics
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/aum2606/portfolio-risk-dashboard.git
   cd portfolio-risk-dashboard
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your portfolio data in the `data/` directory:
   - `Portfolio.csv`: Should contain columns for Ticker, Quantity, Sector, Close, and Weight
   - `Portfolio_prices.csv`: Should contain historical price data with Date, Ticker, and price information

2. Run the dashboard application:
   ```
   python src/app.py
   ```

3. Open your web browser and navigate to `http://localhost:8050` to view the dashboard.

## Data Format

### Portfolio.csv

Column descriptions:
- `Ticker`: Stock symbol/ticker
- `Quantity`: Number of shares held
- `Sector`: Industry sector of the stock
- `Close`: Latest closing price
- `Weight`: Portfolio weight (%) of the holding

### Portfolio_prices.csv

Column descriptions:
- `Date`: Trading date
- `Ticker`: Stock symbol/ticker
- `Open`: Opening price
- `High`: Highest price of the day
- `Low`: Lowest price of the day
- `Close`: Closing price
- `Adjusted`: Adjusted closing price
- `Returns`: Daily returns
- `Volume`: Trading volume

## Development

### Running Tests

To run the unit tests:

```
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines. To format your code:

```
black src/ tests/
isort src/ tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request