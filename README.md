📈 Automated Trading Bot

An automated trading bot built in Python that combines technical analysis, risk management, and portfolio tracking to generate trading signals and evaluate strategy performance through historical backtesting.

🚀 Features
Technical indicator calculations:
Simple Moving Average (SMA)
Exponential Moving Average (EMA)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
Bollinger Bands
Trading strategy implementation:
SMA Crossover Strategy
RSI-based trade filtering
Automated Buy/Sell signal generation
Risk Management:
Position sizing based on risk per trade
Maximum portfolio allocation limits
Stop-loss based risk calculations
Portfolio Management:
Trade execution simulation
Profit/Loss tracking
Portfolio value monitoring
Performance analytics
Backtesting Engine:
Historical strategy testing
Trade statistics
Win rate calculation
Return analysis
🛠️ Technologies Used
Python
Pandas
NumPy
Object-Oriented Programming (OOP)
📂 Project Structure
Automated-Trading-Bot/
│
├── trading_bot.py
├── README.md
│
├── TechnicalIndicators
├── TradingStrategy
├── RiskManager
├── Portfolio
└── AutomatedTradingBot
📊 Trading Indicators
SMA (Simple Moving Average)

Used to identify market trends and generate crossover signals.

EMA (Exponential Moving Average)

Provides higher weight to recent price movements.

RSI (Relative Strength Index)

Measures momentum and identifies overbought or oversold conditions.

MACD

Detects trend direction and momentum shifts.

Bollinger Bands

Measures volatility and potential price breakouts.

🎯 Trading Strategy
Buy Signal

Generated when:

SMA 20 crosses above SMA 50
RSI is below 70
Sell Signal

Generated when:

SMA 20 crosses below SMA 50
OR RSI exceeds 80
⚠️ Risk Management

The bot applies strict risk controls:

Maximum position size: 10% of portfolio
Maximum risk per trade: 2%
Automatic position sizing
Stop-loss based calculations
💰 Portfolio Tracking

Tracks:

Cash balance
Open positions
Trade history
Profit and loss
Portfolio returns
📈 Backtesting

The bot can:

Simulate trades on historical data
Measure strategy performance
Calculate win rate
Generate trade statistics
Compare results across multiple symbols

Example symbols tested:

MSFT
GOOGL
TSLA
AAPL
📦 Installation

Clone the repository:

git clone https://github.com/your-username/automated-trading-bot.git
cd automated-trading-bot

Install dependencies:

pip install pandas numpy
▶️ Run the Project
python trading_bot.py
📋 Sample Output
AUTOMATED TRADING BOT

Strategy: SMA Crossover
Initial Capital: $100,000

BACKTEST RESULTS

Total Return: 12.45%
Win Rate: 63.64%
Total Trades: 22
🔮 Future Enhancements
Real-time stock market data integration
Broker API connectivity (Alpaca, Interactive Brokers, Zerodha)
Machine Learning-based trading signals
Live paper trading
Web dashboard for analytics
Multi-strategy portfolio optimization
👨‍💻 Author

Anirudh Chenna

B.Tech – Artificial Intelligence & Machine Learning

📄 License

This project is intended for educational, research, and learning purposes only. It is not financial advice.
