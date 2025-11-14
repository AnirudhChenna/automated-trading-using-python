import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

class TechnicalIndicators:
    """Calculate various technical indicators for trading signals"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band


class RiskManager:
    """Manage trading risk and position sizing"""
    
    def __init__(self, max_position_size: float = 0.1, max_risk_per_trade: float = 0.02):
        self.max_position_size = max_position_size  # Max 10% of portfolio per position
        self.max_risk_per_trade = max_risk_per_trade  # Max 2% risk per trade
    
    def calculate_position_size(self, portfolio_value: float, entry_price: float, 
                               stop_loss_price: float) -> int:
        """Calculate position size based on risk management rules"""
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        # Calculate max shares based on risk
        max_risk_amount = portfolio_value * self.max_risk_per_trade
        shares_by_risk = int(max_risk_amount / risk_per_share)
        
        # Calculate max shares based on position size
        max_position_value = portfolio_value * self.max_position_size
        shares_by_position = int(max_position_value / entry_price)
        
        # Return the minimum to satisfy both constraints
        return min(shares_by_risk, shares_by_position)


class TradingStrategy:
    """Implement trading strategy logic"""
    
    def __init__(self, name: str = "SMA Crossover"):
        self.name = name
        self.indicators = TechnicalIndicators()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on strategy"""
        # Calculate indicators
        df['SMA_20'] = self.indicators.sma(df['close'], 20)
        df['SMA_50'] = self.indicators.sma(df['close'], 50)
        df['RSI'] = self.indicators.rsi(df['close'], 14)
        df['MACD'], df['MACD_Signal'] = self.indicators.macd(df['close'])
        
        # Generate signals
        df['signal'] = 0
        
        # Buy signal: SMA20 crosses above SMA50, RSI not overbought
        buy_condition = (
            (df['SMA_20'] > df['SMA_50']) & 
            (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1)) &
            (df['RSI'] < 70)
        )
        
        # Sell signal: SMA20 crosses below SMA50, or RSI overbought
        sell_condition = (
            ((df['SMA_20'] < df['SMA_50']) & 
             (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))) |
            (df['RSI'] > 80)
        )
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class Portfolio:
    """Manage portfolio and track performance"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos['shares'] * current_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def buy(self, symbol: str, price: float, shares: int, date: datetime):
        """Execute buy order"""
        cost = price * shares
        if cost <= self.cash:
            self.cash -= cost
            if symbol in self.positions:
                # Average up the position
                old_shares = self.positions[symbol]['shares']
                old_price = self.positions[symbol]['entry_price']
                new_shares = old_shares + shares
                new_avg_price = ((old_price * old_shares) + (price * shares)) / new_shares
                self.positions[symbol] = {
                    'shares': new_shares,
                    'entry_price': new_avg_price,
                    'entry_date': self.positions[symbol]['entry_date']
                }
            else:
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_date': date
                }
            
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'shares': shares,
                'value': cost
            })
            return True
        return False
    
    def sell(self, symbol: str, price: float, shares: int, date: datetime):
        """Execute sell order"""
        if symbol in self.positions and self.positions[symbol]['shares'] >= shares:
            revenue = price * shares
            self.cash += revenue
            
            entry_price = self.positions[symbol]['entry_price']
            profit = (price - entry_price) * shares
            profit_pct = ((price - entry_price) / entry_price) * 100
            
            self.positions[symbol]['shares'] -= shares
            if self.positions[symbol]['shares'] == 0:
                del self.positions[symbol]
            
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'shares': shares,
                'value': revenue,
                'profit': profit,
                'profit_pct': profit_pct
            })
            return True
        return False
    
    def get_performance_summary(self) -> Dict:
        """Calculate performance metrics"""
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
        losing_trades = len([t for t in self.trades if t.get('profit', 0) < 0])
        
        total_profit = sum(t.get('profit', 0) for t in self.trades)
        final_value = self.get_portfolio_value({})
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_profit': total_profit,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'cash': self.cash,
            'open_positions': len(self.positions)
        }


class AutomatedTradingBot:
    """Main trading bot that orchestrates everything"""
    
    def __init__(self, initial_capital: float = 100000):
        self.portfolio = Portfolio(initial_capital)
        self.strategy = TradingStrategy()
        self.risk_manager = RiskManager()
    
    def generate_sample_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Generate sample price data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate random walk price data
        np.random.seed(42)
        returns = np.random.randn(days) * 0.02  # 2% daily volatility
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'open': price * (1 + np.random.randn(days) * 0.005),
            'high': price * (1 + abs(np.random.randn(days) * 0.01)),
            'low': price * (1 - abs(np.random.randn(days) * 0.01)),
            'close': price,
            'volume': np.random.randint(1000000, 10000000, days)
        })
        
        return df
    
    def backtest(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Run backtest on historical data"""
        print(f"\n{'='*60}")
        print(f"Starting Backtest for {symbol}")
        print(f"{'='*60}")
        
        # Generate signals
        data = self.strategy.generate_signals(data)
        
        # Execute trades based on signals
        for idx, row in data.iterrows():
            if pd.isna(row['signal']):
                continue
            
            current_price = row['close']
            portfolio_value = self.portfolio.get_portfolio_value({symbol: current_price})
            
            if row['signal'] == 1:  # Buy signal
                stop_loss = current_price * 0.95  # 5% stop loss
                shares = self.risk_manager.calculate_position_size(
                    portfolio_value, current_price, stop_loss
                )
                if shares > 0:
                    success = self.portfolio.buy(symbol, current_price, shares, row['date'])
                    if success:
                        print(f"{row['date'].date()} - BUY {shares} shares at ${current_price:.2f}")
            
            elif row['signal'] == -1 and symbol in self.portfolio.positions:  # Sell signal
                shares = self.portfolio.positions[symbol]['shares']
                success = self.portfolio.sell(symbol, current_price, shares, row['date'])
                if success:
                    last_trade = self.portfolio.trades[-1]
                    print(f"{row['date'].date()} - SELL {shares} shares at ${current_price:.2f} "
                          f"(Profit: ${last_trade['profit']:.2f}, {last_trade['profit_pct']:.2f}%)")
        
        # Close any remaining positions
        if symbol in self.portfolio.positions:
            final_price = data.iloc[-1]['close']
            shares = self.portfolio.positions[symbol]['shares']
            self.portfolio.sell(symbol, final_price, shares, data.iloc[-1]['date'])
            print(f"\nClosing position: SELL {shares} shares at ${final_price:.2f}")
        
        return self.portfolio.get_performance_summary()
    
    def run(self, symbol: str = "AAPL", days: int = 365):
        """Run the automated trading bot"""
        print("\n" + "="*60)
        print("AUTOMATED TRADING BOT")
        print("="*60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Initial Capital: ${self.portfolio.initial_capital:,.2f}")
        print(f"Symbol: {symbol}")
        print(f"Backtest Period: {days} days")
        
        # Generate sample data
        data = self.generate_sample_data(symbol, days)
        
        # Run backtest
        results = self.backtest(symbol, data)
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"Final Value:        ${results['final_value']:,.2f}")
        print(f"Total Return:       {results['total_return']:.2f}%")
        print(f"Total Profit:       ${results['total_profit']:,.2f}")
        print(f"Cash Remaining:     ${results['cash']:,.2f}")
        print(f"\nTotal Trades:       {results['total_trades']}")
        print(f"Winning Trades:     {results['winning_trades']}")
        print(f"Losing Trades:      {results['losing_trades']}")
        print(f"Win Rate:           {results['win_rate']:.2f}%")
        print(f"Open Positions:     {results['open_positions']}")
        print("="*60)
        
        return results


# Example usage
if __name__ == "__main__":
    # Create and run the trading bot
    bot = AutomatedTradingBot(initial_capital=100000)
    results = bot.run(symbol="AAPL", days=365)
    
    # You can also access individual trades
    print("\nRecent Trades:")
    for trade in bot.portfolio.trades[-5:]:
        print(f"{trade['date'].date()} - {trade['action']} {trade['shares']} shares "
              f"at ${trade['price']:.2f}")
    
    # Additional analysis
    print("\n" + "="*60)
    print("TRADE STATISTICS")
    print("="*60)
    
    if bot.portfolio.trades:
        profits = [t.get('profit', 0) for t in bot.portfolio.trades if 'profit' in t]
        if profits:
            avg_profit = sum(profits) / len(profits)
            max_profit = max(profits)
            max_loss = min(profits)
            print(f"Average Profit per Trade: ${avg_profit:.2f}")
            print(f"Best Trade:              ${max_profit:.2f}")
            print(f"Worst Trade:             ${max_loss:.2f}")
    
    print("\n" + "="*60)
    print("TESTING MULTIPLE SYMBOLS")
    print("="*60)
    
    symbols = ["MSFT", "GOOGL", "TSLA"]
    
    for sym in symbols:
        print(f"\n--- Testing {sym} ---")
        test_bot = AutomatedTradingBot(initial_capital=100000)
        test_results = test_bot.run(symbol=sym, days=365)
        print(f"Final Return for {sym}: {test_results['total_return']:.2f}%")