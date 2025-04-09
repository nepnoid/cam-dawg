"""
Trading Bot - Main Entry Point
"""
import argparse
import json
import logging
import time
import os
import signal
import sys
from datetime import datetime

# Make sure to install these dependencies using pip:
# pip install ccxt pandas numpy matplotlib python-dotenv

# Import your modules
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Set up logging
def setup_logger(name=None):
    """Set up and return a logger instance"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    log_filename = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    f_handler = logging.FileHandler(log_filename)

    # Set levels
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    c_formatter = logging.Formatter(log_format)
    f_formatter = logging.Formatter(log_format)
    c_handler.setFormatter(c_formatter)
    f_handler.setFormatter(f_formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

# Create the platform connector class
class PlatformConnector:
    """Handles communication with the trading platform"""

    def __init__(self, config):
        self.logger = logging.getLogger("PlatformConnector")
        self.config = config

        # Load API credentials from .env file
        load_dotenv()

        # Initialize the exchange
        exchange_id = config['platform']
        self.logger.info(f"Initializing connection to {exchange_id}")

        # Create the exchange instance
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': os.getenv('API_KEY'),
                'secret': os.getenv('API_SECRET'),
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}  # Use spot trading by default
            })
            self.logger.info(f"Successfully connected to {exchange_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise

    def get_market_data(self, symbol, timeframe, limit=100):
        """Fetch OHLCV (candlestick) data for a market"""
        try:
            self.logger.info(f"Fetching {timeframe} data for {symbol}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return None

    def get_balance(self):
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return None

    def create_order(self, symbol, order_type, side, amount, price=None):
        """Create an order on the exchange"""
        try:
            self.logger.info(f"Creating {order_type} {side} order for {symbol}: {amount}")

            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            self.logger.info(f"Order created: {order['id']}")
            return order

        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None

# Strategy class
class Strategy:
    """Trading strategy implementation"""

    def __init__(self, config):
        self.logger = logging.getLogger("Strategy")
        self.config = config
        self.strategy_params = config['strategy_params']
        self.logger.info(f"Initializing {config['trading']['strategy']} strategy")

    def generate_signal(self, market_data):
        """
        Generate trading signals based on market data

        Returns:
            dict: Signal information or None if no signal
        """
        if self.config['trading']['strategy'] == 'simple_moving_average':
            return self._sma_crossover_strategy(market_data)
        else:
            self.logger.warning(f"Strategy {self.config['trading']['strategy']} not implemented")
            return None

    def _sma_crossover_strategy(self, market_data):
        """
        Simple Moving Average crossover strategy
        Buy when short MA crosses above long MA
        Sell when short MA crosses below long MA
        """
        # Ensure we have a DataFrame
        df = market_data.copy()

        # Calculate moving averages
        short_period = self.strategy_params['short_period']
        long_period = self.strategy_params['long_period']

        df['short_ma'] = df['close'].rolling(window=short_period).mean()
        df['long_ma'] = df['close'].rolling(window=long_period).mean()

        # Need at least long_period candles to calculate MAs
        if len(df) < long_period:
            return None

        # Calculate crossover signals
        df['signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1  # Buy signal
        df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1  # Sell signal

        # Detect crossovers (signal change)
        df['signal_change'] = df['signal'].diff()

        # Get the last two rows to check for a crossover
        last_rows = df.iloc[-2:].copy()

        # Check if we have a crossover in the most recent candle
        if len(last_rows) >= 2 and last_rows['signal_change'].iloc[-1] != 0:
            signal_type = 'buy' if last_rows['signal_change'].iloc[-1] > 0 else 'sell'

            self.logger.info(f"Generated {signal_type} signal - Short MA: {last_rows['short_ma'].iloc[-1]:.2f}, Long MA: {last_rows['long_ma'].iloc[-1]:.2f}")

            return {
                'type': signal_type,
                'price': df['close'].iloc[-1],  # Current close price
                'timestamp': df['timestamp'].iloc[-1]
            }

        return None

# Risk Manager class
class RiskManager:
    """Manages trading risk according to configuration"""

    def __init__(self, config):
        self.logger = logging.getLogger("RiskManager")
        self.config = config
        self.risk_config = config['risk']
        self.logger.info("Initializing risk manager")

        # Keep track of open trades
        self.open_trades = {}

    def calculate_position_size(self, balance, current_price):
        """
        Calculate the position size based on risk parameters

        Args:
            balance (float): Available balance
            current_price (float): Current price of the asset

        Returns:
            float: Position size in the base currency
        """
        # Calculate position size as a percentage of available balance
        position_value = balance * self.risk_config['position_size']

        # Convert to quantity
        quantity = position_value / current_price

        self.logger.info(f"Calculated position size: {quantity:.6f} at price {current_price}")
        return quantity

    def check_risk_parameters(self, signal, balance, open_trades=None):
        """
        Check if the trade meets risk parameters

        Args:
            signal (dict): The trading signal
            balance (float): Available balance
            open_trades (list): Currently open trades

        Returns:
            dict: Modified signal with risk parameters or None if trade should not be executed
        """
        # Check if we can open more trades
        if open_trades and len(open_trades) >= self.risk_config['max_open_trades']:
            self.logger.warning("Maximum number of open trades reached")
            return None

        # Modify the signal to include stop loss and take profit
        modified_signal = signal.copy()

        if signal['type'] == 'buy':
            # Set stop loss
            modified_signal['stop_loss'] = signal['price'] * (1 - self.risk_config['stop_loss_pct'] / 100)

            # Set take profit
            modified_signal['take_profit'] = signal['price'] * (1 + self.risk_config['take_profit_pct'] / 100)
        else:
            # For sell orders
            modified_signal['stop_loss'] = signal['price'] * (1 + self.risk_config['stop_loss_pct'] / 100)
            modified_signal['take_profit'] = signal['price'] * (1 - self.risk_config['take_profit_pct'] / 100)

        self.logger.info(f"Risk parameters applied - Stop Loss: {modified_signal['stop_loss']:.2f}, Take Profit: {modified_signal['take_profit']:.2f}")
        return modified_signal

# Main Trader class
class Trader:
    """Main trading bot class that coordinates all components"""

    def __init__(self, config_path):
        self.logger = logging.getLogger("Trader")

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

        # Initialize components
        self.platform = PlatformConnector(self.config['trading'])
        self.strategy = Strategy(self.config)
        self.risk_manager = RiskManager(self.config)

        # Trading parameters
        self.market = self.config['trading']['market']
        self.timeframe = self.config['trading']['timeframe']

        # Runtime flags
        self.is_running = False

        self.logger.info(f"Trader initialized for {self.market} on {self.timeframe} timeframe")

    def start(self):
        """Start the trading bot"""
        self.is_running = True
        self.logger.info("Starting trading bot")

        while self.is_running:
            try:
                # 1. Get market data
                market_data = self.platform.get_market_data(self.market, self.timeframe)

                if market_data is None or len(market_data) < max(self.strategy.strategy_params.get('long_period', 0), self.strategy.strategy_params.get('short_period', 0)):  # Need enough data for analysis
                    self.logger.warning("Insufficient market data, waiting for next cycle")
                    time.sleep(60)
                    continue

                # 2. Generate trading signals
                signal = self.strategy.generate_signal(market_data)

                if signal:
                    # 3. Get account balance
                    balance_data = self.platform.get_balance()
                    if not balance_data:
                        self.logger.error("Failed to fetch balance")
                        time.sleep(60)
                        continue

                    # Get base currency from the market symbol (e.g., "USDT" from "BTC/USDT")
                    quote_currency = self.market.split('/')[1]
                    available_balance = balance_data['free'][quote_currency] if quote_currency in balance_data['free'] else 0

                    # 4. Check risk parameters
                    open_orders = self.platform.exchange.fetch_open_orders(self.market)
                    modified_signal = self.risk_manager.check_risk_parameters(
                        signal, available_balance, open_orders
                    )

                    if modified_signal:
                        # 5. Calculate position size
                        position_size = self.risk_manager.calculate_position_size(
                            available_balance, modified_signal['price']
                        )

                        # 6. Execute trade
                        if position_size > 0:
                            order = self.platform.create_order(
                                self.market,
                                'market',  # Use market orders for simplicity
                                modified_signal['type'],
                                position_size
                            )

                            if order:
                                self.logger.info(f"Order executed: {order['id']}")

                                # Record trade
                                self.risk_manager.open_trades[order['id']] = {
                                    'market': self.market,
                                    'type': modified_signal['type'],
                                    'price': modified_signal['price'],
                                    'size': position_size,
                                    'stop_loss': modified_signal['stop_loss'],
                                    'take_profit': modified_signal['take_profit'],
                                    'time': datetime.now().isoformat(),
                                    'status': 'open'
                                }

                # Wait for next cycle
                self.logger.info("Waiting for next cycle")
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}")
                time.sleep(60)  # Wait before retrying

    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.logger.info("Stopping trading bot")

    def backtest(self, days=30):
        """Run backtesting simulation"""
        self.logger.info(f"Starting backtesting for {days} days of historical data")

        # Get historical data
        end_date = datetime.now()
        # Note: This is a simplified approach. For proper backtesting, you'd want to get precise historical data
        # Assuming '1h' timeframe for backtesting purposes. Adjust limit based on timeframe.
        limit = days * 24 if self.timeframe == '1h' else days * (24 * 60 // int(self.timeframe[:-1])) if self.timeframe[:-1].isdigit() and self.timeframe.endswith('m') else days * (24 // int(self.timeframe[:-1])) if self.timeframe[:-1].isdigit() and self.timeframe.endswith('h') else days # Default to daily if unknown
        market_data = self.platform.get_market_data(self.market, self.timeframe, limit=limit)

        if market_data is None or len(market_data) < max(self.strategy.strategy_params.get('long_period', 0), self.strategy.strategy_params.get('short_period', 0)):
            self.logger.error("Insufficient data for backtesting")
            return None

        # Prepare results containers
        trades = []
        capital = 10000  # Initial capital
        position = None

        # Iterate through each candle
        start_index = max(self.strategy.strategy_params.get('long_period', 0), self.strategy.strategy_params.get('short_period', 0))
        for i in range(start_index, len(market_data)):  # Start after we have enough data for indicators
            # Get data up to current point (to avoid lookahead bias)
            current_data = market_data.iloc[:i+1].copy()

            # Generate signal
            signal = self.strategy.generate_signal(current_data)

            # Process signal
            if signal:
                if signal['type'] == 'buy' and position is None:
                    # Apply risk management
                    risk_signal = self.risk_manager.check_risk_parameters(signal, capital, [])
                    if risk_signal:
                        # Calculate position size
                        size = self.risk_manager.calculate_position_size(capital, risk_signal['price'])
                        position = {
                            'type': 'buy',
                            'price': risk_signal['price'],
                            'size': size,
                            'stop_loss': risk_signal['stop_loss'],
                            'take_profit': risk_signal['take_profit'],
                            'entry_time': current_data['timestamp'].iloc[-1]
                        }
                        trades.append(position)
                        self.logger.info(f"BACKTEST: BUY at {position['price']:.2f}")

                elif signal['type'] == 'sell' and
