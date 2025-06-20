"""Centralized Alpaca API client for the Helios trading bot.

This module provides a secure, robust interface to the Alpaca trading API
with comprehensive error handling, retry logic, and security best practices.

Author: Helios Trading Bot
Version: 1.0
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import json
import configparser

# Import Alpaca API
try:
    from alpaca_trade_api import REST, Stream
    from alpaca_trade_api.rest import APIError, TimeFrame
except ImportError as e:
    print(f"ERROR: Alpaca Trade API not installed. Run: pip install alpaca-trade-api")
    raise e

from .logger_config import get_helios_logger


class AlpacaAPIError(Exception):
    """Custom exception for Alpaca API related errors."""
    pass


class HeliosAlpacaClient:
    """Centralized Alpaca API client for the Helios trading system.

    This class provides a secure, robust interface to Alpaca's trading API
    with built-in error handling, retry logic, and comprehensive logging.
    """

    def __init__(self, config_path: str = "config/config.ini") -> None:
        """Initialize the Alpaca API client.

        Args:
            config_path: Path to the configuration file.

        Raises:
            AlpacaAPIError: If API credentials are missing or invalid.
            ValueError: If configuration is invalid.
        """
        self.logger = get_helios_logger('api_client')
        self.config = self._load_config(config_path)

        # Load credentials from environment
        self._load_credentials()

        # Initialize API client
        self._initialize_client()

        # Connection state tracking
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3  # From PRD SL-3.2
        self._last_successful_call = datetime.now()

        self.logger.info("Alpaca API client initialized successfully")

    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        """Load configuration from config.ini file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            ConfigParser object with loaded configuration.
        """
        config = configparser.ConfigParser()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config.read(config_path)
        return config

    def _load_credentials(self) -> None:
        """Load API credentials from environment variables.

        Raises:
            AlpacaAPIError: If required credentials are missing.
        """
        # Load environment variables
        load_dotenv()

        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise AlpacaAPIError(
                "API credentials not found. Please set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY in your .env file."
            )

        # Get base URL from config
        self.base_url = self.config.get('Alpaca', 'base_url',
                                       fallback='https://paper-api.alpaca.markets')

        # Never log the actual credentials
        self.logger.info(f"API credentials loaded for endpoint: {self.base_url}")

    def _initialize_client(self) -> None:
        """Initialize the Alpaca REST API client.

        Raises:
            AlpacaAPIError: If client initialization fails.
        """
        try:
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )

            # Test connection
            account = self.api.get_account()
            self.logger.info(f"Connected to Alpaca account: {account.account_number}")

        except Exception as e:
            self.logger.critical(f"Failed to initialize Alpaca API client: {str(e)}")
            raise AlpacaAPIError(f"API client initialization failed: {str(e)}")

    def _handle_api_call(self, func, *args, **kwargs) -> Any:
        """Wrapper for API calls with error handling and retry logic.

        Args:
            func: API function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result of the API call.

        Raises:
            AlpacaAPIError: If API call fails after retries.
        """
        max_retries = self.config.getint('System', 'api_retry_attempts', fallback=3)
        timeout = self.config.getint('System', 'api_timeout_seconds', fallback=10)

        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                self._consecutive_failures = 0
                self._last_successful_call = datetime.now()
                return result

            except APIError as e:
                self.logger.warning(f"Alpaca API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    self._consecutive_failures += 1
                    self._check_circuit_breaker()
                    raise AlpacaAPIError(f"API call failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                self.logger.error(f"Unexpected error in API call: {str(e)}")
                self._consecutive_failures += 1
                self._check_circuit_breaker()
                raise AlpacaAPIError(f"Unexpected API error: {str(e)}")

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be triggered (SL-3.2).

        Raises:
            AlpacaAPIError: If circuit breaker is triggered.
        """
        if self._consecutive_failures >= self._max_consecutive_failures:
            error_msg = (f"Circuit breaker triggered: {self._consecutive_failures} "
                        f"consecutive API failures. Halting new trade initiations.")
            self.logger.critical(error_msg)
            raise AlpacaAPIError(error_msg)

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including equity and buying power.

        Returns:
            Dictionary containing account information.
        """
        account = self._handle_api_call(self.api.get_account)

        return {
            'account_number': account.account_number,
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'day_trade_count': int(account.day_trade_count),
            'pattern_day_trader': account.pattern_day_trader
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions.

        Returns:
            List of position dictionaries.
        """
        positions = self._handle_api_call(self.api.list_positions)

        position_list = []
        for pos in positions:
            position_list.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': pos.side,
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'avg_entry_price': float(pos.avg_entry_price)
            })

        return position_list

    def get_historical_data(self,
                          symbols: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          timeframe: str = '1Day') -> pd.DataFrame:
        """Get historical OHLCV data for symbols.

        Args:
            symbols: List of stock symbols.
            start_date: Start date for data.
            end_date: End date for data.
            timeframe: Data timeframe (default: '1Day').

        Returns:
            DataFrame with historical data.
        """
        try:
            # Convert timeframe string to Alpaca TimeFrame
            if timeframe == '1Day':
                tf = TimeFrame.Day
            elif timeframe == '1Hour':
                tf = TimeFrame.Hour
            elif timeframe == '1Min':
                tf = TimeFrame.Minute
            else:
                tf = TimeFrame.Day

            bars = self._handle_api_call(
                self.api.get_bars,
                symbols,
                tf,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adjustment='raw'
            )

            # Convert to DataFrame
            data_dict = {}
            for symbol in symbols:
                if symbol in bars:
                    symbol_data = []
                    for bar in bars[symbol]:
                        symbol_data.append({
                            'timestamp': bar.t,
                            'open': float(bar.o),
                            'high': float(bar.h),
                            'low': float(bar.l),
                            'close': float(bar.c),
                            'volume': int(bar.v)
                        })
                    data_dict[symbol] = symbol_data

            # Create multi-index DataFrame
            dfs = []
            for symbol, data in data_dict.items():
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df.set_index(['timestamp', 'symbol'], inplace=True)
                dfs.append(df)

            if dfs:
                result_df = pd.concat(dfs)
                self.logger.info(f"Retrieved historical data for {len(symbols)} symbols")
                return result_df
            else:
                self.logger.warning("No historical data retrieved")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {str(e)}")
            raise AlpacaAPIError(f"Historical data retrieval failed: {str(e)}")

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            Dictionary mapping symbols to their latest prices.
        """
        try:
            latest_trades = self._handle_api_call(
                self.api.get_latest_trades,
                symbols
            )

            prices = {}
            for symbol in symbols:
                if symbol in latest_trades:
                    prices[symbol] = float(latest_trades[symbol].price)
                else:
                    self.logger.warning(f"No latest price found for {symbol}")

            return prices

        except Exception as e:
            self.logger.error(f"Failed to get latest prices: {str(e)}")
            raise AlpacaAPIError(f"Latest price retrieval failed: {str(e)}")

    def place_order(self,
                   symbol: str,
                   qty: int,
                   side: str,
                   order_type: str = 'market',
                   time_in_force: str = 'day') -> Dict[str, Any]:
        """Place a trading order.

        Args:
            symbol: Stock symbol.
            qty: Quantity of shares.
            side: 'buy' or 'sell'.
            order_type: Order type (default: 'market').
            time_in_force: Time in force (default: 'day').

        Returns:
            Dictionary containing order information.

        Raises:
            AlpacaAPIError: If order placement fails.
        """
        try:
            order = self._handle_api_call(
                self.api.submit_order,
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )

            order_info = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': int(order.qty),
                'side': order.side,
                'order_type': order.order_type,
                'status': order.status,
                'submitted_at': order.submitted_at
            }

            self.logger.info(f"Order placed: {side} {qty} shares of {symbol} (ID: {order.id})")
            return order_info

        except Exception as e:
            self.logger.error(f"Failed to place order for {symbol}: {str(e)}")
            raise AlpacaAPIError(f"Order placement failed: {str(e)}")

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of a specific order.

        Args:
            order_id: Order ID to check.

        Returns:
            Dictionary containing order status information.
        """
        try:
            order = self._handle_api_call(self.api.get_order, order_id)

            return {
                'id': order.id,
                'status': order.status,
                'filled_qty': int(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0),
                'qty': int(order.qty)
            }

        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            raise AlpacaAPIError(f"Order status retrieval failed: {str(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancellation successful, False otherwise.
        """
        try:
            self._handle_api_call(self.api.cancel_order, order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False

    def liquidate_all_positions(self) -> bool:
        """Liquidate all open positions (for kill switch functionality).

        Returns:
            True if liquidation successful, False otherwise.
        """
        try:
            self.logger.critical("LIQUIDATING ALL POSITIONS - KILL SWITCH ACTIVATED")

            positions = self.get_positions()
            if not positions:
                self.logger.info("No positions to liquidate")
                return True

            success = True
            for position in positions:
                symbol = position['symbol']
                qty = abs(int(position['qty']))
                side = 'sell' if position['qty'] > 0 else 'buy'

                try:
                    self.place_order(symbol, qty, side)
                    self.logger.critical(f"Liquidated position: {side} {qty} shares of {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to liquidate {symbol}: {str(e)}")
                    success = False

            return success

        except Exception as e:
            self.logger.critical(f"CRITICAL: Failed to liquidate all positions: {str(e)}")
            return False

    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        Returns:
            True if market is open, False otherwise.
        """
        try:
            clock = self._handle_api_call(self.api.get_clock)
            return clock.is_open

        except Exception as e:
            self.logger.warning(f"Failed to check market status: {str(e)}")
            return False

    def get_market_calendar(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get market calendar for date range.

        Args:
            start_date: Start date.
            end_date: End date.

        Returns:
            List of market calendar entries.
        """
        try:
            calendar = self._handle_api_call(
                self.api.get_calendar,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )

            return [
                {
                    'date': str(day.date),
                    'open': str(day.open),
                    'close': str(day.close)
                }
                for day in calendar
            ]

        except Exception as e:
            self.logger.error(f"Failed to get market calendar: {str(e)}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the API connection.

        Returns:
            Dictionary containing health check results.
        """
        health_status = {
            'api_connected': False,
            'last_successful_call': self._last_successful_call,
            'consecutive_failures': self._consecutive_failures,
            'circuit_breaker_active': self._consecutive_failures >= self._max_consecutive_failures
        }

        try:
            # Simple API call to test connection
            self.api.get_clock()
            health_status['api_connected'] = True
            self.logger.info("API health check passed")

        except Exception as e:
            self.logger.warning(f"API health check failed: {str(e)}")

        return health_status


if __name__ == "__main__":
    # Test the API client
    try:
        client = HeliosAlpacaClient()
        print("API client initialized successfully")

        # Test basic functionality
        account = client.get_account_info()
        print(f"Account equity: ${account['equity']:,.2f}")

        health = client.health_check()
        print(f"API Health: {health}")

    except Exception as e:
        print(f"API client test failed: {e}")
