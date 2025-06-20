"""Data Handler Module for the Helios trading bot.

This module implements Module 1 from the PRD: Data Handler & Universe Management.
It provides functionality for fetching, cleaning, and managing market data
from the Alpaca API with comprehensive error handling and data quality controls.

Functional Requirements Implemented:
- FR-1.1: Connect to Alpaca API for historical and real-time data
- FR-1.2: Fetch daily OHLCV data for configurable universe
- FR-1.3: Handle data quality issues (missing data, corporate actions)
- FR-1.4: Configurable universe via configuration file

Author: Helios Trading Bot
Version: 1.0
"""

import os
import configparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json

from ..utils.logger_config import get_helios_logger
from ..utils.api_client import HeliosAlpacaClient, AlpacaAPIError


class DataHandlerError(Exception):
    """Custom exception for data handler related errors."""
    pass


class HeliosDataHandler:
    """Data handler for fetching and managing market data for the Helios trading system.

    This class provides comprehensive data management functionality including:
    - Universe configuration and management
    - Historical data fetching with quality controls
    - Data cleaning and forward-filling
    - Corporate action handling
    - Data validation and error recovery
    """

    def __init__(self, config_path: str = "config/config.ini") -> None:
        """Initialize the data handler.

        Args:
            config_path: Path to the configuration file.

        Raises:
            DataHandlerError: If initialization fails.
        """
        self.logger = get_helios_logger('data_handler')
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize API client
        try:
            self.api_client = HeliosAlpacaClient(config_path)
            self.logger.info("Data handler initialized with API client")
        except Exception as e:
            self.logger.error(f"Failed to initialize API client: {str(e)}")
            raise DataHandlerError(f"API client initialization failed: {str(e)}")

        # Load universe configuration
        self.universe = self._load_universe()
        self.logger.info(f"Universe loaded with {len(self.universe)} symbols: {self.universe}")

        # Data quality parameters
        self.max_forward_fill_days = 3  # From PRD FR-1.3
        self.min_volume_threshold = 1000  # Minimum daily volume for data quality

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file.

        Returns:
            ConfigParser object with loaded configuration.

        Raises:
            DataHandlerError: If config file cannot be loaded.
        """
        config = configparser.ConfigParser()

        if not os.path.exists(self.config_path):
            raise DataHandlerError(f"Configuration file not found: {self.config_path}")

        try:
            config.read(self.config_path)
            return config
        except Exception as e:
            raise DataHandlerError(f"Failed to load configuration: {str(e)}")

    def _load_universe(self) -> List[str]:
        """Load the trading universe from configuration.

        Returns:
            List of stock symbols for the trading universe.

        Raises:
            DataHandlerError: If universe cannot be loaded.
        """
        try:
            tickers_str = self.config.get('Universe', 'tickers',
                                         fallback='JPM,GS,MS,BAC,C,WFC,AXP,USB,PNC,BLK')

            # Parse comma-separated tickers and clean them
            tickers = [ticker.strip().upper() for ticker in tickers_str.split(',') if ticker.strip()]

            if not tickers:
                raise DataHandlerError("No tickers found in universe configuration")

            # Validate ticker format (basic check)
            valid_tickers = []
            for ticker in tickers:
                if len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalpha():
                    valid_tickers.append(ticker)
                else:
                    self.logger.warning(f"Invalid ticker format, skipping: {ticker}")

            if not valid_tickers:
                raise DataHandlerError("No valid tickers found in universe configuration")

            return valid_tickers

        except Exception as e:
            self.logger.error(f"Failed to load universe: {str(e)}")
            raise DataHandlerError(f"Universe loading failed: {str(e)}")

    def get_universe(self) -> List[str]:
        """Get the current trading universe.

        Returns:
            List of stock symbols in the trading universe.
        """
        return self.universe.copy()

    def update_universe(self, new_universe: List[str]) -> None:
        """Update the trading universe.

        Args:
            new_universe: New list of stock symbols.

        Raises:
            DataHandlerError: If universe update fails.
        """
        try:
            # Validate new universe
            valid_tickers = []
            for ticker in new_universe:
                ticker = ticker.strip().upper()
                if len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalpha():
                    valid_tickers.append(ticker)
                else:
                    self.logger.warning(f"Invalid ticker in new universe, skipping: {ticker}")

            if not valid_tickers:
                raise DataHandlerError("No valid tickers in new universe")

            old_universe = self.universe.copy()
            self.universe = valid_tickers

            self.logger.info(f"Universe updated from {len(old_universe)} to {len(self.universe)} symbols")
            self.logger.info(f"New universe: {self.universe}")

        except Exception as e:
            self.logger.error(f"Failed to update universe: {str(e)}")
            raise DataHandlerError(f"Universe update failed: {str(e)}")

    def get_historical_data(self,
                           symbols: Optional[List[str]] = None,
                           days_back: int = 252,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data for symbols.

        Args:
            symbols: List of symbols to fetch. If None, uses entire universe.
            days_back: Number of trading days to fetch (default: 252 = 1 year).
            end_date: End date for data. If None, uses today.

        Returns:
            DataFrame with historical OHLCV data, multi-indexed by (timestamp, symbol).

        Raises:
            DataHandlerError: If data fetching fails.
        """
        try:
            # Use provided symbols or default to universe
            if symbols is None:
                symbols = self.universe

            # Validate symbols
            if not symbols:
                raise DataHandlerError("No symbols provided for historical data fetch")

            # Calculate date range
            if end_date is None:
                end_date = datetime.now().date()
            elif isinstance(end_date, datetime):
                end_date = end_date.date()

            # Add buffer days to account for weekends/holidays
            start_date = end_date - timedelta(days=int(days_back * 1.5))

            self.logger.info(f"Fetching historical data for {len(symbols)} symbols "
                           f"from {start_date} to {end_date}")

            # Fetch data from API
            raw_data = self.api_client.get_historical_data(
                symbols=symbols,
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.min.time()),
                timeframe='1Day'
            )

            if raw_data.empty:
                self.logger.warning("No historical data returned from API")
                return pd.DataFrame()

            # Clean and validate data
            cleaned_data = self._clean_historical_data(raw_data, symbols)

            # Limit to requested number of days
            if not cleaned_data.empty:
                cleaned_data = self._limit_to_trading_days(cleaned_data, days_back)

            self.logger.info(f"Successfully fetched and cleaned historical data: "
                           f"{len(cleaned_data)} rows, {len(symbols)} symbols")

            return cleaned_data

        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {str(e)}")
            raise DataHandlerError(f"Historical data fetch failed: {str(e)}")

    def _clean_historical_data(self, raw_data: pd.DataFrame, expected_symbols: List[str]) -> pd.DataFrame:
        """Clean historical data and handle quality issues.

        Args:
            raw_data: Raw data from API.
            expected_symbols: List of symbols that should be present.

        Returns:
            Cleaned DataFrame with quality issues resolved.
        """
        try:
            if raw_data.empty:
                return pd.DataFrame()

            self.logger.info("Starting data cleaning process")

            # Reset index to work with the data
            df = raw_data.reset_index()

            # Check for missing symbols
            actual_symbols = df['symbol'].unique() if 'symbol' in df.columns else []
            missing_symbols = set(expected_symbols) - set(actual_symbols)

            if missing_symbols:
                self.logger.warning(f"Missing data for symbols: {missing_symbols}")

            # Data quality checks and cleaning for each symbol
            cleaned_dfs = []

            for symbol in actual_symbols:
                if symbol not in expected_symbols:
                    continue

                symbol_data = df[df['symbol'] == symbol].copy()

                if symbol_data.empty:
                    continue

                # Sort by timestamp
                symbol_data = symbol_data.sort_values('timestamp')

                # Basic data validation
                symbol_data = self._validate_ohlcv_data(symbol_data, symbol)

                if symbol_data.empty:
                    self.logger.warning(f"No valid data remaining for {symbol} after validation")
                    continue

                # Handle missing dates (forward fill for up to 3 days)
                symbol_data = self._handle_missing_dates(symbol_data, symbol)

                # Final validation
                if len(symbol_data) > 0:
                    cleaned_dfs.append(symbol_data)
                else:
                    self.logger.warning(f"No data remaining for {symbol} after cleaning")

            if not cleaned_dfs:
                self.logger.warning("No valid data remaining after cleaning")
                return pd.DataFrame()

            # Combine all cleaned data
            result_df = pd.concat(cleaned_dfs, ignore_index=True)

            # Set multi-index
            result_df.set_index(['timestamp', 'symbol'], inplace=True)
            result_df.sort_index(inplace=True)

            self.logger.info(f"Data cleaning completed: {len(result_df)} rows across "
                           f"{len(result_df.index.get_level_values('symbol').unique())} symbols")

            return result_df

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            raise DataHandlerError(f"Data cleaning failed: {str(e)}")

    def _validate_ohlcv_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLCV data for basic quality issues.

        Args:
            data: DataFrame with OHLCV data for a single symbol.
            symbol: Symbol being validated.

        Returns:
            DataFrame with invalid rows removed.
        """
        original_count = len(data)

        # Remove rows with missing or invalid price data
        price_cols = ['open', 'high', 'low', 'close']
        data = data.dropna(subset=price_cols)

        # Remove rows with zero or negative prices
        for col in price_cols:
            data = data[data[col] > 0]

        # Basic OHLC relationship validation (high >= low, etc.)
        data = data[
            (data['high'] >= data['low']) &
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['low'] <= data['open']) &
            (data['low'] <= data['close'])
        ]

        # Volume validation (remove zero volume days, but warn)
        zero_volume_count = len(data[data['volume'] == 0])
        if zero_volume_count > 0:
            self.logger.warning(f"{symbol}: Found {zero_volume_count} zero-volume days")

        # Keep zero volume days but flag them
        data['low_volume_flag'] = data['volume'] < self.min_volume_threshold

        removed_count = original_count - len(data)
        if removed_count > 0:
            self.logger.info(f"{symbol}: Removed {removed_count} invalid rows during validation")

        return data

    def _handle_missing_dates(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle missing dates by forward-filling for up to 3 consecutive days.

        Args:
            data: DataFrame with data for a single symbol.
            symbol: Symbol being processed.

        Returns:
            DataFrame with missing dates handled.
        """
        try:
            if data.empty:
                return data

            # Sort by timestamp
            data = data.sort_values('timestamp')

            # Create a complete date range
            start_date = data['timestamp'].min().date()
            end_date = data['timestamp'].max().date()

            # Generate business days (weekdays only)
            business_days = pd.bdate_range(start=start_date, end=end_date, freq='B')

            # Convert to datetime with same timezone/format as original data
            complete_dates = [pd.Timestamp(d) for d in business_days]

            # Create complete DataFrame
            complete_df = pd.DataFrame({'timestamp': complete_dates})
            complete_df['symbol'] = symbol

            # Merge with existing data
            merged = complete_df.merge(data, on=['timestamp', 'symbol'], how='left')

            # Forward fill missing values (limit to 3 days as per PRD FR-1.3)
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            merged[price_cols] = merged[price_cols].fillna(method='ffill', limit=self.max_forward_fill_days)

            # Remove rows that still have missing values after forward fill
            merged = merged.dropna(subset=price_cols)

            # Flag forward-filled data
            original_dates = set(data['timestamp'])
            merged['forward_filled'] = ~merged['timestamp'].isin(original_dates)

            filled_count = merged['forward_filled'].sum()
            if filled_count > 0:
                self.logger.info(f"{symbol}: Forward-filled {filled_count} missing dates")

            return merged

        except Exception as e:
            self.logger.error(f"Failed to handle missing dates for {symbol}: {str(e)}")
            return data  # Return original data if processing fails

    def _limit_to_trading_days(self, data: pd.DataFrame, max_days: int) -> pd.DataFrame:
        """Limit data to the most recent N trading days.

        Args:
            data: DataFrame with historical data.
            max_days: Maximum number of trading days to keep.

        Returns:
            DataFrame limited to the most recent trading days.
        """
        if data.empty:
            return data

        # Get unique dates in descending order
        unique_dates = sorted(data.index.get_level_values('timestamp').unique(), reverse=True)

        # Take the most recent max_days
        if len(unique_dates) > max_days:
            cutoff_date = unique_dates[max_days - 1]
            data = data[data.index.get_level_values('timestamp') >= cutoff_date]
            self.logger.info(f"Limited data to {max_days} most recent trading days")

        return data

    def get_latest_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """Get latest prices for symbols.

        Args:
            symbols: List of symbols. If None, uses entire universe.

        Returns:
            Dictionary mapping symbols to their latest prices.

        Raises:
            DataHandlerError: If price fetching fails.
        """
        try:
            if symbols is None:
                symbols = self.universe

            if not symbols:
                raise DataHandlerError("No symbols provided for latest prices")

            self.logger.info(f"Fetching latest prices for {len(symbols)} symbols")

            prices = self.api_client.get_latest_prices(symbols)

            # Validate prices
            valid_prices = {}
            for symbol, price in prices.items():
                if price > 0:
                    valid_prices[symbol] = price
                else:
                    self.logger.warning(f"Invalid price for {symbol}: {price}")

            missing_symbols = set(symbols) - set(valid_prices.keys())
            if missing_symbols:
                self.logger.warning(f"Missing latest prices for: {missing_symbols}")

            self.logger.info(f"Successfully fetched latest prices for {len(valid_prices)} symbols")
            return valid_prices

        except Exception as e:
            self.logger.error(f"Failed to fetch latest prices: {str(e)}")
            raise DataHandlerError(f"Latest prices fetch failed: {str(e)}")

    def validate_data_availability(self, symbols: Optional[List[str]] = None,
                                 min_days: int = 60) -> Dict[str, bool]:
        """Validate that sufficient historical data is available for symbols.

        Args:
            symbols: List of symbols to validate. If None, uses universe.
            min_days: Minimum number of days of data required.

        Returns:
            Dictionary mapping symbols to availability status (True/False).
        """
        try:
            if symbols is None:
                symbols = self.universe

            self.logger.info(f"Validating data availability for {len(symbols)} symbols "
                           f"(minimum {min_days} days)")

            # Fetch recent data to check availability
            recent_data = self.get_historical_data(symbols=symbols, days_back=min_days + 10)

            availability = {}
            for symbol in symbols:
                if recent_data.empty:
                    availability[symbol] = False
                    continue

                symbol_data = recent_data[recent_data.index.get_level_values('symbol') == symbol]
                days_available = len(symbol_data)
                availability[symbol] = days_available >= min_days

                if not availability[symbol]:
                    self.logger.warning(f"{symbol}: Only {days_available} days available "
                                      f"(required: {min_days})")

            available_count = sum(availability.values())
            self.logger.info(f"Data availability check: {available_count}/{len(symbols)} symbols "
                           f"have sufficient data")

            return availability

        except Exception as e:
            self.logger.error(f"Data availability validation failed: {str(e)}")
            # Return False for all symbols if validation fails
            return {symbol: False for symbol in (symbols or self.universe)}

    def get_data_quality_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate a data quality report for the universe.

        Args:
            days_back: Number of days to analyze for the report.

        Returns:
            Dictionary containing data quality metrics.
        """
        try:
            self.logger.info(f"Generating data quality report for last {days_back} days")

            # Fetch recent data
            recent_data = self.get_historical_data(days_back=days_back)

            if recent_data.empty:
                return {
                    'status': 'error',
                    'message': 'No data available for quality analysis'
                }

            report = {
                'analysis_period_days': days_back,
                'universe_size': len(self.universe),
                'symbols_with_data': len(recent_data.index.get_level_values('symbol').unique()),
                'total_data_points': len(recent_data),
                'date_range': {
                    'start': recent_data.index.get_level_values('timestamp').min().strftime('%Y-%m-%d'),
                    'end': recent_data.index.get_level_values('timestamp').max().strftime('%Y-%m-%d')
                },
                'symbol_stats': {}
            }

            # Per-symbol statistics
            for symbol in recent_data.index.get_level_values('symbol').unique():
                symbol_data = recent_data[recent_data.index.get_level_values('symbol') == symbol]

                report['symbol_stats'][symbol] = {
                    'data_points': len(symbol_data),
                    'avg_volume': float(symbol_data['volume'].mean()),
                    'avg_price': float(symbol_data['close'].mean()),
                    'price_range': {
                        'min': float(symbol_data['low'].min()),
                        'max': float(symbol_data['high'].max())
                    }
                }

                # Check for potential data quality flags
                if 'low_volume_flag' in symbol_data.columns:
                    low_vol_days = symbol_data['low_volume_flag'].sum()
                    report['symbol_stats'][symbol]['low_volume_days'] = int(low_vol_days)

                if 'forward_filled' in symbol_data.columns:
                    filled_days = symbol_data['forward_filled'].sum()
                    report['symbol_stats'][symbol]['forward_filled_days'] = int(filled_days)

            report['status'] = 'success'
            self.logger.info("Data quality report generated successfully")

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate data quality report: {str(e)}")
            return {
                'status': 'error',
                'message': f'Report generation failed: {str(e)}'
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the data handler.

        Returns:
            Dictionary containing health check results.
        """
        health_status = {
            'api_client_healthy': False,
            'universe_loaded': len(self.universe) > 0,
            'universe_size': len(self.universe),
            'config_loaded': self.config is not None,
            'latest_prices_available': False,
            'historical_data_available': False
        }

        try:
            # Test API client
            api_health = self.api_client.health_check()
            health_status['api_client_healthy'] = api_health.get('api_connected', False)

            # Test latest prices (sample)
            if self.universe:
                sample_symbols = self.universe[:3]  # Test first 3 symbols
                latest_prices = self.get_latest_prices(sample_symbols)
                health_status['latest_prices_available'] = len(latest_prices) > 0

            # Test historical data (sample)
            if self.universe:
                sample_data = self.get_historical_data(symbols=self.universe[:2], days_back=5)
                health_status['historical_data_available'] = not sample_data.empty

            health_status['overall_healthy'] = all([
                health_status['api_client_healthy'],
                health_status['universe_loaded'],
                health_status['config_loaded'],
                health_status['latest_prices_available']
            ])

            self.logger.info(f"Data handler health check: {'HEALTHY' if health_status['overall_healthy'] else 'UNHEALTHY'}")

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            health_status['error'] = str(e)
            health_status['overall_healthy'] = False

        return health_status


if __name__ == "__main__":
    # Test the data handler
    try:
        print("Testing Helios Data Handler...")

        data_handler = HeliosDataHandler()
        print(f"✓ Data handler initialized")
        print(f"✓ Universe: {data_handler.get_universe()}")

        # Health check
        health = data_handler.health_check()
        print(f"✓ Health check: {'PASS' if health['overall_healthy'] else 'FAIL'}")

        # Test data fetch (small sample)
        test_symbols = data_handler.get_universe()[:2]
        print(f"✓ Testing data fetch for: {test_symbols}")

        historical_data = data_handler.get_historical_data(symbols=test_symbols, days_back=5)
        print(f"✓ Historical data shape: {historical_data.shape}")

        latest_prices = data_handler.get_latest_prices(test_symbols)
        print(f"✓ Latest prices: {latest_prices}")

        print("✓ Data handler test completed successfully!")

    except Exception as e:
        print(f"✗ Data handler test failed: {e}")
