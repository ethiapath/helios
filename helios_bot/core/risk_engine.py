"""Risk & Position Sizing Engine for the Helios trading bot.

This module implements Module 3 from the PRD: Risk & Position Sizing Engine.
It provides comprehensive risk management and position sizing functionality
based on volatility-parity models and configurable risk parameters.

Functional Requirements Implemented:
- FR-3.1: ATR-based position sizing with volatility-parity model
- FR-3.2: Position concentration limits (20% max per position)
- FR-3.3: Max concurrent pairs limit (5 pairs maximum)
- Portfolio equity monitoring and risk factor calculations
- Pre-trade risk validation and position approval

Author: Helios Trading Bot
Version: 1.0
"""

import os
import configparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings

from ..utils.logger_config import get_helios_logger
from ..utils.api_client import HeliosAlpacaClient, AlpacaAPIError
from .data_handler import HeliosDataHandler, DataHandlerError


class RiskEngineError(Exception):
    """Custom exception for Risk Engine related errors."""
    pass


class RiskLevel(Enum):
    """Enumeration for risk assessment levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    REJECTED = "rejected"


class PositionSide(Enum):
    """Enumeration for position sides in pair trades."""
    LONG = "long"
    SHORT = "short"


@dataclass
class PositionSize:
    """Data structure for calculated position sizes."""
    symbol: str
    side: PositionSide
    shares: int
    notional_value: float
    risk_amount: float
    atr_value: float
    price: float
    portfolio_percentage: float
    calculation_timestamp: datetime


@dataclass
class RiskCheck:
    """Data structure for risk assessment results."""
    passed: bool
    risk_level: RiskLevel
    total_risk_amount: float
    portfolio_utilization: float
    concentration_check: bool
    concurrent_pairs_check: bool
    individual_position_checks: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    assessment_timestamp: datetime


@dataclass
class PortfolioRisk:
    """Data structure for overall portfolio risk metrics."""
    total_equity: float
    cash_available: float
    buying_power: float
    current_positions_count: int
    total_exposure: float
    risk_utilization: float
    max_position_size: float
    available_risk_budget: float
    last_updated: datetime


class HeliosRiskEngine:
    """Risk & Position Sizing Engine for the Helios trading system.

    This class provides comprehensive risk management including:
    - ATR-based position sizing with volatility adjustment
    - Portfolio concentration and exposure limits
    - Concurrent position limits and monitoring
    - Pre-trade risk validation and approval
    - Real-time portfolio risk assessment
    """

    def __init__(self,
                 data_handler: HeliosDataHandler,
                 api_client: HeliosAlpacaClient,
                 config_path: str = "config/config.ini") -> None:
        """Initialize the Risk & Position Sizing Engine.

        Args:
            data_handler: Initialized data handler for market data.
            api_client: Initialized API client for account data.
            config_path: Path to the configuration file.

        Raises:
            RiskEngineError: If initialization fails.
        """
        self.logger = get_helios_logger('risk_engine')
        self.data_handler = data_handler
        self.api_client = api_client
        self.config_path = config_path
        self.config = self._load_config()

        # Load risk parameters from config
        self._load_risk_parameters()

        # Portfolio risk tracking
        self._portfolio_risk: Optional[PortfolioRisk] = None
        self._last_portfolio_update: Optional[datetime] = None
        self._portfolio_cache_minutes = 5  # Cache portfolio data for 5 minutes

        # Current positions tracking (for concurrent limits)
        self._current_positions: List[Dict[str, Any]] = []

        self.logger.info("Risk engine initialized successfully")

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file.

        Returns:
            ConfigParser object with loaded configuration.

        Raises:
            RiskEngineError: If config file cannot be loaded.
        """
        if not os.path.exists(self.config_path):
            raise RiskEngineError(f"Configuration file not found: {self.config_path}")

        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
            return config
        except Exception as e:
            raise RiskEngineError(f"Failed to load configuration: {str(e)}")

    def _load_risk_parameters(self) -> None:
        """Load risk management parameters from configuration."""
        try:
            # Position sizing parameters
            self.risk_factor_per_leg = self.config.getfloat(
                'Risk', 'risk_factor_per_leg', fallback=0.005)  # 0.5%
            self.atr_period = self.config.getint(
                'Risk', 'atr_period', fallback=20)
            self.atr_multiplier = self.config.getfloat(
                'Risk', 'atr_multiplier', fallback=2.0)

            # Position limits
            self.max_position_concentration = self.config.getfloat(
                'Risk', 'max_position_concentration', fallback=0.20)  # 20%
            self.max_concurrent_pairs = self.config.getint(
                'Risk', 'max_concurrent_pairs', fallback=5)

            # Portfolio risk limits
            self.max_daily_drawdown_limit = self.config.getfloat(
                'Risk', 'max_daily_drawdown_limit', fallback=0.03)  # 3%

            # Validation parameters
            self.min_price_threshold = 1.00  # Minimum stock price for trading
            self.max_spread_percentage = 0.10  # Maximum bid-ask spread as % of price

            self.logger.info("Risk parameters loaded successfully")
            self.logger.info(f"Risk factor per leg: {self.risk_factor_per_leg:.3%}")
            self.logger.info(f"Max position concentration: {self.max_position_concentration:.1%}")
            self.logger.info(f"Max concurrent pairs: {self.max_concurrent_pairs}")

        except Exception as e:
            raise RiskEngineError(f"Failed to load risk parameters: {str(e)}")

    def calculate_atr(self, symbol: str, period: Optional[int] = None) -> float:
        """Calculate Average True Range (ATR) for a symbol.

        Args:
            symbol: Stock symbol.
            period: ATR period in days (uses config default if None).

        Returns:
            ATR value in dollars.

        Raises:
            RiskEngineError: If ATR calculation fails.
        """
        try:
            if period is None:
                period = self.atr_period

            # Get historical OHLC data
            historical_data = self.data_handler.get_historical_data(
                symbols=[symbol],
                days_back=period + 10  # Extra buffer
            )

            if historical_data.empty:
                raise RiskEngineError(f"No historical data available for ATR calculation: {symbol}")

            # Extract symbol data
            symbol_data = historical_data[
                historical_data.index.get_level_values('symbol') == symbol
            ].sort_index()

            if len(symbol_data) < period:
                raise RiskEngineError(f"Insufficient data for ATR calculation: "
                                    f"{len(symbol_data)} < {period} days")

            # Calculate True Range components
            high = symbol_data['high']
            low = symbol_data['low']
            close_prev = symbol_data['close'].shift(1)

            # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
            tr1 = high - low
            tr2 = np.abs(high - close_prev)
            tr3 = np.abs(low - close_prev)

            true_range = np.maximum(tr1, np.maximum(tr2, tr3))

            # Calculate ATR as simple moving average of True Range
            atr = true_range.rolling(window=period).mean().iloc[-1]

            if np.isnan(atr) or atr <= 0:
                raise RiskEngineError(f"Invalid ATR calculation result: {atr}")

            self.logger.debug(f"ATR calculated for {symbol}: ${atr:.4f} ({period} days)")
            return float(atr)

        except Exception as e:
            self.logger.error(f"ATR calculation failed for {symbol}: {str(e)}")
            raise RiskEngineError(f"ATR calculation failed for {symbol}: {str(e)}")

    def get_portfolio_risk(self, force_update: bool = False) -> PortfolioRisk:
        """Get current portfolio risk metrics.

        Args:
            force_update: Force update even if cache is valid.

        Returns:
            PortfolioRisk object with current metrics.

        Raises:
            RiskEngineError: If portfolio data cannot be retrieved.
        """
        try:
            # Check cache validity
            if not force_update and self._portfolio_risk and self._last_portfolio_update:
                cache_age = (datetime.now() - self._last_portfolio_update).total_seconds() / 60
                if cache_age < self._portfolio_cache_minutes:
                    return self._portfolio_risk

            self.logger.info("Updating portfolio risk metrics")

            # Get account information from API
            account_info = self.api_client.get_account_info()
            current_positions = self.api_client.get_positions()

            # Calculate current exposure
            total_exposure = sum(abs(pos['market_value']) for pos in current_positions)

            # Calculate risk utilization
            total_equity = account_info['equity']
            risk_utilization = total_exposure / total_equity if total_equity > 0 else 0

            # Calculate maximum single position size
            max_position_size = total_equity * self.max_position_concentration

            # Calculate available risk budget
            current_risk_usage = len(current_positions) * self.risk_factor_per_leg * total_equity
            max_risk_budget = self.max_concurrent_pairs * 2 * self.risk_factor_per_leg * total_equity  # 2 legs per pair
            available_risk_budget = max_risk_budget - current_risk_usage

            # Create portfolio risk object
            self._portfolio_risk = PortfolioRisk(
                total_equity=total_equity,
                cash_available=account_info['cash'],
                buying_power=account_info['buying_power'],
                current_positions_count=len(current_positions),
                total_exposure=total_exposure,
                risk_utilization=risk_utilization,
                max_position_size=max_position_size,
                available_risk_budget=max(0, available_risk_budget),
                last_updated=datetime.now()
            )

            self._current_positions = current_positions
            self._last_portfolio_update = datetime.now()

            self.logger.info(f"Portfolio risk updated: Equity=${total_equity:,.2f}, "
                           f"Exposure=${total_exposure:,.2f}, "
                           f"Risk utilization={risk_utilization:.1%}")

            return self._portfolio_risk

        except Exception as e:
            self.logger.error(f"Failed to get portfolio risk: {str(e)}")
            raise RiskEngineError(f"Portfolio risk calculation failed: {str(e)}")

    def calculate_position_size(self,
                              symbol: str,
                              side: PositionSide,
                              current_price: Optional[float] = None) -> PositionSize:
        """Calculate position size using ATR-based volatility adjustment (FR-3.1).

        Args:
            symbol: Stock symbol.
            side: Position side (long or short).
            current_price: Current stock price (fetched if None).

        Returns:
            PositionSize object with calculated size and metrics.

        Raises:
            RiskEngineError: If position size calculation fails.
        """
        try:
            self.logger.info(f"Calculating position size for {symbol} ({side.value})")

            # Get current price if not provided
            if current_price is None:
                latest_prices = self.data_handler.get_latest_prices([symbol])
                current_price = latest_prices.get(symbol)
                if not current_price or current_price <= 0:
                    raise RiskEngineError(f"Invalid current price for {symbol}: {current_price}")

            # Validate minimum price threshold
            if current_price < self.min_price_threshold:
                raise RiskEngineError(f"Price below minimum threshold: ${current_price:.2f} < ${self.min_price_threshold:.2f}")

            # Calculate ATR
            atr_value = self.calculate_atr(symbol)

            # Get portfolio metrics
            portfolio_risk = self.get_portfolio_risk()

            # Calculate position size using the formula from PRD:
            # Shares = (Total_Portfolio_Equity * Risk_Factor_Per_Leg) / (ATR_Multiplier * 20_Day_ATR)
            risk_amount = portfolio_risk.total_equity * self.risk_factor_per_leg
            position_risk_denominator = self.atr_multiplier * atr_value

            if position_risk_denominator <= 0:
                raise RiskEngineError(f"Invalid position risk denominator: {position_risk_denominator}")

            shares_float = risk_amount / position_risk_denominator
            shares = int(round(shares_float))

            # Ensure minimum position size
            if shares < 1:
                shares = 1

            # Calculate notional value and portfolio percentage
            notional_value = shares * current_price
            portfolio_percentage = notional_value / portfolio_risk.total_equity

            # Check concentration limit
            if portfolio_percentage > self.max_position_concentration:
                # Scale down to concentration limit
                max_notional = portfolio_risk.total_equity * self.max_position_concentration
                shares = int(max_notional / current_price)
                notional_value = shares * current_price
                portfolio_percentage = notional_value / portfolio_risk.total_equity

                self.logger.warning(f"Position size scaled down due to concentration limit: "
                                  f"{symbol} {shares} shares ({portfolio_percentage:.1%})")

            position_size = PositionSize(
                symbol=symbol,
                side=side,
                shares=shares,
                notional_value=notional_value,
                risk_amount=risk_amount,
                atr_value=atr_value,
                price=current_price,
                portfolio_percentage=portfolio_percentage,
                calculation_timestamp=datetime.now()
            )

            self.logger.info(f"Position size calculated for {symbol}: {shares} shares "
                           f"(${notional_value:,.2f}, {portfolio_percentage:.1%} of portfolio)")

            return position_size

        except Exception as e:
            self.logger.error(f"Position size calculation failed for {symbol}: {str(e)}")
            raise RiskEngineError(f"Position size calculation failed: {str(e)}")

    def calculate_pair_position_sizes(self,
                                    stock1: str,
                                    stock2: str,
                                    hedge_ratio: float,
                                    signal_type: str) -> Tuple[PositionSize, PositionSize]:
        """Calculate position sizes for both legs of a pair trade.

        Args:
            stock1: First stock symbol.
            stock2: Second stock symbol.
            hedge_ratio: Hedge ratio from cointegration analysis.
            signal_type: Type of signal (determines position directions).

        Returns:
            Tuple of (position1, position2) PositionSize objects.

        Raises:
            RiskEngineError: If pair position calculation fails.
        """
        try:
            self.logger.info(f"Calculating pair position sizes: {stock1}/{stock2} "
                           f"(hedge_ratio: {hedge_ratio:.4f})")

            # Get current prices
            latest_prices = self.data_handler.get_latest_prices([stock1, stock2])
            price1 = latest_prices.get(stock1)
            price2 = latest_prices.get(stock2)

            if not price1 or not price2 or price1 <= 0 or price2 <= 0:
                raise RiskEngineError(f"Invalid prices for pair: {stock1}=${price1}, {stock2}=${price2}")

            # Determine position directions based on signal type
            if signal_type.lower() == 'pairs_long_spread':
                # Long spread: long stock1, short stock2
                side1 = PositionSide.LONG
                side2 = PositionSide.SHORT
            elif signal_type.lower() == 'pairs_short_spread':
                # Short spread: short stock1, long stock2
                side1 = PositionSide.SHORT
                side2 = PositionSide.LONG
            else:
                raise RiskEngineError(f"Unknown signal type for pair positioning: {signal_type}")

            # Calculate individual position sizes
            position1 = self.calculate_position_size(stock1, side1, price1)

            # For the second position, adjust size based on hedge ratio
            # The hedge ratio tells us how many units of stock2 per unit of stock1
            adjusted_shares2 = int(round(position1.shares * abs(hedge_ratio)))

            # Ensure minimum position size
            if adjusted_shares2 < 1:
                adjusted_shares2 = 1

            # Create position2 with adjusted size
            portfolio_risk = self.get_portfolio_risk()
            notional_value2 = adjusted_shares2 * price2
            portfolio_percentage2 = notional_value2 / portfolio_risk.total_equity

            # Check concentration limit for second position
            if portfolio_percentage2 > self.max_position_concentration:
                max_notional2 = portfolio_risk.total_equity * self.max_position_concentration
                adjusted_shares2 = int(max_notional2 / price2)
                notional_value2 = adjusted_shares2 * price2
                portfolio_percentage2 = notional_value2 / portfolio_risk.total_equity

            position2 = PositionSize(
                symbol=stock2,
                side=side2,
                shares=adjusted_shares2,
                notional_value=notional_value2,
                risk_amount=position1.risk_amount,  # Same risk budget for the pair
                atr_value=self.calculate_atr(stock2),
                price=price2,
                portfolio_percentage=portfolio_percentage2,
                calculation_timestamp=datetime.now()
            )

            self.logger.info(f"Pair positions calculated: {stock1} {position1.shares} shares "
                           f"({side1.value}), {stock2} {position2.shares} shares ({side2.value})")

            return position1, position2

        except Exception as e:
            self.logger.error(f"Pair position calculation failed: {str(e)}")
            raise RiskEngineError(f"Pair position calculation failed: {str(e)}")

    def validate_trade_risk(self,
                          positions: List[PositionSize],
                          open_positions: Optional[List[Dict[str, Any]]] = None) -> RiskCheck:
        """Validate risk for a proposed trade (FR-3.2, FR-3.3).

        Args:
            positions: List of PositionSize objects for the proposed trade.
            open_positions: List of current open positions (fetched if None).

        Returns:
            RiskCheck object with validation results.

        Raises:
            RiskEngineError: If risk validation fails.
        """
        try:
            self.logger.info(f"Validating trade risk for {len(positions)} positions")

            # Initialize risk check
            warnings = []
            errors = []
            individual_checks = []

            # Get current portfolio state
            portfolio_risk = self.get_portfolio_risk()

            if open_positions is None:
                open_positions = self._current_positions

            # Calculate total risk amount for this trade
            total_risk_amount = sum(pos.risk_amount for pos in positions)

            # Check 1: Portfolio utilization
            current_exposure = sum(abs(pos['market_value']) for pos in open_positions)
            new_exposure = sum(pos.notional_value for pos in positions)
            total_exposure = current_exposure + new_exposure
            portfolio_utilization = total_exposure / portfolio_risk.total_equity

            if portfolio_utilization > 0.8:  # 80% utilization warning
                warnings.append(f"High portfolio utilization: {portfolio_utilization:.1%}")

            # Check 2: Concentration limits (FR-3.2)
            concentration_check = True
            for position in positions:
                if position.portfolio_percentage > self.max_position_concentration:
                    concentration_check = False
                    errors.append(f"Position concentration exceeds limit: {position.symbol} "
                                f"{position.portfolio_percentage:.1%} > {self.max_position_concentration:.1%}")

                individual_checks.append({
                    'symbol': position.symbol,
                    'concentration': position.portfolio_percentage,
                    'concentration_limit': self.max_position_concentration,
                    'concentration_ok': position.portfolio_percentage <= self.max_position_concentration
                })

            # Check 3: Concurrent pairs limit (FR-3.3)
            current_pairs = self._count_current_pairs(open_positions)
            new_pairs = len(positions) // 2  # Assuming pairs come in sets of 2
            total_pairs = current_pairs + new_pairs

            concurrent_pairs_check = total_pairs <= self.max_concurrent_pairs
            if not concurrent_pairs_check:
                errors.append(f"Concurrent pairs limit exceeded: {total_pairs} > {self.max_concurrent_pairs}")

            # Check 4: Available risk budget
            if total_risk_amount > portfolio_risk.available_risk_budget:
                errors.append(f"Insufficient risk budget: ${total_risk_amount:,.2f} > "
                            f"${portfolio_risk.available_risk_budget:,.2f}")

            # Check 5: Individual position validations
            for position in positions:
                # Minimum viable position size
                if position.shares < 1:
                    errors.append(f"Position size too small: {position.symbol} {position.shares} shares")

                # Price validation
                if position.price < self.min_price_threshold:
                    errors.append(f"Price below minimum threshold: {position.symbol} ${position.price:.2f}")

                # ATR validation
                if position.atr_value <= 0:
                    errors.append(f"Invalid ATR value: {position.symbol} ATR={position.atr_value:.4f}")

            # Determine overall risk level
            if errors:
                risk_level = RiskLevel.REJECTED
                passed = False
            elif portfolio_utilization > 0.9:
                risk_level = RiskLevel.CRITICAL
                passed = False
            elif portfolio_utilization > 0.7 or total_pairs >= self.max_concurrent_pairs:
                risk_level = RiskLevel.HIGH
                passed = True
            elif portfolio_utilization > 0.5 or warnings:
                risk_level = RiskLevel.MODERATE
                passed = True
            else:
                risk_level = RiskLevel.LOW
                passed = True

            risk_check = RiskCheck(
                passed=passed,
                risk_level=risk_level,
                total_risk_amount=total_risk_amount,
                portfolio_utilization=portfolio_utilization,
                concentration_check=concentration_check,
                concurrent_pairs_check=concurrent_pairs_check,
                individual_position_checks=individual_checks,
                warnings=warnings,
                errors=errors,
                assessment_timestamp=datetime.now()
            )

            result_str = "PASSED" if passed else "REJECTED"
            self.logger.info(f"Risk validation {result_str}: {risk_level.value} risk level, "
                           f"{portfolio_utilization:.1%} utilization")

            if errors:
                for error in errors:
                    self.logger.error(f"Risk validation error: {error}")

            if warnings:
                for warning in warnings:
                    self.logger.warning(f"Risk validation warning: {warning}")

            return risk_check

        except Exception as e:
            self.logger.error(f"Risk validation failed: {str(e)}")
            raise RiskEngineError(f"Risk validation failed: {str(e)}")

    def _count_current_pairs(self, positions: List[Dict[str, Any]]) -> int:
        """Count the number of current pair trades.

        Args:
            positions: List of current positions.

        Returns:
            Number of pair trades currently open.
        """
        # This is a simplified count - in a real implementation,
        # we would track pairs more explicitly in the state management
        return len(positions) // 2  # Assume positions come in pairs

    def check_portfolio_drawdown(self, start_of_day_equity: float) -> bool:
        """Check if daily drawdown limit has been breached.

        Args:
            start_of_day_equity: Portfolio equity at start of trading day.

        Returns:
            True if drawdown limit exceeded, False otherwise.

        Raises:
            RiskEngineError: If drawdown check fails.
        """
        try:
            if start_of_day_equity <= 0:
                raise RiskEngineError(f"Invalid start of day equity: ${start_of_day_equity}")

            current_portfolio = self.get_portfolio_risk(force_update=True)
            current_equity = current_portfolio.total_equity

            drawdown = (start_of_day_equity - current_equity) / start_of_day_equity

            drawdown_exceeded = drawdown > self.max_daily_drawdown_limit

            if drawdown_exceeded:
                self.logger.critical(f"DAILY DRAWDOWN LIMIT EXCEEDED: {drawdown:.2%} > "
                                   f"{self.max_daily_drawdown_limit:.2%}")
                self.logger.critical(f"Start equity: ${start_of_day_equity:,.2f}, "
                                   f"Current equity: ${current_equity:,.2f}")
            else:
                self.logger.info(f"Daily drawdown check: {drawdown:.2%} "
                               f"(limit: {self.max_daily_drawdown_limit:.2%})")

            return drawdown_exceeded

        except Exception as e:
            self.logger.error(f"Drawdown check failed: {str(e)}")
            raise RiskEngineError(f"Drawdown check failed: {str(e)}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary for the portfolio.

        Returns:
            Dictionary containing risk metrics and status.
        """
        try:
            portfolio_risk = self.get_portfolio_risk()

            summary = {
                'timestamp': datetime.now(),
                'portfolio_equity': portfolio_risk.total_equity,
                'cash_available': portfolio_risk.cash_available,
                'buying_power': portfolio_risk.buying_power,
                'current_positions': portfolio_risk.current_positions_count,
                'total_exposure': portfolio_risk.total_exposure,
                'risk_utilization': portfolio_risk.risk_utilization,
                'max_position_size': portfolio_risk.max_position_size,
                'available_risk_budget': portfolio_risk.available_risk_budget,
                'risk_parameters': {
                    'risk_factor_per_leg': self.risk_factor_per_leg,
                    'max_position_concentration': self.max_position_concentration,
                    'max_concurrent_pairs': self.max_concurrent_pairs,
                    'atr_period': self.atr_period,
                    'atr_multiplier': self.atr_multiplier
                },
                'limits_status': {
                    'concentration_usage': portfolio_risk.risk_utilization / self.max_position_concentration,
                    'pairs_usage': self._count_current_pairs(self._current_positions) / self.max_concurrent_pairs,
                    'risk_budget_usage': 1.0 - (portfolio_risk.available_risk_budget /
                                               (self.max_concurrent_pairs * 2 * self.risk_factor_per_leg * portfolio_risk.total_equity))
                }
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to generate risk summary: {str(e)}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the risk engine.

        Returns:
            Dictionary containing health check results.
        """
        health_status = {
            'config_loaded': self.config is not None,
            'data_handler_available': self.data_handler is not None,
            'api_client_available': self.api_client is not None,
            'portfolio_data_accessible': False,
            'atr_calculation_working': False,
            'position_sizing_working': False
        }

        try:
            # Test portfolio data access
            try:
                portfolio_risk = self.get_portfolio_risk()
                health_status['portfolio_data_accessible'] = True
                health_status['portfolio_equity'] = portfolio_risk.total_equity
            except Exception as e:
                health_status['portfolio_data_error'] = str(e)

            # Test ATR calculation
            try:
                universe = self.data_handler.get_universe()
                if universe:
                    test_symbol = universe[0]
                    atr_value = self.calculate_atr(test_symbol)
                    health_status['atr_calculation_working'] = atr_value > 0
                    health_status['test_atr_value'] = atr_value
            except Exception as e:
                health_status['atr_calculation_error'] = str(e)

            # Test position sizing
            try:
                if universe:
                    test_symbol = universe[0]
                    position_size = self.calculate_position_size(test_symbol, PositionSide.LONG)
                    health_status['position_sizing_working'] = position_size.shares > 0
            except Exception as e:
                health_status['position_sizing_error'] = str(e)

            # Overall health assessment
            critical_checks = [
                'config_loaded',
                'data_handler_available',
                'api_client_available',
                'portfolio_data_accessible'
            ]

            health_status['overall_healthy'] = all(
                health_status.get(check, False) for check in critical_checks
            )

            if health_status['overall_healthy']:
                self.logger.info("Risk engine health check: HEALTHY")
            else:
                self.logger.warning("Risk engine health check: UNHEALTHY")

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            health_status['error'] = str(e)
            health_status['overall_healthy'] = False

        return health_status


if __name__ == "__main__":
    # Test the risk engine
    try:
        from .data_handler import HeliosDataHandler
        from ..utils.api_client import HeliosAlpacaClient

        print("Testing Helios Risk Engine...")

        # Initialize dependencies
        data_handler = HeliosDataHandler()
        api_client = HeliosAlpacaClient()
        print("✓ Dependencies initialized")

        # Initialize risk engine
        risk_engine = HeliosRiskEngine(data_handler, api_client)
        print("✓ Risk engine initialized")

        # Health check
        health = risk_engine.health_check()
        print(f"✓ Health check: {'PASS' if health['overall_healthy'] else 'FAIL'}")

        # Test portfolio risk
        print("✓ Testing portfolio risk metrics...")
        try:
            portfolio_risk = risk_engine.get_portfolio_risk()
            print(f"✓ Portfolio equity: ${portfolio_risk.total_equity:,.2f}")
            print(f"✓ Risk utilization: {portfolio_risk.risk_utilization:.1%}")
            print(f"✓ Available risk budget: ${portfolio_risk.available_risk_budget:,.2f}")
        except Exception as e:
            print(f"⚠ Portfolio risk test failed: {e}")

        # Test ATR calculation
        print("✓ Testing ATR calculation...")
        try:
            universe = data_handler.get_universe()
            if universe:
                test_symbol = universe[0]
                atr_value = risk_engine.calculate_atr(test_symbol)
                print(f"✓ ATR for {test_symbol}: ${atr_value:.4f}")
            else:
                print("⚠ No universe symbols for ATR test")
        except Exception as e:
            print(f"⚠ ATR calculation test failed: {e}")

        # Test position sizing
        print("✓ Testing position sizing...")
        try:
            if universe:
                test_symbol = universe[0]
                position_size = risk_engine.calculate_position_size(
                    test_symbol, PositionSide.LONG
                )
                print(f"✓ Position size for {test_symbol}: {position_size.shares} shares")
                print(f"  - Notional: ${position_size.notional_value:,.2f}")
                print(f"  - Portfolio %: {position_size.portfolio_percentage:.1%}")
                print(f"  - ATR: ${position_size.atr_value:.4f}")
        except Exception as e:
            print(f"⚠ Position sizing test failed: {e}")

        # Test risk validation
        print("✓ Testing risk validation...")
        try:
            if universe and len(universe) >= 2:
                # Create test positions
                test_positions = [
                    risk_engine.calculate_position_size(universe[0], PositionSide.LONG),
                    risk_engine.calculate_position_size(universe[1], PositionSide.SHORT)
                ]

                risk_check = risk_engine.validate_trade_risk(test_positions)
                print(f"✓ Risk validation: {'PASS' if risk_check.passed else 'FAIL'}")
                print(f"  - Risk level: {risk_check.risk_level.value}")
                print(f"  - Portfolio utilization: {risk_check.portfolio_utilization:.1%}")

                if risk_check.warnings:
                    print(f"  - Warnings: {len(risk_check.warnings)}")
                if risk_check.errors:
                    print(f"  - Errors: {len(risk_check.errors)}")
        except Exception as e:
            print(f"⚠ Risk validation test failed: {e}")

        # Get risk summary
        print("✓ Testing risk summary...")
        try:
            summary = risk_engine.get_risk_summary()
            if 'error' not in summary:
                print(f"✓ Risk summary generated successfully")
                print(f"  - Current positions: {summary.get('current_positions', 0)}")
                print(f"  - Risk utilization: {summary.get('risk_utilization', 0):.1%}")
            else:
                print(f"⚠ Risk summary error: {summary['error']}")
        except Exception as e:
            print(f"⚠ Risk summary test failed: {e}")

        print("✓ Risk engine test completed successfully!")

    except Exception as e:
        print(f"✗ Risk engine test failed: {e}")
