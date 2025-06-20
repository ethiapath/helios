"""Main Controller for the Helios trading bot.

This module implements the central orchestration system that coordinates all
Helios modules and implements comprehensive safety mechanisms as specified
in the Product Requirements Document (PRD).

Safety Mechanisms Implemented:
- SL-1.1: Trade-level Z-score failure stop (|Z-score| > 3.5)
- SL-1.2: Time-in-trade stop (60 trading days maximum)
- SL-2.1: Portfolio-level daily drawdown circuit breaker (3% limit)
- SL-2.2: Correlation check failure for open positions (<0.70)
- SL-3.1: Manual kill switch capability
- SL-3.2: API connectivity failure handler (3 consecutive failures)

Author: Helios Trading Bot
Version: 1.0
"""

import os
import json
import configparser
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import tempfile
import shutil

from .utils.logger_config import get_helios_logger, log_critical_error, log_trade_event, log_risk_event
from .utils.api_client import HeliosAlpacaClient, AlpacaAPIError
from .core.data_handler import HeliosDataHandler, DataHandlerError
from .core.signal_engine import HeliosSignalEngine, SignalEngineError, PairSignal, SignalType
from .core.risk_engine import HeliosRiskEngine, RiskEngineError, PositionSize, RiskCheck
from .core.execution_engine import HeliosExecutionEngine, ExecutionEngineError, TradeExecution


class HeliosControllerError(Exception):
    """Custom exception for Main Controller related errors."""
    pass


class SystemState(Enum):
    """Enumeration for system operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    SHUTDOWN = "shutdown"


class TradingPhase(Enum):
    """Enumeration for daily trading cycle phases."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    ACTIVE_TRADING = "active_trading"
    POSITION_MONITORING = "position_monitoring"
    POST_MARKET = "post_market"
    MAINTENANCE = "maintenance"


@dataclass
class SystemStatus:
    """Data structure for comprehensive system status."""
    state: SystemState
    phase: TradingPhase
    timestamp: datetime
    equity_start_of_day: float
    current_equity: float
    drawdown_today: float
    active_positions: int
    candidate_pairs: int
    last_signal_check: Optional[datetime]
    last_risk_check: Optional[datetime]
    consecutive_api_failures: int
    kill_switch_active: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class Position:
    """Data structure for tracking open positions."""
    id: str
    stock1: str
    stock2: str
    direction: str  # LONG_SPREAD or SHORT_SPREAD
    entry_timestamp: datetime
    entry_zscore: float
    shares1: int
    shares2: int
    entry_price1: float
    entry_price2: float
    hedge_ratio: float
    days_in_trade: int
    current_zscore: Optional[float] = None
    current_correlation: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class HeliosMainController:
    """Main Controller for the Helios systematic trading system.

    This class provides comprehensive system orchestration including:
    - Daily trading cycle management
    - Safety mechanism enforcement
    - State persistence and recovery
    - Emergency stop procedures
    - Health monitoring and alerting
    """

    def __init__(self, config_path: str = "config/config.ini") -> None:
        """Initialize the Helios Main Controller.

        Args:
            config_path: Path to the configuration file.

        Raises:
            HeliosControllerError: If initialization fails.
        """
        self.logger = get_helios_logger('main_controller')
        self.config_path = config_path
        self.config = self._load_config()

        # System state
        self.system_state = SystemState.INITIALIZING
        self.trading_phase = TradingPhase.MAINTENANCE
        self.kill_switch_active = False
        self.emergency_stop_triggered = False

        # Load configuration parameters
        self._load_system_parameters()

        # Initialize modules
        self._initialize_modules()

        # Position and state management
        self.positions: Dict[str, Position] = {}
        self.equity_start_of_day: float = 0.0
        self.last_position_check: Optional[datetime] = None
        self.consecutive_api_failures: int = 0

        # Safety monitoring
        self._safety_thread: Optional[threading.Thread] = None
        self._safety_thread_stop_event = threading.Event()
        self._position_lock = threading.Lock()

        # Load existing state
        self._load_state()

        self.logger.info("Helios Main Controller initialized successfully")
        self.system_state = SystemState.HEALTHY

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file."""
        if not os.path.exists(self.config_path):
            raise HeliosControllerError(f"Configuration file not found: {self.config_path}")

        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
            return config
        except Exception as e:
            raise HeliosControllerError(f"Failed to load configuration: {str(e)}")

    def _load_system_parameters(self) -> None:
        """Load system parameters from configuration."""
        try:
            # State file management
            self.state_file_path = self.config.get('Paths', 'state_file', fallback='state/positions.json')

            # Safety parameters (from PRD)
            self.max_daily_drawdown = self.config.getfloat('Risk', 'max_daily_drawdown_limit', fallback=0.03)
            self.zscore_stop_threshold = self.config.getfloat('Strategy', 'zscore_stop_loss_threshold', fallback=3.5)
            self.time_in_trade_stop_days = self.config.getint('Strategy', 'time_in_trade_stop_days', fallback=60)
            self.correlation_threshold = self.config.getfloat('Risk', 'correlation_threshold', fallback=0.70)

            # System monitoring parameters
            self.position_check_interval_minutes = 5  # Check positions every 5 minutes
            self.health_check_interval_minutes = 15   # Health check every 15 minutes
            self.max_consecutive_api_failures = 3     # SL-3.2 requirement

            # Kill switch parameters
            self.kill_switch_file = "KILL_SWITCH_ACTIVE"  # File-based kill switch

            self.logger.info("System parameters loaded successfully")

        except Exception as e:
            raise HeliosControllerError(f"Failed to load system parameters: {str(e)}")

    def _initialize_modules(self) -> None:
        """Initialize all core modules with error handling."""
        try:
            self.logger.info("Initializing core modules...")

            # Initialize API client
            self.api_client = HeliosAlpacaClient(self.config_path)

            # Initialize data handler
            self.data_handler = HeliosDataHandler(self.config_path)

            # Initialize signal engine
            self.signal_engine = HeliosSignalEngine(self.data_handler, self.config_path)

            # Initialize risk engine
            self.risk_engine = HeliosRiskEngine(self.data_handler, self.api_client, self.config_path)

            # Initialize execution engine
            self.execution_engine = HeliosExecutionEngine(self.api_client, self.risk_engine, self.config_path)

            # Validate all modules are healthy
            self._validate_module_health()

            self.logger.info("All core modules initialized successfully")

        except Exception as e:
            log_critical_error("Failed to initialize core modules", e)
            raise HeliosControllerError(f"Module initialization failed: {str(e)}")

    def _validate_module_health(self) -> None:
        """Validate that all modules are healthy and operational."""
        health_checks = {
            'data_handler': self.data_handler.health_check(),
            'signal_engine': self.signal_engine.health_check(),
            'risk_engine': self.risk_engine.health_check(),
            'execution_engine': self.execution_engine.health_check()
        }

        failed_modules = []
        for module_name, health in health_checks.items():
            if not health.get('overall_healthy', False):
                failed_modules.append(module_name)
                self.logger.error(f"Module health check failed: {module_name}")

        if failed_modules:
            raise HeliosControllerError(f"Module health validation failed: {failed_modules}")

    def _load_state(self) -> None:
        """Load existing position state from persistent storage."""
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, 'r') as f:
                    state_data = json.load(f)

                # Convert JSON data back to Position objects
                for position_id, position_data in state_data.items():
                    position = Position(
                        id=position_data['id'],
                        stock1=position_data['stock1'],
                        stock2=position_data['stock2'],
                        direction=position_data['direction'],
                        entry_timestamp=datetime.fromisoformat(position_data['entry_timestamp']),
                        entry_zscore=position_data['entry_zscore'],
                        shares1=position_data['shares1'],
                        shares2=position_data['shares2'],
                        entry_price1=position_data['entry_price1'],
                        entry_price2=position_data['entry_price2'],
                        hedge_ratio=position_data['hedge_ratio'],
                        days_in_trade=position_data['days_in_trade']
                    )
                    self.positions[position_id] = position

                self.logger.info(f"Loaded {len(self.positions)} positions from state file")
            else:
                self.logger.info("No existing state file found - starting with empty position set")

        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            # Don't raise exception - we can continue with empty state

    def _save_state(self) -> None:
        """Save current position state to persistent storage with atomic writes."""
        try:
            # Ensure state directory exists
            Path(self.state_file_path).parent.mkdir(parents=True, exist_ok=True)

            # Prepare state data for JSON serialization
            state_data = {}
            for position_id, position in self.positions.items():
                state_data[position_id] = {
                    'id': position.id,
                    'stock1': position.stock1,
                    'stock2': position.stock2,
                    'direction': position.direction,
                    'entry_timestamp': position.entry_timestamp.isoformat(),
                    'entry_zscore': position.entry_zscore,
                    'shares1': position.shares1,
                    'shares2': position.shares2,
                    'entry_price1': position.entry_price1,
                    'entry_price2': position.entry_price2,
                    'hedge_ratio': position.hedge_ratio,
                    'days_in_trade': position.days_in_trade
                }

            # Atomic write: write to temporary file then rename
            temp_file = f"{self.state_file_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            # Atomic rename operation
            shutil.move(temp_file, self.state_file_path)

            self.logger.debug(f"State saved successfully: {len(self.positions)} positions")

        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            # Clean up temp file if it exists
            temp_file = f"{self.state_file_path}.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def run_daily_cycle(self) -> None:
        """Execute the complete daily trading cycle."""
        try:
            self.logger.info("Starting daily trading cycle")

            # Check kill switch before starting
            if self._check_kill_switch():
                self.logger.critical("Kill switch is active - aborting daily cycle")
                return

            # Phase 1: Pre-market preparation
            self._run_pre_market_phase()

            # Phase 2: Market open validation
            if not self._wait_for_market_open():
                self.logger.warning("Market not open - aborting daily cycle")
                return

            # Phase 3: Active trading phase
            self._run_active_trading_phase()

            # Phase 4: Position monitoring phase
            self._run_position_monitoring_phase()

            # Phase 5: Post-market cleanup
            self._run_post_market_phase()

            self.logger.info("Daily trading cycle completed successfully")

        except Exception as e:
            log_critical_error("Daily trading cycle failed", e)
            self._trigger_emergency_stop("Daily cycle failure")

    def _run_pre_market_phase(self) -> None:
        """Execute pre-market preparation phase."""
        self.trading_phase = TradingPhase.PRE_MARKET
        self.logger.info("Starting pre-market phase")

        try:
            # Get start-of-day equity for drawdown monitoring
            portfolio_risk = self.risk_engine.get_portfolio_risk(force_update=True)
            self.equity_start_of_day = portfolio_risk.total_equity

            self.logger.info(f"Start-of-day equity: ${self.equity_start_of_day:,.2f}")

            # Update position days in trade
            self._update_position_metrics()

            # Run weekly cointegration screening if needed
            self.signal_engine.run_weekly_cointegration_screening()

            # Run momentum analysis if needed
            self.signal_engine.run_momentum_analysis()

            self.logger.info("Pre-market phase completed")

        except Exception as e:
            log_critical_error("Pre-market phase failed", e)
            raise

    def _wait_for_market_open(self) -> bool:
        """Wait for market to open and validate trading conditions."""
        self.trading_phase = TradingPhase.MARKET_OPEN

        try:
            # Check if market is open
            is_open = self.api_client.is_market_open()

            if not is_open:
                self.logger.info("Market is closed - no trading today")
                return False

            self.logger.info("Market is open - proceeding with trading")
            return True

        except Exception as e:
            self.logger.error(f"Failed to check market status: {str(e)}")
            return False

    def _run_active_trading_phase(self) -> None:
        """Execute active trading phase with signal generation and execution."""
        self.trading_phase = TradingPhase.ACTIVE_TRADING
        self.logger.info("Starting active trading phase")

        try:
            # Generate daily signals
            signals = self.signal_engine.generate_daily_signals()
            self.logger.info(f"Generated {len(signals)} trading signals")

            # Process each signal
            for signal in signals:
                try:
                    # Safety checks before processing signal
                    if self._check_kill_switch():
                        self.logger.critical("Kill switch activated - stopping signal processing")
                        break

                    if self._check_daily_drawdown():
                        self.logger.critical("Daily drawdown limit breached - stopping trading")
                        break

                    # Process the signal
                    self._process_trading_signal(signal)

                except Exception as e:
                    self.logger.error(f"Failed to process signal {signal.stock1}/{signal.stock2}: {str(e)}")
                    continue

            # Check for exit signals on open positions
            self._check_exit_signals()

            self.logger.info("Active trading phase completed")

        except Exception as e:
            log_critical_error("Active trading phase failed", e)
            raise

    def _run_position_monitoring_phase(self) -> None:
        """Execute position monitoring phase with safety checks."""
        self.trading_phase = TradingPhase.POSITION_MONITORING
        self.logger.info("Starting position monitoring phase")

        try:
            # Start safety monitoring thread if not already running
            if not self._safety_thread or not self._safety_thread.is_alive():
                self._start_safety_monitoring()

            # Run comprehensive position safety checks
            self._check_all_safety_mechanisms()

            # Update position metrics
            self._update_position_metrics()

            # Save current state
            self._save_state()

            self.logger.info("Position monitoring phase completed")

        except Exception as e:
            log_critical_error("Position monitoring phase failed", e)
            raise

    def _run_post_market_phase(self) -> None:
        """Execute post-market cleanup phase."""
        self.trading_phase = TradingPhase.POST_MARKET
        self.logger.info("Starting post-market phase")

        try:
            # Final position update
            self._update_position_metrics()

            # Generate daily summary
            self._generate_daily_summary()

            # Save final state
            self._save_state()

            self.logger.info("Post-market phase completed")

        except Exception as e:
            self.logger.error(f"Post-market phase failed: {str(e)}")

    def _process_trading_signal(self, signal: PairSignal) -> None:
        """Process a trading signal through risk validation and execution."""
        try:
            self.logger.info(f"Processing signal: {signal.signal_type.value} {signal.stock1}/{signal.stock2}")

            # Validate signal is still valid
            if not self.signal_engine.validate_signal_conditions(signal):
                self.logger.warning(f"Signal validation failed for {signal.stock1}/{signal.stock2}")
                return

            # Calculate position sizes
            position1, position2 = self.risk_engine.calculate_pair_position_sizes(
                signal.stock1,
                signal.stock2,
                signal.hedge_ratio,
                signal.signal_type.value
            )

            # Validate trade risk
            risk_check = self.risk_engine.validate_trade_risk([position1, position2])

            if not risk_check.passed:
                self.logger.warning(f"Risk validation failed for {signal.stock1}/{signal.stock2}: {risk_check.errors}")
                log_risk_event("TRADE_REJECTED", {
                    'pair': f"{signal.stock1}_{signal.stock2}",
                    'reason': 'risk_validation_failed',
                    'errors': risk_check.errors
                })
                return

            # Execute the trade
            trade_execution = self.execution_engine.execute_pair_trade(
                position1,
                position2,
                {
                    'signal_type': signal.signal_type.value,
                    'entry_zscore': signal.spread_zscore,
                    'hedge_ratio': signal.hedge_ratio,
                    'cointegration_pvalue': signal.cointegration_pvalue
                }
            )

            # If execution successful, add to positions
            if trade_execution.execution_status == "completed":
                position = Position(
                    id=trade_execution.trade_id,
                    stock1=signal.stock1,
                    stock2=signal.stock2,
                    direction=signal.signal_type.value,
                    entry_timestamp=trade_execution.execution_start,
                    entry_zscore=signal.spread_zscore,
                    shares1=position1.shares,
                    shares2=position2.shares,
                    entry_price1=signal.entry_price1,
                    entry_price2=signal.entry_price2,
                    hedge_ratio=signal.hedge_ratio,
                    days_in_trade=0
                )

                with self._position_lock:
                    self.positions[trade_execution.trade_id] = position

                log_trade_event("PAIR_TRADE_OPENED", f"{signal.stock1}_{signal.stock2}", {
                    'trade_id': trade_execution.trade_id,
                    'entry_zscore': signal.spread_zscore,
                    'direction': signal.signal_type.value
                })

                self.logger.info(f"Trade executed successfully: {trade_execution.trade_id}")

        except Exception as e:
            self.logger.error(f"Failed to process trading signal: {str(e)}")

    def _check_exit_signals(self) -> None:
        """Check for exit signals on open positions."""
        try:
            if not self.positions:
                return

            # Get exit signals from signal engine
            positions_data = []
            for position in self.positions.values():
                positions_data.append({
                    'id': position.id,
                    'stock1': position.stock1,
                    'stock2': position.stock2,
                    'entry_zscore': position.entry_zscore
                })

            exit_signals = self.signal_engine.generate_exit_signals(positions_data)

            for exit_signal in exit_signals:
                position_id = exit_signal.metadata.get('position_id')
                if position_id in self.positions:
                    self._close_position(position_id, "zscore_reversal")

        except Exception as e:
            self.logger.error(f"Failed to check exit signals: {str(e)}")

    def _check_all_safety_mechanisms(self) -> None:
        """Execute comprehensive safety checks (SL-1 through SL-3)."""
        try:
            # SL-1: Trade-level safety checks
            self._check_trade_level_stops()

            # SL-2: Portfolio-level safety checks
            self._check_portfolio_level_stops()

            # SL-3: System-level safety checks
            self._check_system_level_stops()

        except Exception as e:
            log_critical_error("Safety mechanism check failed", e)

    def _check_trade_level_stops(self) -> None:
        """Check trade-level stop losses (SL-1.1 and SL-1.2)."""
        try:
            positions_to_close = []

            with self._position_lock:
                for position_id, position in self.positions.items():
                    # SL-1.2: Time-in-trade stop (60 days)
                    if position.days_in_trade >= self.time_in_trade_stop_days:
                        positions_to_close.append((position_id, "time_in_trade_stop"))
                        continue

                    # SL-1.1: Z-score failure stop (|Z-score| > 3.5)
                    if position.current_zscore and abs(position.current_zscore) > self.zscore_stop_threshold:
                        positions_to_close.append((position_id, "zscore_stop_loss"))
                        continue

            # Close positions that hit stops
            for position_id, reason in positions_to_close:
                self._close_position(position_id, reason)

        except Exception as e:
            self.logger.error(f"Trade-level stop check failed: {str(e)}")

    def _check_portfolio_level_stops(self) -> None:
        """Check portfolio-level circuit breakers (SL-2.1 and SL-2.2)."""
        try:
            # SL-2.1: Daily drawdown limit (3%)
            if self._check_daily_drawdown():
                self._trigger_emergency_stop("Daily drawdown limit exceeded")
                return

            # SL-2.2: Correlation check for open positions
            positions_to_close = []

            with self._position_lock:
                for position_id, position in self.positions.items():
                    if position.current_correlation and position.current_correlation < self.correlation_threshold:
                        positions_to_close.append((position_id, "correlation_failure"))

            for position_id, reason in positions_to_close:
                self._close_position(position_id, reason)

        except Exception as e:
            log_critical_error("Portfolio-level stop check failed", e)

    def _check_system_level_stops(self) -> None:
        """Check system-level kill switches (SL-3.1 and SL-3.2)."""
        try:
            # SL-3.1: Manual kill switch
            if self._check_kill_switch():
                self._trigger_emergency_stop("Manual kill switch activated")
                return

            # SL-3.2: API connectivity failure
            if self.consecutive_api_failures >= self.max_consecutive_api_failures:
                log_critical_error("API connectivity circuit breaker triggered", None)
                self._trigger_emergency_stop("API connectivity failure")
                return

        except Exception as e:
            log_critical_error("System-level stop check failed", e)

    def _check_daily_drawdown(self) -> bool:
        """Check if daily drawdown limit has been breached (SL-2.1)."""
        try:
            if self.equity_start_of_day <= 0:
                return False

            current_portfolio = self.risk_engine.get_portfolio_risk(force_update=True)
            drawdown_exceeded = self.risk_engine.check_portfolio_drawdown(self.equity_start_of_day)

            if drawdown_exceeded:
                log_critical_error("DAILY DRAWDOWN LIMIT EXCEEDED", None)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Drawdown check failed: {str(e)}")
            return False

    def _check_kill_switch(self) -> bool:
        """Check if manual kill switch is active (SL-3.1)."""
        try:
            # File-based kill switch
            if os.path.exists(self.kill_switch_file):
                return True

            # Class-level kill switch
            return self.kill_switch_active

        except Exception as e:
            self.logger.error(f"Kill switch check failed: {str(e)}")
            return False

    def _trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop procedures."""
        try:
            self.emergency_stop_triggered = True
            self.system_state = SystemState.EMERGENCY_STOP

            log_critical_error(f"EMERGENCY STOP TRIGGERED: {reason}", None)

            # Liquidate all positions
            self._liquidate_all_positions()

            # Stop safety monitoring
            self._stop_safety_monitoring()

            # Create kill switch file
            with open(self.kill_switch_file, 'w') as f:
                f.write(f"Emergency stop triggered: {reason}\nTimestamp: {datetime.now().isoformat()}\n")

            self.logger.critical(f"Emergency stop procedures completed: {reason}")

        except Exception as e:
            log_critical_error("Emergency stop procedure failed", e)

    def _close_position(self, position_id: str, reason: str) -> None:
        """Close a specific position."""
        try:
            if position_id not in self.positions:
                self.logger.warning(f"Position not found for closure: {position_id}")
                return

            position = self.positions[position_id]

            self.logger.info(f"Closing position {position_id}: {reason}")

            # Close both legs
            try:
                # Close leg 1
                result1 = self.execution_engine.close_position(
                    position.stock1,
                    position.shares1 if position.direction == "PAIRS_LONG_SPREAD" else -position.shares1,
                    reason
                )

                # Close leg 2
                result2 = self.execution_engine.close_position(
                    position.stock2,
                    -position.shares2 if position.direction == "PAIRS_LONG_SPREAD" else position.shares2,
                    reason
                )

                # Remove from positions if both closes successful
                if result1.status.value == "filled" and result2.status.value == "filled":
                    with self._position_lock:
                        del self.positions[position_id]

                    log_trade_event("PAIR_TRADE_CLOSED", f"{position.stock1}_{position.stock2}", {
                        'position_id': position_id,
                        'reason': reason,
                        'days_in_trade': position.days_in_trade
                    })

                    self.logger.info(f"Position closed successfully: {position_id}")

            except Exception as e:
                self.logger.error(f"Failed to close position {position_id}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Position closure failed: {str(e)}")

    def _liquidate_all_positions(self) -> None:
        """Liquidate all open positions (emergency procedure)."""
        try:
            self.logger.critical("LIQUIDATING ALL POSITIONS")

            # Use API client's liquidate all function
            success = self.api_client.liquidate_all_positions()

            if success:
                # Clear position tracking
                with self._position_lock:
                    self.positions.clear()

                self.logger.critical("All positions liquidated successfully")
            else:
                self.logger.critical("Position liquidation may have failed - manual intervention required")

        except Exception as e:
            log_critical_error("Position liquidation failed", e)

    def _update_position_metrics(self) -> None:
        """Update metrics for all open positions."""
        try:
            if not self.positions:
                return

            # Get current prices for Z-score and correlation updates
            all_symbols = set()
            for position in self.positions.values():
                all_symbols.update([position.stock1, position.stock2])

            if not all_symbols:
                return

            # Get recent data for calculations
            recent_data = self.data_handler.get_historical_data(
                symbols=list(all_symbols),
                days_back=self.signal_engine.zscore_window + 5
            )

            latest_prices = self.data_handler.get_latest_prices(list(all_symbols))

            # Update each position's metrics
            with self._position_lock:
                for position in self.positions.values():
                    # Update days in trade
                    position.days_in_trade = (datetime.now() - position.entry_timestamp).days

                    # Calculate current Z-score if data available
                    try:
                        # Get price series for both stocks
                        stock1_data = recent_data[
                            recent_data.index.get_level_values('symbol') == position.stock1
                        ]['close'].sort_index()

                        stock2_data = recent_data[
                            recent_data.index.get_level_values('symbol') == position.stock2
                        ]['close'].sort_index()

                        if len(stock1_data) >= self.signal_engine.zscore_window and len(stock2_data) >= self.signal_engine.zscore_window:
                            # Align data
                            aligned_data = pd.DataFrame({
                                position.stock1: stock1_data,
                                position.stock2: stock2_data
                            }).dropna()

                            if len(aligned_data) >= self.signal_engine.zscore_window:
                                # Calculate spread and Z-score
                                spread = aligned_data[position.stock1] - position.hedge_ratio * aligned_data[position.stock2]
                                rolling_mean = spread.rolling(window=self.signal_engine.zscore_window).mean()
                                rolling_std = spread.rolling(window=self.signal_engine.zscore_window).std()
                                z_scores = (spread - rolling_mean) / rolling_std
                                position.current_zscore = float(z_scores.iloc[-1])

                                # Calculate current correlation
                                position.current_correlation = float(
                                    aligned_data[position.stock1].tail(self.signal_engine.zscore_window).corr(
                                        aligned_data[position.stock2].tail(self.signal_engine.zscore_window)
                                    )
                                )

                    except Exception as e:
                        self.logger.warning(f"Failed to update metrics for position {position.id}: {str(e)}")

            self.last_position_check = datetime.now()

        except Exception as e:
            self.logger.error(f"Failed to update position metrics: {str(e)}")

    def _start_safety_monitoring(self) -> None:
        """Start the safety monitoring background thread."""
        try:
            if self._safety_thread and self._safety_thread.is_alive():
                return

            self._safety_thread_stop_event.clear()
            self._safety_thread = threading.Thread(
                target=self._safety_monitoring_loop,
                name="HeliosSafetyMonitor",
                daemon=True
            )
            self._safety_thread.start()
            self.logger.info("Safety monitoring thread started")

        except Exception as e:
            self.logger.error(f"Failed to start safety monitoring: {str(e)}")

    def _stop_safety_monitoring(self) -> None:
        """Stop the safety monitoring background thread."""
        try:
            if self._safety_thread and self._safety_thread.is_alive():
                self._safety_thread_stop_event.set()
                self._safety_thread.join(timeout=10)
                self.logger.info("Safety monitoring thread stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop safety monitoring: {str(e)}")

    def _safety_monitoring_loop(self) -> None:
        """Background safety monitoring loop."""
        while not self._safety_thread_stop_event.is_set():
            try:
                # Check safety mechanisms every minute
                self._check_all_safety_mechanisms()

                # Update position metrics every 5 minutes
                if (not self.last_position_check or
                    (datetime.now() - self.last_position_check).total_seconds() > 300):
                    self._update_position_metrics()

                # Wait before next check
                self._safety_thread_stop_event.wait(60)  # Check every 60 seconds

            except Exception as e:
                self.logger.error(f"Safety monitoring loop error: {str(e)}")
                time.sleep(60)

    def _generate_daily_summary(self) -> None:
        """Generate and log daily trading summary."""
        try:
            portfolio_risk = self.risk_engine.get_portfolio_risk()
            execution_summary = self.execution_engine.get_execution_summary()

            summary = {
                'date': datetime.now().date().isoformat(),
                'start_of_day_equity': self.equity_start_of_day,
                'end_of_day_equity': portfolio_risk.total_equity,
                'daily_pnl': portfolio_risk.total_equity - self.equity_start_of_day,
                'daily_return': (portfolio_risk.total_equity - self.equity_start_of_day) / self.equity_start_of_day if self.equity_start_of_day > 0 else 0,
                'active_positions': len(self.positions),
                'trades_executed': execution_summary.get('completed_executions', 0),
                'success_rate': execution_summary.get('success_rate', 0),
                'total_commission': execution_summary.get('total_commission_paid', 0),
                'system_state': self.system_state.value,
                'emergency_stops': 1 if self.emergency_stop_triggered else 0
            }

            self.logger.info(f"Daily Summary: {summary}")

            # Log to trade events for record keeping
            log_trade_event("DAILY_SUMMARY", "SYSTEM", summary)

        except Exception as e:
            self.logger.error(f"Failed to generate daily summary: {str(e)}")

    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status."""
        try:
            portfolio_risk = self.risk_engine.get_portfolio_risk()

            return SystemStatus(
                state=self.system_state,
                phase=self.trading_phase,
                timestamp=datetime.now(),
                equity_start_of_day=self.equity_start_of_day,
                current_equity=portfolio_risk.total_equity,
                drawdown_today=(self.equity_start_of_day - portfolio_risk.total_equity) / self.equity_start_of_day if self.equity_start_of_day > 0 else 0,
                active_positions=len(self.positions),
                candidate_pairs=len(self.signal_engine.get_candidate_pairs()),
                last_signal_check=getattr(self.signal_engine, '_last_signal_check', None),
                last_risk_check=self.last_position_check,
                consecutive_api_failures=self.consecutive_api_failures,
                kill_switch_active=self.kill_switch_active,
                errors=[],
                warnings=[]
            )

        except Exception as e:
            self.logger.error(f"Failed to get system status: {str(e)}")
            return SystemStatus(
                state=SystemState.CRITICAL,
                phase=self.trading_phase,
                timestamp=datetime.now(),
                equity_start_of_day=0,
                current_equity=0,
                drawdown_today=0,
                active_positions=0,
                candidate_pairs=0,
                last_signal_check=None,
                last_risk_check=None,
                consecutive_api_failures=self.consecutive_api_failures,
                kill_switch_active=True,
                errors=[str(e)],
                warnings=[]
            )

    def activate_kill_switch(self, reason: str = "Manual activation") -> None:
        """Manually activate the kill switch (SL-3.1)."""
        try:
            self.kill_switch_active = True
            self.logger.critical(f"Kill switch manually activated: {reason}")
            self._trigger_emergency_stop(f"Manual kill switch: {reason}")

        except Exception as e:
            log_critical_error("Failed to activate kill switch", e)

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            health_status = {
                'system_state': self.system_state.value,
                'trading_phase': self.trading_phase.value,
                'kill_switch_active': self.kill_switch_active,
                'emergency_stop_triggered': self.emergency_stop_triggered,
                'active_positions': len(self.positions),
                'consecutive_api_failures': self.consecutive_api_failures,
                'modules_healthy': True,
                'safety_monitoring_active': self._safety_thread and self._safety_thread.is_alive(),
                'last_position_check': self.last_position_check.isoformat() if self.last_position_check else None
            }

            # Check module health
            module_healths = {
                'data_handler': self.data_handler.health_check(),
                'signal_engine': self.signal_engine.health_check(),
                'risk_engine': self.risk_engine.health_check(),
                'execution_engine': self.execution_engine.health_check()
            }

            health_status['module_details'] = module_healths

            # Overall health assessment
            modules_healthy = all(h.get('overall_healthy', False) for h in module_healths.values())
            health_status['modules_healthy'] = modules_healthy

            health_status['overall_healthy'] = (
                modules_healthy and
                not self.kill_switch_active and
                not self.emergency_stop_triggered and
                self.system_state not in [SystemState.CRITICAL, SystemState.EMERGENCY_STOP]
            )

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'overall_healthy': False,
                'error': str(e),
                'system_state': 'error'
            }

    def shutdown(self) -> None:
        """Gracefully shutdown the Helios system."""
        try:
            self.logger.info("Initiating Helios system shutdown...")
            self.system_state = SystemState.SHUTDOWN

            # Stop safety monitoring
            self._stop_safety_monitoring()

            # Save final state
            self._save_state()

            # Shutdown execution engine
            self.execution_engine.shutdown()

            # Generate final summary
            self._generate_daily_summary()

            self.logger.info("Helios system shutdown completed")

        except Exception as e:
            log_critical_error("System shutdown failed", e)


if __name__ == "__main__":
    # Test the main controller
    try:
        print("Testing Helios Main Controller...")

        # Initialize main controller
        controller = HeliosMainController()
        print("✓ Main controller initialized")

        # Health check
        health = controller.health_check()
        print(f"✓ Health check: {'PASS' if health['overall_healthy'] else 'FAIL'}")

        # Get system status
        status = controller.get_system_status()
        print(f"✓ System status: {status.state.value}")
        print(f"  - Current equity: ${status.current_equity:,.2f}")
        print(f"  - Active positions: {status.active_positions}")
        print(f"  - Candidate pairs: {status.candidate_pairs}")

        print("✓ Main controller test completed successfully!")
        print("Note: No trading executed in test mode")

        # Clean shutdown
        controller.shutdown()

    except Exception as e:
        print(f"✗ Main controller test failed: {e}")
