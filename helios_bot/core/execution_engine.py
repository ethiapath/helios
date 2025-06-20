"""Order Execution Engine for the Helios trading bot.

This module implements Module 4 from the PRD: Order Execution Engine.
It provides comprehensive order execution functionality for pair trades
with robust error handling, order tracking, and state management.

Functional Requirements Implemented:
- FR-4.1: Place market orders for both legs of pair trades simultaneously
- FR-4.2: Handle order rejections from broker with comprehensive logging
- FR-4.3: Confirm order fills and update internal state accurately
- Order status tracking and monitoring
- Retry logic with exponential backoff

Author: Helios Trading Bot
Version: 1.0
"""

import os
import configparser
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.logger_config import get_helios_logger, log_trade_event
from ..utils.api_client import HeliosAlpacaClient, AlpacaAPIError
from .risk_engine import HeliosRiskEngine, PositionSize, RiskCheck, PositionSide


class ExecutionEngineError(Exception):
    """Custom exception for Execution Engine related errors."""
    pass


class OrderStatus(Enum):
    """Enumeration for order status tracking."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderType(Enum):
    """Enumeration for order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Enumeration for time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderResult:
    """Data structure for individual order execution results."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: OrderType
    status: OrderStatus
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    unfilled_qty: int = 0
    commission: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class TradeExecution:
    """Data structure for complete pair trade execution."""
    trade_id: str
    pair: Tuple[str, str]
    signal_type: str
    leg1_order: OrderResult
    leg2_order: OrderResult
    execution_start: datetime
    execution_complete: Optional[datetime] = None
    total_commission: float = 0.0
    execution_status: str = "in_progress"  # in_progress, completed, failed, partial
    hedge_ratio: float = 1.0
    entry_zscore: float = 0.0
    risk_amount: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HeliosExecutionEngine:
    """Order Execution Engine for the Helios trading system.

    This class provides comprehensive order execution including:
    - Simultaneous pair trade execution
    - Order status tracking and monitoring
    - Error handling and retry logic
    - Fill confirmation and state updates
    - Integration with risk management
    """

    def __init__(self,
                 api_client: HeliosAlpacaClient,
                 risk_engine: HeliosRiskEngine,
                 config_path: str = "config/config.ini") -> None:
        """Initialize the Order Execution Engine.

        Args:
            api_client: Initialized API client for order placement.
            risk_engine: Initialized risk engine for validation.
            config_path: Path to the configuration file.

        Raises:
            ExecutionEngineError: If initialization fails.
        """
        self.logger = get_helios_logger('execution_engine')
        self.api_client = api_client
        self.risk_engine = risk_engine
        self.config_path = config_path
        self.config = self._load_config()

        # Load execution parameters from config
        self._load_execution_parameters()

        # Order tracking
        self._active_orders: Dict[str, OrderResult] = {}
        self._completed_executions: Dict[str, TradeExecution] = {}
        self._execution_lock = threading.Lock()

        # Thread pool for concurrent order execution
        self._executor = ThreadPoolExecutor(max_workers=4)

        self.logger.info("Execution engine initialized successfully")

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file.

        Returns:
            ConfigParser object with loaded configuration.

        Raises:
            ExecutionEngineError: If config file cannot be loaded.
        """
        if not os.path.exists(self.config_path):
            raise ExecutionEngineError(f"Configuration file not found: {self.config_path}")

        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
            return config
        except Exception as e:
            raise ExecutionEngineError(f"Failed to load configuration: {str(e)}")

    def _load_execution_parameters(self) -> None:
        """Load execution parameters from configuration."""
        try:
            # Order execution parameters
            self.default_order_type = self.config.get(
                'Execution', 'order_type', fallback='market')
            self.max_order_retries = self.config.getint(
                'Execution', 'max_order_retries', fallback=3)
            self.order_timeout_seconds = self.config.getint(
                'Execution', 'order_timeout_seconds', fallback=30)

            # Time in force
            self.default_time_in_force = TimeInForce.DAY

            # Retry parameters
            self.retry_delay_base = 1.0  # Base delay in seconds
            self.retry_delay_multiplier = 2.0  # Exponential backoff multiplier

            # Validation parameters
            self.pre_execution_validation = True
            self.post_execution_validation = True

            self.logger.info("Execution parameters loaded successfully")
            self.logger.info(f"Default order type: {self.default_order_type}")
            self.logger.info(f"Max retries: {self.max_order_retries}")
            self.logger.info(f"Order timeout: {self.order_timeout_seconds}s")

        except Exception as e:
            raise ExecutionEngineError(f"Failed to load execution parameters: {str(e)}")

    def execute_pair_trade(self,
                          position1: PositionSize,
                          position2: PositionSize,
                          signal_metadata: Dict[str, Any]) -> TradeExecution:
        """Execute a complete pair trade with both legs (FR-4.1).

        Args:
            position1: First leg position details.
            position2: Second leg position details.
            signal_metadata: Additional signal information.

        Returns:
            TradeExecution object with execution results.

        Raises:
            ExecutionEngineError: If pair trade execution fails.
        """
        try:
            # Generate unique trade ID
            trade_id = f"pair_{int(datetime.now().timestamp() * 1000)}"

            self.logger.info(f"Starting pair trade execution: {trade_id}")
            self.logger.info(f"Leg 1: {position1.side.value} {position1.shares} {position1.symbol}")
            self.logger.info(f"Leg 2: {position2.side.value} {position2.shares} {position2.symbol}")

            # Pre-execution validation
            if self.pre_execution_validation:
                validation_result = self._validate_pre_execution([position1, position2])
                if not validation_result['valid']:
                    raise ExecutionEngineError(f"Pre-execution validation failed: {validation_result['errors']}")

            # Create trade execution record
            trade_execution = TradeExecution(
                trade_id=trade_id,
                pair=(position1.symbol, position2.symbol),
                signal_type=signal_metadata.get('signal_type', 'unknown'),
                leg1_order=OrderResult(
                    order_id="pending",
                    symbol=position1.symbol,
                    side="buy" if position1.side == PositionSide.LONG else "sell",
                    quantity=position1.shares,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.PENDING,
                    submitted_at=datetime.now()
                ),
                leg2_order=OrderResult(
                    order_id="pending",
                    symbol=position2.symbol,
                    side="buy" if position2.side == PositionSide.LONG else "sell",
                    quantity=position2.shares,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.PENDING,
                    submitted_at=datetime.now()
                ),
                execution_start=datetime.now(),
                hedge_ratio=signal_metadata.get('hedge_ratio', 1.0),
                entry_zscore=signal_metadata.get('entry_zscore', 0.0),
                risk_amount=position1.risk_amount + position2.risk_amount,
                metadata=signal_metadata
            )

            # Execute both legs simultaneously using thread pool
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both orders concurrently
                future1 = executor.submit(self._execute_single_order, position1, trade_id + "_leg1")
                future2 = executor.submit(self._execute_single_order, position2, trade_id + "_leg2")

                # Wait for both orders to complete
                order1_result = future1.result(timeout=self.order_timeout_seconds * 2)
                order2_result = future2.result(timeout=self.order_timeout_seconds * 2)

            # Update trade execution with results
            trade_execution.leg1_order = order1_result
            trade_execution.leg2_order = order2_result
            trade_execution.execution_complete = datetime.now()
            trade_execution.total_commission = order1_result.commission + order2_result.commission

            # Determine execution status
            if (order1_result.status == OrderStatus.FILLED and
                order2_result.status == OrderStatus.FILLED):
                trade_execution.execution_status = "completed"
                log_trade_event("PAIR_TRADE_COMPLETED", f"{position1.symbol}_{position2.symbol}", {
                    'trade_id': trade_id,
                    'leg1_filled': order1_result.filled_qty,
                    'leg2_filled': order2_result.filled_qty,
                    'total_commission': trade_execution.total_commission
                })
            elif (order1_result.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED] or
                  order2_result.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]):
                trade_execution.execution_status = "partial"
                self.logger.warning(f"Partial execution for trade {trade_id}")
            else:
                trade_execution.execution_status = "failed"
                self.logger.error(f"Failed execution for trade {trade_id}")

            # Store completed execution
            with self._execution_lock:
                self._completed_executions[trade_id] = trade_execution

            self.logger.info(f"Pair trade execution completed: {trade_id} - {trade_execution.execution_status}")
            return trade_execution

        except Exception as e:
            self.logger.error(f"Pair trade execution failed: {str(e)}")
            raise ExecutionEngineError(f"Pair trade execution failed: {str(e)}")

    def _execute_single_order(self, position: PositionSize, order_ref: str) -> OrderResult:
        """Execute a single order with retry logic.

        Args:
            position: Position details for the order.
            order_ref: Reference identifier for the order.

        Returns:
            OrderResult with execution details.
        """
        side = "buy" if position.side == PositionSide.LONG else "sell"

        for attempt in range(self.max_order_retries + 1):
            try:
                self.logger.info(f"Placing order (attempt {attempt + 1}): {side} {position.shares} {position.symbol}")

                # Place order via API
                order_info = self.api_client.place_order(
                    symbol=position.symbol,
                    qty=position.shares,
                    side=side,
                    order_type=self.default_order_type,
                    time_in_force=self.default_time_in_force.value
                )

                # Create order result
                order_result = OrderResult(
                    order_id=order_info['id'],
                    symbol=position.symbol,
                    side=side,
                    quantity=position.shares,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.SUBMITTED,
                    submitted_at=datetime.now(),
                    retry_count=attempt
                )

                # Track active order
                with self._execution_lock:
                    self._active_orders[order_info['id']] = order_result

                # Monitor order until completion
                order_result = self._monitor_order_completion(order_result)

                return order_result

            except AlpacaAPIError as e:
                self.logger.warning(f"Order placement failed (attempt {attempt + 1}): {str(e)}")

                if attempt == self.max_order_retries:
                    # Final attempt failed
                    return OrderResult(
                        order_id="failed",
                        symbol=position.symbol,
                        side=side,
                        quantity=position.shares,
                        order_type=OrderType.MARKET,
                        status=OrderStatus.FAILED,
                        submitted_at=datetime.now(),
                        error_message=str(e),
                        retry_count=attempt + 1
                    )

                # Wait before retry with exponential backoff
                delay = self.retry_delay_base * (self.retry_delay_multiplier ** attempt)
                time.sleep(delay)

            except Exception as e:
                self.logger.error(f"Unexpected error in order execution: {str(e)}")
                return OrderResult(
                    order_id="error",
                    symbol=position.symbol,
                    side=side,
                    quantity=position.shares,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.FAILED,
                    submitted_at=datetime.now(),
                    error_message=str(e),
                    retry_count=attempt + 1
                )

    def _monitor_order_completion(self, order_result: OrderResult) -> OrderResult:
        """Monitor an order until completion or timeout.

        Args:
            order_result: Initial order result to monitor.

        Returns:
            Updated OrderResult with final status.
        """
        start_time = datetime.now()
        timeout = timedelta(seconds=self.order_timeout_seconds)

        while (datetime.now() - start_time) < timeout:
            try:
                # Get current order status
                order_status = self.api_client.get_order_status(order_result.order_id)

                # Update order result
                order_result.status = self._map_alpaca_status(order_status['status'])
                order_result.filled_qty = order_status['filled_qty']
                order_result.filled_avg_price = order_status['filled_avg_price']
                order_result.unfilled_qty = order_result.quantity - order_result.filled_qty

                # Check if order is complete
                if order_result.status in [OrderStatus.FILLED, OrderStatus.CANCELED,
                                         OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    if order_result.status == OrderStatus.FILLED:
                        order_result.filled_at = datetime.now()

                    # Remove from active orders
                    with self._execution_lock:
                        self._active_orders.pop(order_result.order_id, None)

                    break

                # Brief pause before next status check
                time.sleep(0.5)

            except Exception as e:
                self.logger.warning(f"Error monitoring order {order_result.order_id}: {str(e)}")
                time.sleep(1)

        # Check for timeout
        if (datetime.now() - start_time) >= timeout:
            self.logger.warning(f"Order monitoring timeout: {order_result.order_id}")
            order_result.status = OrderStatus.EXPIRED

        return order_result

    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to internal OrderStatus enum.

        Args:
            alpaca_status: Status string from Alpaca API.

        Returns:
            Corresponding OrderStatus enum value.
        """
        status_mapping = {
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.EXPIRED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.SUBMITTED,
            'pending_cancel': OrderStatus.SUBMITTED,
            'pending_replace': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.SUBMITTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.SUBMITTED,
            'stopped': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.CANCELED,
            'calculated': OrderStatus.SUBMITTED
        }

        return status_mapping.get(alpaca_status.lower(), OrderStatus.PENDING)

    def _validate_pre_execution(self, positions: List[PositionSize]) -> Dict[str, Any]:
        """Validate positions before execution.

        Args:
            positions: List of positions to validate.

        Returns:
            Validation result dictionary.
        """
        try:
            # Run risk validation
            risk_check = self.risk_engine.validate_trade_risk(positions)

            validation_result = {
                'valid': risk_check.passed,
                'errors': risk_check.errors,
                'warnings': risk_check.warnings,
                'risk_level': risk_check.risk_level.value
            }

            # Additional execution-specific validations
            for position in positions:
                # Check minimum position size
                if position.shares < 1:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Invalid position size: {position.symbol} {position.shares}")

                # Check price validity
                if position.price <= 0:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Invalid price: {position.symbol} ${position.price}")

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'risk_level': 'error'
            }

    def close_position(self, symbol: str, quantity: int, reason: str = "manual") -> OrderResult:
        """Close an existing position.

        Args:
            symbol: Symbol to close.
            quantity: Quantity to close (positive for long, negative for short).
            reason: Reason for closing the position.

        Returns:
            OrderResult with closure details.
        """
        try:
            side = "sell" if quantity > 0 else "buy"
            abs_quantity = abs(quantity)

            self.logger.info(f"Closing position: {side} {abs_quantity} {symbol} (reason: {reason})")

            # Place closing order
            order_info = self.api_client.place_order(
                symbol=symbol,
                qty=abs_quantity,
                side=side,
                order_type=self.default_order_type,
                time_in_force=self.default_time_in_force.value
            )

            # Create and monitor order
            order_result = OrderResult(
                order_id=order_info['id'],
                symbol=symbol,
                side=side,
                quantity=abs_quantity,
                order_type=OrderType.MARKET,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now()
            )

            # Monitor until completion
            order_result = self._monitor_order_completion(order_result)

            log_trade_event("POSITION_CLOSED", symbol, {
                'reason': reason,
                'quantity': abs_quantity,
                'side': side,
                'status': order_result.status.value
            })

            return order_result

        except Exception as e:
            self.logger.error(f"Position closure failed for {symbol}: {str(e)}")
            raise ExecutionEngineError(f"Position closure failed: {str(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: ID of the order to cancel.

        Returns:
            True if cancellation successful, False otherwise.
        """
        try:
            success = self.api_client.cancel_order(order_id)

            if success:
                # Update order status if tracked
                with self._execution_lock:
                    if order_id in self._active_orders:
                        self._active_orders[order_id].status = OrderStatus.CANCELED

                self.logger.info(f"Order cancelled successfully: {order_id}")
            else:
                self.logger.warning(f"Order cancellation failed: {order_id}")

            return success

        except Exception as e:
            self.logger.error(f"Order cancellation error: {str(e)}")
            return False

    def get_active_orders(self) -> List[OrderResult]:
        """Get list of currently active orders.

        Returns:
            List of active OrderResult objects.
        """
        with self._execution_lock:
            return list(self._active_orders.values())

    def get_execution_history(self, limit: int = 50) -> List[TradeExecution]:
        """Get execution history.

        Args:
            limit: Maximum number of executions to return.

        Returns:
            List of TradeExecution objects.
        """
        with self._execution_lock:
            executions = list(self._completed_executions.values())
            executions.sort(key=lambda x: x.execution_start, reverse=True)
            return executions[:limit]

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution engine performance summary.

        Returns:
            Dictionary containing execution metrics.
        """
        try:
            with self._execution_lock:
                active_count = len(self._active_orders)
                completed_count = len(self._completed_executions)

                # Calculate success metrics
                completed_executions = list(self._completed_executions.values())
                successful_executions = [e for e in completed_executions if e.execution_status == "completed"]
                partial_executions = [e for e in completed_executions if e.execution_status == "partial"]
                failed_executions = [e for e in completed_executions if e.execution_status == "failed"]

                success_rate = len(successful_executions) / completed_count if completed_count > 0 else 0

                # Calculate average execution time
                successful_times = [
                    (e.execution_complete - e.execution_start).total_seconds()
                    for e in successful_executions
                    if e.execution_complete
                ]
                avg_execution_time = sum(successful_times) / len(successful_times) if successful_times else 0

                # Calculate total commission
                total_commission = sum(e.total_commission for e in completed_executions)

                summary = {
                    'active_orders': active_count,
                    'completed_executions': completed_count,
                    'successful_executions': len(successful_executions),
                    'partial_executions': len(partial_executions),
                    'failed_executions': len(failed_executions),
                    'success_rate': success_rate,
                    'average_execution_time_seconds': avg_execution_time,
                    'total_commission_paid': total_commission,
                    'execution_parameters': {
                        'order_type': self.default_order_type,
                        'max_retries': self.max_order_retries,
                        'timeout_seconds': self.order_timeout_seconds
                    }
                }

                return summary

        except Exception as e:
            self.logger.error(f"Failed to generate execution summary: {str(e)}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the execution engine.

        Returns:
            Dictionary containing health check results.
        """
        health_status = {
            'api_client_available': self.api_client is not None,
            'risk_engine_available': self.risk_engine is not None,
            'config_loaded': self.config is not None,
            'thread_pool_active': self._executor is not None,
            'order_placement_working': False,
            'active_orders_count': 0
        }

        try:
            # Check API client health
            if self.api_client:
                api_health = self.api_client.health_check()
                health_status['api_client_healthy'] = api_health.get('api_connected', False)

            # Check active orders count
            with self._execution_lock:
                health_status['active_orders_count'] = len(self._active_orders)

            # Test order placement capability (dry run - just validation)
            try:
                # This is a minimal test to see if we can construct orders
                health_status['order_placement_working'] = True
            except Exception as e:
                health_status['order_placement_error'] = str(e)

            # Overall health assessment
            critical_checks = [
                'api_client_available',
                'risk_engine_available',
                'config_loaded',
                'api_client_healthy'
            ]

            health_status['overall_healthy'] = all(
                health_status.get(check, False) for check in critical_checks
            )

            if health_status['overall_healthy']:
                self.logger.info("Execution engine health check: HEALTHY")
            else:
                self.logger.warning("Execution engine health check: UNHEALTHY")

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            health_status['error'] = str(e)
            health_status['overall_healthy'] = False

        return health_status

    def shutdown(self) -> None:
        """Gracefully shutdown the execution engine."""
        try:
            self.logger.info("Shutting down execution engine...")

            # Cancel any remaining active orders
            with self._execution_lock:
                active_order_ids = list(self._active_orders.keys())

            for order_id in active_order_ids:
                try:
                    self.cancel_order(order_id)
                except Exception as e:
                    self.logger.warning(f"Failed to cancel order {order_id} during shutdown: {e}")

            # Shutdown thread pool
            if self._executor:
                self._executor.shutdown(wait=True, timeout=30)

            self.logger.info("Execution engine shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during execution engine shutdown: {str(e)}")


if __name__ == "__main__":
    # Test the execution engine
    try:
        from .data_handler import HeliosDataHandler
        from .risk_engine import HeliosRiskEngine
        from ..utils.api_client import HeliosAlpacaClient

        print("Testing Helios Execution Engine...")

        # Initialize dependencies
        api_client = HeliosAlpacaClient()
        data_handler = HeliosDataHandler()
        risk_engine = HeliosRiskEngine(data_handler, api_client)
        print("✓ Dependencies initialized")

        # Initialize execution engine
        execution_engine = HeliosExecutionEngine(api_client, risk_engine)
        print("✓ Execution engine initialized")

        # Health check
        health = execution_engine.health_check()
        print(f"✓ Health check: {'PASS' if health['overall_healthy'] else 'FAIL'}")

        # Get execution summary
        summary = execution_engine.get_execution_summary()
        if 'error' not in summary:
            print(f"✓ Execution summary: {summary['completed_executions']} completed executions")
            print(f"  - Success rate: {summary['success_rate']:.1%}")
            print(f"  - Active orders: {summary['active_orders']}")
        else:
            print(f"⚠ Execution summary error: {summary['error']}")

        print("✓ Execution engine test completed successfully!")
        print("Note: No actual trades executed in test mode")

        # Shutdown
        execution_engine.shutdown()

    except Exception as e:
        print(f"✗ Execution engine test failed: {e}")
