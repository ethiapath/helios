"""Utilities package for the Helios trading bot.

This package contains shared utilities used across the Helios trading system,
including logging configuration and API client management.

Author: Helios Trading Bot
Version: 1.0
"""

from .logger_config import (
    setup_helios_logging,
    get_helios_logger,
    log_critical_error,
    log_trade_event,
    log_risk_event,
    HeliosLoggerConfig
)

from .api_client import (
    HeliosAlpacaClient,
    AlpacaAPIError
)

__all__ = [
    'setup_helios_logging',
    'get_helios_logger',
    'log_critical_error',
    'log_trade_event',
    'log_risk_event',
    'HeliosLoggerConfig',
    'HeliosAlpacaClient',
    'AlpacaAPIError'
]
