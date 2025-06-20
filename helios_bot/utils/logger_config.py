"""Centralized logging configuration for the Helios trading bot.

This module provides a unified logging setup that all other modules in the
Helios system should use. It configures dual output (console + file) with
proper formatting, rotation, and error handling.

Author: Helios Trading Bot
Version: 1.0
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import configparser


class HeliosLoggerConfig:
    """Centralized logger configuration for the Helios trading system.

    This class provides methods to set up and configure logging for the entire
    Helios application with proper formatting, file rotation, and error handling.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the logger configuration.

        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        self.config_path = config_path or "config/config.ini"
        self.config = self._load_config()
        self._setup_complete = False

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file.

        Returns:
            ConfigParser object with loaded configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            configparser.Error: If config file is malformed.
        """
        config = configparser.ConfigParser()

        if not os.path.exists(self.config_path):
            # Create minimal default config if file doesn't exist
            config.add_section('Logging')
            config.set('Logging', 'log_level', 'INFO')
            config.set('Logging', 'max_log_file_size_mb', '100')
            config.set('Logging', 'backup_count', '5')

            config.add_section('Paths')
            config.set('Paths', 'log_file', 'logs/helios_bot.log')
        else:
            config.read(self.config_path)

        return config

    def _get_log_level(self) -> int:
        """Get the configured log level.

        Returns:
            Logging level constant (e.g., logging.INFO).
        """
        level_str = self.config.get('Logging', 'log_level', fallback='INFO').upper()
        return getattr(logging, level_str, logging.INFO)

    def _get_log_file_path(self) -> Path:
        """Get the configured log file path and ensure directory exists.

        Returns:
            Path object for the log file.
        """
        log_file = self.config.get('Paths', 'log_file', fallback='logs/helios_bot.log')
        log_path = Path(log_file)

        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        return log_path

    def _create_formatter(self) -> logging.Formatter:
        """Create a standardized formatter for all log messages.

        Returns:
            Configured logging formatter.
        """
        return logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _setup_console_handler(self, formatter: logging.Formatter) -> logging.StreamHandler:
        """Set up console logging handler.

        Args:
            formatter: Logging formatter to use.

        Returns:
            Configured console handler.
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self._get_log_level())
        return console_handler

    def _setup_file_handler(self, formatter: logging.Formatter) -> logging.handlers.RotatingFileHandler:
        """Set up rotating file logging handler.

        Args:
            formatter: Logging formatter to use.

        Returns:
            Configured rotating file handler.
        """
        log_file_path = self._get_log_file_path()
        max_size_mb = self.config.getint('Logging', 'max_log_file_size_mb', fallback=100)
        backup_count = self.config.getint('Logging', 'backup_count', fallback=5)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        return file_handler

    def setup_logging(self, logger_name: str = 'helios_bot') -> logging.Logger:
        """Set up comprehensive logging for the Helios trading bot.

        This method configures logging with:
        - Dual output to console and rotating file
        - Proper formatting with timestamps and context
        - Configurable log levels
        - Automatic log rotation

        Args:
            logger_name: Name for the root logger (default: 'helios_bot').

        Returns:
            Configured logger instance.

        Raises:
            OSError: If log directory cannot be created.
            PermissionError: If log file cannot be written to.
        """
        if self._setup_complete:
            return logging.getLogger(logger_name)

        try:
            # Create formatter
            formatter = self._create_formatter()

            # Get root logger for our application
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)  # Allow all messages, handlers will filter

            # Clear any existing handlers to avoid duplicates
            logger.handlers.clear()

            # Set up console handler
            console_handler = self._setup_console_handler(formatter)
            logger.addHandler(console_handler)

            # Set up file handler
            file_handler = self._setup_file_handler(formatter)
            logger.addHandler(file_handler)

            # Prevent propagation to root logger to avoid duplicate messages
            logger.propagate = False

            # Log initial setup message
            logger.info("Helios logging system initialized successfully")
            logger.info(f"Console log level: {logging.getLevelName(self._get_log_level())}")
            logger.info(f"File log path: {self._get_log_file_path()}")

            self._setup_complete = True
            return logger

        except Exception as e:
            # If logging setup fails, at least print to stderr
            print(f"CRITICAL: Failed to set up logging: {e}", file=sys.stderr)
            raise

    def get_logger(self, module_name: str) -> logging.Logger:
        """Get a logger instance for a specific module.

        Args:
            module_name: Name of the module requesting the logger.

        Returns:
            Logger instance configured for the module.
        """
        if not self._setup_complete:
            self.setup_logging()

        return logging.getLogger(f'helios_bot.{module_name}')


# Global logger configuration instance
_logger_config = HeliosLoggerConfig()


def setup_helios_logging(config_path: Optional[str] = None) -> logging.Logger:
    """Convenience function to set up Helios logging.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Configured root logger for the Helios system.
    """
    global _logger_config
    if config_path:
        _logger_config = HeliosLoggerConfig(config_path)
    return _logger_config.setup_logging()


def get_helios_logger(module_name: str) -> logging.Logger:
    """Convenience function to get a logger for a specific module.

    Args:
        module_name: Name of the module requesting the logger.

    Returns:
        Logger instance for the module.
    """
    return _logger_config.get_logger(module_name)


def log_critical_error(message: str, exception: Optional[Exception] = None) -> None:
    """Log a critical error that should trigger system shutdown.

    Args:
        message: Critical error message.
        exception: Optional exception that caused the error.
    """
    logger = get_helios_logger('critical')
    if exception:
        logger.critical(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.critical(message)


def log_trade_event(event_type: str, pair: str, details: Dict[str, Any]) -> None:
    """Log trading events with structured format.

    Args:
        event_type: Type of trade event (e.g., 'ENTRY', 'EXIT', 'STOP_LOSS').
        pair: Trading pair identifier (e.g., 'JPM_GS').
        details: Dictionary containing trade details.
    """
    logger = get_helios_logger('trades')
    logger.info(f"TRADE_EVENT: {event_type} | PAIR: {pair} | DETAILS: {details}")


def log_risk_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log risk management events with structured format.

    Args:
        event_type: Type of risk event (e.g., 'POSITION_LIMIT', 'DRAWDOWN_WARNING').
        details: Dictionary containing risk event details.
    """
    logger = get_helios_logger('risk')
    logger.warning(f"RISK_EVENT: {event_type} | DETAILS: {details}")


if __name__ == "__main__":
    # Test the logging configuration
    test_logger = setup_helios_logging()
    test_logger.info("Testing Helios logging configuration")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")

    # Test module-specific logger
    module_logger = get_helios_logger('test_module')
    module_logger.info("Testing module-specific logger")

    print("Logging configuration test completed successfully")
