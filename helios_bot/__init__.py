"""Helios Trading Bot - Main Package.

This is the main package for the Helios systematic trading bot, a mean-reversion
and momentum algorithm designed for independent retail traders.

The Helios system implements a fully automated, risk-first approach to trading
US equities using statistical arbitrage techniques, primarily pairs trading
with cointegration analysis.

Modules Overview:
- core: Core trading logic modules (data, signals, risk, execution)
- utils: Shared utilities (logging, API client)
- main_controller: Main orchestrator that coordinates all modules

Key Features:
- Market-neutral pairs trading with cointegration screening
- Cross-sectional momentum monitoring
- Comprehensive risk management with multiple stop-loss levels
- Automated position sizing based on volatility (ATR)
- Real-time monitoring and alerting
- Full state persistence and recovery

Safety Features:
- Trade-level stop losses (Z-score and time-based)
- Portfolio-level circuit breakers (drawdown limits)
- System-level kill switches (manual and automatic)
- Comprehensive logging and error handling

Author: Helios Trading Bot
Version: 1.0
License: Private/Proprietary

Usage:
    from helios_bot import HeliosMainController

    # Initialize and run the trading bot
    controller = HeliosMainController()
    controller.run_daily_cycle()
"""

__version__ = "1.0.0"
__author__ = "Helios Trading Bot"
__email__ = "support@heliostrading.bot"
__status__ = "Production"

# Core module imports (import available modules only)
try:
    from .core import (
        HeliosDataHandler,
        DataHandlerError,
        HeliosSignalEngine,
        SignalEngineError,
        TradingSignal,
        PairSignal,
        HeliosRiskEngine,
        RiskEngineError,
        PositionSize,
        RiskCheck
    )
    _CORE_MODULES_AVAILABLE = True
except ImportError:
    # Core modules not fully implemented yet
    _CORE_MODULES_AVAILABLE = False

# Utilities imports
from .utils import (
    setup_helios_logging,
    get_helios_logger,
    log_critical_error,
    log_trade_event,
    log_risk_event,
    HeliosLoggerConfig,
    HeliosAlpacaClient,
    AlpacaAPIError
)

# Main controller import (will be available after implementation)
try:
    from .main_controller import HeliosMainController, HeliosControllerError
    _MAIN_CONTROLLER_AVAILABLE = True
except ImportError:
    # Main controller not yet implemented
    _MAIN_CONTROLLER_AVAILABLE = False

# Execution engine imports (will be available after implementation)
try:
    from .core import (
        HeliosExecutionEngine,
        ExecutionEngineError,
        TradeExecution,
        OrderResult
    )
    _EXECUTION_ENGINE_AVAILABLE = True
except ImportError:
    # Execution engine not yet implemented
    _EXECUTION_ENGINE_AVAILABLE = False

# Package-level constants
DEFAULT_CONFIG_PATH = "config/config.ini"
DEFAULT_STATE_PATH = "state/positions.json"
DEFAULT_LOG_PATH = "logs/helios_bot.log"

# Risk management constants (from PRD)
MAX_DAILY_DRAWDOWN = 0.03  # 3%
MAX_POSITION_CONCENTRATION = 0.20  # 20%
MAX_CONCURRENT_PAIRS = 5
ZSCORE_ENTRY_THRESHOLD = 2.0
ZSCORE_EXIT_THRESHOLD = 0.0
ZSCORE_STOP_THRESHOLD = 3.5
TIME_IN_TRADE_STOP_DAYS = 60

# Define what gets imported with "from helios_bot import *"
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__status__',

    # Utilities (always available)
    'setup_helios_logging',
    'get_helios_logger',
    'log_critical_error',
    'log_trade_event',
    'log_risk_event',
    'HeliosLoggerConfig',
    'HeliosAlpacaClient',
    'AlpacaAPIError',

    # Constants
    'DEFAULT_CONFIG_PATH',
    'DEFAULT_STATE_PATH',
    'DEFAULT_LOG_PATH',
    'MAX_DAILY_DRAWDOWN',
    'MAX_POSITION_CONCENTRATION',
    'MAX_CONCURRENT_PAIRS',
    'ZSCORE_ENTRY_THRESHOLD',
    'ZSCORE_EXIT_THRESHOLD',
    'ZSCORE_STOP_THRESHOLD',
    'TIME_IN_TRADE_STOP_DAYS'
]

# Add core modules to exports if available
if _CORE_MODULES_AVAILABLE:
    __all__.extend([
        'HeliosDataHandler',
        'DataHandlerError',
        'HeliosSignalEngine',
        'SignalEngineError',
        'TradingSignal',
        'PairSignal',
        'HeliosRiskEngine',
        'RiskEngineError',
        'PositionSize',
        'RiskCheck'
    ])

# Add execution engine to exports if available
if _EXECUTION_ENGINE_AVAILABLE:
    __all__.extend([
        'HeliosExecutionEngine',
        'ExecutionEngineError',
        'TradeExecution',
        'OrderResult'
    ])

# Add main controller to exports if available
if _MAIN_CONTROLLER_AVAILABLE:
    __all__.extend(['HeliosMainController', 'HeliosControllerError'])


def get_version_info() -> dict:
    """Get comprehensive version and system information.

    Returns:
        Dictionary containing version and system details.
    """
    return {
        'version': __version__,
        'author': __author__,
        'status': __status__,
        'main_controller_available': _MAIN_CONTROLLER_AVAILABLE,
        'package_name': __name__
    }


def quick_start_check() -> dict:
    """Perform a quick system check to verify Helios is ready to run.

    Returns:
        Dictionary containing system readiness status.
    """
    import os
    from pathlib import Path

    checks = {
        'config_file_exists': os.path.exists(DEFAULT_CONFIG_PATH),
        'state_directory_exists': Path(DEFAULT_STATE_PATH).parent.exists(),
        'log_directory_exists': Path(DEFAULT_LOG_PATH).parent.exists(),
        'env_file_exists': os.path.exists('.env'),
        'core_modules_available': _CORE_MODULES_AVAILABLE,
        'execution_engine_available': _EXECUTION_ENGINE_AVAILABLE,
        'main_controller_available': _MAIN_CONTROLLER_AVAILABLE
    }

    checks['ready_to_run'] = all([
        checks['config_file_exists'],
        checks['state_directory_exists'],
        checks['log_directory_exists'],
        checks['core_modules_available'],
        checks['main_controller_available']
    ])

    return checks


# Initialize logging when package is imported
try:
    _logger = setup_helios_logging()
    _logger.info(f"Helios Trading Bot v{__version__} package loaded successfully")
except Exception as e:
    print(f"Warning: Failed to initialize logging during package import: {e}")


# Package-level configuration validation
def validate_environment() -> bool:
    """Validate that the environment is properly configured for Helios.

    Returns:
        True if environment is valid, False otherwise.
    """
    try:
        from dotenv import load_dotenv
        import os

        load_dotenv()

        required_env_vars = ['APCA_API_KEY_ID', 'APCA_API_SECRET_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]

        if missing_vars:
            print(f"ERROR: Missing required environment variables: {missing_vars}")
            print("Please ensure your .env file contains all required API credentials.")
            return False

        return True

    except Exception as e:
        print(f"ERROR: Failed to validate environment: {e}")
        return False


if __name__ == "__main__":
    # Package self-test when run directly
    print(f"Helios Trading Bot v{__version__}")
    print("=" * 50)

    version_info = get_version_info()
    for key, value in version_info.items():
        print(f"{key}: {value}")

    print("\nSystem Readiness Check:")
    print("-" * 30)

    readiness = quick_start_check()
    for check, status in readiness.items():
        status_str = "✓" if status else "✗"
        print(f"{status_str} {check}: {status}")

    print(f"\nEnvironment Validation: {'✓' if validate_environment() else '✗'}")

    if readiness['ready_to_run'] and validate_environment():
        print("\n🎯 Helios is ready to run!")
    else:
        print("\n⚠️  Helios requires additional setup before running.")
        print("Please refer to the README.md for setup instructions.")
