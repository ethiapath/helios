"""Core modules package for the Helios trading bot.

This package contains the core trading logic modules that implement the
main functionality of the Helios systematic trading system as defined
in the Product Requirements Document (PRD).

Modules:
- data_handler: Module 1 - Data Handler & Universe Management
- signal_engine: Module 2 - Signal Generation Engine
- risk_engine: Module 3 - Risk & Position Sizing Engine
- execution_engine: Module 4 - Order Execution Engine (pending)

Author: Helios Trading Bot
Version: 1.0
"""

# Import existing modules
from .data_handler import (
    HeliosDataHandler,
    DataHandlerError
)

from .signal_engine import (
    HeliosSignalEngine,
    SignalEngineError,
    TradingSignal,
    PairSignal,
    CointegrationResult,
    MomentumRanking,
    SignalType,
    SignalStrength
)

from .risk_engine import (
    HeliosRiskEngine,
    RiskEngineError,
    PositionSize,
    RiskCheck,
    PortfolioRisk,
    RiskLevel,
    PositionSide
)

# Execution engine will be imported when implemented
try:
    from .execution_engine import (
        HeliosExecutionEngine,
        ExecutionEngineError,
        TradeExecution,
        OrderResult
    )
    _EXECUTION_ENGINE_AVAILABLE = True
except ImportError:
    # Execution engine not yet implemented
    _EXECUTION_ENGINE_AVAILABLE = False

__all__ = [
    # Data Handler
    'HeliosDataHandler',
    'DataHandlerError',

    # Signal Engine
    'HeliosSignalEngine',
    'SignalEngineError',
    'TradingSignal',
    'PairSignal',
    'CointegrationResult',
    'MomentumRanking',
    'SignalType',
    'SignalStrength',

    # Risk Engine
    'HeliosRiskEngine',
    'RiskEngineError',
    'PositionSize',
    'RiskCheck',
    'PortfolioRisk',
    'RiskLevel',
    'PositionSide'
]

# Add execution engine to exports if available
if _EXECUTION_ENGINE_AVAILABLE:
    __all__.extend([
        'HeliosExecutionEngine',
        'ExecutionEngineError',
        'TradeExecution',
        'OrderResult'
    ])
