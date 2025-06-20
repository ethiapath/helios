"""Signal Generation Engine for the Helios trading bot.

This module implements Module 2 from the PRD: Signal Generation Engine.
It provides comprehensive signal generation for pairs trading using cointegration
analysis and momentum monitoring for the Helios systematic trading system.

Functional Requirements Implemented:
- FR-2.1: Pairs trading signal generation with cointegration screening
- FR-2.1.1: Weekly cointegration screening using Engle-Granger method
- FR-2.1.2: Daily Z-score calculation on 60-day rolling window
- FR-2.1.3: Entry/exit logic based on Z-score thresholds
- FR-2.2: Cross-sectional momentum monitoring (V1 - monitoring only)

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

# Statistical analysis imports
try:
    from statsmodels.tsa.stattools import coint
    from statsmodels.api import OLS
    import statsmodels.api as sm
except ImportError as e:
    print(f"ERROR: statsmodels not installed. Run: pip install statsmodels")
    raise e

from ..utils.logger_config import get_helios_logger
from .data_handler import HeliosDataHandler, DataHandlerError


class SignalEngineError(Exception):
    """Custom exception for Signal Engine related errors."""
    pass


class SignalType(Enum):
    """Enumeration for different types of trading signals."""
    PAIRS_LONG_SPREAD = "pairs_long_spread"    # Long first stock, short second
    PAIRS_SHORT_SPREAD = "pairs_short_spread"  # Short first stock, long second
    MOMENTUM_LONG = "momentum_long"            # Momentum long signal
    MOMENTUM_SHORT = "momentum_short"          # Momentum short signal
    NO_SIGNAL = "no_signal"                    # No trading signal


class SignalStrength(Enum):
    """Enumeration for signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


@dataclass
class TradingSignal:
    """Base data structure for trading signals."""
    signal_type: SignalType
    timestamp: datetime
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


@dataclass
class PairSignal(TradingSignal):
    """Specialized trading signal for pairs trading."""
    stock1: str
    stock2: str
    spread_zscore: float
    cointegration_pvalue: float
    hedge_ratio: float
    entry_price1: float
    entry_price2: float
    expected_profit_target: Optional[float] = None
    expected_stop_loss: Optional[float] = None


@dataclass
class CointegrationResult:
    """Results from cointegration analysis."""
    stock1: str
    stock2: str
    cointegration_stat: float
    pvalue: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    is_cointegrated: bool
    analysis_date: datetime


@dataclass
class MomentumRanking:
    """Momentum ranking for a single stock."""
    symbol: str
    momentum_score: float  # 12M-1M return
    rank: int
    percentile: float
    analysis_date: datetime


class HeliosSignalEngine:
    """Signal Generation Engine for the Helios trading system.

    This class implements comprehensive signal generation including:
    - Pairs trading signals based on cointegration analysis
    - Cross-sectional momentum analysis and monitoring
    - Signal strength assessment and filtering
    - Integration with risk management parameters
    """

    def __init__(self,
                 data_handler: HeliosDataHandler,
                 config_path: str = "config/config.ini") -> None:
        """Initialize the Signal Generation Engine.

        Args:
            data_handler: Initialized data handler for market data.
            config_path: Path to the configuration file.

        Raises:
            SignalEngineError: If initialization fails.
        """
        self.logger = get_helios_logger('signal_engine')
        self.data_handler = data_handler
        self.config_path = config_path
        self.config = self._load_config()

        # Load strategy parameters from config
        self._load_strategy_parameters()

        # Cache for cointegration results and momentum rankings
        self._cointegration_cache: Dict[str, CointegrationResult] = {}
        self._momentum_cache: Dict[str, MomentumRanking] = {}
        self._last_cointegration_screening: Optional[datetime] = None
        self._last_momentum_analysis: Optional[datetime] = None

        # Current candidate pairs (cointegrated pairs)
        self.candidate_pairs: List[Tuple[str, str]] = []

        self.logger.info("Signal engine initialized successfully")

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file.

        Returns:
            ConfigParser object with loaded configuration.

        Raises:
            SignalEngineError: If config file cannot be loaded.
        """
        if not os.path.exists(self.config_path):
            raise SignalEngineError(f"Configuration file not found: {self.config_path}")

        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
            return config
        except Exception as e:
            raise SignalEngineError(f"Failed to load configuration: {str(e)}")

    def _load_strategy_parameters(self) -> None:
        """Load strategy parameters from configuration."""
        try:
            # Cointegration parameters
            self.coint_pvalue_threshold = self.config.getfloat(
                'Strategy', 'coint_pvalue_threshold', fallback=0.05)
            self.coint_screening_frequency_days = self.config.getint(
                'Strategy', 'coint_screening_frequency_days', fallback=7)

            # Z-score parameters
            self.zscore_window = self.config.getint(
                'Strategy', 'zscore_window', fallback=60)
            self.zscore_entry_threshold = self.config.getfloat(
                'Strategy', 'zscore_entry_threshold', fallback=2.0)
            self.zscore_exit_threshold = self.config.getfloat(
                'Strategy', 'zscore_exit_threshold', fallback=0.0)

            # Momentum parameters
            self.momentum_window_long = self.config.getint(
                'Strategy', 'momentum_window_long', fallback=252)  # 12 months
            self.momentum_window_short = self.config.getint(
                'Strategy', 'momentum_window_short', fallback=21)   # 1 month
            self.momentum_rebalance_frequency_days = self.config.getint(
                'Strategy', 'momentum_rebalance_frequency_days', fallback=30)

            # Risk parameters (for signal strength assessment)
            self.correlation_threshold = self.config.getfloat(
                'Risk', 'correlation_threshold', fallback=0.70)

            self.logger.info("Strategy parameters loaded successfully")
            self.logger.info(f"Cointegration p-value threshold: {self.coint_pvalue_threshold}")
            self.logger.info(f"Z-score entry threshold: {self.zscore_entry_threshold}")
            self.logger.info(f"Z-score window: {self.zscore_window} days")

        except Exception as e:
            raise SignalEngineError(f"Failed to load strategy parameters: {str(e)}")

    def run_weekly_cointegration_screening(self, force_update: bool = False) -> List[CointegrationResult]:
        """Perform weekly cointegration screening on all universe pairs (FR-2.1.1).

        Args:
            force_update: Force screening even if recently completed.

        Returns:
            List of cointegration results for all tested pairs.

        Raises:
            SignalEngineError: If screening fails.
        """
        try:
            # Check if screening is needed
            if not force_update and self._last_cointegration_screening:
                days_since_last = (datetime.now() - self._last_cointegration_screening).days
                if days_since_last < self.coint_screening_frequency_days:
                    self.logger.info(f"Cointegration screening not needed: "
                                   f"last done {days_since_last} days ago")
                    return list(self._cointegration_cache.values())

            self.logger.info("Starting weekly cointegration screening")

            # Get universe for analysis
            universe = self.data_handler.get_universe()
            if len(universe) < 2:
                raise SignalEngineError("Insufficient stocks in universe for pairs analysis")

            # Fetch historical data for cointegration analysis
            # Need more data for reliable cointegration testing
            lookback_days = max(252, self.zscore_window * 2)  # At least 1 year or 2x Z-score window
            historical_data = self.data_handler.get_historical_data(
                symbols=universe,
                days_back=lookback_days
            )

            if historical_data.empty:
                raise SignalEngineError("No historical data available for cointegration screening")

            # Perform cointegration tests on all pairs
            cointegration_results = []
            candidate_pairs = []

            total_pairs = len(universe) * (len(universe) - 1) // 2
            tested_pairs = 0

            self.logger.info(f"Testing {total_pairs} pairs for cointegration")

            for i, stock1 in enumerate(universe):
                for j, stock2 in enumerate(universe[i+1:], i+1):
                    tested_pairs += 1

                    try:
                        result = self._test_cointegration(historical_data, stock1, stock2)
                        cointegration_results.append(result)

                        # Cache result
                        pair_key = f"{stock1}_{stock2}"
                        self._cointegration_cache[pair_key] = result

                        # Add to candidate pairs if cointegrated
                        if result.is_cointegrated:
                            candidate_pairs.append((stock1, stock2))
                            self.logger.info(f"Cointegrated pair found: {stock1}/{stock2} "
                                           f"(p-value: {result.pvalue:.4f})")

                    except Exception as e:
                        self.logger.warning(f"Failed to test cointegration for {stock1}/{stock2}: {str(e)}")
                        continue

                    # Progress logging
                    if tested_pairs % 50 == 0:
                        self.logger.info(f"Cointegration testing progress: {tested_pairs}/{total_pairs} pairs")

            # Update candidate pairs
            self.candidate_pairs = candidate_pairs
            self._last_cointegration_screening = datetime.now()

            cointegrated_count = len(candidate_pairs)
            self.logger.info(f"Cointegration screening completed: {cointegrated_count} "
                           f"cointegrated pairs found out of {tested_pairs} tested")

            return cointegration_results

        except Exception as e:
            self.logger.error(f"Cointegration screening failed: {str(e)}")
            raise SignalEngineError(f"Cointegration screening failed: {str(e)}")

    def _test_cointegration(self,
                          historical_data: pd.DataFrame,
                          stock1: str,
                          stock2: str) -> CointegrationResult:
        """Test cointegration between two stocks using Engle-Granger method.

        Args:
            historical_data: Historical price data.
            stock1: First stock symbol.
            stock2: Second stock symbol.

        Returns:
            CointegrationResult object with test results.

        Raises:
            SignalEngineError: If cointegration test fails.
        """
        try:
            # Extract price series for both stocks
            stock1_data = historical_data[
                historical_data.index.get_level_values('symbol') == stock1
            ]['close'].sort_index()

            stock2_data = historical_data[
                historical_data.index.get_level_values('symbol') == stock2
            ]['close'].sort_index()

            # Align data on common dates
            aligned_data = pd.DataFrame({
                stock1: stock1_data,
                stock2: stock2_data
            }).dropna()

            if len(aligned_data) < 30:  # Minimum data points for reliable test
                raise SignalEngineError(f"Insufficient data for cointegration test: "
                                      f"only {len(aligned_data)} observations")

            # Perform Engle-Granger cointegration test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress statsmodels warnings
                coint_stat, pvalue, critical_values = coint(
                    aligned_data[stock1],
                    aligned_data[stock2]
                )

            # Calculate hedge ratio using OLS regression
            X = sm.add_constant(aligned_data[stock2])
            model = OLS(aligned_data[stock1], X).fit()
            hedge_ratio = model.params[stock2]

            # Determine if pair is cointegrated
            is_cointegrated = pvalue < self.coint_pvalue_threshold

            # Format critical values
            crit_values_dict = {
                '1%': critical_values[0],
                '5%': critical_values[1],
                '10%': critical_values[2]
            }

            return CointegrationResult(
                stock1=stock1,
                stock2=stock2,
                cointegration_stat=coint_stat,
                pvalue=pvalue,
                critical_values=crit_values_dict,
                hedge_ratio=hedge_ratio,
                is_cointegrated=is_cointegrated,
                analysis_date=datetime.now()
            )

        except Exception as e:
            raise SignalEngineError(f"Cointegration test failed for {stock1}/{stock2}: {str(e)}")

    def generate_daily_signals(self) -> List[PairSignal]:
        """Generate daily trading signals for all candidate pairs (FR-2.1.2, FR-2.1.3).

        Returns:
            List of pair trading signals.

        Raises:
            SignalEngineError: If signal generation fails.
        """
        try:
            self.logger.info("Starting daily signal generation")

            if not self.candidate_pairs:
                self.logger.info("No candidate pairs available for signal generation")
                return []

            # Get recent price data for Z-score calculation
            all_symbols = set()
            for stock1, stock2 in self.candidate_pairs:
                all_symbols.update([stock1, stock2])

            recent_data = self.data_handler.get_historical_data(
                symbols=list(all_symbols),
                days_back=self.zscore_window + 10  # Extra buffer for calculation
            )

            if recent_data.empty:
                raise SignalEngineError("No recent data available for signal generation")

            # Get latest prices for entry price calculation
            latest_prices = self.data_handler.get_latest_prices(list(all_symbols))

            signals = []

            for stock1, stock2 in self.candidate_pairs:
                try:
                    signal = self._generate_pair_signal(
                        recent_data, stock1, stock2, latest_prices
                    )
                    if signal and signal.signal_type != SignalType.NO_SIGNAL:
                        signals.append(signal)
                        self.logger.info(f"Signal generated: {signal.signal_type.value} "
                                       f"for {stock1}/{stock2} (Z-score: {signal.spread_zscore:.2f})")

                except Exception as e:
                    self.logger.warning(f"Failed to generate signal for {stock1}/{stock2}: {str(e)}")
                    continue

            self.logger.info(f"Daily signal generation completed: {len(signals)} signals generated")
            return signals

        except Exception as e:
            self.logger.error(f"Daily signal generation failed: {str(e)}")
            raise SignalEngineError(f"Daily signal generation failed: {str(e)}")

    def _generate_pair_signal(self,
                            historical_data: pd.DataFrame,
                            stock1: str,
                            stock2: str,
                            latest_prices: Dict[str, float]) -> Optional[PairSignal]:
        """Generate trading signal for a specific pair.

        Args:
            historical_data: Historical price data.
            stock1: First stock symbol.
            stock2: Second stock symbol.
            latest_prices: Dictionary of latest prices.

        Returns:
            PairSignal object or None if no signal.
        """
        try:
            # Get cointegration parameters for this pair
            pair_key = f"{stock1}_{stock2}"
            if pair_key not in self._cointegration_cache:
                self.logger.warning(f"No cointegration data found for {stock1}/{stock2}")
                return None

            coint_result = self._cointegration_cache[pair_key]

            # Extract price series
            stock1_prices = historical_data[
                historical_data.index.get_level_values('symbol') == stock1
            ]['close'].sort_index()

            stock2_prices = historical_data[
                historical_data.index.get_level_values('symbol') == stock2
            ]['close'].sort_index()

            # Align data
            aligned_data = pd.DataFrame({
                stock1: stock1_prices,
                stock2: stock2_prices
            }).dropna()

            if len(aligned_data) < self.zscore_window:
                self.logger.warning(f"Insufficient data for Z-score calculation: "
                                  f"{len(aligned_data)} < {self.zscore_window}")
                return None

            # Calculate spread using the hedge ratio
            spread = aligned_data[stock1] - coint_result.hedge_ratio * aligned_data[stock2]

            # Calculate rolling Z-score
            rolling_mean = spread.rolling(window=self.zscore_window).mean()
            rolling_std = spread.rolling(window=self.zscore_window).std()
            z_scores = (spread - rolling_mean) / rolling_std

            # Get current Z-score
            current_zscore = z_scores.iloc[-1]

            # Check for entry signal
            if abs(current_zscore) < self.zscore_entry_threshold:
                return None  # No signal

            # Determine signal type and strength
            if current_zscore > self.zscore_entry_threshold:
                # Spread is too high, short the spread (short stock1, long stock2)
                signal_type = SignalType.PAIRS_SHORT_SPREAD
            else:
                # Spread is too low, long the spread (long stock1, short stock2)
                signal_type = SignalType.PAIRS_LONG_SPREAD

            # Assess signal strength based on Z-score magnitude
            zscore_abs = abs(current_zscore)
            if zscore_abs >= 3.0:
                strength = SignalStrength.CRITICAL
                confidence = 0.95
            elif zscore_abs >= 2.5:
                strength = SignalStrength.STRONG
                confidence = 0.85
            elif zscore_abs >= 2.0:
                strength = SignalStrength.MODERATE
                confidence = 0.70
            else:
                strength = SignalStrength.WEAK
                confidence = 0.55

            # Get entry prices
            entry_price1 = latest_prices.get(stock1, 0.0)
            entry_price2 = latest_prices.get(stock2, 0.0)

            if entry_price1 <= 0 or entry_price2 <= 0:
                self.logger.warning(f"Invalid entry prices for {stock1}/{stock2}: "
                                  f"{entry_price1}, {entry_price2}")
                return None

            # Create signal
            signal = PairSignal(
                signal_type=signal_type,
                timestamp=datetime.now(),
                strength=strength,
                confidence=confidence,
                stock1=stock1,
                stock2=stock2,
                spread_zscore=current_zscore,
                cointegration_pvalue=coint_result.pvalue,
                hedge_ratio=coint_result.hedge_ratio,
                entry_price1=entry_price1,
                entry_price2=entry_price2,
                metadata={
                    'zscore_window': self.zscore_window,
                    'spread_mean': float(rolling_mean.iloc[-1]),
                    'spread_std': float(rolling_std.iloc[-1]),
                    'current_spread': float(spread.iloc[-1]),
                    'correlation': float(aligned_data[stock1].corr(aligned_data[stock2]))
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"Failed to generate pair signal for {stock1}/{stock2}: {str(e)}")
            return None

    def generate_exit_signals(self,
                            open_positions: List[Dict[str, Any]]) -> List[PairSignal]:
        """Generate exit signals for open positions.

        Args:
            open_positions: List of open position dictionaries.

        Returns:
            List of exit signals.
        """
        try:
            self.logger.info(f"Checking exit signals for {len(open_positions)} open positions")

            if not open_positions:
                return []

            # Get all symbols from open positions
            all_symbols = set()
            for position in open_positions:
                all_symbols.update([position['stock1'], position['stock2']])

            # Get recent data for Z-score calculation
            recent_data = self.data_handler.get_historical_data(
                symbols=list(all_symbols),
                days_back=self.zscore_window + 5
            )

            exit_signals = []

            for position in open_positions:
                try:
                    stock1 = position['stock1']
                    stock2 = position['stock2']
                    entry_zscore = position.get('entry_zscore', 0.0)

                    # Calculate current Z-score
                    pair_key = f"{stock1}_{stock2}"
                    if pair_key not in self._cointegration_cache:
                        self.logger.warning(f"No cointegration data for open position {stock1}/{stock2}")
                        continue

                    coint_result = self._cointegration_cache[pair_key]

                    # Extract and align price data
                    stock1_prices = recent_data[
                        recent_data.index.get_level_values('symbol') == stock1
                    ]['close'].sort_index()

                    stock2_prices = recent_data[
                        recent_data.index.get_level_values('symbol') == stock2
                    ]['close'].sort_index()

                    aligned_data = pd.DataFrame({
                        stock1: stock1_prices,
                        stock2: stock2_prices
                    }).dropna()

                    if len(aligned_data) < self.zscore_window:
                        continue

                    # Calculate current spread and Z-score
                    spread = aligned_data[stock1] - coint_result.hedge_ratio * aligned_data[stock2]
                    rolling_mean = spread.rolling(window=self.zscore_window).mean()
                    rolling_std = spread.rolling(window=self.zscore_window).std()
                    z_scores = (spread - rolling_mean) / rolling_std
                    current_zscore = z_scores.iloc[-1]

                    # Check exit condition: Z-score crosses back through zero
                    should_exit = False
                    if entry_zscore > 0 and current_zscore <= self.zscore_exit_threshold:
                        should_exit = True
                    elif entry_zscore < 0 and current_zscore >= -self.zscore_exit_threshold:
                        should_exit = True

                    if should_exit:
                        # Get current prices for exit
                        latest_prices = self.data_handler.get_latest_prices([stock1, stock2])

                        exit_signal = PairSignal(
                            signal_type=SignalType.NO_SIGNAL,  # Exit signal
                            timestamp=datetime.now(),
                            strength=SignalStrength.MODERATE,
                            confidence=0.80,
                            stock1=stock1,
                            stock2=stock2,
                            spread_zscore=current_zscore,
                            cointegration_pvalue=coint_result.pvalue,
                            hedge_ratio=coint_result.hedge_ratio,
                            entry_price1=latest_prices.get(stock1, 0.0),
                            entry_price2=latest_prices.get(stock2, 0.0),
                            metadata={
                                'exit_reason': 'zscore_reversal',
                                'entry_zscore': entry_zscore,
                                'exit_zscore': current_zscore,
                                'position_id': position.get('id', 'unknown')
                            }
                        )

                        exit_signals.append(exit_signal)
                        self.logger.info(f"Exit signal generated for {stock1}/{stock2}: "
                                       f"Z-score {entry_zscore:.2f} → {current_zscore:.2f}")

                except Exception as e:
                    self.logger.warning(f"Failed to check exit for position {position}: {str(e)}")
                    continue

            self.logger.info(f"Exit signal generation completed: {len(exit_signals)} exit signals")
            return exit_signals

        except Exception as e:
            self.logger.error(f"Exit signal generation failed: {str(e)}")
            raise SignalEngineError(f"Exit signal generation failed: {str(e)}")

    def run_momentum_analysis(self, force_update: bool = False) -> List[MomentumRanking]:
        """Perform cross-sectional momentum analysis (FR-2.2).

        Args:
            force_update: Force analysis even if recently completed.

        Returns:
            List of momentum rankings for all universe stocks.

        Raises:
            SignalEngineError: If momentum analysis fails.
        """
        try:
            # Check if analysis is needed
            if not force_update and self._last_momentum_analysis:
                days_since_last = (datetime.now() - self._last_momentum_analysis).days
                if days_since_last < self.momentum_rebalance_frequency_days:
                    self.logger.info(f"Momentum analysis not needed: "
                                   f"last done {days_since_last} days ago")
                    return list(self._momentum_cache.values())

            self.logger.info("Starting cross-sectional momentum analysis")

            universe = self.data_handler.get_universe()
            if not universe:
                raise SignalEngineError("No stocks in universe for momentum analysis")

            # Get sufficient historical data for momentum calculation
            lookback_days = self.momentum_window_long + 30  # Buffer for weekends/holidays
            historical_data = self.data_handler.get_historical_data(
                symbols=universe,
                days_back=lookback_days
            )

            if historical_data.empty:
                raise SignalEngineError("No historical data for momentum analysis")

            momentum_scores = []

            for symbol in universe:
                try:
                    symbol_data = historical_data[
                        historical_data.index.get_level_values('symbol') == symbol
                    ]['close'].sort_index()

                    if len(symbol_data) < self.momentum_window_long:
                        self.logger.warning(f"Insufficient data for momentum calculation: {symbol}")
                        continue

                    # Calculate 12M-1M momentum
                    # Get price from 12 months ago and 1 month ago
                    price_12m_ago = symbol_data.iloc[-(self.momentum_window_long)]
                    price_1m_ago = symbol_data.iloc[-(self.momentum_window_short)]

                    # Momentum score = (Price_1M_ago / Price_12M_ago) - 1
                    momentum_score = (price_1m_ago / price_12m_ago) - 1.0

                    momentum_scores.append({
                        'symbol': symbol,
                        'momentum_score': momentum_score
                    })

                except Exception as e:
                    self.logger.warning(f"Failed to calculate momentum for {symbol}: {str(e)}")
                    continue

            if not momentum_scores:
                raise SignalEngineError("No valid momentum scores calculated")

            # Sort by momentum score and assign ranks
            momentum_scores.sort(key=lambda x: x['momentum_score'], reverse=True)

            momentum_rankings = []
            for rank, data in enumerate(momentum_scores, 1):
                percentile = (len(momentum_scores) - rank + 1) / len(momentum_scores) * 100

                ranking = MomentumRanking(
                    symbol=data['symbol'],
                    momentum_score=data['momentum_score'],
                    rank=rank,
                    percentile=percentile,
                    analysis_date=datetime.now()
                )

                momentum_rankings.append(ranking)
                self._momentum_cache[data['symbol']] = ranking

            self._last_momentum_analysis = datetime.now()

            # Log top and bottom performers
            top_performers = momentum_rankings[:5]
            bottom_performers = momentum_rankings[-5:]

            self.logger.info("Top 5 momentum performers:")
            for ranking in top_performers:
                self.logger.info(f"  {ranking.symbol}: {ranking.momentum_score:.2%} "
                               f"(Rank {ranking.rank})")

            self.logger.info("Bottom 5 momentum performers:")
            for ranking in bottom_performers:
                self.logger.info(f"  {ranking.symbol}: {ranking.momentum_score:.2%} "
                               f"(Rank {ranking.rank})")

            self.logger.info(f"Momentum analysis completed: {len(momentum_rankings)} stocks ranked")
            return momentum_rankings

        except Exception as e:
            self.logger.error(f"Momentum analysis failed: {str(e)}")
            raise SignalEngineError(f"Momentum analysis failed: {str(e)}")

    def get_candidate_pairs(self) -> List[Tuple[str, str]]:
        """Get the current list of candidate pairs (cointegrated pairs).

        Returns:
            List of tuples containing cointegrated stock pairs.
        """
        return self.candidate_pairs.copy()

    def get_cointegration_result(self, stock1: str, stock2: str) -> Optional[CointegrationResult]:
        """Get cointegration result for a specific pair.

        Args:
            stock1: First stock symbol.
            stock2: Second stock symbol.

        Returns:
            CointegrationResult object or None if not found.
        """
        pair_key = f"{stock1}_{stock2}"
        return self._cointegration_cache.get(pair_key)

    def get_momentum_ranking(self, symbol: str) -> Optional[MomentumRanking]:
        """Get momentum ranking for a specific symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            MomentumRanking object or None if not found.
        """
        return self._momentum_cache.get(symbol)

    def get_all_momentum_rankings(self) -> List[MomentumRanking]:
        """Get all current momentum rankings.

        Returns:
            List of MomentumRanking objects sorted by rank.
        """
        rankings = list(self._momentum_cache.values())
        return sorted(rankings, key=lambda x: x.rank)

    def validate_signal_conditions(self, signal: PairSignal) -> bool:
        """Validate that signal conditions are still valid.

        Args:
            signal: Trading signal to validate.

        Returns:
            True if signal is valid, False otherwise.
        """
        try:
            # Check if cointegration is still valid
            coint_result = self.get_cointegration_result(signal.stock1, signal.stock2)
            if not coint_result or not coint_result.is_cointegrated:
                self.logger.warning(f"Signal invalid: pair {signal.stock1}/{signal.stock2} "
                                  f"no longer cointegrated")
                return False

            # Check correlation threshold
            if 'correlation' in signal.metadata:
                correlation = signal.metadata['correlation']
                if correlation < self.correlation_threshold:
                    self.logger.warning(f"Signal invalid: correlation {correlation:.3f} "
                                      f"below threshold {self.correlation_threshold}")
                    return False

            # Check signal age (signals should be recent)
            signal_age = (datetime.now() - signal.timestamp).total_seconds() / 3600  # hours
            if signal_age > 24:  # Signals older than 24 hours are stale
                self.logger.warning(f"Signal invalid: signal age {signal_age:.1f} hours")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Signal validation failed: {str(e)}")
            return False

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signal generation status.

        Returns:
            Dictionary containing signal summary information.
        """
        try:
            summary = {
                'candidate_pairs_count': len(self.candidate_pairs),
                'last_cointegration_screening': self._last_cointegration_screening,
                'last_momentum_analysis': self._last_momentum_analysis,
                'cointegration_cache_size': len(self._cointegration_cache),
                'momentum_cache_size': len(self._momentum_cache),
                'strategy_parameters': {
                    'coint_pvalue_threshold': self.coint_pvalue_threshold,
                    'zscore_entry_threshold': self.zscore_entry_threshold,
                    'zscore_window': self.zscore_window,
                    'momentum_window_long': self.momentum_window_long,
                    'momentum_window_short': self.momentum_window_short
                }
            }

            # Add screening status
            if self._last_cointegration_screening:
                days_since_screening = (datetime.now() - self._last_cointegration_screening).days
                summary['days_since_cointegration_screening'] = days_since_screening
                summary['cointegration_screening_needed'] = days_since_screening >= self.coint_screening_frequency_days

            if self._last_momentum_analysis:
                days_since_momentum = (datetime.now() - self._last_momentum_analysis).days
                summary['days_since_momentum_analysis'] = days_since_momentum
                summary['momentum_analysis_needed'] = days_since_momentum >= self.momentum_rebalance_frequency_days

            return summary

        except Exception as e:
            self.logger.error(f"Failed to generate signal summary: {str(e)}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the signal engine.

        Returns:
            Dictionary containing health check results.
        """
        health_status = {
            'data_handler_available': self.data_handler is not None,
            'config_loaded': self.config is not None,
            'candidate_pairs_available': len(self.candidate_pairs) > 0,
            'cointegration_cache_populated': len(self._cointegration_cache) > 0,
            'momentum_cache_populated': len(self._momentum_cache) > 0,
            'recent_cointegration_screening': False,
            'recent_momentum_analysis': False
        }

        try:
            # Check if data handler is healthy
            if self.data_handler:
                data_health = self.data_handler.health_check()
                health_status['data_handler_healthy'] = data_health.get('overall_healthy', False)

            # Check screening recency
            if self._last_cointegration_screening:
                days_since = (datetime.now() - self._last_cointegration_screening).days
                health_status['recent_cointegration_screening'] = days_since <= self.coint_screening_frequency_days

            if self._last_momentum_analysis:
                days_since = (datetime.now() - self._last_momentum_analysis).days
                health_status['recent_momentum_analysis'] = days_since <= self.momentum_rebalance_frequency_days

            # Test signal generation capability
            try:
                universe = self.data_handler.get_universe()
                if len(universe) >= 2:
                    # Try to generate signals (dry run)
                    health_status['signal_generation_capable'] = True
                else:
                    health_status['signal_generation_capable'] = False
                    health_status['signal_generation_error'] = 'Insufficient universe size'
            except Exception as e:
                health_status['signal_generation_capable'] = False
                health_status['signal_generation_error'] = str(e)

            # Overall health assessment
            critical_checks = [
                'data_handler_available',
                'config_loaded',
                'data_handler_healthy'
            ]

            health_status['overall_healthy'] = all(
                health_status.get(check, False) for check in critical_checks
            )

            if health_status['overall_healthy']:
                self.logger.info("Signal engine health check: HEALTHY")
            else:
                self.logger.warning("Signal engine health check: UNHEALTHY")

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            health_status['error'] = str(e)
            health_status['overall_healthy'] = False

        return health_status

    def cleanup_cache(self, max_age_days: int = 30) -> None:
        """Clean up old cache entries.

        Args:
            max_age_days: Maximum age in days for cache entries.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)

            # Clean cointegration cache
            old_coint_keys = [
                key for key, result in self._cointegration_cache.items()
                if result.analysis_date < cutoff_date
            ]

            for key in old_coint_keys:
                del self._cointegration_cache[key]

            # Clean momentum cache
            old_momentum_keys = [
                key for key, ranking in self._momentum_cache.items()
                if ranking.analysis_date < cutoff_date
            ]

            for key in old_momentum_keys:
                del self._momentum_cache[key]

            if old_coint_keys or old_momentum_keys:
                self.logger.info(f"Cache cleanup: removed {len(old_coint_keys)} cointegration "
                               f"entries and {len(old_momentum_keys)} momentum entries")

        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {str(e)}")


if __name__ == "__main__":
    # Test the signal engine
    try:
        from .data_handler import HeliosDataHandler

        print("Testing Helios Signal Engine...")

        # Initialize data handler
        data_handler = HeliosDataHandler()
        print("✓ Data handler initialized")

        # Initialize signal engine
        signal_engine = HeliosSignalEngine(data_handler)
        print("✓ Signal engine initialized")

        # Health check
        health = signal_engine.health_check()
        print(f"✓ Health check: {'PASS' if health['overall_healthy'] else 'FAIL'}")

        # Test cointegration screening (small sample)
        print("✓ Testing cointegration screening...")
        try:
            coint_results = signal_engine.run_weekly_cointegration_screening(force_update=True)
            print(f"✓ Cointegration screening: {len(coint_results)} pairs tested")

            # Show cointegrated pairs
            cointegrated = [r for r in coint_results if r.is_cointegrated]
            print(f"✓ Found {len(cointegrated)} cointegrated pairs")

            for result in cointegrated[:3]:  # Show first 3
                print(f"  - {result.stock1}/{result.stock2}: p-value={result.pvalue:.4f}")

        except Exception as e:
            print(f"⚠ Cointegration screening test failed: {e}")

        # Test momentum analysis
        print("✓ Testing momentum analysis...")
        try:
            momentum_rankings = signal_engine.run_momentum_analysis(force_update=True)
            print(f"✓ Momentum analysis: {len(momentum_rankings)} stocks ranked")

            # Show top 5 performers
            top_5 = momentum_rankings[:5]
            print("  Top 5 momentum performers:")
            for ranking in top_5:
                print(f"    {ranking.symbol}: {ranking.momentum_score:.2%}")

        except Exception as e:
            print(f"⚠ Momentum analysis test failed: {e}")

        # Test signal generation
        print("✓ Testing signal generation...")
        try:
            signals = signal_engine.generate_daily_signals()
            print(f"✓ Signal generation: {len(signals)} signals generated")

            for signal in signals[:3]:  # Show first 3 signals
                print(f"  - {signal.signal_type.value}: {signal.stock1}/{signal.stock2} "
                      f"(Z-score: {signal.spread_zscore:.2f})")

        except Exception as e:
            print(f"⚠ Signal generation test failed: {e}")

        # Get summary
        summary = signal_engine.get_signal_summary()
        print(f"✓ Signal summary: {summary.get('candidate_pairs_count', 0)} candidate pairs")

        print("✓ Signal engine test completed successfully!")

    except Exception as e:
        print(f"✗ Signal engine test failed: {e}")
