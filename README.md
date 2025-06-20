# Helios Trading Bot

A systematic, fully automated trading algorithm designed for mean-reversion and momentum strategies in the US equity market. Helios operates on a market-neutral basis, focusing on statistical arbitrage through pairs trading while maintaining strict risk management protocols.

## 🎯 Overview

**Helios** is built for independent retail traders who want to automate their trading strategies with institutional-grade risk management. The system exploits statistical mispricings in stock pairs through cointegration analysis and Z-score mean reversion, while maintaining comprehensive safety mechanisms to protect capital.

### Key Features

- **Market-Neutral Strategy**: Pairs trading approach insulated from broad market direction
- **Statistical Arbitrage**: Cointegration-based pair selection with mean-reversion signals
- **Comprehensive Risk Management**: Multi-level stop losses and circuit breakers
- **Automated Execution**: Full automation with Alpaca API integration
- **Safety-First Design**: Emergency stop mechanisms and fail-safes
- **Production-Ready**: Containerized deployment with comprehensive logging

### Performance Targets

- **Sharpe Ratio**: > 1.0
- **Maximum Drawdown**: < 15%
- **Annualized Return**: > 10%
- **Win/Loss Ratio**: > 50%

## 🚨 Safety Notice

**⚠️ IMPORTANT: This system is designed to trade real money. Never run this system without:**
1. Thoroughly understanding the code and strategy
2. Testing extensively in paper trading mode
3. Implementing proper risk management settings
4. Having emergency stop procedures in place

## 📋 Requirements

### System Requirements
- Python 3.12 or higher
- Unix-like environment (Linux/macOS recommended)
- Minimum 4GB RAM
- Stable internet connection

### API Requirements
- Active Alpaca trading account
- Paper trading API credentials (for testing)
- Live trading API credentials (for production)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd helios

# Create virtual environment
python -m venv helios_env
source helios_env/bin/activate  # On Windows: helios_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API credentials
nano .env
```

Add your Alpaca API credentials to `.env`:
```
APCA_API_KEY_ID="your_paper_trading_key_id"
APCA_API_SECRET_KEY="your_paper_trading_secret_key"
```

### 3. Configuration Review

Review and customize `config/config.ini`:
- Trading universe (default: XLF financial sector components)
- Risk parameters (position sizing, stops, limits)
- Strategy parameters (Z-score thresholds, cointegration settings)

### 4. First Run (Paper Trading)

```bash
# Ensure you're in the virtual environment
source helios_env/bin/activate

# Run the bot
python run.py
```

## 🏗️ Architecture

### Core Modules

```
helios_bot/
├── core/
│   ├── data_handler.py      # Market data fetching and processing
│   ├── signal_engine.py     # Cointegration analysis and signal generation
│   ├── risk_engine.py       # Position sizing and risk management
│   └── execution_engine.py  # Order placement and management
├── utils/
│   ├── api_client.py        # Alpaca API integration
│   └── logger_config.py     # Centralized logging system
└── main_controller.py       # System orchestration and safety
```

### Trading Workflow

1. **Data Collection**: Fetch daily OHLCV data for configured universe
2. **Cointegration Screening**: Weekly analysis to identify candidate pairs
3. **Signal Generation**: Daily Z-score calculation and entry/exit signals
4. **Risk Assessment**: Position sizing and risk validation
5. **Order Execution**: Simultaneous pair trade placement
6. **Position Monitoring**: Continuous safety mechanism monitoring
7. **Exit Management**: Profit-taking and stop-loss execution

## ⚙️ Configuration

### Trading Parameters

Key settings in `config/config.ini`:

```ini
[Strategy]
coint_pvalue_threshold = 0.05        # Cointegration significance level
zscore_window = 60                   # Rolling window for Z-score calculation
zscore_entry_threshold = 2.0         # Entry signal threshold
zscore_exit_threshold = 0.0          # Exit signal threshold (mean reversion)

[Risk]
risk_factor_per_leg = 0.005          # 0.5% risk per position leg
max_position_concentration = 0.20     # 20% max position size
max_concurrent_pairs = 5             # Maximum simultaneous pairs
max_daily_drawdown_limit = 0.03      # 3% daily drawdown circuit breaker
```

### Universe Configuration

Default universe is XLF (Financial Select Sector SPDR Fund) components:
```ini
[Universe]
tickers = JPM,GS,MS,BAC,C,WFC,AXP,USB,PNC,BLK,SCHW,TFC,COF,MTB,FITB,HBAN,RF,CFG,KEY,ZION,CMA,PBCT,SIVB,CBOE,NTRS
```

## 🛡️ Safety Mechanisms

### Trade-Level Stops (SL-1)
- **Z-Score Failure Stop**: Position closed if |Z-score| > 3.5
- **Time-in-Trade Stop**: Automatic closure after 60 trading days

### Portfolio-Level Stops (SL-2)
- **Daily Drawdown Circuit Breaker**: 3% daily loss limit triggers full liquidation
- **Correlation Check**: Positions closed if 60-day correlation < 0.70

### System-Level Stops (SL-3)
- **Manual Kill Switch**: Independent script for emergency liquidation
- **API Connectivity Failure**: Halt new trades after 3 consecutive API failures

### Emergency Procedures

```bash
# Emergency stop - liquidate all positions immediately
python kill_switch.py

# Check system status
python -c "from helios_bot.main_controller import HeliosMainController; controller = HeliosMainController(); print(controller.get_system_status())"
```

## 📊 Monitoring

### Logging

All system activity is logged to `logs/helios_bot.log` with structured format:
- **INFO**: Routine operations and decisions
- **WARNING**: Recoverable issues requiring attention
- **ERROR**: Failed operations that don't halt the system
- **CRITICAL**: Fatal errors requiring immediate intervention

### State Tracking

System state is persisted in `state/positions.json`:
- Current open positions
- Entry prices and Z-scores
- Position metadata and timing
- Risk metrics and monitoring data

## 🐳 Docker Deployment

### Build Container

```bash
# Build Docker image
docker build -t helios-trading-bot .

# Run container
docker run -d \
  --name helios-bot \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/state:/app/state \
  helios-trading-bot
```

### Production Deployment

For production deployment:
1. Use live API credentials (not paper trading)
2. Set up proper monitoring and alerting
3. Configure automated backups for state files
4. Implement log rotation and archival
5. Set up health checks and restart policies

## 🧪 Testing

### Paper Trading Validation

**Required**: Run in paper trading mode for minimum 3 months before live trading:

```bash
# Ensure paper trading credentials in .env
python run.py
```

Monitor performance metrics:
- Sharpe ratio progression
- Maximum drawdown periods
- Win/loss ratio stability
- Safety mechanism activations

### Unit Testing

```bash
# Run test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=helios_bot --cov-report=html
```

## 📈 Performance Analysis

### Key Metrics

Monitor these metrics in logs and state files:
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Time in market per trade
- **Correlation Stability**: Pair relationship strength

### Analysis Tools

```python
# Example: Analyze trading history
from helios_bot.utils.analysis import TradingAnalyzer

analyzer = TradingAnalyzer('state/positions.json', 'logs/helios_bot.log')
analyzer.generate_performance_report()
```

## ⚠️ Risk Warnings

1. **Capital Risk**: Trading involves substantial risk of loss
2. **Systematic Risk**: Algorithm may fail during market stress
3. **Technology Risk**: System failures could result in losses
4. **Model Risk**: Statistical relationships may break down
5. **Execution Risk**: Slippage and timing issues in volatile markets

**Never risk more than you can afford to lose.**

## 🔧 Troubleshooting

### Common Issues

**API Connection Failures**:
```bash
# Check API credentials
python -c "from helios_bot.utils.api_client import HeliosAlpacaClient; client = HeliosAlpacaClient(); print(client.health_check())"
```

**Permission Errors**:
```bash
# Fix file permissions
chmod +x run.py
chmod 755 logs/ state/
```

**Module Import Errors**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Emergency Procedures

1. **Immediate Stop**: Run `python kill_switch.py`
2. **Check Positions**: Review `state/positions.json`
3. **Review Logs**: Check `logs/helios_bot.log` for errors
4. **API Status**: Verify Alpaca API connectivity
5. **System Health**: Run health check diagnostics

## 📚 Additional Resources

- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [Statistical Arbitrage Research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=872435)
- [Risk Management Best Practices](https://www.risk.net/)
- [Cointegration Analysis](https://en.wikipedia.org/wiki/Cointegration)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚡ Support

For issues and questions:
1. Check logs for error details
2. Review configuration settings
3. Verify API connectivity
4. Run system health checks

**Remember: This system trades real money. Always prioritize safety over returns.**

---

**Disclaimer**: This software is for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this system.