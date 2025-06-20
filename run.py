#!/usr/bin/env python3
"""
Helios Trading Bot - Main Entry Point

This is the primary entry point for the Helios systematic trading algorithm.
It initializes all components, manages the trading cycle, and ensures safe operation.

Safety Features:
- Comprehensive error handling and graceful shutdown
- Environment validation before startup
- Proper logging initialization
- Signal handling for clean exits
- Emergency stop capabilities

Usage:
    python run.py

Requirements:
    - .env file with valid API credentials
    - config/config.ini with trading parameters
    - All required Python dependencies installed

Author: Helios Trading Bot
Version: 1.0
"""

import os
import sys
import signal
import time
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    from helios_bot.main_controller import HeliosMainController, HeliosControllerError, SystemState
    from helios_bot.utils.logger_config import get_helios_logger, log_critical_error
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import required modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class HeliosRunner:
    """Main runner class for the Helios trading bot."""

    def __init__(self):
        self.controller: Optional[HeliosMainController] = None
        self.logger = None
        self.shutdown_requested = False

    def setup_environment(self) -> bool:
        """Load and validate environment variables."""
        try:
            # Load environment variables from .env file
            env_file = project_root / ".env"
            if not env_file.exists():
                print("ERROR: .env file not found. Please copy .env.example to .env and configure your API keys.")
                print("Expected location: .env")
                return False

            load_dotenv(env_file)

            # Validate required environment variables
            required_vars = [
                "APCA_API_KEY_ID",
                "APCA_API_SECRET_KEY"
            ]

            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)

            if missing_vars:
                print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
                print("Please check your .env file and ensure all API credentials are set.")
                return False

            print("✓ Environment variables loaded successfully")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load environment: {e}")
            return False

    def setup_logging(self) -> bool:
        """Initialize the logging system."""
        try:
            self.logger = get_helios_logger(__name__)
            self.logger.info("=" * 60)
            self.logger.info("HELIOS TRADING BOT STARTING UP")
            self.logger.info("=" * 60)
            self.logger.info("Logging system initialized successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize logging: {e}")
            return False

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_names = {
                signal.SIGINT: "SIGINT (Ctrl+C)",
                signal.SIGTERM: "SIGTERM"
            }
            signal_name = signal_names.get(signum, f"Signal {signum}")

            if self.logger:
                self.logger.warning(f"Received {signal_name} - initiating graceful shutdown...")
            else:
                print(f"Received {signal_name} - initiating graceful shutdown...")

            self.shutdown_requested = True

            if self.controller:
                try:
                    self.controller.shutdown()
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during controller shutdown: {e}")
                    else:
                        print(f"Error during controller shutdown: {e}")

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if self.logger:
            self.logger.info("Signal handlers registered for graceful shutdown")

    def initialize_controller(self) -> bool:
        """Initialize the main controller."""
        try:
            config_path = project_root / "config" / "config.ini"
            if not config_path.exists():
                self.logger.error(f"Configuration file not found: {config_path}")
                return False

            self.logger.info("Initializing Helios Main Controller...")
            self.controller = HeliosMainController(str(config_path))

            # Perform health check
            health_status = self.controller.health_check()
            if not health_status.get('overall_healthy', False):
                self.logger.error("Controller health check failed!")
                self.logger.error(f"Health status: {health_status}")
                return False

            self.logger.info("✓ Controller initialized and health check passed")
            return True

        except Exception as e:
            log_critical_error(self.logger, f"Failed to initialize controller: {e}")
            return False

    def run_main_loop(self) -> None:
        """Run the main trading loop."""
        try:
            self.logger.info("Starting main trading loop...")

            while not self.shutdown_requested:
                try:
                    # Run daily trading cycle
                    self.logger.info("Starting daily trading cycle...")
                    cycle_result = self.controller.run_daily_cycle()

                    if cycle_result.get('emergency_stop', False):
                        self.logger.critical("Emergency stop triggered during trading cycle!")
                        break

                    # Check system status
                    status = self.controller.get_system_status()
                    if status.state in [SystemState.CRITICAL, SystemState.EMERGENCY_STOP, SystemState.KILL_SWITCH_ACTIVE]:
                        self.logger.critical(f"System in critical state: {status.state}")
                        break

                    # Log cycle completion
                    self.logger.info(f"Daily cycle completed successfully. Status: {status.state}")

                    # Wait for next cycle (daily execution)
                    # In production, this would be managed by cron/scheduler
                    # For now, we'll wait 24 hours or until shutdown
                    sleep_duration = 24 * 60 * 60  # 24 hours in seconds

                    self.logger.info(f"Sleeping for {sleep_duration} seconds until next cycle...")
                    for _ in range(sleep_duration):
                        if self.shutdown_requested:
                            break
                        time.sleep(1)

                except HeliosControllerError as e:
                    self.logger.error(f"Controller error during main loop: {e}")
                    # Continue loop unless it's a critical error
                    if "critical" in str(e).lower() or "emergency" in str(e).lower():
                        break
                    time.sleep(60)  # Wait a minute before retrying

                except Exception as e:
                    log_critical_error(self.logger, f"Unexpected error in main loop: {e}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt - shutting down...")
        except Exception as e:
            log_critical_error(self.logger, f"Fatal error in main loop: {e}")

    def shutdown(self) -> None:
        """Perform clean shutdown procedures."""
        try:
            if self.logger:
                self.logger.info("Initiating shutdown procedures...")

            if self.controller:
                try:
                    # Get final status
                    status = self.controller.get_system_status()
                    if self.logger:
                        self.logger.info(f"Final system status: {status.state}")
                        self.logger.info(f"Active positions: {status.active_positions}")

                    # Shutdown controller
                    self.controller.shutdown()
                    if self.logger:
                        self.logger.info("Controller shutdown completed")

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during controller shutdown: {e}")

            if self.logger:
                self.logger.info("=" * 60)
                self.logger.info("HELIOS TRADING BOT SHUTDOWN COMPLETE")
                self.logger.info("=" * 60)

        except Exception as e:
            print(f"Error during shutdown: {e}")

    def run(self) -> int:
        """Main run method."""
        try:
            print("Starting Helios Trading Bot...")

            # Setup environment
            if not self.setup_environment():
                return 1

            # Setup logging
            if not self.setup_logging():
                return 1

            # Setup signal handlers
            self.setup_signal_handlers()

            # Initialize controller
            if not self.initialize_controller():
                return 1

            # Run main loop
            self.run_main_loop()

            return 0

        except Exception as e:
            error_msg = f"Fatal error in main run method: {e}"
            if self.logger:
                log_critical_error(self.logger, error_msg)
            else:
                print(f"CRITICAL ERROR: {error_msg}")
            return 1

        finally:
            self.shutdown()


def main() -> int:
    """Entry point for the Helios trading bot."""
    runner = HeliosRunner()
    return runner.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
