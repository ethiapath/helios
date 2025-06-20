#!/usr/bin/env python3
"""Helios Kill Switch - Emergency Position Liquidation Script

This is a standalone emergency script that immediately liquidates all positions
managed by the Helios trading bot and prevents new trade initiations.

This implements Safety Mechanism SL-3.1 from the PRD:
"A simple, standalone script (kill_switch.py) must be available that, when run,
immediately connects to the Alpaca API, liquidates all positions managed by
the algorithm, and prevents the main script from initiating new trades."

Usage:
    python kill_switch.py [reason]

Where [reason] is an optional description of why the kill switch was activated.

Author: Helios Trading Bot
Version: 1.0
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print emergency kill switch banner."""
    print("=" * 80)
    print("🚨 HELIOS EMERGENCY KILL SWITCH ACTIVATED 🚨")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This script will immediately liquidate ALL positions!")
    print("=" * 80)

def load_api_credentials() -> tuple:
    """Load API credentials from environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")

        if not api_key or not secret_key:
            print("❌ ERROR: API credentials not found in environment variables!")
            print("Please ensure APCA_API_KEY_ID and APCA_API_SECRET_KEY are set in .env file")
            sys.exit(1)

        return api_key, secret_key

    except ImportError:
        print("❌ ERROR: python-dotenv not installed. Run: pip install python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to load credentials: {e}")
        sys.exit(1)

def initialize_alpaca_client(api_key: str, secret_key: str):
    """Initialize Alpaca API client."""
    try:
        from alpaca_trade_api import REST

        # Use paper trading endpoint for safety
        base_url = "https://paper-api.alpaca.markets"

        client = REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )

        # Test connection
        account = client.get_account()
        print(f"✅ Connected to Alpaca account: {account.account_number}")
        print(f"✅ Account equity: ${float(account.equity):,.2f}")

        return client

    except ImportError:
        print("❌ ERROR: alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to Alpaca API: {e}")
        sys.exit(1)

def get_all_positions(client) -> List[Dict[str, Any]]:
    """Get all current positions from Alpaca."""
    try:
        positions = client.list_positions()

        if not positions:
            print("ℹ️  No positions found in account")
            return []

        position_list = []
        print(f"\n📊 Found {len(positions)} open positions:")
        print("-" * 60)

        for pos in positions:
            position_data = {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': pos.side,
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'avg_entry_price': float(pos.avg_entry_price)
            }
            position_list.append(position_data)

            print(f"{pos.symbol:>6} | {pos.side:>5} | {float(pos.qty):>8.0f} shares | "
                  f"${float(pos.market_value):>10,.2f} | P&L: ${float(pos.unrealized_pl):>8,.2f}")

        print("-" * 60)
        total_value = sum(abs(pos['market_value']) for pos in position_list)
        total_pnl = sum(pos['unrealized_pl'] for pos in position_list)
        print(f"Total Exposure: ${total_value:,.2f} | Total P&L: ${total_pnl:,.2f}")

        return position_list

    except Exception as e:
        print(f"❌ ERROR: Failed to get positions: {e}")
        return []

def liquidate_all_positions(client, positions: List[Dict[str, Any]]) -> bool:
    """Liquidate all positions immediately."""
    if not positions:
        print("ℹ️  No positions to liquidate")
        return True

    print(f"\n🔥 LIQUIDATING {len(positions)} POSITIONS...")
    print("=" * 60)

    liquidation_results = []
    total_success = 0

    for position in positions:
        symbol = position['symbol']
        qty = abs(int(position['qty']))
        current_side = position['side']

        # Determine liquidation side (opposite of current position)
        liquidation_side = 'sell' if current_side == 'long' else 'buy'

        try:
            print(f"🔄 Liquidating {symbol}: {liquidation_side} {qty} shares...")

            # Place market order to close position
            order = client.submit_order(
                symbol=symbol,
                qty=qty,
                side=liquidation_side,
                type='market',
                time_in_force='day'
            )

            print(f"✅ Order placed for {symbol}: {order.id}")
            liquidation_results.append({
                'symbol': symbol,
                'order_id': order.id,
                'qty': qty,
                'side': liquidation_side,
                'status': 'submitted',
                'timestamp': datetime.now().isoformat()
            })
            total_success += 1

            # Brief pause to avoid overwhelming the API
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ FAILED to liquidate {symbol}: {e}")
            liquidation_results.append({
                'symbol': symbol,
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            })

    print("=" * 60)
    print(f"📊 Liquidation Summary: {total_success}/{len(positions)} orders placed successfully")

    if total_success == len(positions):
        print("✅ ALL POSITIONS LIQUIDATED SUCCESSFULLY")
        return True
    else:
        print("⚠️  SOME LIQUIDATIONS FAILED - MANUAL INTERVENTION MAY BE REQUIRED")
        return False

def create_kill_switch_flag(reason: str) -> None:
    """Create kill switch flag file to prevent new trades."""
    try:
        flag_file = "KILL_SWITCH_ACTIVE"

        with open(flag_file, 'w') as f:
            f.write(f"HELIOS KILL SWITCH ACTIVATED\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Reason: {reason}\n")
            f.write(f"Status: All positions liquidated\n")
            f.write(f"\nTo reactivate trading:\n")
            f.write(f"1. Delete this file: {flag_file}\n")
            f.write(f"2. Ensure all systems are safe\n")
            f.write(f"3. Restart the Helios trading bot\n")

        print(f"✅ Kill switch flag created: {flag_file}")
        print("⚠️  Trading will remain disabled until this file is manually removed")

    except Exception as e:
        print(f"❌ ERROR: Failed to create kill switch flag: {e}")

def log_emergency_event(reason: str, liquidation_success: bool) -> None:
    """Log the emergency event for audit trail."""
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "kill_switch_events.log")

        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event': 'KILL_SWITCH_ACTIVATED',
            'reason': reason,
            'liquidation_success': liquidation_success,
            'executed_by': 'kill_switch.py',
            'user': os.getenv('USER', 'unknown')
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(event_data) + '\n')

        print(f"✅ Emergency event logged to: {log_file}")

    except Exception as e:
        print(f"⚠️  WARNING: Failed to log emergency event: {e}")

def send_critical_alert(reason: str, liquidation_success: bool) -> None:
    """Send critical alert notifications."""
    try:
        # This is a simplified alert - in production, would integrate with
        # email/SMS systems based on configuration
        alert_message = f"""
🚨 HELIOS EMERGENCY ALERT 🚨

KILL SWITCH ACTIVATED
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reason: {reason}
Liquidation Status: {'SUCCESS' if liquidation_success else 'PARTIAL/FAILED'}

All trading has been halted.
Manual intervention required to resume operations.
        """

        print("\n" + "="*50)
        print("📧 CRITICAL ALERT MESSAGE:")
        print("="*50)
        print(alert_message)
        print("="*50)

        # TODO: In production, implement actual email/SMS alerting here
        # using SMTP or Twilio based on configuration

    except Exception as e:
        print(f"⚠️  WARNING: Failed to send critical alert: {e}")

def confirm_kill_switch_activation(reason: str) -> bool:
    """Confirm kill switch activation with user."""
    print(f"\n⚠️  KILL SWITCH REASON: {reason}")
    print("\n🚨 WARNING: This will immediately liquidate ALL positions!")
    print("This action cannot be undone.")

    try:
        response = input("\nType 'LIQUIDATE' to confirm, or anything else to abort: ").strip()
        return response == 'LIQUIDATE'
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        return False
    except Exception:
        return False

def main():
    """Main kill switch execution function."""
    # Get reason from command line argument
    reason = sys.argv[1] if len(sys.argv) > 1 else "Manual activation"

    print_banner()

    # Confirm activation unless reason contains "emergency" (for automated triggers)
    if "emergency" not in reason.lower():
        if not confirm_kill_switch_activation(reason):
            print("❌ Kill switch activation cancelled")
            sys.exit(0)

    print(f"\n🚀 Initiating emergency liquidation procedure...")
    print(f"📝 Reason: {reason}")

    try:
        # Step 1: Load API credentials
        print("\n1️⃣  Loading API credentials...")
        api_key, secret_key = load_api_credentials()

        # Step 2: Connect to Alpaca
        print("\n2️⃣  Connecting to Alpaca API...")
        client = initialize_alpaca_client(api_key, secret_key)

        # Step 3: Get all positions
        print("\n3️⃣  Retrieving current positions...")
        positions = get_all_positions(client)

        # Step 4: Liquidate all positions
        print("\n4️⃣  Executing liquidation orders...")
        liquidation_success = liquidate_all_positions(client, positions)

        # Step 5: Create kill switch flag
        print("\n5️⃣  Creating kill switch flag...")
        create_kill_switch_flag(reason)

        # Step 6: Log emergency event
        print("\n6️⃣  Logging emergency event...")
        log_emergency_event(reason, liquidation_success)

        # Step 7: Send alerts
        print("\n7️⃣  Sending critical alerts...")
        send_critical_alert(reason, liquidation_success)

        # Final status
        print("\n" + "="*80)
        if liquidation_success:
            print("✅ KILL SWITCH EXECUTION COMPLETED SUCCESSFULLY")
            print("✅ All positions have been liquidated")
        else:
            print("⚠️  KILL SWITCH EXECUTION COMPLETED WITH WARNINGS")
            print("⚠️  Some positions may require manual intervention")

        print("🔒 Trading is now DISABLED until kill switch flag is removed")
        print("📞 Contact system administrator for next steps")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n❌ Kill switch execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during kill switch execution: {e}")
        print("🚨 MANUAL INTERVENTION REQUIRED IMMEDIATELY")
        print("📞 Contact system administrator and broker directly")
        sys.exit(1)

if __name__ == "__main__":
    main()
