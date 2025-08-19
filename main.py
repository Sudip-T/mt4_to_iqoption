"""
MT4 Signal Processing and IQ Option Trade Execution System

A comprehensive trading signal processing system that receives, validates, stores, and
executes trading signals from MetaTrader 4 (MT4) Expert Advisors. The system supports
both file-based and HTTP-based signal reception with a SQLite database for persistence,
execution pipelines and an optional GUI for monitoring.

Author: C2AwithSudip
Version: 2.0.0
License: MIT

Usage:
    python mt4_signal_processor.py

Configuration:
    - Database: SQLite database stored in 'signals.db'
    - File monitoring: Default MT4 common files directory
    - HTTP server: Default port 8080
    - File Receiver - Monitors MT4's common files directory for signals
    - HTTP Receiver - Flask-based API endpoint for signal reception
    - Signal Processor - Core business logic for handling signals
    - Monitoring GUI - Tkinter-based interface for real-time monitoring
    - Logging: File and console logging enabled
"""


import sys
import time
import signal
from gui import TradingSignalGUI
from utilities import get_logger
from processor import TradingSignalProcessor


logger = get_logger(__name__)


try:
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def signal_handler(signum, frame) -> None:
    """Handle OS signals for graceful shutdown."""
    logger.info(f"Received Signal {signum}, Shutting Down...")
    raise SystemExit(0)


def main() -> None:
    try:        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize processor
        processor = TradingSignalProcessor(
            db_path="trading_signals.db",
            http_port=8080
        )
        
        # Run with or without GUI
        if GUI_AVAILABLE and '--nogui' not in sys.argv:
            logger.info("Processor running in console mode. Press Ctrl+C to exit.")
            gui = TradingSignalGUI(processor)
            gui.run()
        else:
            with processor:
                logger.info("Processor running in console mode. Press Ctrl+C to exit.")
                while True:
                    time.sleep(1)
                    
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()