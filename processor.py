import time
import random
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from models import MT4Signal
from receivers.file_receiver import FileSignalReceiver
from receivers.http_receiver import HTTPSignalReceiver
from db_manager.dbmanager import DatabaseManager

import logging

logger = logging.getLogger(__name__)



class TradingSignalProcessor:
    """
    Advanced trading signal processor with comprehensive validation and execution capabilities.
    
    This is the main orchestration class that coordinates signal reception, validation,
    storage, and execution. It provides a complete trading signal processing pipeline
    with robust error handling and monitoring capabilities.
    
    Attributes:
        db: Database manager instance
        file_receiver: File-based signal receiver
        http_receiver: HTTP-based signal receiver
        running: Whether the processor is currently running
        executor: Thread pool for parallel signal processing
        
    Methods:
        start(): Start signal processing
        stop(): Stop signal processing
        process_signal(): Process incoming signals
        execute_trade(): Execute trading logic
        get_statistics(): Get processing statistics
    """
    
    def __init__(self, 
                 db_path: str = "signals.db",
                 file_path: Optional[str] = None,
                 http_port: int = 8080,
                 max_workers: int = 5):
        """
        Initialize the trading signal processor.
        
        Args:
            db_path: Path to SQLite database file
            file_path: Path to signal file (optional)
            http_port: HTTP server port
            max_workers: Maximum number of worker threads
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        # Initialize components
        self.db = DatabaseManager(db_path)
        self.file_receiver = FileSignalReceiver(file_path)
        self.http_receiver = HTTPSignalReceiver(http_port)
        
        # Set up signal callbacks
        self.file_receiver.set_callback(self.process_signal)
        self.http_receiver.set_callback(self.process_signal)
        
        # Threading and execution
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self._shutdown_event = threading.Event()
        
        # Statistics and monitoring
        self.stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'processed_signals': 0,
            'failed_signals': 0,
            'start_time': None
        }
        self._stats_lock = threading.Lock()
        
        # Risk management
        self.risk_limits = {
            'max_lot_size': 10.0,
            'max_daily_trades': 50,
            'max_concurrent_trades': 10,
            'min_account_balance': 1000.0
        }
        
        logger.info("Trading signal processor initialized")
    
    def process_signal(self, signal: MT4Signal) -> None:
        """
        Process an incoming trading signal asynchronously.
        
        This method handles the complete signal processing pipeline including
        validation, risk management, storage, and execution.
        
        Args:
            signal: MT4Signal instance to process
            
        Raises:
            None: All exceptions are caught and logged
        """
        try:
            logger.info(f"Processing signal: {signal.signal_type} {signal.symbol} at {signal.price}")
            
            # Submit to thread pool for async processing
            future = self.executor.submit(self._process_signal_sync, signal)
            
            # Don't wait for completion to avoid blocking
            # Results will be handled in the async method
            
        except Exception as e:
            logger.error(f"Error submitting signal for processing: {e}")
            self._update_stats('failed_signals', 1)
    
    def _process_signal_sync(self, signal: MT4Signal) -> None:
        """
        Synchronous signal processing method run in thread pool.
        
        Args:
            signal: MT4Signal instance to process
        """
        try:
            # Validate signal
            validation_result = signal.validate()
            if not validation_result.is_valid:
                logger.warning(f"Signal validation failed: {validation_result.error_message}")
                self._update_stats('failed_signals', 1)
                return
            
            # Check risk limits
            if not self._check_risk_limits(signal):
                logger.warning(f"Signal rejected due to risk limits: {signal.symbol}")
                self._update_stats('failed_signals', 1)
                return
            
            # Save to database
            signal_id = self.db.save_signal(signal)
            
            # Update statistics
            with self._stats_lock:
                self.stats['total_signals'] += 1
                if signal.signal_type == 'BUY':
                    self.stats['buy_signals'] += 1
                elif signal.signal_type == 'SELL':
                    self.stats['sell_signals'] += 1
            
            # Execute trade
            success = self.execute_trade(signal)
            
            if success:
                # Mark as processed
                self.db.mark_processed(signal_id)
                self._update_stats('processed_signals', 1)
                logger.info(f"Signal processed successfully: {signal.symbol}")
            else:
                self._update_stats('failed_signals', 1)
                logger.warning(f"Signal execution failed: {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Error in synchronous signal processing: {e}")
            self._update_stats('failed_signals', 1)
    
    def _check_risk_limits(self, signal: MT4Signal) -> bool:
        """
        Check if signal passes risk management limits.
        
        Args:
            signal: MT4Signal to validate
            
        Returns:
            bool: True if signal passes risk checks
        """
        try:
            # Check lot size limit
            if signal.lot_size > self.risk_limits['max_lot_size']:
                logger.warning(f"Signal exceeds max lot size: {signal.lot_size}")
                return False
            
            # Check daily trade limit
            today = datetime.now().date()
            today_signals = self.db.get_signals(
                start_date=datetime.combine(today, datetime.min.time()),
                end_date=datetime.combine(today, datetime.max.time())
            )
            
            if len(today_signals) >= self.risk_limits['max_daily_trades']:
                logger.warning("Daily trade limit exceeded")
                return False
            
            # Check concurrent trades (unprocessed signals)
            unprocessed_signals = self.db.get_signals(processed=False)
            if len(unprocessed_signals) >= self.risk_limits['max_concurrent_trades']:
                logger.warning("Concurrent trade limit exceeded")
                return False
            
            # Additional risk checks can be added here
            # - Account balance validation
            # - Symbol-specific limits
            # - Time-based restrictions
            # - Correlation checks
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def execute_trade(self, signal: MT4Signal) -> bool:
        """
        Execute trading logic based on the signal.
        
        This method contains the core trading logic that should be customized
        based on your specific trading strategy and broker integration.
        
        Args:
            signal: MT4Signal to execute
            
        Returns:
            bool: True if trade was executed successfully
            
        Raises:
            None: All exceptions are caught and logged
        """
        try:
            logger.info(f"Executing {signal.signal_type} trade:")
            logger.info(f"  {signal.symbol}, Price: {signal.price}, Reason: {signal.reason}")
            # logger.info(f"  Price: {signal.price}")
            # logger.info(f"  Lot Size: {signal.lot_size}")
            # logger.info(f"  Stop Loss: {signal.stop_loss}")
            # logger.info(f"  Take Profit: {signal.take_profit}")
            # logger.info(f"  Reason: {signal.reason}")
            
            # Pre-execution validation
            if not self._validate_market_conditions(signal):
                logger.warning("Market conditions not suitable for trading")
                return False
            
            # Execute based on signal type
            if signal.signal_type == 'BUY':
                return self._execute_buy_order(signal)
            elif signal.signal_type == 'SELL':
                return self._execute_sell_order(signal)
            elif signal.signal_type == 'CLOSE':
                return self._execute_close_order(signal)
            else:
                logger.error(f"Unknown signal type: {signal.signal_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _validate_market_conditions(self, signal: MT4Signal) -> bool:
        """
        Validate market conditions before trade execution.
        
        Args:
            signal: MT4Signal to validate
            
        Returns:
            bool: True if market conditions are suitable
        """
        try:
            # Check if market is open (simplified check)
            current_time = datetime.now()
            weekday = current_time.weekday()
            
            # Skip weekend (Saturday = 5, Sunday = 6)
            if weekday >= 5:
                logger.info("Market is closed (weekend)")
                return False
            
            # Check trading hours (simplified - assumes 24/5 forex market)
            # In reality, you'd want more sophisticated market hours checking
            hour = current_time.hour
            if hour < 6 or hour > 22:  # Avoid low liquidity hours
                logger.info("Outside recommended trading hours")
                return False
            
            # Additional market condition checks can be added here:
            # - Economic news events
            # - Volatility levels
            # - Spread conditions
            # - Liquidity checks
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating market conditions: {e}")
            return False
    
    def _execute_buy_order(self, signal: MT4Signal) -> bool:
        """
        Execute a buy order based on the signal.
        
        Args:
            signal: MT4Signal containing buy order details
            
        Returns:
            bool: True if order was executed successfully
        """
        try:
            # logger.info(f"Executing BUY order for {signal.symbol}")
            
            # Here you would integrate with your broker's API
            # Example integration points:
            # - MetaTrader 5 Python API
            # - FIX protocol connection
            # - Broker-specific REST API
            # - Trading platform API
            
            # Placeholder implementation
            order_details = {
                'symbol': signal.symbol,
                'action': 'BUY',
                'quantity': signal.lot_size,
                'price': signal.price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'magic_number': signal.magic_number
            }
            
            # Simulate order execution
            success = self._simulate_order_execution(order_details)
            
            if success:
                logger.info(f"BUY order executed successfully for {signal.symbol}")
            else:
                logger.error(f"BUY order execution failed for {signal.symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing BUY order: {e}")
            return False
    
    def _execute_sell_order(self, signal: MT4Signal) -> bool:
        """
        Execute a sell order based on the signal.
        
        Args:
            signal: MT4Signal containing sell order details
            
        Returns:
            bool: True if order was executed successfully
        """
        try:
            # logger.info(f"Executing SELL order for {signal.symbol}")
            
            # Similar to buy order, integrate with broker API
            order_details = {
                'symbol': signal.symbol,
                'action': 'SELL',
                'quantity': signal.lot_size,
                'price': signal.price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'magic_number': signal.magic_number
            }
            
            # Simulate order execution
            success = self._simulate_order_execution(order_details)
            
            if success:
                logger.info(f"SELL order executed successfully for {signal.symbol}")
            else:
                logger.error(f"SELL order execution failed for {signal.symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing SELL order: {e}")
            return False
    
    def _execute_close_order(self, signal: MT4Signal) -> bool:
        """
        Execute a close order based on the signal.
        
        Args:
            signal: MT4Signal containing close order details
            
        Returns:
            bool: True if order was closed successfully
        """
        try:
            logger.info(f"Executing CLOSE order for {signal.symbol}")
            
            # Close order logic would depend on your position management system
            # You might need to:
            # - Find existing positions for the symbol
            # - Close specific positions by magic number
            # - Close all positions for the symbol
            
            # Placeholder implementation
            close_details = {
                'symbol': signal.symbol,
                'action': 'CLOSE',
                'magic_number': signal.magic_number
            }
            
            success = self._simulate_order_execution(close_details)
            
            if success:
                logger.info(f"CLOSE order executed successfully for {signal.symbol}")
            else:
                logger.error(f"CLOSE order execution failed for {signal.symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing CLOSE order: {e}")
            return False
    
    def _simulate_order_execution(self, order_details: Dict[str, Any]) -> bool:
        """
        Simulate order execution for testing purposes.
        
        In a real implementation, this would be replaced with actual
        broker API calls.
        
        Args:
            order_details: Dictionary containing order parameters
            
        Returns:
            bool: True if simulation succeeds
        """
        try:
            # Simulate network latency
            time.sleep(0.1)
            
            # Randomly fail 5% of the time to simulate real-world conditions
            if random.random() < 0.05:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in order simulation: {e}")
            return False

    def _update_stats(self, key: str, value: int = 1) -> None:
        """
        Update statistics in a thread-safe manner.
        
        Args:
            key: Statistic key to update
            value: Value to add (default 1)
        """
        with self._stats_lock:
            if key in self.stats:
                self.stats[key] += value
            else:
                self.stats[key] = value

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        with self._stats_lock:
            stats = self.stats.copy()
            
            # Add database statistics
            db_stats = self.db.get_statistics()
            stats.update(db_stats)
            
            # Calculate uptime
            if self.stats['start_time']:
                stats['uptime'] = str(datetime.now() - self.stats['start_time'])
            else:
                stats['uptime'] = "0:00:00"
                
            return stats

    def start(self) -> None:
        """
        Start all signal processing components.
        """
        if self.running:
            logger.warning("Processor is already running")
            return
            
        self.running = True
        self._shutdown_event.clear()
        self.stats['start_time'] = datetime.now()
        
        # Start receivers
        self.file_receiver.start()
        self.http_receiver.start()
        
        # Start background tasks
        threading.Thread(target=self._monitor_processor, daemon=True).start()
        
        logger.info("Trading signal processor started")

    def stop(self) -> None:
        """
        Stop all signal processing components gracefully.
        """
        if not self.running:
            return
            
        self.running = False
        self._shutdown_event.set()
        
        # Stop receivers
        self.file_receiver.stop()
        self.http_receiver.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Trading signal processor stopped")

    def _monitor_processor(self) -> None:
        """
        Background monitoring thread for processor health.
        """
        logger.info("Processor monitoring started")
        
        while self.running:
            try:
                # Check database connection
                try:
                    self.db.get_statistics()
                except Exception as e:
                    logger.error(f"Database connection check failed: {e}")
                
                # Log statistics periodically
                stats = self.get_statistics()
                logger.info(f"Processor stats - Total: {stats['total_signals']}, "
                          f"Processed: {stats['processed_signals']}, "
                          f"Failed: {stats['failed_signals']}")
                
                # Cleanup old signals
                try:
                    self.db.cleanup_old_signals(days_old=7)
                except Exception as e:
                    logger.error(f"Failed to cleanup old signals: {e}")
                
                # Sleep until next check
                self._shutdown_event.wait(timeout=300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(60)

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop()


    
    # def _export_to_csv(self) -> None:
    #     """Export signals to CSV file."""
    #     try:
    #         filename = tk.filedialog.asksaveasfilename(
    #             defaultextension=".csv",
    #             filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    #             title="Export Signals"
    #         )
            
    #         if filename:
    #             signals = self.processor.get_recent_signals(1000)  # Get up to 1000 signals
    #             df = pd.DataFrame(signals)
    #             df.to_csv(filename, index=False)
    #             self._log_message(f"Exported {len(df)} signals to {filename}")
    #     except Exception as e:
    #         self._log_message(f"Export failed: {e}")