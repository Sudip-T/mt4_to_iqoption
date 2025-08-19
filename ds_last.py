# Add this to your imports at the top of the first script
from enum import Enum, auto
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import asyncio
import time
from datetime import datetime, timedelta


# Add these enums and dataclasses (if not already present)
class TradeSignal(Enum):
    BUY = 'BUY'
    SELL = 'SELL'
    CLOSE = 'CLOSE'
    HOLD = 'HOLD'

class TradeOutcome(Enum):
    WIN = 'WIN'
    LOSS = 'LOSS'
    PENDING = 'PENDING'
    CANCELLED = 'CANCELLED'

@dataclass
class TradingConfig:
    """Configuration for the trading bot"""
    initial_balance: float = 200.00
    stop_loss_amount: float = 10.00
    daily_target: float = 100.00
    martingale_levels: int = 7
    payout_ratio: float = 0.85
    max_trades_per_day: int = 100
    trade_expiry: int = 1
    min_bet_amount: float = 1.0
    max_bet_amount: float = 100.0
    enable_notifications: bool = False

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    signal: TradeSignal
    amount: float
    price: float
    filled_at: datetime = None
    status: TradeOutcome = TradeOutcome.PENDING
    pnl: float = 0.0
    martingale_step: int = 0
    strategy: str = None

    def __post_init__(self):
        if self.filled_at is None:
            self.filled_at = datetime.now()
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['filled_at'] = self.filled_at.isoformat()
        data['symbol'] = self.symbol
        data['signal'] = self.signal.value
        data['status'] = self.status.value
        data['strategy'] = self.strategy
        return data

class AutoTradingBot:
    """Integrated trading bot that works with TradingSignalProcessor"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.current_balance = config.initial_balance
        self.running = False
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Statistics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0
        }
        
        logger.info("AutoTradingBot initialized")
    
    def start(self):
        """Start the trading bot"""
        if self.running:
            logger.warning("Bot is already running")
            return
            
        self.running = True
        # Start the event loop in a separate thread
        self.executor.submit(self._run_event_loop)
        logger.info("AutoTradingBot started")
    
    def _run_event_loop(self):
        """Run the asyncio event loop"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self.loop.close()
    
    def stop(self):
        """Stop the trading bot"""
        if not self.running:
            return
            
        self.running = False
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.executor.shutdown(wait=True)
        logger.info("AutoTradingBot stopped")
    
    async def execute_trade(self, signal: MT4Signal) -> bool:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: MT4Signal to execute
            
        Returns:
            bool: True if trade was executed successfully
        """
        try:
            # Convert MT4Signal to our Order format
            order = Order(
                order_id=f"ORDER_{int(time.time() * 1000)}",
                symbol=signal.symbol,
                signal=TradeSignal[signal.signal_type],
                amount=signal.lot_size,
                price=signal.price,
                strategy=signal.reason
            )
            
            logger.info(f"Executing trade: {order}")
            
            # Here you would implement actual trade execution logic
            # For now we'll simulate it
            success = await self._simulate_order_execution(order)
            
            if success:
                # Update metrics
                self.metrics['total_trades'] += 1
                if order.pnl > 0:
                    self.metrics['winning_trades'] += 1
                else:
                    self.metrics['losing_trades'] += 1
                self.metrics['total_pnl'] += order.pnl
                self.metrics['daily_pnl'] += order.pnl
                
                logger.info(f"Trade executed successfully: {order.order_id}")
            else:
                logger.warning(f"Trade execution failed: {order.order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def _simulate_order_execution(self, order: Order) -> bool:
        """
        Simulate order execution for testing.
        In a real implementation, this would connect to your broker API.
        """
        try:
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Randomly determine success (90% success rate in simulation)
            success = random.random() < 0.9
            
            if success:
                # Simulate P&L - random between -5% and +10% of amount
                order.pnl = order.amount * random.uniform(-0.05, 0.10)
                order.status = TradeOutcome.WIN if order.pnl > 0 else TradeOutcome.LOSS
            else:
                order.pnl = -order.amount  # Lose the entire amount on failure
                order.status = TradeOutcome.LOSS
            
            return success
            
        except Exception as e:
            logger.error(f"Order simulation error: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current trading metrics"""
        return self.metrics.copy()


# Now modify the TradingSignalProcessor class to include the AutoTradingBot
class TradingSignalProcessor:
    """
    Modified TradingSignalProcessor with integrated AutoTradingBot
    """
    
    def __init__(self, 
                 db_path: str = "signals.db",
                 file_path: Optional[str] = None,
                 http_port: int = 8080,
                 max_workers: int = 5,
                 trading_config: Optional[Dict] = None):
        """
        Initialize with trading bot configuration
        """
        # ... existing initialization code ...
        
        # Initialize trading bot
        trading_config = trading_config or {}
        self.trading_bot = AutoTradingBot(TradingConfig(**trading_config))
        
    def start(self) -> None:
        """
        Start all components including trading bot
        """
        if self.running:
            logger.warning("Processor is already running")
            return
            
        # Start signal processing components
        self.running = True
        self._shutdown_event.clear()
        self.stats['start_time'] = datetime.now()
        
        # Start receivers
        self.file_receiver.start()
        self.http_receiver.start()
        
        # Start trading bot
        self.trading_bot.start()
        
        # Start background tasks
        threading.Thread(target=self._monitor_processor, daemon=True).start()
        
        logger.info("Trading signal processor and bot started")

    def stop(self) -> None:
        """
        Stop all components including trading bot
        """
        if not self.running:
            return
            
        self.running = False
        self._shutdown_event.set()
        
        # Stop receivers
        self.file_receiver.stop()
        self.http_receiver.stop()
        
        # Stop trading bot
        self.trading_bot.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Trading signal processor and bot stopped")

    def _process_signal_sync(self, signal: MT4Signal) -> None:
        """
        Modified to include trading bot execution
        """
        try:
            # ... existing validation and risk checks ...
            
            # Save to database
            signal_id = self.db.save_signal(signal)
            
            # Update statistics
            with self._stats_lock:
                self.stats['total_signals'] += 1
                if signal.signal_type == 'BUY':
                    self.stats['buy_signals'] += 1
                elif signal.signal_type == 'SELL':
                    self.stats['sell_signals'] += 1
            
            # Execute trade through trading bot
            future = asyncio.run_coroutine_threadsafe(
                self.trading_bot.execute_trade(signal),
                self.trading_bot.loop
            )
            
            try:
                success = future.result(timeout=30)  # Wait up to 30 seconds
                if success:
                    self.db.mark_processed(signal_id)
                    self._update_stats('processed_signals', 1)
                    logger.info(f"Signal processed successfully: {signal.symbol}")
                else:
                    self._update_stats('failed_signals', 1)
                    logger.warning(f"Signal execution failed: {signal.symbol}")
            except TimeoutError:
                logger.error(f"Trade execution timed out for signal: {signal.symbol}")
                self._update_stats('failed_signals', 1)
                
        except Exception as e:
            logger.error(f"Error in synchronous signal processing: {e}")
            self._update_stats('failed_signals', 1)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get combined statistics from processor and trading bot
        """
        with self._stats_lock:
            stats = self.stats.copy()
            
            # Add database statistics
            db_stats = self.db.get_statistics()
            stats.update(db_stats)
            
            # Add trading bot metrics
            bot_stats = self.trading_bot.get_metrics()
            stats.update({
                'bot_total_trades': bot_stats['total_trades'],
                'bot_winning_trades': bot_stats['winning_trades'],
                'bot_losing_trades': bot_stats['losing_trades'],
                'bot_total_pnl': bot_stats['total_pnl'],
                'bot_daily_pnl': bot_stats['daily_pnl']
            })
            
            # Calculate uptime
            if self.stats['start_time']:
                stats['uptime'] = str(datetime.now() - self.stats['start_time'])
            else:
                stats['uptime'] = "0:00:00"
                
            return stats
        



# Example usage
if __name__ == "__main__":
    # Configure trading bot
    trading_config = {
        'initial_balance': 1000.0,
        'stop_loss_amount': 100.0,
        'daily_target': 500.0,
        'max_trades_per_day': 50
    }
    
    # Create processor with trading bot
    processor = TradingSignalProcessor(
        db_path="trading_signals.db",
        http_port=8080,
        max_workers=5,
        trading_config=trading_config
    )
    
    try:
        # Start processing
        processor.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        processor.stop()