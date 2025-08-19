import os
import csv
import json
import time
import random
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
from queue import Queue
import signal
import sys
from enum import Enum, auto
from utilities import setup_logging


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ===========================================================================



class AutoNameEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name
    
class BotState(Enum):
    IDLE = "IDLE"
    TRADING = "TRADING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

class TradeSignal(Enum):
    BUY = 'call'
    SELL = 'put'
    HOLD = 'hold'

class TradeOutcome(AutoNameEnum):
    WIN = auto()
    LOSS = auto()
    PENDING = auto()
    CANCELLED = auto()

class OrderStatus(AutoNameEnum):
    WIN = auto()
    LOSS = auto()
    PENDING = auto()
    CANCELLED = auto()

class StopReason(AutoNameEnum):
    MANUAL_STOP = auto()
    SYSTEM_ERROR = auto()
    STOP_LOSS_HIT = auto()
    INSUFFICIENT_FUNDS = auto()
    DAILY_TARGET_REACHED = auto()

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
    enable_notifications:bool=False

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    signal: TradeSignal
    betamount: float
    filled_at: datetime = None
    status: TradeOutcome = TradeOutcome.PENDING
    pnl: float = 0.0
    martingale_step: int = 0
    strategy:str = None

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    signal: TradeSignal
    betamount: float
    filled_at: datetime = None
    status: TradeOutcome = TradeOutcome.PENDING
    pnl: float = 0.0
    martingale_step: int = 0
    strategy:str = None

    def __post_init__(self):
        if self.filled_at is None:
            self.filled_at = datetime.now()
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['filled_at'] = self.filled_at.isoformat()
        data['symbol'] = self.symbol
        data['order_type'] = self.signal.value
        data['status'] = self.status.value
        data['strategy'] = self.strategy
        return data

@dataclass
class TradingSessionStats:
    start_time: datetime = datetime.now()
    end_time: Optional[datetime] = 'Open'
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_consecutive_wins: int = 0
    current_consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    stop_reason: Optional[StopReason] = None
    
    def update(self, order: Order):
        """Update metrics with new order"""
        self.total_trades += 1
        self.total_pnl += order.pnl
        self.daily_pnl += order.pnl
        
        if order.pnl > 0:
            self.winning_trades += 1
            self.current_consecutive_wins += 1
            self.current_consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.current_consecutive_wins)
        else:
            self.losing_trades += 1
            self.current_consecutive_losses += 1
            self.current_consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.current_consecutive_losses)
        
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        if self.winning_trades > 0:
            self.avg_win = sum(o.pnl for o in self.get_winning_orders()) / self.winning_trades
        if self.losing_trades > 0:
            self.avg_loss = sum(o.pnl for o in self.get_losing_orders()) / self.losing_trades
    
    def get_winning_orders(self) -> List[Order]:
        # In a real implementation, this would return actual winning orders
        return []
    
    def get_losing_orders(self) -> List[Order]:
        return []
    



# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class AutoTradingBot:
    """Main trading bot class"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.state = BotState.IDLE
        # self.broker = SimulatedBroker(config)
        # self.risk_manager = RiskManager(config)
        # self.martingale = MartingaleSystem(config)
        # self.signal_generator = SignalGenerator(config)
        # self.logger = TradingLogger(config)
        # self.notifications = NotificationSystem(config)
        self.metrics = TradingSessionStats()
        
        # Trading state
        self.is_running = False
        self.current_balance = config.initial_balance
        self.trade_queue = Queue()
        self.stop_reason = None

        self.logger = setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal")
        self.stop_trading("SIGNAL_RECEIVED")
    
    async def initialize(self) -> bool:
        """Initialize bot components"""
        try:
            self.logger.info("Initializing trading bot...")
            
            # Connect to broker
            connected = await self.broker.connect()
            if not connected:
                self.logger.error("Failed to connect to broker")
                return False
            
            # Update balance
            self.current_balance = await self.broker.get_balance()
            
            # Reset daily limits
            self.risk_manager.reset_daily_limits()
            
            self.logger.info(f"Bot initialized successfully. Balance: ${self.current_balance:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}")
            return False
    
    async def start_trading(self):
        """Start trading session"""
        if not await self.initialize():
            return
        
        self.is_running = True
        self.state = BotState.TRADING
        
        self.logger.info("Starting trading session...")
        self.logger.info(f"Configuration: {self.config}")
        
        try:
            while self.is_running and self.state == BotState.TRADING:
                # Check risk limits
                can_trade, reason = self.risk_manager.check_daily_limits()
                if not can_trade:
                    self.stop_trading(reason)
                    break
                
                # Check maximum trades
                if self.metrics.total_trades >= self.config.max_trades_per_day:
                    self.stop_trading("MAX_TRADES_REACHED")
                    break
                
                # Execute trading cycle
                await self._trading_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Trading session error: {e}")
            self.state = BotState.ERROR
        finally:
            await self.shutdown()
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Get market data
            market_data = await self.broker.get_market_data(self.config.symbol)
            
            # Generate signal
            signal = await self.signal_generator.generate_signal(market_data)
            signal_strength = self.signal_generator.get_signal_strength()
            
            # Check if we should trade
            if signal in [TradeSignal.BUY, TradeSignal.SELL] and signal_strength > 0.7:
                await self._execute_trade(signal, signal_strength, market_data)
            
        except Exception as e:
            self.logger.error(f"Trading cycle error: {e}")
    
    async def _execute_trade(self, signal: TradeSignal, signal_strength: float, market_data: MarketData):
        """Execute a trade"""
        try:
            # Calculate bet amount
            bet_amount = self.martingale.get_next_amount()
            
            # Risk check
            if bet_amount > self.current_balance:
                if self.current_balance >= self.config.min_bet_amount:
                    bet_amount = self.current_balance
                else:
                    self.stop_trading("INSUFFICIENT_FUNDS")
                    return
            
            # Create order
            order = Order(
                order_id=f"ORDER_{int(time.time() * 1000)}",
                symbol=self.config.symbol,
                order_type=TradeSignal.BUY.value if signal == TradeSignal.BUY else TradeSignal.SELL.value,
                amount=bet_amount,
                price=market_data.close,
                martingale_step=self.martingale.current_step
            )
            
            # Place order
            success = await self.broker.place_order(order)
            if not success:
                self.logger.warning(f"Order placement failed: {order.order_id}")
                return
            
            # Update balance
            self.current_balance = await self.broker.get_balance()
            
            # Update martingale system
            if order.pnl > 0:
                self.martingale.reset()
            else:
                self.martingale.on_loss()
            
            # Update metrics
            self.metrics.update(order)
            
            # Update risk manager
            self.risk_manager.update_pnl(order.pnl)
            
            # Log trade
            self.logger.log_trade(order, self.current_balance, self.metrics, signal, signal_strength)
            
            # Send notification
            await self.notifications.notify_trade_executed(order)
            
            # Print trade summary
            outcome = "WIN" if order.pnl > 0 else "LOSS"
            self.logger.info(
                f"Trade {outcome}: {order.order_type.value} ${order.amount:.2f} "
                f"-> P&L: ${order.pnl:.2f} | Balance: ${self.current_balance:.2f} "
                f"| Step: {self.martingale.current_step + 1}/{self.config.martingale_levels}"
            )
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
    
    def stop_trading(self, reason: str = "USER_STOPPED"):
        """Stop trading session"""
        self.is_running = False
        self.state = BotState.STOPPED
        self.stop_reason = reason
        self.logger.info(f"Trading stopped: {reason}")
    
    async def shutdown(self):
        """Shutdown bot"""
        self.logger.info("Shutting down trading bot...")
        
        # Disconnect from broker
        await self.broker.disconnect()
        
        # Print final summary
        self._print_summary()
        
        self.logger.info("Bot shutdown complete")
    
    def _print_summary(self):
        """Print trading session summary"""
        print("\n" + "="*80)
        print("TRADING SESSION SUMMARY")
        print("="*80)
        print(f"Stop Reason: {self.stop_reason}")
        print(f"Initial Balance: ${self.config.initial_balance:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total P&L: ${self.metrics.total_pnl:.2f}")
        print(f"Daily P&L: ${self.metrics.daily_pnl:.2f}")
        print(f"Total Trades: {self.metrics.total_trades}")
        print(f"Winning Trades: {self.metrics.winning_trades}")
        print(f"Losing Trades: {self.metrics.losing_trades}")
        print(f"Win Rate: {self.metrics.win_rate:.1f}%")
        print(f"Max Consecutive Losses: {self.metrics.max_consecutive_losses}")
        print(f"Max Consecutive Wins: {self.metrics.max_consecutive_wins}")
        print(f"CSV Log: {self.logger.csv_filename}")
        print("="*80)
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'state': self.state.value,
            'balance': self.current_balance,
            'metrics': asdict(self.metrics),
            'martingale_step': self.martingale.current_step + 1,
            'stop_reason': self.stop_reason,
            'is_running': self.is_running
        }




# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function"""

    # Configuration
    config = TradingConfig(
        initial_balance=1000.0,
        stop_loss_amount=300.0,
        daily_target=500.0,
        martingale_levels=7,
        payout_ratio=0.85,
        max_trades_per_day=50,
        min_bet_amount=1.0,
        max_bet_amount=100.0,
        enable_notifications=True,
    )
    
    # Create and start bot
    bot = AutoTradingBot(config)
    await bot.start_trading()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)