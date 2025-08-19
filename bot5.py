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
    start_time: datetime
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
# RISK MANAGEMENT
# ============================================================================

class RiskManager:
    """Risk management system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.max_concurrent_trades = 1
        self.current_trades = 0
        self.last_reset_date = datetime.now().date()
    
    def check_daily_limits(self) -> Tuple[bool, str]:
        """Check if daily limits are exceeded"""
        if abs(self.daily_loss) >= self.config.stop_loss_amount:
            return False, "Daily stop loss exceeded"
        
        if self.daily_profit >= self.config.daily_target:
            return False, "Daily target reached"
        
        return True, "OK"
    
    def calculate_position_size(self, balance: float, risk_percentage: float = None) -> float:
        """Calculate position size based on risk"""
        risk_pct = risk_percentage or self.config.risk_percentage
        position_size = balance * risk_pct
        return min(max(position_size, self.config.min_bet_amount), self.config.max_bet_amount)
    
    def reset_daily_limits(self):
        """Reset daily limits"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.last_reset_date = current_date
    
    def update_pnl(self, pnl: float):
        """Update P&L tracking"""
        if pnl > 0:
            self.daily_profit += pnl
        else:
            self.daily_loss += abs(pnl)


# ============================================================================
# MARTINGALE SYSTEM
# ============================================================================

class MartingaleSystem:
    """Advanced Martingale system with risk controls"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.current_step = 0
        self.sequence = []
        self.consecutive_losses = 0
        self.base_amount = 0.0
        self.total_investment = 0.0
        self.reset()
    
    def reset(self):
        """Reset martingale system"""
        self.current_step = 0
        self.consecutive_losses = 0
        self.total_investment = 0.0
        self.sequence = self._calculate_sequence()
    
    def _calculate_sequence(self) -> List[float]:
        """Calculate optimal martingale sequence"""
        return self.get_martingale_sequence(
            capital=self.config.initial_balance,
            levels=self.config.martingale_levels,
            payout=self.config.payout_ratio
        )
    
    def get_next_amount(self) -> float:
        """Get next bet amount in sequence"""
        if self.current_step >= len(self.sequence):
            return self.sequence[-1]  # Use last amount if exceeded
        
        return self.sequence[self.current_step]
    
    def on_loss(self):
        """Handle losing trade"""
        self.current_step += 1
        self.consecutive_losses += 1
        self.total_investment += self.get_next_amount()
    
    def should_reset(self) -> bool:
        """Check if system should reset"""
        return self.current_step >= self.config.martingale_levels
    
    def get_martingale_sequence(self,
        capital: Optional[float] = None,
        levels: int = None, 
        payout: float = None, 
        precision: float = 0.01,
        target_profit: Optional[float] = None  # Optional profit per trade
    ) -> List[float]:
        
        capital = capital if capital is not None else self.config.initial_balance
        levels = levels if levels is not None else self.config.martingale_levels
        payout = payout if payout is not None else self.config.payout_ratio

        if capital <= 0:
            raise ValueError("Capital must be positive")
        if levels <= 0:
            raise ValueError("Levels must be positive")
        if not 0 < payout < 1:
            raise ValueError("Payout must be between 0 and 1")


        def simulate_martingale(initial_bet: float) -> Tuple[List[float], float]:
            bet_sequence = []
            cumulative_loss = 0.0
            for i in range(levels):
                # On each step, calculate the required bet
                if target_profit is not None:
                    required_return = cumulative_loss + target_profit
                    current_bet = required_return / payout
                else:
                    current_bet = cumulative_loss / payout if cumulative_loss > 0 else initial_bet

                current_bet = round(current_bet, 2)
                bet_sequence.append(current_bet)
                cumulative_loss += current_bet

            return bet_sequence, cumulative_loss

        # Binary search for best initial bet
        low = 0.01
        high = capital
        best_sequence = []

        while high - low > precision:
            mid = (low + high) / 2
            sequence, total = simulate_martingale(mid)
            if total <= capital:
                best_sequence = sequence
                low = mid
            else:
                high = mid

        return best_sequence


# ============================================================================
# BROKER INTERFACE
# ============================================================================

class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> bool:
        """Place order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> float:
        """Get account balance"""
        pass


class SimulatedBroker(BrokerInterface):
    """Simulated broker for testing"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.balance = config.initial_balance
        self.orders: Dict[str, Order] = {}
        self.connected = False
        # self.market_data = MarketData(
        #     timestamp=datetime.now(),
        #     open=1.1000,
        #     high=1.1050,
        #     low=1.0950,
        #     close=1.1025,
        #     volume=1000000
        # )
    
    async def connect(self) -> bool:
        """Simulate connection"""
        await asyncio.sleep(0.1)
        self.connected = True
        return True
    
    async def disconnect(self):
        """Simulate disconnection"""
        self.connected = False
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Simulate market data"""
        # Simulate price movement
        price_change = random.uniform(-0.001, 0.001)
        self.market_data.close += price_change
        self.market_data.timestamp = datetime.now()
        return self.market_data
    
    async def place_order(self, order: Order) -> bool:
        """Simulate order placement"""
        await asyncio.sleep(0.1)
        
        if self.balance < order.amount:
            order.status = OrderStatus.REJECTED
            return False
        
        self.balance -= order.amount
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        self.orders[order.order_id] = order
        
        # Simulate trade outcome
        win_probability = 0.47  # Slightly less than 50% to simulate real conditions
        if random.random() < win_probability:
            order.pnl = order.amount * self.config.payout_ratio
        else:
            order.pnl = -order.amount
        
        self.balance += order.amount + order.pnl
        return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED
    
    async def get_balance(self) -> float:
        """Get current balance"""
        return self.balance


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Trading signal generator"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.last_signal = TradeSignal.HOLD
        self.signal_history = []
    
    async def generate_signal(self, market_data: MarketData) -> TradeSignal:
        """Generate trading signal based on market data"""
        # Simple random signal generator for simulation
        # In real implementation, this would use technical indicators
        
        signals = [TradeSignal.BUY, TradeSignal.SELL]
        weights = [0.48, 0.52]  # Slightly favor sell to simulate market conditions
        
        signal = random.choices(signals, weights=weights)[0]
        self.last_signal = signal
        self.signal_history.append((datetime.now(), signal))
        
        return signal
    
    def get_signal_strength(self) -> float:
        """Get signal strength (0-1)"""
        return random.uniform(0.6, 0.9)


# ============================================================================
# LOGGER SYSTEM
# ============================================================================

class TradingLogger:
    """Advanced logging system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.setup_logging()
        self.csv_filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.setup_csv()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("TradingBot")
    
    def setup_csv(self):
        """Setup CSV logging"""
        headers = [
            'timestamp', 'order_id', 'symbol', 'order_type', 'amount', 'price',
            'status', 'pnl', 'balance', 'martingale_step', 'consecutive_losses',
            'win_rate', 'daily_pnl', 'signal', 'signal_strength'
        ]
        
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    def log_trade(self, order: Order, balance: float, metrics: TradingSessionStats, 
                  signal: TradeSignal, signal_strength: float):
        """Log trade to CSV"""
        trade_data = [
            datetime.now().isoformat(),
            order.order_id,
            order.symbol,
            order.order_type.value,
            order.amount,
            order.price or 0,
            order.status.value,
            order.pnl,
            balance,
            order.martingale_step,
            metrics.current_consecutive_losses,
            metrics.win_rate,
            metrics.daily_pnl,
            signal.value,
            signal_strength
        ]
        
        with open(self.csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(trade_data)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================

class NotificationSystem:
    """Notification system for trading events"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = config.enable_notifications
    
    async def send_notification(self, title: str, message: str, priority: str = "INFO"):
        """Send notification"""
        if not self.enabled:
            return
        
        # In real implementation, this would send email, SMS, or push notifications
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{priority}] {timestamp} - {title}: {message}")
    
    async def notify_trade_executed(self, order: Order):
        """Notify trade execution"""
        outcome = "WIN" if order.pnl > 0 else "LOSS"
        await self.send_notification(
            f"Trade {outcome}",
            f"Order {order.order_id}: {order.order_type.value} {order.amount} - P&L: {order.pnl:.2f}"
        )
    
    async def notify_daily_target_reached(self, profit: float):
        """Notify daily target reached"""
        await self.send_notification(
            "Daily Target Reached",
            f"Daily profit target reached: ${profit:.2f}",
            "SUCCESS"
        )
    
    async def notify_stop_loss_hit(self, loss: float):
        """Notify stop loss hit"""
        await self.send_notification(
            "Stop Loss Hit",
            f"Daily stop loss hit: ${loss:.2f}",
            "WARNING"
        )


# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class AutoTradingBot:
    """Main trading bot class"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.state = BotState.IDLE
        self.broker = SimulatedBroker(config)
        self.risk_manager = RiskManager(config)
        self.martingale = MartingaleSystem(config)
        self.signal_generator = SignalGenerator(config)
        self.logger = TradingLogger(config)
        self.notifications = NotificationSystem(config)
        self.metrics = TradingSessionStats()
        
        # Trading state
        self.is_running = False
        self.current_balance = config.initial_balance
        self.trade_queue = Queue()
        self.stop_reason = None
        
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