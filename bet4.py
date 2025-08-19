import os
import csv
import time
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum, auto
import abc

# ====================== Constants and Enums ======================

class TradeOutcome(Enum):
    WIN = auto()
    LOSS = auto()
    PENDING = auto()
    CANCELLED = auto()

class TradeSignal(Enum):
    BUY = auto()
    SELL = auto()
    NEUTRAL = auto()

class StopReason(Enum):
    DAILY_TARGET_REACHED = auto()
    STOP_LOSS_HIT = auto()
    INSUFFICIENT_FUNDS = auto()
    MAX_TRADES_REACHED = auto()
    MANUAL_STOP = auto()
    SYSTEM_ERROR = auto()

# ====================== Data Classes ======================

@dataclass
class Trade:
    id: str
    symbol: str
    signal: TradeSignal
    amount: float
    timestamp: datetime
    outcome: TradeOutcome = TradeOutcome.PENDING
    payout: float = 0.0
    closing_time: Optional[datetime] = None

@dataclass
class TradingSessionStats:
    start_time: datetime
    end_time: Optional[datetime] = None
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    stop_reason: Optional[StopReason] = None

# ====================== Interfaces ======================

class IExchangeClient(abc.ABC):
    @abc.abstractmethod
    def place_trade(self, symbol: str, amount: float, signal: TradeSignal) -> Trade:
        pass
    
    @abc.abstractmethod
    def check_trade_status(self, trade_id: str) -> TradeOutcome:
        pass
    
    @abc.abstractmethod
    def get_current_price(self, symbol: str) -> float:
        pass

class IRiskManager(abc.ABC):
    @abc.abstractmethod
    def assess_trade_risk(self, trade: Trade) -> bool:
        pass
    
    @abc.abstractmethod
    def should_stop_trading(self) -> Tuple[bool, Optional[StopReason]]:
        pass

class IStrategy(abc.ABC):
    @abc.abstractmethod
    def generate_signal(self) -> TradeSignal:
        pass
    
    @abc.abstractmethod
    def get_bet_amount(self) -> float:
        pass
    
    @abc.abstractmethod
    def update_strategy(self, last_trade: Trade):
        pass

# ====================== Implementations ======================

class MartingaleStrategy(IStrategy):
    def __init__(self, capital: float, levels: int = 7, payout: float = 0.85):
        self.initial_capital = capital
        self.current_capital = capital
        self.levels = levels
        self.payout = payout
        self.current_step = 0
        self.consecutive_losses = 0
        self.bet_sequence = self._calculate_martingale_sequence()
        
    def _calculate_martingale_sequence(self) -> List[float]:
        """Calculate optimal martingale sequence for current capital"""
        def simulate_sequence(initial_bet: float) -> Tuple[List[float], float]:
            sequence = []
            current_bet = initial_bet
            total_risk = 0
            
            for _ in range(self.levels):
                sequence.append(round(current_bet, 2))
                total_risk += current_bet
                current_bet = total_risk / self.payout
            
            return sequence, total_risk
        
        # Binary search to find optimal initial bet
        low, high = 0.01, self.current_capital
        best_sequence, best_total = [], 0
        precision = 0.01
        
        while high - low > precision:
            mid = (low + high) / 2
            sequence, total = simulate_sequence(mid)
            
            if total <= self.current_capital:
                best_sequence = sequence
                best_total = total
                low = mid
            else:
                high = mid
                
        return best_sequence
    
    def generate_signal(self) -> TradeSignal:
        """Simple random signal generation for demo purposes"""
        return random.choice([TradeSignal.BUY, TradeSignal.SELL])
    
    def get_bet_amount(self) -> float:
        """Get next bet amount based on martingale sequence"""
        if self.current_step >= len(self.bet_sequence):
            self.current_step = 0  # Reset sequence if we exceed levels
            
        amount = self.bet_sequence[self.current_step]
        return min(amount, self.current_capital)  # Don't bet more than we have
    
    def update_strategy(self, last_trade: Trade):
        """Update strategy state based on last trade outcome"""
        if last_trade.outcome == TradeOutcome.WIN:
            self.current_step = 0
            self.consecutive_losses = 0
            self.current_capital += last_trade.payout
            # Recalculate sequence with new capital
            self.bet_sequence = self._calculate_martingale_sequence()
        else:
            self.current_step += 1
            self.consecutive_losses += 1
            self.current_capital -= last_trade.amount

class DemoExchangeClient(IExchangeClient):
    def __init__(self, symbols: List[str], win_prob: float = 0.45):
        self.symbols = symbols
        self.win_prob = win_prob
        self.open_trades = {}
        self.prices = {symbol: random.uniform(100, 200) for symbol in symbols}
        
    def place_trade(self, symbol: str, amount: float, signal: TradeSignal) -> Trade:
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")
            
        trade_id = f"TRADE-{time.time_ns()}"
        trade = Trade(
            id=trade_id,
            symbol=symbol,
            signal=signal,
            amount=amount,
            timestamp=datetime.now()
        )
        
        # Simulate trade execution delay
        time.sleep(random.uniform(0.1, 0.5))
        
        # Store trade and simulate market movement
        self.open_trades[trade_id] = trade
        return trade
    
    def check_trade_status(self, trade_id: str) -> TradeOutcome:
        if trade_id not in self.open_trades:
            return TradeOutcome.CANCELLED
            
        trade = self.open_trades[trade_id]
        
        # Simulate trade outcome after some time
        if datetime.now() - trade.timestamp < timedelta(seconds=2):
            return TradeOutcome.PENDING
            
        # Determine outcome
        if random.random() < self.win_prob:
            trade.outcome = TradeOutcome.WIN
            trade.payout = trade.amount * (1 + random.uniform(0.7, 0.9))  # 70-90% payout
        else:
            trade.outcome = TradeOutcome.LOSS
            trade.payout = 0
            
        trade.closing_time = datetime.now()
        return trade.outcome
    
    def get_current_price(self, symbol: str) -> float:
        # Simulate price movement
        self.prices[symbol] *= random.uniform(0.99, 1.01)
        return round(self.prices[symbol], 2)

class BasicRiskManager(IRiskManager):
    def __init__(self, initial_balance: float, daily_stop_loss: float, daily_target: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_stop_loss = daily_stop_loss
        self.daily_target = daily_target
        self.start_time = datetime.now()
        self.max_trades = 1000  # Safety limit
        
    def assess_trade_risk(self, trade: Trade) -> bool:
        """Check if trade meets risk parameters"""
        if trade.amount > self.current_balance:
            return False
        if trade.amount > self.current_balance * 0.1:  # Don't risk more than 10% per trade
            return False
        return True
    
    def should_stop_trading(self) -> Tuple[bool, Optional[StopReason]]:
        """Check if we should stop trading"""
        daily_pnl = self.current_balance - self.initial_balance
        
        if daily_pnl >= self.daily_target:
            return True, StopReason.DAILY_TARGET_REACHED
        if daily_pnl <= -self.daily_stop_loss:
            return True, StopReason.STOP_LOSS_HIT
        if self.current_balance < 1:  # Minimum trade size
            return True, StopReason.INSUFFICIENT_FUNDS
        if (datetime.now() - self.start_time) > timedelta(hours=24):
            return True, StopReason.MAX_TRADES_REACHED
            
        return False, None
    
    def update_balance(self, amount: float):
        """Update current balance"""
        self.current_balance += amount

class TradingBot:
    def __init__(self, 
                 exchange: IExchangeClient,
                 strategy: IStrategy,
                 risk_manager: IRiskManager,
                 symbol: str = "BTCUSD"):
        self.exchange = exchange
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.stats = TradingSessionStats(
            start_time=datetime.now(),
            initial_balance=self.risk_manager.current_balance
        )
        self.logger = self._setup_logger()
        self.is_running = False
        
    def _setup_logger(self):
        """Configure logging"""
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # File handler
        log_filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger
    
    def _log_trade(self, trade: Trade):
        """Log trade details to CSV"""
        csv_filename = f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'trade_id', 'symbol', 'signal', 'amount',
                    'outcome', 'payout', 'balance_before', 'balance_after'
                ])
                
            writer.writerow([
                trade.timestamp.isoformat(),
                trade.id,
                trade.symbol,
                trade.signal.name,
                trade.amount,
                trade.outcome.name if trade.outcome else 'PENDING',
                trade.payout,
                self.risk_manager.current_balance,
                self.risk_manager.current_balance + (trade.payout if trade.outcome == TradeOutcome.WIN else -trade.amount)
            ])
    
    def _update_stats(self, trade: Trade):
        """Update session statistics"""
        self.stats.total_trades += 1
        
        if trade.outcome == TradeOutcome.WIN:
            self.stats.winning_trades += 1
            self.stats.max_consecutive_wins = max(
                self.stats.max_consecutive_wins,
                self.stats.winning_trades - self.stats.losing_trades
            )
        else:
            self.stats.losing_trades += 1
            self.stats.max_consecutive_losses = max(
                self.stats.max_consecutive_losses,
                self.stats.losing_trades - self.stats.winning_trades
            )
        
        self.risk_manager.update_balance(
            trade.payout if trade.outcome == TradeOutcome.WIN else -trade.amount
        )
    
    def _print_status(self):
        """Print current trading status"""
        win_rate = (self.stats.winning_trades / self.stats.total_trades * 100) if self.stats.total_trades > 0 else 0
        pnl = self.risk_manager.current_balance - self.stats.initial_balance
        
        print(f"\n=== Trading Status ===")
        print(f"Trades: {self.stats.total_trades} (W: {self.stats.winning_trades} / L: {self.stats.losing_trades})")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Balance: {self.risk_manager.current_balance:.2f} (PNL: {pnl:.2f})")
        print(f"Max Consecutive Wins: {self.stats.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {self.stats.max_consecutive_losses}")
        print("=====================")
    
    def start(self):
        """Start the trading session"""
        self.is_running = True
        self.logger.info("Starting trading session")
        
        try:
            while self.is_running:
                # Check if we should stop trading
                should_stop, reason = self.risk_manager.should_stop_trading()
                if should_stop:
                    self.stats.stop_reason = reason
                    self.logger.info(f"Stopping trading session. Reason: {reason.name}")
                    break
                
                # Generate trade signal
                signal = self.strategy.generate_signal()
                bet_amount = self.strategy.get_bet_amount()
                
                # Create trade
                trade = self.exchange.place_trade(
                    symbol=self.symbol,
                    amount=bet_amount,
                    signal=signal
                )
                
                # Check risk parameters
                if not self.risk_manager.assess_trade_risk(trade):
                    self.logger.warning(f"Trade rejected by risk manager: {trade.id}")
                    continue
                
                self.logger.info(f"Placed trade {trade.id}: {trade.signal.name} {trade.amount:.2f}")
                
                # Wait for trade to complete
                while True:
                    outcome = self.exchange.check_trade_status(trade.id)
                    if outcome != TradeOutcome.PENDING:
                        break
                    time.sleep(0.5)
                
                trade.outcome = outcome
                self._update_stats(trade)
                self.strategy.update_strategy(trade)
                self._log_trade(trade)
                
                # Print status periodically
                if self.stats.total_trades % 5 == 0:
                    self._print_status()
                
                # Random delay between trades
                time.sleep(random.uniform(0.5, 2.0))
                
        except Exception as e:
            self.logger.error(f"Error in trading session: {str(e)}")
            self.stats.stop_reason = StopReason.SYSTEM_ERROR
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading session"""
        self.is_running = False
        self.stats.end_time = datetime.now()
        self.stats.final_balance = self.risk_manager.current_balance
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final session summary"""
        duration = (self.stats.end_time - self.stats.start_time).total_seconds() / 60
        pnl = self.stats.final_balance - self.stats.initial_balance
        win_rate = (self.stats.winning_trades / self.stats.total_trades * 100) if self.stats.total_trades > 0 else 0
        
        print("\n=== TRADING SESSION SUMMARY ===")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Initial Balance: {self.stats.initial_balance:.2f}")
        print(f"Final Balance: {self.stats.final_balance:.2f}")
        print(f"Total P/L: {pnl:.2f}")
        print(f"Total Trades: {self.stats.total_trades}")
        print(f"Winning Trades: {self.stats.winning_trades}")
        print(f"Losing Trades: {self.stats.losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Max Consecutive Wins: {self.stats.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {self.stats.max_consecutive_losses}")
        print(f"Stop Reason: {self.stats.stop_reason.name if self.stats.stop_reason else 'N/A'}")
        print("==============================")

# ====================== Main Execution ======================

if __name__ == "__main__":
    # Configuration
    INITIAL_BALANCE = 1000.0
    DAILY_STOP_LOSS = -300.0  # Negative value
    DAILY_TARGET = 500.0
    MARTINGALE_LEVELS = 7
    TRADING_SYMBOL = "BTCUSD"
    
    # Setup components
    exchange = DemoExchangeClient(symbols=[TRADING_SYMBOL], win_prob=0.47)
    strategy = MartingaleStrategy(
        capital=INITIAL_BALANCE,
        levels=MARTINGALE_LEVELS,
        payout=0.85
    )
    risk_manager = BasicRiskManager(
        initial_balance=INITIAL_BALANCE,
        daily_stop_loss=DAILY_STOP_LOSS,
        daily_target=DAILY_TARGET
    )
    
    # Create and start bot
    bot = TradingBot(
        exchange=exchange,
        strategy=strategy,
        risk_manager=risk_manager,
        symbol=TRADING_SYMBOL
    )
    
    bot.start()