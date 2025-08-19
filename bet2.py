import asyncio
import json
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import uuid4

import aiofiles
import aiohttp
import pandas as pd
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool


# ==============================================================================
# Configuration and Models
# ==============================================================================

class TradingConfig(BaseModel):
    """Trading configuration with validation"""
    
    initial_capital: float = Field(gt=0, description="Initial trading capital")
    max_daily_loss: float = Field(gt=0, description="Maximum daily loss limit")
    daily_profit_target: float = Field(gt=0, description="Daily profit target")
    max_position_size: float = Field(gt=0, le=1, description="Maximum position size as fraction of capital")
    
    # Risk Management
    max_drawdown_pct: float = Field(default=0.20, ge=0, le=1, description="Maximum drawdown percentage")
    max_consecutive_losses: int = Field(default=10, ge=1, description="Maximum consecutive losses")
    volatility_adjustment: bool = Field(default=True, description="Enable volatility-based position sizing")
    
    # Strategy Parameters
    martingale_levels: int = Field(default=7, ge=1, le=15, description="Number of martingale levels")
    kelly_fraction: float = Field(default=0.25, ge=0, le=1, description="Kelly criterion fraction")
    
    # Execution
    slippage_tolerance: float = Field(default=0.001, ge=0, description="Slippage tolerance")
    execution_timeout: int = Field(default=30, ge=1, description="Order execution timeout in seconds")
    
    # Monitoring
    metrics_interval: int = Field(default=60, ge=1, description="Metrics update interval in seconds")
    alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "drawdown": 0.15,
        "consecutive_losses": 5,
        "low_balance": 0.3
    })

    @validator('max_daily_loss')
    def validate_daily_loss(cls, v, values):
        if 'initial_capital' in values and v >= values['initial_capital']:
            raise ValueError("Daily loss limit must be less than initial capital")
        return v


class TradeSignal(Enum):
    """Trade signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderStatus(Enum):
    """Order status types"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Trade:
    """Trade execution record"""
    
    trade_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = "DEFAULT"
    signal: TradeSignal = TradeSignal.HOLD
    quantity: Decimal = Decimal('0')
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    pnl: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    strategy: str = "default"
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: Decimal = Decimal('0')
    average_loss: Decimal = Decimal('0')
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    
    def update_from_trade(self, trade: Trade):
        """Update metrics from a completed trade"""
        if trade.pnl is None:
            return
            
        self.total_trades += 1
        self.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # Calculate derived metrics
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # if self.winning_trades > 0:
        #     self.average_win = sum(t.pnl for t in self._trades if t.pnl > 0) / self.winning_trades
        # if self.losing_trades > 0:
        #     self.average_loss = sum(t.pnl for t in self._trades if t.pnl < 0) / self.losing_trades
            
        # if self.average_loss != 0:
        #     self.profit_factor = abs(self.average_win / self.average_loss)


# ==============================================================================
# Database Models
# ==============================================================================

Base = declarative_base()


class TradeRecord(Base):
    """Database model for trade records"""
    
    __tablename__ = 'trades'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    symbol = Column(String, nullable=False)
    signal = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    status = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    risk_metrics = Column(Text)
    # metadata = Column(Text)


class PerformanceRecord(Base):
    """Database model for performance metrics"""
    
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    consecutive_losses = Column(Integer, default=0)


# ==============================================================================
# Position Sizing Strategies
# ==============================================================================

class PositionSizer(ABC):
    """Abstract base class for position sizing strategies"""
    
    @abstractmethod
    async def calculate_position_size(self, 
                                    capital: Decimal, 
                                    signal_strength: float,
                                    volatility: float,
                                    **kwargs) -> Decimal:
        """Calculate position size based on strategy"""
        pass


class KellyPositionSizer(PositionSizer):
    """Kelly Criterion position sizing"""
    
    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction
    
    async def calculate_position_size(self, 
                                    capital: Decimal,
                                    signal_strength: float,
                                    volatility: float,
                                    win_rate: float = 0.5,
                                    avg_win: float = 1.0,
                                    avg_loss: float = 1.0,
                                    **kwargs) -> Decimal:
        """Calculate Kelly optimal position size"""
        
        if avg_loss == 0:
            return Decimal('0')
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - p
        
        kelly_f = (b * p - q) / b
        kelly_f = max(0, min(kelly_f, 1))  # Clamp between 0 and 1
        
        # Apply fraction and signal strength
        position_fraction = kelly_f * self.kelly_fraction * signal_strength
        
        # Adjust for volatility
        volatility_adj = 1 / (1 + volatility)
        position_fraction *= volatility_adj
        
        return capital * Decimal(str(position_fraction))


class MartingalePositionSizer(PositionSizer):
    """Martingale position sizing with risk controls"""
    
    def __init__(self, levels: int = 7, max_multiplier: float = 2.0):
        self.levels = levels
        self.max_multiplier = max_multiplier
        self.current_level = 0
        self.base_size = None
    
    async def calculate_position_size(self,
                                    capital: Decimal,
                                    signal_strength: float,
                                    volatility: float,
                                    consecutive_losses: int = 0,
                                    **kwargs) -> Decimal:
        """Calculate martingale position size"""
        
        if self.base_size is None:
            self.base_size = capital * Decimal('0.02')  # 2% of capital
        
        # Reset on win or max levels reached
        if consecutive_losses == 0 or consecutive_losses >= self.levels:
            self.current_level = 0
        else:
            self.current_level = consecutive_losses
        
        # Calculate multiplier with exponential decay for high levels
        if self.current_level == 0:
            multiplier = 1.0
        else:
            multiplier = min(self.max_multiplier ** self.current_level, 
                           capital.quantize(Decimal('0.01')) / self.base_size)
        
        # Apply volatility adjustment
        volatility_adj = 1 / (1 + volatility * 2)
        
        position_size = self.base_size * Decimal(str(multiplier * volatility_adj))
        
        # Ensure position doesn't exceed capital
        return min(position_size, capital * Decimal('0.5'))


# ==============================================================================
# Risk Management
# ==============================================================================

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = Decimal('0')
        self.peak_capital = Decimal(str(config.initial_capital))
        self.current_drawdown = Decimal('0')
        self.risk_breaches = []
        self.last_reset = datetime.now().date()
    
    async def check_risk_limits(self, 
                               current_capital: Decimal,
                               trade: Trade,
                               metrics: PerformanceMetrics) -> Tuple[bool, List[str]]:
        """Check if trade violates risk limits"""
        
        violations = []
        
        # Daily reset check
        if datetime.now().date() > self.last_reset:
            self.daily_pnl = Decimal('0')
            self.last_reset = datetime.now().date()
        
        # Daily loss limit
        if abs(self.daily_pnl) >= Decimal(str(self.config.max_daily_loss)):
            violations.append("DAILY_LOSS_LIMIT_EXCEEDED")
        
        # Maximum drawdown
        self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        if self.current_drawdown >= Decimal(str(self.config.max_drawdown_pct)):
            violations.append("MAX_DRAWDOWN_EXCEEDED")
        
        # Consecutive losses
        if metrics.consecutive_losses >= self.config.max_consecutive_losses:
            violations.append("MAX_CONSECUTIVE_LOSSES")
        
        # Position size limit
        max_position = current_capital * Decimal(str(self.config.max_position_size))
        if trade.quantity > max_position:
            violations.append("POSITION_SIZE_EXCEEDED")
        
        # Update peak capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        return len(violations) == 0, violations
    
    async def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 10:
            return 0.0
        
        returns_sorted = sorted(returns)
        index = int((1 - confidence) * len(returns_sorted))
        return returns_sorted[index]


# ==============================================================================
# Strategy Interface
# ==============================================================================

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    async def generate_signal(self, market_data: Dict[str, Any]) -> Tuple[TradeSignal, float]:
        """Generate trading signal with strength"""
        pass
    
    @abstractmethod
    async def should_exit(self, trade: Trade, current_price: Decimal) -> bool:
        """Determine if position should be closed"""
        pass


class RandomStrategy(TradingStrategy):
    """Random trading strategy for testing"""
    
    def __init__(self, win_rate: float = 0.55):
        self.win_rate = win_rate
    
    async def generate_signal(self, market_data: Dict[str, Any]) -> Tuple[TradeSignal, float]:
        """Generate random signal"""
        import random
        
        # Simulate market analysis delay
        await asyncio.sleep(0.1)
        
        signal = TradeSignal.BUY if random.random() < 0.6 else TradeSignal.SELL
        strength = random.uniform(0.3, 1.0)
        
        return signal, strength
    
    async def should_exit(self, trade: Trade, current_price: Decimal) -> bool:
        """Random exit decision"""
        import random
        return random.random() < self.win_rate


# ==============================================================================
# Execution Engine
# ==============================================================================

class ExecutionEngine:
    """Advanced order execution engine"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.pending_orders = {}
        self.execution_lock = asyncio.Lock()
    
    async def execute_trade(self, trade: Trade) -> Tuple[bool, str]:
        """Execute trade with slippage and timeout handling"""
        
        async with self.execution_lock:
            try:
                # Simulate order placement
                await asyncio.sleep(0.1)
                
                # Simulate slippage
                import random
                slippage = random.uniform(-self.config.slippage_tolerance, 
                                        self.config.slippage_tolerance)
                
                # Simulate fill
                fill_price = trade.entry_price * (1 + Decimal(str(slippage)))
                trade.entry_price = fill_price
                trade.status = OrderStatus.FILLED
                
                return True, "ORDER_FILLED"
                
            except asyncio.TimeoutError:
                trade.status = OrderStatus.CANCELLED
                return False, "EXECUTION_TIMEOUT"
            except Exception as e:
                trade.status = OrderStatus.REJECTED
                return False, f"EXECUTION_ERROR: {str(e)}"


# ==============================================================================
# Main Trading Bot
# ==============================================================================

class AdvancedTradingBot:
    """Professional trading bot with comprehensive features"""
    
    def __init__(self, config: TradingConfig, strategy: TradingStrategy):
        self.config = config
        self.strategy = strategy
        
        # Core components
        self.risk_manager = RiskManager(config)
        self.execution_engine = ExecutionEngine(config)
        self.position_sizer = KellyPositionSizer(config.kelly_fraction)
        
        # State management
        self.current_capital = Decimal(str(config.initial_capital))
        self.is_running = False
        self.metrics = PerformanceMetrics()
        self.active_trades = {}
        
        # Database
        self.db_engine = create_engine(
            "sqlite:///trading_bot.db",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.db_engine)
        self.SessionLocal = sessionmaker(bind=self.db_engine)
        
        # Logging
        self.logger = self._setup_logging()
        
        # Event system
        self.event_handlers = {
            'trade_executed': [],
            'risk_violation': [],
            'daily_target_reached': [],
            'system_error': []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('AdvancedTradingBot')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
    
    async def save_trade(self, trade: Trade):
        """Save trade to database"""
        try:
            with self.SessionLocal() as session:
                db_trade = TradeRecord(
                    id=trade.trade_id,
                    timestamp=trade.timestamp,
                    symbol=trade.symbol,
                    signal=trade.signal.value,
                    quantity=float(trade.quantity),
                    entry_price=float(trade.entry_price) if trade.entry_price else None,
                    exit_price=float(trade.exit_price) if trade.exit_price else None,
                    pnl=float(trade.pnl) if trade.pnl else None,
                    status=trade.status.value,
                    strategy=trade.strategy,
                    risk_metrics=json.dumps(trade.risk_metrics),
                    metadata=json.dumps(trade.metadata)
                )
                session.add(db_trade)
                session.commit()
        except Exception as e:
            self.logger.error(f"Failed to save trade: {e}")
    
    async def process_trade(self) -> bool:
        """Process a single trade"""
        try:
            # Generate signal
            signal, strength = await self.strategy.generate_signal({})
            
            if signal == TradeSignal.HOLD:
                return True
            
            # Calculate position size
            position_size = await self.position_sizer.calculate_position_size(
                capital=self.current_capital,
                signal_strength=strength,
                volatility=self.metrics.volatility,
                win_rate=self.metrics.win_rate,
                avg_win=float(self.metrics.average_win),
                avg_loss=float(self.metrics.average_loss),
                consecutive_losses=self.metrics.consecutive_losses
            )
            
            # Create trade
            trade = Trade(
                symbol="DEFAULT",
                signal=signal,
                quantity=position_size,
                entry_price=Decimal('100'),  # Simulated price
                strategy="advanced_strategy",
                risk_metrics={
                    'signal_strength': strength,
                    'volatility': self.metrics.volatility,
                    'kelly_fraction': self.config.kelly_fraction
                }
            )
            
            # Risk check
            risk_ok, violations = await self.risk_manager.check_risk_limits(
                self.current_capital, trade, self.metrics
            )
            
            if not risk_ok:
                self.logger.warning(f"Risk violations: {violations}")
                await self.emit_event('risk_violation', {
                    'violations': violations,
                    'trade': trade
                })
                return False
            
            # Execute trade
            executed, message = await self.execution_engine.execute_trade(trade)
            
            if executed:
                # Simulate trade outcome
                import random
                win = await self.strategy.should_exit(trade, trade.entry_price)
                
                if win:
                    trade.pnl = trade.quantity * Decimal('0.85')  # 85% payout
                    trade.exit_price = trade.entry_price * Decimal('1.85')
                else:
                    trade.pnl = -trade.quantity
                    trade.exit_price = Decimal('0')
                
                # Update capital and metrics
                self.current_capital += trade.pnl
                self.metrics.update_from_trade(trade)
                self.risk_manager.daily_pnl += trade.pnl
                
                # Save trade
                await self.save_trade(trade)
                
                # Emit event
                await self.emit_event('trade_executed', {
                    'trade': trade,
                    'capital': self.current_capital,
                    'metrics': self.metrics
                })
                
                self.logger.info(f"Trade executed: {trade.trade_id}, "
                               f"PnL: {trade.pnl}, Capital: {self.current_capital}")
                
                return True
            else:
                self.logger.warning(f"Trade execution failed: {message}")
                return False
                
        except Exception as e:
            self.logger.error(f"Trade processing error: {e}")
            await self.emit_event('system_error', {'error': str(e)})
            return False
    
    async def run_monitoring_loop(self):
        """Background monitoring and metrics updates"""
        while self.is_running:
            try:
                # Update performance metrics
                if self.metrics.total_trades > 0:
                    # Calculate additional metrics
                    returns = []  # Would be populated with actual returns
                    var = await self.risk_manager.calculate_var(returns)
                    
                    # Save performance snapshot
                    with self.SessionLocal() as session:
                        perf_record = PerformanceRecord(
                            total_trades=self.metrics.total_trades,
                            winning_trades=self.metrics.winning_trades,
                            total_pnl=float(self.metrics.total_pnl),
                            max_drawdown=float(self.metrics.max_drawdown),
                            sharpe_ratio=self.metrics.sharpe_ratio,
                            win_rate=self.metrics.win_rate,
                            consecutive_losses=self.metrics.consecutive_losses
                        )
                        session.add(perf_record)
                        session.commit()
                
                # Check alert thresholds
                await self._check_alerts()
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _check_alerts(self):
        """Check alert thresholds and send notifications"""
        thresholds = self.config.alert_thresholds
        
        # Drawdown alert
        if self.risk_manager.current_drawdown >= Decimal(str(thresholds['drawdown'])):
            await self.emit_event('risk_violation', {
                'type': 'HIGH_DRAWDOWN',
                'value': float(self.risk_manager.current_drawdown),
                'threshold': thresholds['drawdown']
            })
        
        # Consecutive losses alert
        if self.metrics.consecutive_losses >= thresholds['consecutive_losses']:
            await self.emit_event('risk_violation', {
                'type': 'CONSECUTIVE_LOSSES',
                'value': self.metrics.consecutive_losses,
                'threshold': thresholds['consecutive_losses']
            })
        
        # Low balance alert
        balance_ratio = self.current_capital / Decimal(str(self.config.initial_capital))
        if balance_ratio <= Decimal(str(thresholds['low_balance'])):
            await self.emit_event('risk_violation', {
                'type': 'LOW_BALANCE',
                'value': float(balance_ratio),
                'threshold': thresholds['low_balance']
            })
    
    async def start(self):
        """Start the trading bot"""
        self.logger.info("Starting Advanced Trading Bot...")
        self.logger.info(f"Initial Capital: ${self.current_capital}")
        self.logger.info(f"Configuration: {self.config}")
        
        self.is_running = True
        
        # Start monitoring loop
        monitoring_task = asyncio.create_task(self.run_monitoring_loop())
        
        try:
            trade_count = 0
            max_trades = 100  # Safety limit
            
            while self.is_running and trade_count < max_trades:
                success = await self.process_trade()
                
                if success:
                    trade_count += 1
                
                # Check daily target
                daily_pnl = self.current_capital - Decimal(str(self.config.initial_capital))
                if daily_pnl >= Decimal(str(self.config.daily_profit_target)):
                    self.logger.info(f"Daily target reached: ${daily_pnl}")
                    await self.emit_event('daily_target_reached', {
                        'daily_pnl': float(daily_pnl),
                        'target': self.config.daily_profit_target
                    })
                    break
                
                # Check stop conditions
                if abs(self.risk_manager.daily_pnl) >= Decimal(str(self.config.max_daily_loss)):
                    self.logger.warning("Daily loss limit reached")
                    break
                
                await asyncio.sleep(1)  # Rate limiting
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")
        finally:
            self.is_running = False
            monitoring_task.cancel()
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down trading bot...")
        
        # Close any open positions
        for trade_id, trade in self.active_trades.items():
            self.logger.info(f"Closing position: {trade_id}")
            # Implementation would close actual positions
        
        # Final metrics report
        self.logger.info(f"Final Performance:")
        self.logger.info(f"  Total Trades: {self.metrics.total_trades}")
        self.logger.info(f"  Win Rate: {self.metrics.win_rate:.2%}")
        self.logger.info(f"  Total PnL: ${self.metrics.total_pnl}")
        self.logger.info(f"  Final Capital: ${self.current_capital}")
        self.logger.info(f"  Max Drawdown: {self.metrics.max_drawdown:.2%}")
        self.logger.info(f"  Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}")
        
        self.logger.info("Trading bot shutdown complete")


# ==============================================================================
# Usage Example
# ==============================================================================

async def main():
    """Main execution function"""
    
    # Configuration
    config = TradingConfig(
        initial_capital=10000.0,
        max_daily_loss=1000.0,
        daily_profit_target=500.0,
        max_position_size=0.1,
        martingale_levels=5,
        kelly_fraction=0.25,
        max_consecutive_losses=7
    )
    
    # Strategy
    strategy = RandomStrategy(win_rate=0.55)
    
    # Create bot
    bot = AdvancedTradingBot(config, strategy)
    
    # Add event handlers
    async def on_trade_executed(data):
        print(f"Trade executed: {data['trade'].trade_id}, "
              f"PnL: ${data['trade'].pnl}, "
              f"Capital: ${data['capital']}")
    
    async def on_risk_violation(data):
        print(f"Risk violation: {data}")
    
    bot.add_event_handler('trade_executed', on_trade_executed)
    bot.add_event_handler('risk_violation', on_risk_violation)
    
    # Start trading
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())