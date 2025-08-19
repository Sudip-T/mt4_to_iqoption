import asyncio
from enum import Enum
from uuid import uuid4
from decimal import Decimal
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, DateTime, Integer, Text


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
        
        if self.winning_trades > 0:
            self.average_win = sum(t.pnl for t in self._trades if t.pnl > 0) / self.winning_trades
        if self.losing_trades > 0:
            self.average_loss = sum(t.pnl for t in self._trades if t.pnl < 0) / self.losing_trades
            
        if self.average_loss != 0:
            self.profit_factor = abs(self.average_win / self.average_loss)


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
    metadata = Column(Text)


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