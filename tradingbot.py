import json
import asyncio
import logging
from decimal import Decimal
from typing import Any, Dict, Callable

from bet3 import *


class TradingBot:
    """Professional trading bot with comprehensive features"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        
        # Core components
        self.risk_manager = RiskManager(config)
        self.execution_engine = ExecutionEngine(config)
        self.position_sizer = MartingalePositionSizer(config.kelly_fraction)
        
        # State management
        self.is_running = False
        self.active_trades = {}
        self.metrics = PerformanceMetrics()
        self.current_capital = Decimal(str(config.initial_capital))
        
        # Event system
        self.event_handlers = {
            'system_error': [],
            'trade_executed': [],
            'risk_violation': [],
            'daily_target_reached': [],
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
        self.logger.info("Starting Trading Bot...")
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



bot = TradingBot()