from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


@dataclass
class SignalValidationResult:
    """
    Result of signal validation with details about success/failure.
    
    Attributes:
        is_valid: Whether the signal passed validation
        error_message: Error message if validation failed
        warnings: List of warning messages
        risk_score: Risk assessment score (0-100)
    """
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    risk_score: int = 0


@dataclass
class MT4Signal:
    """
    Data class representing a trading signal from MT4.
    
    This class encapsulates all information about a trading signal including
    timing, symbol, trade direction, pricing, and metadata.
    
    Attributes:
        timestamp: When the signal was generated
        symbol: Trading symbol (e.g., 'EURUSD', 'GBPJPY')
        signal_type: Type of signal ('BUY', 'SELL', 'CLOSE')
        price: Entry price for the trade
        lot_size: Position size in lots
        reason: Human-readable reason for the signal
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        magic_number: MT4 magic number for order identification
        processed: Whether the signal has been processed
        validation_result: Result of signal validation
        created_at: When the signal object was created
        
    Methods:
        validate(): Perform basic validation of signal data
        to_dict(): Convert signal to dictionary representation
        from_dict(): Create signal from dictionary
    """
    timestamp: datetime
    symbol: str
    signal_type: str
    price: float
    lot_size: float
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic_number: Optional[int] = None
    processed: bool = False
    validation_result: Optional[SignalValidationResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> SignalValidationResult:
        """
        Perform comprehensive validation of the signal data.
        
        Validates signal integrity, data types, ranges, and trading logic.
        
        Returns:
            SignalValidationResult: Validation result with details
            
        Raises:
            None: All validation errors are captured in the result
        """
        warnings = []
        
        try:
            # Validate signal type
            valid_types = ['BUY', 'SELL', 'CLOSE', 'BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP']
            if self.signal_type not in valid_types:
                return SignalValidationResult(
                    is_valid=False,
                    error_message=f"Invalid signal type: {self.signal_type}. Must be one of {valid_types}"
                )
            
            # Validate price
            if self.price <= 0:
                return SignalValidationResult(
                    is_valid=False,
                    error_message=f"Invalid price: {self.price}. Price must be positive"
                )
            
            # Validate lot size
            if self.lot_size <= 0 or self.lot_size > 100:
                return SignalValidationResult(
                    is_valid=False,
                    error_message=f"Invalid lot size: {self.lot_size}. Must be between 0 and 100"
                )
            
            # Validate symbol format
            if not self.symbol or len(self.symbol) < 3:
                return SignalValidationResult(
                    is_valid=False,
                    error_message=f"Invalid symbol: {self.symbol}. Must be at least 3 characters"
                )
            
            # Validate timestamp
            if self.timestamp > datetime.now() + timedelta(minutes=5):
                return SignalValidationResult(
                    is_valid=False,
                    error_message="Signal timestamp is too far in the future"
                )
            
            # Validate stop loss and take profit
            if self.stop_loss is not None:
                if self.stop_loss <= 0:
                    warnings.append("Stop loss should be positive")
                elif self.signal_type == 'BUY' and self.stop_loss >= self.price:
                    warnings.append("Stop loss should be below entry price for BUY signals")
                elif self.signal_type == 'SELL' and self.stop_loss <= self.price:
                    warnings.append("Stop loss should be above entry price for SELL signals")
            
            if self.take_profit is not None:
                if self.take_profit <= 0:
                    warnings.append("Take profit should be positive")
                elif self.signal_type == 'BUY' and self.take_profit <= self.price:
                    warnings.append("Take profit should be above entry price for BUY signals")
                elif self.signal_type == 'SELL' and self.take_profit >= self.price:
                    warnings.append("Take profit should be below entry price for SELL signals")
            
            # Calculate risk score
            risk_score = self._calculate_risk_score()
            
            self.validation_result = SignalValidationResult(
                is_valid=True,
                warnings=warnings,
                risk_score=risk_score
            )
            
            return self.validation_result
            
        except Exception as e:
            return SignalValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def _calculate_risk_score(self) -> int:
        """
        Calculate a risk score for the signal based on various factors.
        
        Returns:
            int: Risk score from 0 (low risk) to 100 (high risk)
        """
        risk_score = 0
        
        # Lot size risk
        if self.lot_size > 1.0:
            risk_score += min(int(self.lot_size * 10), 30)
        
        # Missing stop loss increases risk
        if self.stop_loss is None:
            risk_score += 25
        
        # Missing take profit increases risk slightly
        if self.take_profit is None:
            risk_score += 10
        
        # Time-based risk (older signals are riskier)
        age_minutes = (datetime.now() - self.timestamp).total_seconds() / 60
        if age_minutes > 60:
            risk_score += min(int(age_minutes / 10), 20)
        
        return min(risk_score, 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert signal to dictionary representation.
        
        Returns:
            Dict containing all signal attributes
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'price': self.price,
            'lot_size': self.lot_size,
            'reason': self.reason,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'magic_number': self.magic_number,
            'processed': self.processed,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MT4Signal':
        """
        Create MT4Signal instance from dictionary.
        
        Args:
            data: Dictionary containing signal data
            
        Returns:
            MT4Signal instance
            
        Raises:
            ValueError: If required fields are missing
            TypeError: If data types are incorrect
        """
        try:
            return cls(
                timestamp=datetime.fromisoformat(data['timestamp']),
                symbol=data['symbol'],
                signal_type=data['signal_type'],
                price=float(data['price']),
                lot_size=float(data['lot_size']),
                reason=data.get('reason', ''),
                stop_loss=float(data['stop_loss']) if data.get('stop_loss') else None,
                take_profit=float(data['take_profit']) if data.get('take_profit') else None,
                magic_number=int(data['magic_number']) if data.get('magic_number') else None,
                processed=bool(data.get('processed', False)),
                created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid signal data: {str(e)}")