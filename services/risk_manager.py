"""
Risk management service.
"""

import logging
from typing import Dict
from core.models import MT4Signal
from data.repositories import SignalRepository

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages trading risk assessment and limits."""
    
    def __init__(self, repository: SignalRepository, limits: Dict):
        self.repo = repository
        self.limits = limits
    
    def assess_risk(self, signal: MT4Signal) -> bool:
        """Check if signal complies with risk limits."""
        if signal.lot_size > self.limits['max_lot_size']:
            logger.warning(f"Signal exceeds max lot size: {signal.lot_size}")
            return False
        
        # Add more risk checks as needed
        return True