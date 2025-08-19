"""
General helper functions for trading operations.
"""

from datetime import datetime
from typing import Optional

def format_timestamp(timestamp: str) -> Optional[str]:
    """
    Format timestamp string for display.
    
    Args:
        timestamp: ISO format timestamp string
        
    Returns:
        Formatted string or None if invalid
    """
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None

def calculate_pip_value(symbol: str, price: float) -> float:
    """
    Calculate pip value for a currency pair.
    
    Args:
        symbol: Currency pair (e.g. 'EURUSD')
        price: Current price
        
    Returns:
        Pip value in account currency
    """
    # Simplified calculation - adjust based on your needs
    if 'JPY' in symbol:
        return 0.01
    return 0.0001