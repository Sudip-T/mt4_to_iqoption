import os
import re
import logging
from pathlib import Path
from typing import Optional



def validate_symbol(symbol: str) -> Optional[str]:
    """
    Validate currency pair symbol format.
    
    Args:
        symbol: Currency pair to validate
        
    Returns:
        Error message if invalid, None if valid
    """
    if not re.match(r'^[A-Z]{6}$', symbol):
        return "Symbol must be 6 uppercase letters (e.g. EURUSD)"
    return None

def validate_price(price: float) -> Optional[str]:
    """
    Validate price value.
    
    Args:
        price: Price to validate
        
    Returns:
        Error message if invalid, None if valid
    """
    if price <= 0:
        return "Price must be positive"
    return None



def validate_signal_data(data: dict) -> bool:
    """
    Validate signal data structure.
    
    Args:
        data: Dictionary of signal data
        
    Returns:
        bool: True if valid, False otherwise
    """
    required = ['timestamp', 'symbol', 'signal_type', 'price', 'lot_size']
    return all(field in data for field in required)



def configure_logging(log_file: str = "signal_processor.log") -> None:
    """
    Configure application-wide logging.
    
    Args:
        log_file: Path to log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_mt4_common_path() -> Optional[Path]:
    """
    Get the default MT4 common files directory path.
    
    Returns:
        Path object or None if not found
    """
    appdata = os.getenv('APPDATA')
    if not appdata:
        return None
        
    mt4_common = Path(appdata) / 'MetaQuotes' / 'Terminal' / 'Common' / 'Files'
    return mt4_common if mt4_common.exists() else None
