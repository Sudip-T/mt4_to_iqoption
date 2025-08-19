import sys
import logging
from enum import Enum, auto


# Configure comprehensive logging
def setup_logging(log_level: str = "INFO", log_file: str = "signal_processor.log") -> logging.Logger:
    """
    Configure comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        
    Returns:
        Configured logger instance
        
    Raises:
        ValueError: If invalid log level is provided
    """
    try:
        level = getattr(logging, log_level.upper())
    except AttributeError:
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create custom formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger




def setup_logging2(log_file: str = "signal_processor.log") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to log file
    """
    global logger
    
    logger = logging.getLogger("TradingSignalProcessor")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class SignalType(Enum):
    """Enumeration of valid signal types."""
    BUY = auto()
    SELL = auto()
    HOLD = "HOLD"







from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional



@dataclass
class MartingaleResult:
    """Data class for martingale sequence calculation results."""
    levels: int
    payout_ratio: float
    initial_bet: float
    bet_sequence: List[float]
    total_risk: float
    calculation_timestamp: str


def calculate_martingale_sequence(
    levels: int = 7, 
    payout: float = 0.85, 
    initial_bet: float = 1.0
) -> MartingaleResult:
    """
    Calculate a martingale bet sequence for loss recovery.

    Args:
        levels (int): Number of martingale levels
        payout (float): Broker payout ratio (e.g., 0.85 = 85%)
        initial_bet (float): Starting bet amount

    Returns:
        MartingaleResult: Complete martingale calculation results

    Raises:
        ValueError: If input parameters are invalid
    """
    if levels <= 0:
        raise ValueError("Levels must be positive")
    if not 0 < payout < 1:
        raise ValueError("Payout must be between 0 and 1")
    if initial_bet <= 0:
        raise ValueError("Initial bet must be positive")


    bet_sequence = []
    current_bet = initial_bet
    cumulative_loss = 0

    for level in range(levels):
        bet_sequence.append(round(current_bet, 2))
        cumulative_loss += current_bet
        current_bet = cumulative_loss / payout

    return bet_sequence


def get_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s',
        handlers=[
            logging.FileHandler("my_log.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)