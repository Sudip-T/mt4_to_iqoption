import os
from pathlib import Path
from typing import Dict, Any

# Database settings
DB_PATH = Path(os.getenv('DB_PATH', 'data/trading_signals.db'))
DB_BACKUP_INTERVAL = int(os.getenv('DB_BACKUP_INTERVAL', '3600'))  # seconds

# File receiver settings
FILE_RECEIVER_PATH = os.getenv('FILE_RECEIVER_PATH')  # Default to MT4 common files directory
FILE_POLLING_INTERVAL = float(os.getenv('FILE_POLLING_INTERVAL', '0.5'))  # seconds

# HTTP receiver settings
HTTP_PORT = int(os.getenv('HTTP_PORT', '8080'))
HTTP_AUTH_TOKEN = os.getenv('HTTP_AUTH_TOKEN')

# Risk management settings
RISK_LIMITS: Dict[str, Any] = {
    'max_lot_size': float(os.getenv('MAX_LOT_SIZE', '10.0')),
    'max_daily_trades': int(os.getenv('MAX_DAILY_TRADES', '50')),
    'max_concurrent_trades': int(os.getenv('MAX_CONCURRENT_TRADES', '10')),
    'min_account_balance': float(os.getenv('MIN_ACCOUNT_BALANCE', '1000.0'))
}

# Processing settings
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '5'))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_signals.log')

# Ensure directories exist
Path('data').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)