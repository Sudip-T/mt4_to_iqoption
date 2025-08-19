"""
Configuration package initialization.
Exports the main configuration items for easy access.
"""

from .settings import DB_PATH, DB_BACKUP_INTERVAL, FILE_RECEIVER_PATH
from .settings import FILE_POLLING_INTERVAL, HTTP_PORT, HTTP_AUTH_TOKEN
from .settings import RISK_LIMITS, MAX_WORKERS
from .logging import setup_logging

__all__ = [
    'DB_PATH',
    'DB_BACKUP_INTERVAL',
    'FILE_RECEIVER_PATH',
    'FILE_POLLING_INTERVAL',
    'HTTP_PORT',
    'HTTP_AUTH_TOKEN',
    'RISK_LIMITS',
    'MAX_WORKERS',
    'setup_logging'
]