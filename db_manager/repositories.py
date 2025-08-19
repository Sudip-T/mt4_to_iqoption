"""
Data access layer for trading signals.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from .database import DatabaseManager
from core.models import MT4Signal

logger = logging.getLogger(__name__)

class SignalRepository:
    """Handles all database operations for trading signals."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def save_signal(self, signal: MT4Signal) -> int:
        """Save a signal to the database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (
                    timestamp, symbol, signal_type, price, lot_size, reason,
                    stop_loss, take_profit, magic_number, processed, risk_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.signal_type,
                signal.price,
                signal.lot_size,
                signal.reason,
                signal.stop_loss,
                signal.take_profit,
                signal.magic_number,
                signal.processed,
                signal.validation_result.risk_score if signal.validation_result else 0
            ))
            return cursor.lastrowid
    
    def get_signals(self, limit: int = 100, **filters) -> List[Dict[str, Any]]:
        """Retrieve signals with optional filters."""
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        
        if 'symbol' in filters:
            query += " AND symbol = ?"
            params.append(filters['symbol'])
        
        if 'processed' in filters:
            query += " AND processed = ?"
            params.append(filters['processed'])
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_processed(self, signal_id: int) -> bool:
        """Mark a signal as processed."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE signals SET processed = TRUE WHERE id = ?",
                (signal_id,)
            )
            return cursor.rowcount > 0