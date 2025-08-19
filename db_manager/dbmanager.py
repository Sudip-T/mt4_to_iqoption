import time
import sqlite3
import logging
import threading
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from models import MT4Signal

logger = logging.getLogger(__name__)


class DatabaseManager:
    """    
    This class provides a robust interface for storing, retrieving, and managing
    trading signals in a SQLite database. It includes connection pooling, 
    transaction management, and comprehensive error handling.
    
    Attributes:
        db_path: Path to the SQLite database file
        _lock: Thread lock for database operations
        
    """
    
    def __init__(self, db_path: str = "mt4_signals.db", backup_interval: int = 3600):
        self.db_path = Path(db_path)
        self.backup_interval = backup_interval
        self._lock = threading.RLock()
        self._last_backup = 0

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Database manager initialized: {self.db_path}")
        
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic cleanup.
        
        Yields:
            sqlite3.Connection: Database connection
            
        Raises:
            sqlite3.Error: If connection cannot be established
        """
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None  # Autocommit mode
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _init_db(self) -> None:
        """
        Initialize the database schema with optimized structure.
        
        Creates tables, indexes, and triggers for the signal processing system.
        The schema is designed for high performance with appropriate indexes.
        
        Raises:
            sqlite3.Error: If schema creation fails
        """
        try:
            with self.get_connection() as conn:
                # Create main signals table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        price REAL NOT NULL,
                        lot_size REAL NOT NULL,
                        reason TEXT,
                        stop_loss REAL,
                        take_profit REAL,
                        magic_number INTEGER,
                        processed BOOLEAN DEFAULT FALSE,
                        risk_score INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create performance indexes
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
                    ON signals(timestamp)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol 
                    ON signals(symbol)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_signals_processed 
                    ON signals(processed)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_signals_created_at 
                    ON signals(created_at)
                ''')
                
                # Create trigger for updated_at timestamp
                conn.execute('''
                    CREATE TRIGGER IF NOT EXISTS update_signals_timestamp
                    AFTER UPDATE ON signals
                    BEGIN
                        UPDATE signals SET updated_at = CURRENT_TIMESTAMP 
                        WHERE id = NEW.id;
                    END
                ''')
                
                # Create statistics view
                conn.execute('''
                    CREATE VIEW IF NOT EXISTS signal_statistics AS
                    SELECT 
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
                        SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
                        SUM(CASE WHEN processed = 1 THEN 1 ELSE 0 END) as processed_signals,
                        AVG(risk_score) as avg_risk_score,
                        MAX(created_at) as last_signal_time
                    FROM signals
                ''')
                
                logger.info("Database schema initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_signal(self, signal: MT4Signal) -> int:
        """
        Save a trading signal to the database with validation.
        
        Args:
            signal: MT4Signal instance to save
            
        Returns:
            int: Database ID of the saved signal
            
        Raises:
            sqlite3.Error: If save operation fails
            ValueError: If signal validation fails
        """
        # Validate signal before saving
        validation_result = signal.validate()
        if not validation_result.is_valid:
            raise ValueError(f"Signal validation failed: {validation_result.error_message}")
        
        try:
            with self._lock:
                with self.get_connection() as conn:
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
                        validation_result.risk_score
                    ))
                    
                    signal_id = cursor.lastrowid
                    logger.debug(f"Signal saved with ID: {signal_id}")
                    
                    # Auto-backup if needed
                    # self._auto_backup()
                    
                    return signal_id
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to save signal: {e}")
            raise
    
    def get_signals(self, 
                   limit: int = 100, 
                   offset: int = 0, 
                   symbol: Optional[str] = None,
                   signal_type: Optional[str] = None,
                   processed: Optional[bool] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> List[MT4Signal]:
        """
        Retrieve signals from database with comprehensive filtering options.
        
        Args:
            limit: Maximum number of signals to return
            offset: Number of signals to skip
            symbol: Filter by trading symbol
            signal_type: Filter by signal type
            processed: Filter by processed status
            start_date: Filter signals after this date
            end_date: Filter signals before this date
            
        Returns:
            List of MT4Signal instances
            
        Raises:
            sqlite3.Error: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                # Build dynamic query
                query = "SELECT * FROM signals WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if signal_type:
                    query += " AND signal_type = ?"
                    params.append(signal_type)
                
                if processed is not None:
                    query += " AND processed = ?"
                    params.append(processed)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                signals = []
                for row in cursor.fetchall():
                    signal = MT4Signal(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        symbol=row['symbol'],
                        signal_type=row['signal_type'],
                        price=row['price'],
                        lot_size=row['lot_size'],
                        reason=row['reason'] or '',
                        stop_loss=row['stop_loss'],
                        take_profit=row['take_profit'],
                        magic_number=row['magic_number'],
                        processed=bool(row['processed']),
                        created_at=datetime.fromisoformat(row['created_at'])
                    )
                    signals.append(signal)
                
                return signals
                
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve signals: {e}")
            raise
    
    def mark_processed(self, signal_id: int) -> bool:
        """
        Mark a signal as processed in the database.
        
        Args:
            signal_id: Database ID of the signal to mark as processed
            
        Returns:
            bool: True if signal was successfully marked as processed
            
        Raises:
            sqlite3.Error: If update operation fails
        """
        try:
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE signals SET processed = TRUE WHERE id = ?",
                        (signal_id,)
                    )
                    
                    success = cursor.rowcount > 0
                    if success:
                        logger.debug(f"Signal {signal_id} marked as processed")
                    else:
                        logger.warning(f"Signal {signal_id} not found for processing")
                    
                    return success
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to mark signal as processed: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive signal processing statistics.
        
        Returns:
            Dictionary containing various statistics about signals
            
        Raises:
            sqlite3.Error: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM signal_statistics")
                row = cursor.fetchone()
                
                if row:
                    return {
                        'total_signals': row['total_signals'],
                        'buy_signals': row['buy_signals'],
                        'sell_signals': row['sell_signals'],
                        'processed_signals': row['processed_signals'],
                        'avg_risk_score': round(row['avg_risk_score'] or 0, 2),
                        'last_signal_time': row['last_signal_time']
                    }
                else:
                    return {
                        'total_signals': 0,
                        'buy_signals': 0,
                        'sell_signals': 0,
                        'processed_signals': 0,
                        'avg_risk_score': 0,
                        'last_signal_time': None
                    }
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
    
    def _auto_backup(self) -> None:
        """
        Automatically backup database if backup interval has passed.
        """
        current_time = time.time()
        if current_time - self._last_backup > self.backup_interval:
            try:
                self.backup_database()
                self._last_backup = current_time
            except Exception as e:
                logger.error(f"Auto-backup failed: {e}")
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup file (optional)
            
        Returns:
            str: Path to the backup file
            
        Raises:
            sqlite3.Error: If backup operation fails
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path.stem}_backup_{timestamp}.db"
        
        try:
            with self.get_connection() as source_conn:
                backup_conn = sqlite3.connect(backup_path)
                source_conn.backup(backup_conn)
                backup_conn.close()
            
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except sqlite3.Error as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    def cleanup_old_signals(self, days_old: int = 30) -> int:
        """
        Remove signals older than specified number of days.
        
        Args:
            days_old: Number of days after which signals should be removed
            
        Returns:
            int: Number of signals removed
            
        Raises:
            sqlite3.Error: If cleanup operation fails
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM signals WHERE created_at < ?",
                        (cutoff_date.isoformat(),)
                    )
                    
                    removed_count = cursor.rowcount
                    logger.info(f"Removed {removed_count} signals older than {days_old} days")
                    
                    return removed_count
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to cleanup old signals: {e}")
            raise


    def erase_database(self) -> None:
        """Clear all signals from the database."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM signals")
            conn.commit()





# class SignalDatabase:
#     """
#     Persistent storage for trading signals using SQLite.
    
#     Features:
#     - Thread-safe operations
#     - Connection pooling
#     - Schema versioning
#     """
    
#     SCHEMA_VERSION = 1
    
#     def __init__(self, db_path: str = "signals.db"):
#         """
#         Initialize the signal database.
        
#         Args:
#             db_path: Path to SQLite database file
#         """
#         self.db_path = db_path
#         self._lock = threading.Lock()
#         self._init_database()
    
#     def _init_database(self) -> None:
#         """Initialize database schema and verify version."""
#         with self._lock, sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
            
#             # Create version table if not exists
#             cursor.execute('''
#                 CREATE TABLE IF NOT EXISTS db_version (
#                     version INTEGER PRIMARY KEY
#                 )
#             ''')
            
#             # Get current version
#             cursor.execute('SELECT version FROM db_version')
#             version = cursor.fetchone()
            
#             if version is None:
#                 # New database - set version and create schema
#                 cursor.execute('INSERT INTO db_version (version) VALUES (?)', 
#                              (self.SCHEMA_VERSION,))
#                 self._create_schema(cursor)
#             elif version[0] < self.SCHEMA_VERSION:
#                 # Existing database - migrate schema
#                 self._migrate_schema(cursor, version[0])
            
#             conn.commit()
    
#     def _create_schema(self, cursor: sqlite3.Cursor) -> None:
#         """Create initial database schema."""
#         cursor.execute('''
#             CREATE TABLE signals (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 timestamp TEXT NOT NULL,
#                 symbol TEXT NOT NULL,
#                 signal_type TEXT NOT NULL,
#                 price REAL NOT NULL,
#                 lot_size REAL NOT NULL,
#                 reason TEXT,
#                 processed BOOLEAN DEFAULT FALSE,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 processed_at TIMESTAMP,
#                 execution_status TEXT
#             )
#         ''')
        
#         # Create indexes for faster queries
#         cursor.execute('CREATE INDEX idx_signals_processed ON signals(processed)')
#         cursor.execute('CREATE INDEX idx_signals_timestamp ON signals(timestamp)')
    
#     def _migrate_schema(self, cursor: sqlite3.Cursor, from_version: int) -> None:
#         """Migrate database schema from older version."""
#         # Example migration - add execution_status column in future version
#         pass
    
#     def save_signal(self, signal: 'MT4Signal') -> None:
#         """
#         Save a signal to the database.
        
#         Args:
#             signal: MT4Signal object to save
#         """
#         with self._lock, sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
            
#             cursor.execute('''
#                 INSERT INTO signals (
#                     timestamp, symbol, signal_type, price, 
#                     lot_size, reason, processed
#                 ) VALUES (?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 signal.timestamp.isoformat(),
#                 signal.symbol,
#                 signal.signal_type.name,
#                 signal.price,
#                 signal.lot_size,
#                 signal.reason,
#                 signal.processed
#             ))
            
#             conn.commit()
    
#     def get_unprocessed_signals(self) -> List[Dict[str, Any]]:
#         """
#         Retrieve all unprocessed signals from the database.
        
#         Returns:
#             List of signal dictionaries
#         """
#         with self._lock, sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
            
#             cursor.execute('''
#                 SELECT timestamp, symbol, signal_type, price, lot_size, reason
#                 FROM signals 
#                 WHERE processed = FALSE
#                 ORDER BY created_at ASC
#             ''')
            
#             return [dict(row) for row in cursor.fetchall()]
    
#     def mark_processed(self, signal: 'MT4Signal', status: str = "EXECUTED") -> None:
#         """
#         Mark a signal as processed in the database.
        
#         Args:
#             signal: MT4Signal to mark as processed
#             status: Execution status (e.g., "EXECUTED", "FAILED")
#         """
#         with self._lock, sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
            
#             cursor.execute('''
#                 UPDATE signals 
#                 SET processed = TRUE,
#                     processed_at = CURRENT_TIMESTAMP,
#                     execution_status = ?
#                 WHERE timestamp = ? AND symbol = ? AND signal_type = ?
#             ''', (
#                 status,
#                 signal.timestamp.isoformat(),
#                 signal.symbol,
#                 signal.signal_type.name
#             ))
            
#             conn.commit()
    
#     def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
#         """
#         Retrieve recent signals from database.
        
#         Args:
#             limit: Maximum number of signals to return
            
#         Returns:
#             List of signal dictionaries
#         """
#         with sqlite3.connect(self.db_path) as conn:
#             query = f'''
#                 SELECT 
#                     id, timestamp, symbol, signal_type, 
#                     price, lot_size, processed, reason
#                 FROM signals 
#                 ORDER BY created_at DESC 
#                 LIMIT {limit}
#             '''
#             df = pd.read_sql_query(query, conn)
#             return df.to_dict('records')
    
#     def clear_database(self) -> None:
#         """Clear all signals from the database."""
#         with self._lock, sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("DELETE FROM signals")
#             conn.commit()