import os
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from .base import ISignalReceiver
from typing import Optional, Callable
from models import MT4Signal

logger = logging.getLogger(__name__)

class FileSignalReceiver(ISignalReceiver):
    """
    Advanced file-based signal receiver with robust monitoring and parsing.
    
    This class monitors a specified file for new trading signals from MT4 Expert Advisors.
    It provides robust file monitoring, parsing, and error handling capabilities.
    
    Attributes:
        file_path: Path to the signal file
        running: Whether the receiver is currently running
        last_modified: Last modification time of the signal file
        signal_callback: Callback function for processed signals
        
    Methods:
        start(): Start file monitoring
        stop(): Stop file monitoring
        set_callback(): Set signal processing callback
        parse_signal(): Parse signal from file content
    """
    
    def __init__(self, file_path: Optional[str] = None, polling_interval: float = 0.5):
        """
        Initialize the file signal receiver.
        
        Args:
            file_path: Path to signal file (defaults to MT4 common files directory)
            polling_interval: File polling interval in seconds
            
        Raises:
            FileNotFoundError: If default MT4 directory cannot be found
        """
        self.polling_interval = polling_interval
        self.running = False
        self.last_modified = 0
        self.signal_callback: Optional[Callable[[MT4Signal], None]] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Determine file path
        if file_path is None:
            try:
                mt4_common = Path(os.getenv('APPDATA', '')) / 'MetaQuotes' / 'Terminal' / 'Common' / 'Files'
                self.file_path = mt4_common / 'mt4_signals.txt'
            except Exception:
                self.file_path = Path('./mt4_signals.txt')
        else:
            self.file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"File signal receiver initialized: {self.file_path}")
    
    def set_callback(self, callback: Callable[[MT4Signal], None]) -> None:
        """
        Set the callback function for signal processing.
        
        Args:
            callback: Function to call when a signal is received
            
        Raises:
            TypeError: If callback is not callable
        """
        if not callable(callback):
            raise TypeError("Callback must be callable")
        
        self.signal_callback = callback
        logger.debug("Signal callback set for file receiver")
    
    def parse_signal(self, content: str) -> Optional[MT4Signal]:
        """
        Parse a trading signal from file content.
        
        Expected format: timestamp,symbol,signal_type,price,lot_size,reason[,stop_loss,take_profit,magic_number]
        
        Args:
            content: Raw signal content from file
            
        Returns:
            MT4Signal instance or None if parsing fails
            
        Raises:
            None: All parsing errors are logged and None is returned
        """
        try:
            content = content.strip()
            if not content:
                return None
            
            # Support both comma and semicolon separators
            parts = content.replace(';', ',').split(',')
            
            if len(parts) < 6:
                logger.warning(f"Invalid signal format - insufficient parts: {content}")
                return None
            
            # Parse required fields
            timestamp_str = parts[0].strip()
            symbol = parts[1].strip().upper()
            signal_type = parts[2].strip().upper()
            price = float(parts[3].strip())
            lot_size = float(parts[4].strip())
            reason = parts[5].strip()
            
            # Parse timestamp with multiple format support
            timestamp = self._parse_timestamp(timestamp_str)
            if timestamp is None:
                logger.warning(f"Invalid timestamp format: {timestamp_str}")
                return None
            
            # Parse optional fields
            stop_loss = None
            take_profit = None
            magic_number = None
            
            if len(parts) > 6 and parts[6].strip():
                try:
                    stop_loss = float(parts[6].strip())
                except ValueError:
                    logger.warning(f"Invalid stop_loss value: {parts[6]}")
            
            if len(parts) > 7 and parts[7].strip():
                try:
                    take_profit = float(parts[7].strip())
                except ValueError:
                    logger.warning(f"Invalid take_profit value: {parts[7]}")
            
            if len(parts) > 8 and parts[8].strip():
                try:
                    magic_number = int(parts[8].strip())
                except ValueError:
                    logger.warning(f"Invalid magic_number value: {parts[8]}")
            
            signal = MT4Signal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                price=price,
                lot_size=lot_size,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                magic_number=magic_number
            )
            
            logger.debug(f"Parsed signal: {signal.signal_type} {signal.symbol} at {signal.price}")
            return signal
            
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing signal '{content}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing signal: {e}")
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse timestamp string with multiple format support.
        
        Args:
            timestamp_str: Timestamp string in various formats
            
        Returns:
            datetime object or None if parsing fails
        """
        formats = [
            '%Y.%m.%d %H:%M:%S',  # MT4 default format
            '%Y-%m-%d %H:%M:%S',  # ISO format
            '%Y/%m/%d %H:%M:%S',  # Alternative format
            '%Y.%m.%d %H:%M',     # Without seconds
            '%Y-%m-%d %H:%M',     # ISO without seconds
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # Try ISO format parsing
        try:
            return datetime.fromisoformat(timestamp_str.replace('.', '-'))
        except ValueError:
            pass
        
        return None
    
    def check_file_changes(self) -> None:
        """
        Check for file changes and process new signals.
        
        This method monitors the signal file for modifications and processes
        any new signals found. It includes robust error handling and file locking.
        """
        try:
            if not self.file_path.exists():
                return
            
            current_modified = self.file_path.stat().st_mtime
            if current_modified <= self.last_modified:
                return
            
            self.last_modified = current_modified
            
            # Read file with proper encoding and error handling
            try:
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    
                if not content:
                    return
                
                # Handle multiple signals in one file
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    signal = self.parse_signal(line)
                    if signal and self.signal_callback:
                        try:
                            self.signal_callback(signal)
                        except Exception as e:
                            logger.error(f"Error in signal callback: {e}")
                
                # Clear file after processing
                try:
                    with open(self.file_path, 'w', encoding='utf-8') as f:
                        f.write('')
                except Exception as e:
                    logger.warning(f"Could not clear signal file: {e}")
                    
            except Exception as e:
                logger.error(f"Error reading signal file: {e}")
        
        except Exception as e:
            logger.error(f"Error checking file changes: {e}")
    
    def start(self) -> None:
        """
        Start file monitoring in a separate thread.
        
        Creates a daemon thread that continuously monitors the signal file
        for changes and processes any new signals found.
        """
        if self.running:
            logger.warning("File receiver is already running")
            return
        
        self.running = True
        
        def monitor_loop():
            """Main monitoring loop that runs in the background thread."""
            logger.info("File monitoring started")
            
            while self.running:
                try:
                    self.check_file_changes()
                    time.sleep(self.polling_interval)
                except Exception as e:
                    logger.error(f"Error in file monitoring loop: {e}")
                    time.sleep(1)  # Brief pause before retrying
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("File signal receiver started")
    
    def stop(self) -> None:
        """
        Stop file monitoring gracefully.
        
        Signals the monitoring thread to stop and waits for it to complete.
        """
        if not self.running:
            return
        
        self.running = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
            if self._monitor_thread.is_alive():
                logger.warning("File monitoring thread did not stop gracefully")
        
        logger.info("File signal receiver stopped")
    

    # @staticmethod
    # def _get_default_file_path() -> str:
    #     """Get default MT4 common files directory path."""
    #     mt4_common = os.path.join(
    #         os.getenv('APPDATA', ''),
    #         'MetaQuotes',
    #         'Terminal',
    #         'Common',
    #         'Files'
    #     )
    #     return os.path.join(mt4_common, 'mt4_signals.txt')