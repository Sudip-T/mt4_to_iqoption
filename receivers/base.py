"""
Base classes for signal receivers.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
from models import MT4Signal

class ISignalReceiver(ABC):
    """Interface for signal receivers."""
    
    @abstractmethod
    def set_callback(self, callback: Callable[[MT4Signal], None]) -> None:
        """Set the callback for processed signals."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the receiver."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the receiver."""
        pass