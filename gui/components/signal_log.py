"""
Signal log component displaying recent trading signals.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.processor import TradingSignalProcessor

class SignalLog(ttk.LabelFrame):
    """
    Scrollable log of recent trading signals.
    """
    
    def __init__(self, parent, processor: 'TradingSignalProcessor'):
        """
        Initialize signal log.
        
        Args:
            parent: Parent widget
            processor: TradingSignalProcessor instance
        """
        super().__init__(parent, text="Signal Log", padding=10)
        self.processor = processor
        
        self._create_widgets()
        self._update_log()
        
    def _create_widgets(self) -> None:
        """Create log display widgets."""
        self.text = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=('Consolas', 10)
        )
        self.text.pack(fill=tk.BOTH, expand=True)
        
    def _update_log(self) -> None:
        """Update log with recent signals."""
        self.text.delete(1.0, tk.END)
        
        signals = self.processor.repository.get_signals(limit=20)
        for signal in reversed(signals):
            self.text.insert(tk.END, 
                f"{signal['timestamp']} - {signal['symbol']} {signal['signal_type']}\n"
                f"Price: {signal['price']}  Lots: {signal['lot_size']}\n"
                f"Reason: {signal['reason']}\n\n")
        
        self.after(5000, self._update_log)