"""
Status panel component showing processor state and controls.
"""

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.processor import TradingSignalProcessor

class StatusPanel(ttk.LabelFrame):
    """
    Panel showing processor status and control buttons.
    """
    
    def __init__(self, parent, processor: 'TradingSignalProcessor'):
        """
        Initialize status panel.
        
        Args:
            parent: Parent widget
            processor: TradingSignalProcessor instance
        """
        super().__init__(parent, text="Processor Status", padding=10)
        self.processor = processor
        
        self._create_widgets()
        self._update_status()
        
    def _create_widgets(self) -> None:
        """Create and arrange child widgets."""
        # Status label
        self.status_label = ttk.Label(self, text="Status: Unknown", font=('Arial', 10, 'bold'))
        self.status_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        # Control buttons
        self.start_btn = ttk.Button(self, text="Start", command=self._start_processor)
        self.start_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = ttk.Button(self, text="Stop", command=self._stop_processor, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=5)
        
    def _update_status(self) -> None:
        """Update status display based on processor state."""
        if self.processor.running:
            self.status_label.config(text="Status: RUNNING", foreground='green')
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Status: STOPPED", foreground='red')
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
        
        self.after(1000, self._update_status)
        
    def _start_processor(self) -> None:
        """Start the signal processor."""
        self.processor.start()
        self._update_status()
        
    def _stop_processor(self) -> None:
        """Stop the signal processor."""
        self.processor.stop()
        self._update_status()