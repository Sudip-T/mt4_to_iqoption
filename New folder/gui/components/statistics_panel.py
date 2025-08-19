"""
Statistics display panel component.
"""

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ...processor import TradingSignalProcessor

class StatisticsPanel(ttk.LabelFrame):
    """
    Panel for displaying processing statistics.
    """
    
    def __init__(self, parent, processor: 'TradingSignalProcessor'):
        """
        Initialize the statistics panel.
        
        Args:
            parent: Parent widget
            processor: TradingSignalProcessor instance
        """
        super().__init__(parent, text="Processing Statistics", padding="10")
        self.processor = processor
        self._create_widgets()
        self._setup_bindings()
        
    def _create_widgets(self) -> None:
        """Create and arrange all child widgets."""
        self.stats_vars = {}
        stats_grid = ttk.Frame(self)
        stats_grid.pack(fill=tk.X)
        
        stats = [
            ('total_signals', 'Total Signals'),
            ('buy_signals', 'Buy Signals'),
            ('sell_signals', 'Sell Signals'),
            ('processed_signals', 'Processed'),
            ('failed_signals', 'Failed'),
            ('last_signal_time', 'Last Signal')
        ]
        
        for i, (key, label) in enumerate(stats):
            ttk.Label(stats_grid, text=label + ":").grid(
                row=i//3, column=(i%3)*2, padx=5, pady=2, sticky=tk.W)
            
            self.stats_vars[key] = tk.StringVar(value="0")
            ttk.Label(stats_grid, 
                     textvariable=self.stats_vars[key],
                     font=('Arial', 10, 'bold')).grid(
                row=i//3, column=(i%3)*2+1, padx=5, pady=2, sticky=tk.W)
    
    def _setup_bindings(self) -> None:
        """Set up event bindings."""
        self.bind("<Button-1>", self.refresh)
    
    def refresh(self, event=None) -> None:
        """Refresh the statistics display."""
        stats = self.processor.get_statistics()
        for key, var in self.stats_vars.items():
            value = stats.get(key, "")
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            var.set(str(value))