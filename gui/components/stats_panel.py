"""
Statistics panel showing processing metrics.
"""

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..core.processor import TradingSignalProcessor

class StatsPanel(ttk.LabelFrame):
    """
    Panel displaying signal processing statistics.
    """
    
    def __init__(self, parent, processor: 'TradingSignalProcessor'):
        """
        Initialize stats panel.
        
        Args:
            parent: Parent widget
            processor: TradingSignalProcessor instance
        """
        super().__init__(parent, text="Processing Statistics", padding=10)
        self.processor = processor
        
        self.stats_vars: Dict[str, tk.StringVar] = {}
        self._create_widgets()
        self._update_stats()
        
    def _create_widgets(self) -> None:
        """Create and arrange statistics display widgets."""
        stats = [
            ('total_signals', 'Total Signals:'),
            ('buy_signals', 'Buy Signals:'),
            ('sell_signals', 'Sell Signals:'),
            ('processed_signals', 'Processed:'),
            ('failed_signals', 'Failed:'),
            ('uptime', 'Uptime:')
        ]
        
        for i, (key, label) in enumerate(stats):
            ttk.Label(self, text=label).grid(row=i//3, column=(i%3)*2, padx=5, pady=2, sticky=tk.W)
            self.stats_vars[key] = tk.StringVar(value="0")
            ttk.Label(self, textvariable=self.stats_vars[key]).grid(
                row=i//3, column=(i%3)*2+1, padx=5, pady=2, sticky=tk.W)
                
    def _update_stats(self) -> None:
        """Update statistics display."""
        stats = self.processor.get_statistics()
        for key, var in self.stats_vars.items():
            if key in stats:
                var.set(str(stats[key]))
        
        self.after(1000, self._update_stats)