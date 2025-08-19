"""
Main application window for the trading signal processor GUI.
"""

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.processor import TradingSignalProcessor

class MainWindow:
    """
    Main application window that coordinates all GUI components.
    """
    
    def __init__(self, processor: 'TradingSignalProcessor'):
        """
        Initialize main window with processor reference.
        
        Args:
            processor: TradingSignalProcessor instance
        """
        self.processor = processor
        self.root = tk.Tk()
        self.root.title("Trading Signal Processor")
        self.root.geometry("1000x800")
        
        self._setup_styles()
        self._create_widgets()
        
    def _setup_styles(self) -> None:
        """Configure ttk styles."""
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', padding=5)
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        
    def _create_widgets(self) -> None:
        """Create and arrange all GUI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status panel
        self.status_panel = StatusPanel(main_frame, self.processor)
        self.status_panel.pack(fill=tk.X, pady=5)
        
        # Stats panel
        self.stats_panel = StatsPanel(main_frame, self.processor)
        self.stats_panel.pack(fill=tk.X, pady=5)
        
        # Signal log
        self.signal_log = SignalLog(main_frame, self.processor)
        self.signal_log.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def run(self) -> None:
        """Start the GUI main loop."""
        self.root.mainloop()