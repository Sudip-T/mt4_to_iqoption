"""
Main application window for the signal processor GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING
from processor import TradingSignalProcessor
from gui.components.statistics_panel import StatisticsPanel
from gui.components.signal_table import SignalTable
from gui.components.log_panel import LogPanel

if TYPE_CHECKING:
    from processor import TradingSignalProcessor


class MainWindow:
    """
    Main application window container.
    """
    
    def __init__(self, processor: 'TradingSignalProcessor'):
        """
        Initialize the main window.
        
        Args:
            processor: TradingSignalProcessor instance
        """
        self.processor = processor
        self.root = tk.Tk()
        self._setup_window()
        self._create_widgets()
        self._setup_menu()
        
    def _setup_window(self) -> None:
        """Configure main window properties."""
        self.root.title("MT4 Signal Processor")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _create_widgets(self) -> None:
        """Create and arrange all GUI components."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create components
        self.stats_panel = StatisticsPanel(self.main_frame, self.processor)
        self.signal_table = SignalTable(self.main_frame, self.processor)
        self.log_panel = LogPanel(self.main_frame)
        
        # Layout components
        self.stats_panel.pack(fill=tk.X, pady=(0, 10))
        self.signal_table.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.log_panel.pack(fill=tk.BOTH, expand=True)
        
    def _setup_menu(self) -> None:
        """Create the main menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export Signals...", command=self._export_signals)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Manual Trade...", command=self._show_manual_trade)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _export_signals(self) -> None:
        """Export signals to CSV file."""
        self.signal_table.export_to_csv()
    
    def _show_manual_trade(self) -> None:
        """Show manual trade dialog."""
        messagebox.showinfo("Manual Trade", "Manual trade functionality coming soon")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "MT4 Signal Processor\nVersion 1.0\n\nAdvanced trading signal processing system"
        )
    
    def _on_close(self) -> None:
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.root.destroy()
    
    def run(self) -> None:
        """Run the main application loop."""
        self.root.mainloop()