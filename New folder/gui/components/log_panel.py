"""
System log display panel component.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime

class LogPanel(ttk.LabelFrame):
    """
    Panel for displaying system logs and status messages.
    """
    
    def __init__(self, parent):
        """
        Initialize the log panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, text="System Log", padding="10")
        self._create_widgets()
        
    def _create_widgets(self) -> None:
        """Create and arrange all child widgets."""
        # Log display
        self.log_text = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=('Consolas', 9),
            state='disabled'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            style='Status.TLabel'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def log_message(self, message: str) -> None:
        """Add a message to the log display."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.config(state='disabled')
        self.log_text.see(tk.END)
    
    def set_status(self, message: str) -> None:
        """Set the status bar message."""
        self.status_var.set(message)