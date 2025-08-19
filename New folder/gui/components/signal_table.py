"""
Signal display table component.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import TYPE_CHECKING, List, Dict, Any
import pandas as pd

if TYPE_CHECKING:
    from ...processor import TradingSignalProcessor

class SignalTable(ttk.LabelFrame):
    """
    Table for displaying recent trading signals.
    """
    
    def __init__(self, parent, processor: 'TradingSignalProcessor'):
        """
        Initialize the signal table.
        
        Args:
            parent: Parent widget
            processor: TradingSignalProcessor instance
        """
        super().__init__(parent, text="Recent Signals", padding="10")
        self.processor = processor
        self._create_widgets()
        self._setup_bindings()
        self.refresh()
        
    def _create_widgets(self) -> None:
        """Create and arrange all child widgets."""
        # Create treeview with scrollbars
        self.tree = ttk.Treeview(
            self,
            columns=('id', 'timestamp', 'symbol', 'type', 
                    'price', 'lots', 'processed', 'reason'),
            show='headings',
            selectmode='browse'
        )
        
        # Configure columns
        columns = {
            'id': {'text': 'ID', 'width': 50, 'anchor': tk.CENTER},
            'timestamp': {'text': 'Timestamp', 'width': 150},
            'symbol': {'text': 'Symbol', 'width': 80, 'anchor': tk.CENTER},
            'type': {'text': 'Type', 'width': 60, 'anchor': tk.CENTER},
            'price': {'text': 'Price', 'width': 80, 'anchor': tk.CENTER},
            'lots': {'text': 'Lots', 'width': 60, 'anchor': tk.CENTER},
            'processed': {'text': 'Processed', 'width': 80, 'anchor': tk.CENTER},
            'reason': {'text': 'Reason', 'width': 300}
        }
        
        for col, config in columns.items():
            self.tree.heading(col, text=config['text'])
            self.tree.column(col, width=config.get('width', 100),
                            anchor=config.get('anchor', tk.W))
        
        # Add scrollbars
        v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Layout components
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_bindings(self) -> None:
        """Set up event bindings."""
        self.tree.bind('<Double-1>', self._show_signal_details)
    
    def refresh(self) -> None:
        """Refresh the table data."""
        self.tree.delete(*self.tree.get_children())
        signals = self.processor.get_recent_signals(100)
        
        for signal in signals:
            self.tree.insert('', 'end', values=(
                signal.get('id', ''),
                signal.get('timestamp', ''),
                signal.get('symbol', ''),
                signal.get('signal_type', ''),
                f"{signal.get('price', 0):.5f}",
                f"{signal.get('lot_size', 0):.2f}",
                "Yes" if signal.get('processed') else "No",
                signal.get('reason', '')[:100] + "..." 
                if len(signal.get('reason', '')) > 100 
                else signal.get('reason', '')
            ))
    
    def export_to_csv(self) -> None:
        """Export signals to CSV file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Signals"
            )
            
            if filename:
                signals = self.processor.get_recent_signals(1000)
                df = pd.DataFrame(signals)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export Complete", 
                                   f"Exported {len(df)} signals to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def _show_signal_details(self, event=None) -> None:
        """Show detailed view of selected signal."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        details = self._format_signal_details(item['values'])
        
        # Create details window
        detail_win = tk.Toplevel(self)
        detail_win.title("Signal Details")
        detail_win.geometry("600x400")
        
        text = tk.Text(detail_win, wrap=tk.WORD, font=('Consolas', 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text.insert(tk.END, details)
        text.config(state='disabled')
    
    def _format_signal_details(self, values: List) -> str:
        """Format signal details for display."""
        return f"""
Signal Details
-------------
ID: {values[0]}
Timestamp: {values[1]}
Symbol: {values[2]}
Type: {values[3]}
Price: {values[4]}
Lot Size: {values[5]}
Processed: {values[6]}
Reason: {values[7]}
"""