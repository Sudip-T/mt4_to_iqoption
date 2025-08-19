import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from utilities import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)


class TradingSignalGUI:
    """
    Combined GUI for monitoring and controlling the trading signal processor.
    
    This class provides a comprehensive graphical interface combining:
    - Basic processor controls (start/stop)
    - Advanced monitoring features
    - Signal statistics and history
    - System logs and export functionality
    """
    
    def __init__(self, processor):
        """
        Initialize the combined GUI with a processor instance.
        
        Args:
            processor: TradingSignalProcessor to monitor/control
        """
        self.processor = processor
        self.root = tk.Tk()
        self.root.title("MT4 Signal Processor - Advanced Monitor")
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0')
        self.style.configure('TButton', padding=5)
        self.style.configure('Red.TLabel', foreground='red', background='#f0f0f0')
        self.style.configure('Green.TLabel', foreground='green', background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Arial', 10, 'bold'), background='#f0f0f0')
        
        # Build UI
        self.create_widgets()
        self.running = False
        
        # Start update loop
        self.update_interval = 1000  # ms
        self.update_display()
        
    def create_widgets(self) -> None:
        """Create and arrange all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Top section with statistics and controls
        self.create_top_section(main_frame)
        
        # Middle section with signals table
        self.create_signals_table(main_frame)
        
        # Bottom section with controls and logs
        self.create_bottom_section(main_frame)
        
    def create_top_section(self, parent) -> None:
        """Create the top section with statistics."""
        stats_frame = ttk.LabelFrame(parent, text="Processing Statistics", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure(0, weight=1)
        
        # Create grid for statistics
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Statistics labels
        self.stats_labels = {}
        
        # Row 1: Total, Buy, Sell signals
        ttk.Label(stats_grid, text="Total Signals:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.stats_labels['total_signals'] = ttk.Label(stats_grid, text="0")
        self.stats_labels['total_signals'].grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(stats_grid, text="Buy Signals:", style='Header.TLabel').grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.stats_labels['buy_signals'] = ttk.Label(stats_grid, text="0")
        self.stats_labels['buy_signals'].grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(stats_grid, text="Sell Signals:", style='Header.TLabel').grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        self.stats_labels['sell_signals'] = ttk.Label(stats_grid, text="0")
        self.stats_labels['sell_signals'].grid(row=0, column=5, sticky=tk.W)
        
        # Row 2: Processed, Failed, Last Signal
        ttk.Label(stats_grid, text="Processed:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.stats_labels['processed_signals'] = ttk.Label(stats_grid, text="0")
        self.stats_labels['processed_signals'].grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(stats_grid, text="Failed:", style='Header.TLabel').grid(row=1, column=2, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.stats_labels['failed_signals'] = ttk.Label(stats_grid, text="0")
        self.stats_labels['failed_signals'].grid(row=1, column=3, sticky=tk.W, padx=(0, 20), pady=(10, 0))
        
        ttk.Label(stats_grid, text="Last Signal:", style='Header.TLabel').grid(row=1, column=4, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.stats_labels['last_signal'] = ttk.Label(stats_grid, text="Never")
        self.stats_labels['last_signal'].grid(row=1, column=5, sticky=tk.W, pady=(10, 0))
        
    def create_signals_table(self, parent) -> None:
        """Create the signals table section."""
        signals_frame = ttk.LabelFrame(parent, text="Recent Signals", padding="10")
        signals_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        signals_frame.columnconfigure(0, weight=1)
        signals_frame.rowconfigure(0, weight=1)
        
        # Create treeview for signals table
        columns = ('ID', 'Timestamp', 'Symbol', 'Type', 'Price', 'Lots', 'Processed', 'Reason')
        self.signals_tree = ttk.Treeview(signals_frame, columns=columns, show='headings', height=10)
        
        # Configure column headings and widths
        column_widths = {'ID': 50, 'Timestamp': 130, 'Symbol': 80, 'Type': 60, 'Price': 80, 'Lots': 60, 'Processed': 80, 'Reason': 150}
        
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=column_widths.get(col, 100), anchor=tk.CENTER if col in ['ID', 'Type', 'Price', 'Lots', 'Processed'] else tk.W)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(signals_frame, orient=tk.VERTICAL, command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(signals_frame, orient=tk.HORIZONTAL, command=self.signals_tree.xview)
        self.signals_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.signals_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def create_bottom_section(self, parent) -> None:
        """Create the bottom section with controls and logs."""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.rowconfigure(1, weight=1)
        
        # Control buttons
        control_frame = ttk.Frame(bottom_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Processor status and controls
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(status_frame, text="Status:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.status_label = ttk.Label(status_frame, text="Stopped", style='Red.TLabel')
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.start_button = ttk.Button(status_frame, text="Start", command=self.start_processor)
        self.start_button.grid(row=0, column=2, padx=(0, 5))
        
        self.stop_button = ttk.Button(status_frame, text="Stop", command=self.stop_processor, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=(0, 20))
        
        # Additional control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Button(button_frame, text="Refresh", command=self.refresh_data).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Clear DB", command=self.clear_database).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(button_frame, text="Export CSV", command=self.export_csv).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(button_frame, text="Settings", command=self.show_settings).grid(row=0, column=3, padx=(0, 5))
        ttk.Button(button_frame, text="Manual Trade", command=self.manual_trade).grid(row=0, column=4)
        
        # System log
        log_frame = ttk.LabelFrame(bottom_frame, text="System Log", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=('Courier', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add some initial log entries
        self.log_message("System initialized")
        
    def start_processor(self) -> None:
        """Start the signal processor."""
        try:
            self.processor.start()
            self.running = True
            self.status_label.config(text="Running", style='Green.TLabel')
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.log_message("Processor started successfully")
        except Exception as e:
            self.log_message(f"Failed to start processor: {str(e)}")
            messagebox.showerror("Error", f"Failed to start processor: {str(e)}")
            
    def stop_processor(self) -> None:
        """Stop the signal processor."""
        try:
            self.processor.stop()
            self.running = False
            self.status_label.config(text="Stopped", style='Red.TLabel')
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.log_message("Processor stopped")
        except Exception as e:
            self.log_message(f"Failed to stop processor: {str(e)}")
            messagebox.showerror("Error", f"Failed to stop processor: {str(e)}")
            
    def refresh_data(self) -> None:
        """Refresh the display data."""
        self.log_message("Refreshing data...")
        self.update_display()
        
    def clear_database(self) -> None:
        """Clear the database."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the database?"):
            try:
                # Assuming the processor has a method to clear database
                if hasattr(self.processor.db, 'erase_database'):
                    self.processor.db.erase_database()
                    self.log_message("Database cleared successfully")
                    self.update_display()
                else:
                    self.log_message("Operation failed: method not available")
            except Exception as e:
                self.log_message(f"Failed to clear database: {str(e)}")
                
    def export_csv(self) -> None:
        """Export signals to CSV."""
        try:
            # Assuming the processor has an export method
            if hasattr(self.processor, 'export_to_csv'):
                filename = f"signals_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.processor.export_to_csv(filename)
                self.log_message(f"Data exported to {filename}")
            else:
                self.log_message("Export failed: method not available")
        except Exception as e:
            self.log_message(f"Export failed: {str(e)}")
            
    def show_settings(self) -> None:
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        ttk.Label(settings_window, text="Settings dialog - To be implemented").pack(pady=50)
        ttk.Button(settings_window, text="Close", command=settings_window.destroy).pack()
        
    def manual_trade(self) -> None:
        """Open manual trade dialog."""
        trade_window = tk.Toplevel(self.root)
        trade_window.title("Manual Trade")
        trade_window.geometry("300x200")
        trade_window.transient(self.root)
        trade_window.grab_set()
        
        ttk.Label(trade_window, text="Manual Trade dialog - To be implemented").pack(pady=50)
        ttk.Button(trade_window, text="Close", command=trade_window.destroy).pack()
        
    def log_message(self, message: str) -> None:
        """Add a message to the system log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
    def update_display(self) -> None:
        """Update the GUI with current processor state."""
        try:
            # Update statistics
            stats = self.processor.get_statistics() if hasattr(self.processor, 'get_statistics') else {}
            
            # Default values if stats not available
            default_stats = {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'processed_signals': 0,
                'failed_signals': 0,
                'last_signal': 'Never'
            }
            
            for key, label in self.stats_labels.items():
                value = stats.get(key, default_stats.get(key, 0))
                if key == 'last_signal' and value != 'Never':
                    print('hello')
                    # Format timestamp if available
                    if hasattr(value, 'strftime'):
                        value = value.strftime('%Y-%m-%d %H:%M:%S')
                label.config(text=str(value))
            
            # Update signals table
            self.update_signals_table()
            
        except Exception as e:
            self.log_message(f"Error updating display: {str(e)}")
            logger.error(f"Error updating GUI: {e}")
        
        # Schedule next update
        self.root.after(self.update_interval, self.update_display)
        
    def update_signals_table(self) -> None:
        """Update the signals table with recent signals."""
        try:
            # Clear existing items
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            # Get recent signals
            if hasattr(self.processor, 'db') and hasattr(self.processor.db, 'get_signals'):
                signals = self.processor.db.get_signals(limit=50)
            else:
                # Mock data for demonstration
                signals = self.get_mock_signals()
            
            # Populate table
            for i, signal in enumerate(signals):
                values = (
                    getattr(signal, 'id', i+1),
                    getattr(signal, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    getattr(signal, 'symbol', 'EURUSD'),
                    getattr(signal, 'signal_type', 'BUY'),
                    getattr(signal, 'price', '1.15800'),
                    getattr(signal, 'lot_size', '0.01'),
                    'Yes' if getattr(signal, 'processed', True) else 'No',
                    getattr(signal, 'reason', 'MANUAL_TRADE')
                )
                self.signals_tree.insert('', 'end', values=values)
                
        except Exception as e:
            self.log_message(f"Error updating signals table: {str(e)}")
            
    def get_mock_signals(self):
        """Generate mock signals for demonstration."""
        class MockSignal:
            def __init__(self, id, timestamp, symbol, signal_type, price, lot_size, processed, reason):
                self.id = id
                self.timestamp = timestamp
                self.symbol = symbol
                self.signal_type = signal_type
                self.price = price
                self.lot_size = lot_size
                self.processed = processed
                self.reason = reason
        
        return [
            MockSignal(1, '2025-07-16 16:33:10', 'EURUSD', 'SELL', '1.15800', '0.01', True, 'MANUAL_SELL_BUTTON'),
            MockSignal(2, '2025-07-16 16:31:49', 'EURUSD', 'BUY', '1.15807', '0.01', True, 'MANUAL_BUY_BUTTON'),
            MockSignal(3, '2025-07-16 16:32:03', 'EURUSD', 'SELL', '1.15807', '0.01', True, 'MANUAL_SELL_BUTTON'),
        ]
        
    def on_close(self) -> None:
        """Handle window close event."""
        if self.running:
            self.stop_processor()
        self.root.destroy()
        
    def run(self) -> None:
        """Run the GUI main loop."""
        self.root.mainloop()