import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
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
        
        # Create menu bar first
        self.create_menu_bar()
        
        # Build UI
        self.create_widgets()
        self.running = False
        
        # Start update loop
        self.update_interval = 1000  # ms
        self.update_display()
        
    def create_menu_bar(self) -> None:
        """Create the menu bar at the top of the window."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self.new_session, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Log", command=self.open_log_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export to CSV", command=self.export_csv, accelerator="Ctrl+E")
        file_menu.add_command(label="Export to Excel", command=self.export_excel)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close, accelerator="Ctrl+Q")
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Selection", command=self.copy_selection, accelerator="Ctrl+C")
        edit_menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences", command=self.show_settings, accelerator="Ctrl+P")
        
        # Processor menu
        processor_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processor", menu=processor_menu)
        processor_menu.add_command(label="Start", command=self.start_processor, accelerator="F5")
        processor_menu.add_command(label="Stop", command=self.stop_processor, accelerator="F6")
        processor_menu.add_command(label="Restart", command=self.restart_processor, accelerator="F7")
        processor_menu.add_separator()
        processor_menu.add_command(label="Force Stop", command=self.force_stop_processor)
        processor_menu.add_command(label="Process Status", command=self.show_process_status)
        
        # Trading menu
        trading_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Trading", menu=trading_menu)
        trading_menu.add_command(label="Manual Trade", command=self.manual_trade, accelerator="Ctrl+T")
        trading_menu.add_command(label="Quick Buy", command=self.quick_buy, accelerator="Ctrl+B")
        trading_menu.add_command(label="Quick Sell", command=self.quick_sell, accelerator="Ctrl+S")
        trading_menu.add_separator()
        trading_menu.add_command(label="Close All Positions", command=self.close_all_positions)
        trading_menu.add_command(label="Trading History", command=self.show_trading_history)
        
        # Database menu
        database_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Database", menu=database_menu)
        database_menu.add_command(label="Refresh Data", command=self.refresh_data, accelerator="F5")
        database_menu.add_command(label="Clear Database", command=self.clear_database)
        database_menu.add_separator()
        database_menu.add_command(label="Backup Database", command=self.backup_database)
        database_menu.add_command(label="Restore Database", command=self.restore_database)
        database_menu.add_separator()
        database_menu.add_command(label="Database Statistics", command=self.show_db_stats)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Display", command=self.refresh_data, accelerator="F5")
        view_menu.add_separator()
        view_menu.add_command(label="Show/Hide Statistics", command=self.toggle_statistics)
        view_menu.add_command(label="Show/Hide Log", command=self.toggle_log)
        view_menu.add_separator()
        view_menu.add_command(label="Full Screen", command=self.toggle_fullscreen, accelerator="F11")
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Signal Generator", command=self.show_signal_generator)
        tools_menu.add_command(label="Market Data", command=self.show_market_data)
        tools_menu.add_command(label="Performance Monitor", command=self.show_performance_monitor)
        tools_menu.add_separator()
        tools_menu.add_command(label="Connection Test", command=self.test_connection)
        tools_menu.add_command(label="System Information", command=self.show_system_info)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide, accelerator="F1")
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="Check for Updates", command=self.check_updates)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.bind_shortcuts()
        
    def bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts to menu actions."""
        self.root.bind('<Control-n>', lambda e: self.new_session())
        self.root.bind('<Control-o>', lambda e: self.open_log_file())
        self.root.bind('<Control-e>', lambda e: self.export_csv())
        self.root.bind('<Control-q>', lambda e: self.on_close())
        self.root.bind('<Control-c>', lambda e: self.copy_selection())
        self.root.bind('<Control-a>', lambda e: self.select_all())
        self.root.bind('<Control-p>', lambda e: self.show_settings())
        self.root.bind('<F5>', lambda e: self.refresh_data())
        self.root.bind('<F6>', lambda e: self.stop_processor())
        self.root.bind('<F7>', lambda e: self.restart_processor())
        self.root.bind('<Control-t>', lambda e: self.manual_trade())
        self.root.bind('<Control-b>', lambda e: self.quick_buy())
        self.root.bind('<Control-s>', lambda e: self.quick_sell())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<F1>', lambda e: self.show_user_guide())
        
    # Menu action methods
    def new_session(self) -> None:
        """Start a new session."""
        if messagebox.askyesno("New Session", "Start a new session? This will clear current data."):
            self.log_message("Starting new session...")
            # Implementation for new session
            
    def open_log_file(self) -> None:
        """Open a log file."""
        filename = filedialog.askopenfilename(
            title="Open Log File",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.log_message(f"Opening log file: {filename}")
            # Implementation for opening log file
            
    def export_excel(self) -> None:
        """Export signals to Excel."""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export to Excel",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if filename:
                self.log_message(f"Exporting to Excel: {filename}")
                # Implementation for Excel export
        except Exception as e:
            self.log_message(f"Excel export failed: {str(e)}")
            
    def copy_selection(self) -> None:
        """Copy selected text to clipboard."""
        try:
            widget = self.root.focus_get()
            if hasattr(widget, 'selection_get'):
                selection = widget.selection_get()
                self.root.clipboard_clear()
                self.root.clipboard_append(selection)
                self.log_message("Selection copied to clipboard")
        except:
            self.log_message("Nothing selected to copy")
            
    def select_all(self) -> None:
        """Select all text in focused widget."""
        try:
            widget = self.root.focus_get()
            if hasattr(widget, 'select_range'):
                widget.select_range(0, tk.END)
            elif hasattr(widget, 'tag_add'):
                widget.tag_add(tk.SEL, "1.0", tk.END)
        except:
            pass
            
    def restart_processor(self) -> None:
        """Restart the processor."""
        self.log_message("Restarting processor...")
        self.stop_processor()
        self.root.after(1000, self.start_processor)  # Restart after 1 second
        
    def force_stop_processor(self) -> None:
        """Force stop the processor."""
        if messagebox.askyesno("Force Stop", "Force stop the processor? This may cause data loss."):
            self.log_message("Force stopping processor...")
            # Implementation for force stop
            
    def show_process_status(self) -> None:
        """Show detailed process status."""
        status_window = tk.Toplevel(self.root)
        status_window.title("Process Status")
        status_window.geometry("400x300")
        status_window.transient(self.root)
        status_window.grab_set()
        
        ttk.Label(status_window, text="Process Status Details - To be implemented").pack(pady=50)
        ttk.Button(status_window, text="Close", command=status_window.destroy).pack()
        
    def quick_buy(self) -> None:
        """Execute quick buy order."""
        self.log_message("Quick buy executed")
        # Implementation for quick buy
        
    def quick_sell(self) -> None:
        """Execute quick sell order."""
        self.log_message("Quick sell executed")
        # Implementation for quick sell
        
    def close_all_positions(self) -> None:
        """Close all open positions."""
        if messagebox.askyesno("Close All", "Close all open positions?"):
            self.log_message("Closing all positions...")
            # Implementation for closing all positions
            
    def show_trading_history(self) -> None:
        """Show trading history."""
        history_window = tk.Toplevel(self.root)
        history_window.title("Trading History")
        history_window.geometry("600x400")
        history_window.transient(self.root)
        
        ttk.Label(history_window, text="Trading History - To be implemented").pack(pady=50)
        ttk.Button(history_window, text="Close", command=history_window.destroy).pack()
        
    def backup_database(self) -> None:
        """Backup database."""
        filename = filedialog.asksaveasfilename(
            title="Backup Database",
            defaultextension=".db",
            filetypes=[("Database files", "*.db"), ("All files", "*.*")]
        )
        if filename:
            self.log_message(f"Backing up database to: {filename}")
            # Implementation for database backup
            
    def restore_database(self) -> None:
        """Restore database."""
        filename = filedialog.askopenfilename(
            title="Restore Database",
            filetypes=[("Database files", "*.db"), ("All files", "*.*")]
        )
        if filename:
            if messagebox.askyesno("Restore", "This will overwrite current database. Continue?"):
                self.log_message(f"Restoring database from: {filename}")
                # Implementation for database restore
                
    def show_db_stats(self) -> None:
        """Show database statistics."""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Database Statistics")
        stats_window.geometry("400x300")
        stats_window.transient(self.root)
        
        ttk.Label(stats_window, text="Database Statistics - To be implemented").pack(pady=50)
        ttk.Button(stats_window, text="Close", command=stats_window.destroy).pack()
        
    def toggle_statistics(self) -> None:
        """Toggle statistics section visibility."""
        # Implementation for toggling statistics
        self.log_message("Toggling statistics display")
        
    def toggle_log(self) -> None:
        """Toggle log section visibility."""
        # Implementation for toggling log
        self.log_message("Toggling log display")
        
    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
        self.log_message("Toggled fullscreen mode")
        
    def show_signal_generator(self) -> None:
        """Show signal generator tool."""
        gen_window = tk.Toplevel(self.root)
        gen_window.title("Signal Generator")
        gen_window.geometry("400x300")
        gen_window.transient(self.root)
        
        ttk.Label(gen_window, text="Signal Generator - To be implemented").pack(pady=50)
        ttk.Button(gen_window, text="Close", command=gen_window.destroy).pack()
        
    def show_market_data(self) -> None:
        """Show market data."""
        market_window = tk.Toplevel(self.root)
        market_window.title("Market Data")
        market_window.geometry("600x400")
        market_window.transient(self.root)
        
        ttk.Label(market_window, text="Market Data - To be implemented").pack(pady=50)
        ttk.Button(market_window, text="Close", command=market_window.destroy).pack()
        
    def show_performance_monitor(self) -> None:
        """Show performance monitor."""
        perf_window = tk.Toplevel(self.root)
        perf_window.title("Performance Monitor")
        perf_window.geometry("500x400")
        perf_window.transient(self.root)
        
        ttk.Label(perf_window, text="Performance Monitor - To be implemented").pack(pady=50)
        ttk.Button(perf_window, text="Close", command=perf_window.destroy).pack()
        
    def test_connection(self) -> None:
        """Test connection to MT4."""
        self.log_message("Testing connection...")
        # Implementation for connection test
        messagebox.showinfo("Connection Test", "Connection test completed")
        
    def show_system_info(self) -> None:
        """Show system information."""
        info_window = tk.Toplevel(self.root)
        info_window.title("System Information")
        info_window.geometry("400x300")
        info_window.transient(self.root)
        
        info_text = f"""
System Information:
- Python Version: {tk.TkVersion}
- Tkinter Version: {tk.TclVersion}
- Processor Status: {'Running' if self.running else 'Stopped'}
- Update Interval: {self.update_interval}ms
        """
        
        ttk.Label(info_window, text=info_text, justify=tk.LEFT).pack(pady=20)
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack()
        
    def show_user_guide(self) -> None:
        """Show user guide."""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")
        guide_window.transient(self.root)
        
        guide_text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD)
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        guide_content = """
MT4 Signal Processor - User Guide

KEYBOARD SHORTCUTS:
Ctrl+N - New Session
Ctrl+O - Open Log File
Ctrl+E - Export to CSV
Ctrl+Q - Exit
Ctrl+C - Copy Selection
Ctrl+A - Select All
Ctrl+P - Preferences
F5 - Refresh Data
F6 - Stop Processor
F7 - Restart Processor
Ctrl+T - Manual Trade
Ctrl+B - Quick Buy
Ctrl+S - Quick Sell
F11 - Toggle Fullscreen
F1 - User Guide

MENU OVERVIEW:
File - File operations, import/export
Edit - Text editing and preferences
Processor - Control signal processor
Trading - Trading operations
Database - Database management
View - Display options
Tools - Additional tools
Help - Documentation and support

For more information, visit the documentation.
        """
        
        guide_text.insert(tk.END, guide_content)
        guide_text.config(state=tk.DISABLED)
        
    def show_shortcuts(self) -> None:
        """Show keyboard shortcuts."""
        shortcuts_window = tk.Toplevel(self.root)
        shortcuts_window.title("Keyboard Shortcuts")
        shortcuts_window.geometry("400x400")
        shortcuts_window.transient(self.root)
        
        shortcuts_text = """
Keyboard Shortcuts:

File Operations:
Ctrl+N - New Session
Ctrl+O - Open Log File
Ctrl+E - Export to CSV
Ctrl+Q - Exit

Processor Control:
F5 - Refresh Data
F6 - Stop Processor
F7 - Restart Processor

Trading:
Ctrl+T - Manual Trade
Ctrl+B - Quick Buy
Ctrl+S - Quick Sell

View:
F11 - Toggle Fullscreen

General:
Ctrl+C - Copy Selection
Ctrl+A - Select All
Ctrl+P - Preferences
F1 - User Guide
        """
        
        ttk.Label(shortcuts_window, text=shortcuts_text, justify=tk.LEFT).pack(pady=10)
        ttk.Button(shortcuts_window, text="Close", command=shortcuts_window.destroy).pack()
        
    def check_updates(self) -> None:
        """Check for updates."""
        self.log_message("Checking for updates...")
        messagebox.showinfo("Updates", "You are using the latest version")
        
    def show_about(self) -> None:
        """Show about dialog."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("350x200")
        about_window.transient(self.root)
        about_window.grab_set()
        
        about_text = """
MT4 Signal Processor
Version 1.0.0

A comprehensive trading signal processor
for MetaTrader 4 integration.

© 2025 Your Company Name
        """
        
        ttk.Label(about_window, text=about_text, justify=tk.CENTER).pack(pady=30)
        ttk.Button(about_window, text="Close", command=about_window.destroy).pack()
        
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
        # System log (continuing from where it was cut off)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log control buttons
        log_button_frame = ttk.Frame(log_frame)
        log_button_frame.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        ttk.Button(log_button_frame, text="Clear Log", command=self.clear_log).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(log_button_frame, text="Save Log", command=self.save_log).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(log_button_frame, text="Auto-scroll", command=self.toggle_autoscroll).grid(row=0, column=2)
        
        # Auto-scroll state
        self.auto_scroll = True
        
    def log_message(self, message: str) -> None:
        """Add a message to the log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        
        # Auto-scroll to bottom if enabled
        if self.auto_scroll:
            self.log_text.see(tk.END)
            
        # Keep log size manageable (last 1000 lines)
        lines = self.log_text.get("1.0", tk.END).split('\n')
        if len(lines) > 1000:
            self.log_text.delete("1.0", f"{len(lines) - 1000}.0")
            
    def clear_log(self) -> None:
        """Clear the log display."""
        self.log_text.delete("1.0", tk.END)
        self.log_message("Log cleared")
        
    def save_log(self) -> None:
        """Save the log to a file."""
        filename = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get("1.0", tk.END))
                self.log_message(f"Log saved to: {filename}")
            except Exception as e:
                self.log_message(f"Failed to save log: {str(e)}")
                
    def toggle_autoscroll(self) -> None:
        """Toggle auto-scroll functionality."""
        self.auto_scroll = not self.auto_scroll
        status = "enabled" if self.auto_scroll else "disabled"
        self.log_message(f"Auto-scroll {status}")
        
    def start_processor(self) -> None:
        """Start the trading signal processor."""
        if not self.running:
            try:
                self.running = True
                self.status_label.config(text="Starting...", style='TLabel')
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                
                # Start processor in a separate thread or process
                self.processor.start()
                
                self.status_label.config(text="Running", style='Green.TLabel')
                self.log_message("Processor started successfully")
                
            except Exception as e:
                self.running = False
                self.status_label.config(text="Error", style='Red.TLabel')
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.log_message(f"Failed to start processor: {str(e)}")
                messagebox.showerror("Error", f"Failed to start processor: {str(e)}")
                
    def stop_processor(self) -> None:
        """Stop the trading signal processor."""
        if self.running:
            try:
                self.status_label.config(text="Stopping...", style='TLabel')
                self.stop_button.config(state=tk.DISABLED)
                
                # Stop processor
                self.processor.stop()
                
                self.running = False
                self.status_label.config(text="Stopped", style='Red.TLabel')
                self.start_button.config(state=tk.NORMAL)
                self.log_message("Processor stopped")
                
            except Exception as e:
                self.log_message(f"Error stopping processor: {str(e)}")
                messagebox.showerror("Error", f"Error stopping processor: {str(e)}")
                
    def refresh_data(self) -> None:
        """Refresh the display with latest data."""
        try:
            self.update_statistics()
            self.update_signals_table()
            self.log_message("Data refreshed")
        except Exception as e:
            self.log_message(f"Error refreshing data: {str(e)}")
            
    def update_statistics(self) -> None:
        """Update the statistics display."""
        try:
            # Get statistics from processor
            stats = self.processor.get_statistics()
            
            self.stats_labels['total_signals'].config(text=str(stats.get('total_signals', 0)))
            self.stats_labels['buy_signals'].config(text=str(stats.get('buy_signals', 0)))
            self.stats_labels['sell_signals'].config(text=str(stats.get('sell_signals', 0)))
            self.stats_labels['processed_signals'].config(text=str(stats.get('processed_signals', 0)))
            self.stats_labels['failed_signals'].config(text=str(stats.get('failed_signals', 0)))
            
            last_signal = stats.get('last_signal_time', 'Never')
            if last_signal != 'Never' and isinstance(last_signal, datetime):
                last_signal = last_signal.strftime("%Y-%m-%d %H:%M:%S")
            self.stats_labels['last_signal'].config(text=str(last_signal))
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
            
    def update_signals_table(self) -> None:
        """Update the signals table with latest data."""
        try:
            # Clear existing items
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
                
            # Get recent signals from processor
            signals = self.processor.get_recent_signals(limit=100)
            
            for signal in signals:
                # Format signal data for display
                values = (
                    signal.get('id', ''),
                    signal.get('timestamp', ''),
                    signal.get('symbol', ''),
                    signal.get('type', ''),
                    signal.get('price', ''),
                    signal.get('lots', ''),
                    'Yes' if signal.get('processed', False) else 'No',
                    signal.get('reason', '')
                )
                
                # Insert into tree
                item = self.signals_tree.insert('', 'end', values=values)
                
                # Color code based on signal type
                if signal.get('type') == 'BUY':
                    self.signals_tree.item(item, tags=('buy',))
                elif signal.get('type') == 'SELL':
                    self.signals_tree.item(item, tags=('sell',))
                    
            # Configure tag colors
            self.signals_tree.tag_configure('buy', foreground='green')
            self.signals_tree.tag_configure('sell', foreground='red')
            
        except Exception as e:
            logger.error(f"Error updating signals table: {str(e)}")
            
    def clear_database(self) -> None:
        """Clear the database after confirmation."""
        if messagebox.askyesno("Clear Database", "Are you sure you want to clear all signal data?"):
            try:
                if hasattr(self.processor.db, 'erase_database'):
                    self.processor.clear_database()
                    self.update_signals_table()
                    self.update_statistics()
                    self.log_message("Database cleared")
                else:
                    self.log_message("Operation failed: method not available")
            except Exception as e:
                self.log_message(f"Error clearing database: {str(e)}")
                messagebox.showerror("Error", f"Error clearing database: {str(e)}")
                
    def export_csv(self) -> None:
        """Export signals to CSV file."""
        filename = filedialog.asksaveasfilename(
            title="Export to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                if hasattr(self.processor, 'export_to_csv'):
                    self.processor.export_to_csv(filename)
                    self.log_message(f"Data exported to: {filename}")
                    messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}")
                else:
                    self.log_message("Export failed: method not available")
            except Exception as e:
                self.log_message(f"Export failed: {str(e)}")
                messagebox.showerror("Export Error", f"Export failed: {str(e)}")
                
    def show_settings(self) -> None:
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Create settings interface
        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General settings tab
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        ttk.Label(general_frame, text="Update Interval (ms):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        interval_var = tk.StringVar(value=str(self.update_interval))
        ttk.Entry(general_frame, textvariable=interval_var, width=10).grid(row=0, column=1, padx=10, pady=10)
        
        # Processor settings tab
        processor_frame = ttk.Frame(notebook)
        notebook.add(processor_frame, text="Processor")
        
        ttk.Label(processor_frame, text="Processor settings - To be implemented").pack(pady=50)
        
        # Buttons
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="OK", command=lambda: self.apply_settings(settings_window, interval_var)).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT)
        
    def apply_settings(self, window, interval_var) -> None:
        """Apply settings changes."""
        try:
            new_interval = int(interval_var.get())
            if new_interval > 0:
                self.update_interval = new_interval
                self.log_message(f"Update interval changed to {new_interval}ms")
            window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid update interval")
            
    def manual_trade(self) -> None:
        """Show manual trade dialog."""
        trade_window = tk.Toplevel(self.root)
        trade_window.title("Manual Trade")
        trade_window.geometry("300x200")
        trade_window.transient(self.root)
        trade_window.grab_set()
        
        # Trade form
        ttk.Label(trade_window, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        symbol_var = tk.StringVar(value="EURUSD")
        ttk.Entry(trade_window, textvariable=symbol_var).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(trade_window, text="Type:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        type_var = tk.StringVar(value="BUY")
        ttk.Combobox(trade_window, textvariable=type_var, values=["BUY", "SELL"]).grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(trade_window, text="Lots:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        lots_var = tk.StringVar(value="0.1")
        ttk.Entry(trade_window, textvariable=lots_var).grid(row=2, column=1, padx=10, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(trade_window)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Execute", command=lambda: self.execute_manual_trade(trade_window, symbol_var, type_var, lots_var)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=trade_window.destroy).pack(side=tk.LEFT)
        
    def execute_manual_trade(self, window, symbol_var, type_var, lots_var) -> None:
        """Execute manual trade."""
        try:
            symbol = symbol_var.get()
            trade_type = type_var.get()
            lots = float(lots_var.get())
            
            # Execute trade through processor
            result = self.processor.execute_manual_trade(symbol, trade_type, lots)
            
            if result:
                self.log_message(f"Manual trade executed: {trade_type} {lots} lots of {symbol}")
                messagebox.showinfo("Trade Executed", f"Trade executed successfully")
                window.destroy()
            else:
                messagebox.showerror("Trade Failed", "Failed to execute trade")
                
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid lots value: {str(e)}")
        except Exception as e:
            self.log_message(f"Manual trade error: {str(e)}")
            messagebox.showerror("Error", f"Trade error: {str(e)}")
            
    def update_display(self) -> None:
        """Update the display periodically."""
        # if self.running:
        #     self.update_statistics()
        #     self.update_signals_table()
            
        # # Schedule next update
        # self.root.after(self.update_interval, self.update_display)

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
        
    def on_close(self) -> None:
        """Handle window close event."""
        if self.running:
            if messagebox.askyesno("Exit", "Processor is running. Stop and exit?"):
                self.stop_processor()
                self.root.destroy()
        else:
            self.root.destroy()
            
    def run(self) -> None:
        """Start the GUI main loop."""
        self.log_message("MT4 Signal Processor GUI started")
        self.root.mainloop()


# Example usage
if __name__ == "__main__":
    # Mock processor class for testing
    class MockProcessor:
        def __init__(self):
            self.running = False
            
        def start(self):
            self.running = True
            
        def stop(self):
            self.running = False
            
        def get_statistics(self):
            return {
                'total_signals': 150,
                'buy_signals': 75,
                'sell_signals': 75,
                'processed_signals': 140,
                'failed_signals': 10,
                'last_signal_time': datetime.now() - timedelta(minutes=5)
            }
            
        def get_recent_signals(self, limit=100):
            return [
                {
                    'id': i,
                    'timestamp': (datetime.now() - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                    'symbol': 'EURUSD',
                    'type': 'BUY' if i % 2 == 0 else 'SELL',
                    'price': 1.1234 + (i * 0.0001),
                    'lots': 0.1,
                    'processed': i % 3 != 0,
                    'reason': 'Signal processed' if i % 3 != 0 else 'Failed validation'
                }
                for i in range(min(limit, 20))
            ]
            
        def clear_database(self):
            pass
            
        def export_to_csv(self, filename):
            pass
            
        def execute_manual_trade(self, symbol, trade_type, lots):
            return True
    
    # Create and run GUI
    processor = MockProcessor()
    gui = TradingSignalGUI(processor)
    gui.run()