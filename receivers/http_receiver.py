"""
HTTP-based signal receiver using Flask web framework.
"""

import os
import time
import logging
import threading
from typing import List
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from models import MT4Signal

from flask import Flask, request, jsonify, abort
from werkzeug.serving import make_server


logger = logging.getLogger(__name__)

class HTTPSignalReceiver:
    """
    Advanced HTTP-based signal receiver with comprehensive API and security features.
    
    This class provides a robust HTTP server for receiving trading signals from
    MT4 Expert Advisors. It includes authentication, rate limiting, and comprehensive
    error handling.
    
    Attributes:
        app: Flask application instance
        port: HTTP server port
        signal_callback: Callback function for processed signals
        
    Methods:
        start(): Start HTTP server
        stop(): Stop HTTP server
        set_callback(): Set signal processing callback
        setup_routes(): Configure Flask routes
    """
    
    def __init__(self, port: int = 8080, auth_token: Optional[str] = None):
        """
        Initialize the HTTP signal receiver.
        
        Args:
            port: Port number for HTTP server
            auth_token: Optional authentication token for API security
            
        Raises:
            ValueError: If port is invalid
        """
        if not (1024 <= port <= 65535):
            raise ValueError(f"Invalid port number: {port}")
        
        self.port = port
        self.auth_token = auth_token
        self.signal_callback: Optional[Callable[[MT4Signal], None]] = None
        self.running = False
        self._server_thread: Optional[threading.Thread] = None
        
        # Initialize Flask app with security configurations
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.urandom(24)
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
        
        # Rate limiting storage
        self._rate_limit_storage = {}
        self._rate_limit_lock = threading.Lock()
        
        self.setup_routes()
        self.setup_error_handlers()
        
        logger.info(f"HTTP signal receiver initialized on port {port}")
    
    def set_callback(self, callback: Callable[[MT4Signal], None]) -> None:
        """
        Set the callback function for signal processing.
        
        Args:
            callback: Function to call when a signal is received
            
        Raises:
            TypeError: If callback is not callable
        """
        if not callable(callback):
            raise TypeError("Callback must be callable")
        
        self.signal_callback = callback
        logger.debug("Signal callback set for HTTP receiver")
    
    def _check_rate_limit(self, client_ip: str, limit: int = 100, window: int = 3600) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_ip: Client IP address
            limit: Maximum requests per window
            window: Time window in seconds
            
        Returns:
            bool: True if within rate limit, False if exceeded
        """
        current_time = time.time()
        
        with self._rate_limit_lock:
            if client_ip not in self._rate_limit_storage:
                self._rate_limit_storage[client_ip] = []
            
            # Clean old entries
            self._rate_limit_storage[client_ip] = [
                timestamp for timestamp in self._rate_limit_storage[client_ip]
                if current_time - timestamp < window
            ]
            
            # Check limit
            if len(self._rate_limit_storage[client_ip]) >= limit:
                return False
            
            # Add current request
            self._rate_limit_storage[client_ip].append(current_time)
            return True
    
    def _authenticate_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Authenticate incoming request.
        
        Args:
            request_data: Request data dictionary
            
        Returns:
            bool: True if authenticated, False otherwise
        """
        if not self.auth_token:
            return True  # No authentication required
        
        # Check for token in headers
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            return token == self.auth_token
        
        # Check for token in request data
        return request_data.get('auth_token') == self.auth_token
    
    def setup_routes(self) -> None:
        """
        Configure Flask routes for the HTTP API.
        
        Sets up endpoints for signal reception, status checking, and system health.
        """
        
        @self.app.route('/signal', methods=['POST'])
        def receive_signal():
            """
            Receive and process a trading signal via HTTP POST.
            
            Expected JSON format:
            {
                "timestamp": "2024-01-01 12:00:00",
                "symbol": "EURUSD",
                "signal_type": "BUY",
                "price": 1.2345,
                "lot_size": 0.1,
                "reason": "Signal reason",
                "stop_loss": 1.2300,
                "take_profit": 1.2400,
                "magic_number": 12345,
                "auth_token": "optional_auth_token"
            }
            
            Returns:
                JSON response with success/error status
            """
            try:
                # Check rate limit
                client_ip = request.remote_addr
                if not self._check_rate_limit(client_ip):
                    abort(429)  # Too Many Requests
                
                # Validate content type
                if not request.is_json:
                    return jsonify({
                        "status": "error",
                        "message": "Content-Type must be application/json"
                    }), 400
                
                data = request.get_json()
                if not data:
                    return jsonify({
                        "status": "error",
                        "message": "Invalid JSON data"
                    }), 400
                
                # Authenticate request
                if not self._authenticate_request(data):
                    return jsonify({
                        "status": "error",
                        "message": "Authentication failed"
                    }), 401
                
                # Create signal from data
                signal = MT4Signal(
                    timestamp=self._parse_timestamp(data.get('timestamp', '')),
                    symbol=data.get('symbol', '').upper(),
                    signal_type=data.get('signal_type', '').upper(),
                    price=float(data.get('price', 0)),
                    lot_size=float(data.get('lot_size', 0)),
                    reason=data.get('reason', ''),
                    stop_loss=float(data['stop_loss']) if data.get('stop_loss') else None,
                    take_profit=float(data['take_profit']) if data.get('take_profit') else None,
                    magic_number=int(data['magic_number']) if data.get('magic_number') else None
                )
                
                # Validate signal
                validation_result = signal.validate()
                if not validation_result.is_valid:
                    return jsonify({
                        "status": "error",
                        "message": f"Signal validation failed: {validation_result.error_message}"
                    }), 400
                
                # Process signal
                if self.signal_callback:
                    self.signal_callback(signal)
                
                response_data = {
                    "status": "success",
                    "message": "Signal received and processed",
                    "signal_id": f"{signal.symbol}_{signal.timestamp.isoformat()}",
                    "warnings": validation_result.warnings
                }
                
                if validation_result.warnings:
                    response_data["warnings"] = validation_result.warnings
                
                return jsonify(response_data), 200
                
            except ValueError as e:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid data format: {str(e)}"
                }), 400
            except Exception as e:
                logger.error(f"Error processing HTTP signal: {e}")
                return jsonify({
                    "status": "error",
                    "message": "Internal server error"
                }), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """
            Get server status and health information.
            
            Returns:
                JSON response with server status
            """
            return jsonify({
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "port": self.port,
                "version": "2.0.0",
                "auth_required": self.auth_token is not None
            })
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """
            Health check endpoint for monitoring systems.
            
            Returns:
                JSON response with health status
            """
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - getattr(self, '_start_time', time.time())
            })
        
        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            """
            Get server metrics for monitoring.
            
            Returns:
                JSON response with server metrics
            """
            with self._rate_limit_lock:
                active_clients = len(self._rate_limit_storage)
                total_requests = sum(len(requests) for requests in self._rate_limit_storage.values())
            
            return jsonify({
                "active_clients": active_clients,
                "total_requests": total_requests,
                "uptime": time.time() - getattr(self, '_start_time', time.time())
            })
    
    def setup_error_handlers(self) -> None:
        """
        Configure error handlers for the Flask application.
        """
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                "status": "error",
                "message": "Bad request"
            }), 400
        
        @self.app.errorhandler(401)
        def unauthorized(error):
            return jsonify({
                "status": "error",
                "message": "Unauthorized"
            }), 401
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                "status": "error",
                "message": "Endpoint not found"
            }), 404
        
        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify({
                "status": "error",
                "message": "Method not allowed"
            }), 405
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            return jsonify({
                "status": "error",
                "message": "Rate limit exceeded"
            }), 429
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                "status": "error",
                "message": "Internal server error"
            }), 500
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp string with multiple format support.
        
        Args:
            timestamp_str: Timestamp string in various formats
            
        Returns:
            datetime object
            
        Raises:
            ValueError: If timestamp cannot be parsed
        """
        if not timestamp_str:
            return datetime.now()
        
        formats = [
            '%Y.%m.%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y.%m.%d %H:%M',
            '%Y-%m-%d %H:%M',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        try:
            return datetime.fromisoformat(timestamp_str.replace('.', '-'))
        except ValueError:
            pass
        
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")
    
    def start(self) -> None:
        """
        Start the HTTP server in a separate thread.
        
        Creates a daemon thread that runs the Flask development server.
        For production use, consider using a proper WSGI server like Gunicorn.
        """
        if self.running:
            logger.warning("HTTP receiver is already running")
            return
        
        self.running = True
        self._start_time = time.time()
        
        def run_server():
            """Run the Flask server in the background thread."""
            try:
                # Disable Flask's request logging for cleaner output
                logging.getLogger('werkzeug').setLevel(logging.WARNING)
                
                self.app.run(
                    host='0.0.0.0',
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e:
                logger.error(f"HTTP server error: {e}")
                self.running = False
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"HTTP signal receiver started on port {self.port}")
    
    def stop(self) -> None:
        """
        Stop the HTTP server gracefully.
        
        Note: Flask's development server doesn't support graceful shutdown.
        For production, use a proper WSGI server with shutdown capabilities.
        """
        if not self.running:
            return
        
        self.running = False
        logger.info("HTTP signal receiver stopped")