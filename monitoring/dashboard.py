"""HTTP health check & real-time monitoring dashboard endpoint.

Provides a lightweight HTTP server that exposes system health, positions,
PnL, risk metrics, and alpha attribution as JSON endpoints.

Endpoints:
    GET /health          → System health check (200/503)
    GET /status          → Full system status (JSON)
    GET /positions       → Current positions
    GET /risk            → Risk metrics snapshot
    GET /alpha           → Alpha attribution summary
    GET /execution       → Execution quality metrics
    GET /alerts          → Recent alerts

Usage:
    dashboard = MonitoringDashboard(port=8080)
    dashboard.register_components(
        risk_manager=portfolio_risk_mgr,
        execution_analytics=exec_analytics,
        alpha_attribution=attribution,
    )
    dashboard.start()  # Starts in background thread
"""

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for monitoring endpoints."""

    # These are set by the MonitoringDashboard class
    _status_fn: Optional[Callable] = None
    _custom_handlers: Dict[str, Callable] = {}

    def do_GET(self):
        path = self.path.rstrip('/')

        # Route to handler
        if path == '/health':
            self._handle_health()
        elif path == '/status':
            self._handle_status()
        elif path in self._custom_handlers:
            self._handle_custom(path)
        else:
            self._send_json(404, {"error": "Not found", "available": [
                "/health", "/status", "/positions", "/risk",
                "/alpha", "/execution", "/alerts",
            ]})

    def _handle_health(self):
        if self._status_fn:
            try:
                status = self._status_fn()
                is_healthy = status.get('system_healthy', True)
                code = 200 if is_healthy else 503
                self._send_json(code, {
                    "status": "healthy" if is_healthy else "degraded",
                    "timestamp": datetime.now().isoformat(),
                    "uptime_s": status.get('uptime_s', 0),
                })
            except Exception as e:
                self._send_json(503, {"status": "error", "message": str(e)})
        else:
            self._send_json(200, {"status": "healthy", "timestamp": datetime.now().isoformat()})

    def _handle_status(self):
        if self._status_fn:
            try:
                status = self._status_fn()
                self._send_json(200, status)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(200, {"message": "No status provider registered"})

    def _handle_custom(self, path: str):
        try:
            handler = self._custom_handlers[path]
            data = handler()
            self._send_json(200, data)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _send_json(self, code: int, data: Dict):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        body = json.dumps(data, default=str, indent=2)
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        # Suppress default server logging
        pass


class MonitoringDashboard:
    """Lightweight HTTP monitoring dashboard."""

    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._start_time = time.time()

        # Component references
        self._risk_manager = None
        self._execution_analytics = None
        self._alpha_attribution = None
        self._alpha_ensemble = None
        self._data_quality = None
        self._trade_db = None
        self._strategy = None

    def register_components(
        self,
        risk_manager=None,
        execution_analytics=None,
        alpha_attribution=None,
        alpha_ensemble=None,
        data_quality=None,
        trade_db=None,
        strategy=None,
    ) -> None:
        """Register system components for monitoring."""
        self._risk_manager = risk_manager
        self._execution_analytics = execution_analytics
        self._alpha_attribution = alpha_attribution
        self._alpha_ensemble = alpha_ensemble
        self._data_quality = data_quality
        self._trade_db = trade_db
        self._strategy = strategy

    def start(self) -> None:
        """Start the monitoring HTTP server in a background thread."""
        # Set up handler
        handler = type('Handler', (_DashboardHandler,), {
            '_status_fn': self._get_full_status,
            '_custom_handlers': {
                '/positions': self._get_positions,
                '/risk': self._get_risk,
                '/alpha': self._get_alpha,
                '/execution': self._get_execution,
                '/alerts': self._get_alerts,
                '/data_quality': self._get_data_quality,
            },
        })

        try:
            self._server = HTTPServer((self.host, self.port), handler)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="monitoring-dashboard",
            )
            self._thread.start()
            logger.info(f"Monitoring dashboard started: http://{self.host}:{self.port}")
        except OSError as e:
            logger.warning(f"Failed to start monitoring dashboard on port {self.port}: {e}")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("Monitoring dashboard stopped")

    # ── Data Providers ──

    def _get_full_status(self) -> Dict:
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_s': time.time() - self._start_time,
            'system_healthy': True,
        }

        try:
            # Risk
            if self._risk_manager:
                risk_status = self._risk_manager.get_status()
                status['risk'] = risk_status
                if risk_status.get('risk_regime') == 'CRISIS':
                    status['system_healthy'] = False

            # Alpha
            if self._alpha_ensemble:
                status['alpha'] = self._alpha_ensemble.get_metrics()

            # Attribution
            if self._alpha_attribution:
                status['attribution'] = self._alpha_attribution.get_metrics()

            # Execution
            if self._execution_analytics:
                status['execution'] = self._execution_analytics.generate_report(recent_n=100)

            # Data quality
            if self._data_quality:
                status['data_quality'] = self._data_quality.get_summary()

            # Strategy
            if self._strategy and hasattr(self._strategy, 'get_system_status'):
                status['strategy'] = self._strategy.get_system_status()

        except Exception as e:
            status['error'] = str(e)
            status['system_healthy'] = False

        return status

    def _get_positions(self) -> Dict:
        if self._risk_manager:
            return {
                'positions': self._risk_manager._positions,
                'prices': self._risk_manager._position_prices,
                'exposure': self._risk_manager.get_exposure(),
            }
        if self._trade_db:
            return {'positions': self._trade_db.get_all_positions()}
        return {'positions': {}}

    def _get_risk(self) -> Dict:
        if self._risk_manager:
            return self._risk_manager.get_status()
        return {'message': 'No risk manager registered'}

    def _get_alpha(self) -> Dict:
        result = {}
        if self._alpha_ensemble:
            result['ensemble'] = self._alpha_ensemble.get_metrics()
        if self._alpha_attribution:
            result['attribution'] = self._alpha_attribution.get_metrics()
        return result or {'message': 'No alpha components registered'}

    def _get_execution(self) -> Dict:
        if self._execution_analytics:
            return self._execution_analytics.generate_report()
        return {'message': 'No execution analytics registered'}

    def _get_alerts(self) -> Dict:
        if self._trade_db:
            return {'alerts': self._trade_db.get_recent_alerts(50)}
        return {'alerts': []}

    def _get_data_quality(self) -> Dict:
        if self._data_quality:
            return self._data_quality.get_summary()
        return {'message': 'No data quality sentinel registered'}
