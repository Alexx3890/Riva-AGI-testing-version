"""
Metrics collector for Riva-AGI.
Exposes standard Prometheus exposition format and JSON format.
Zero external heavy dependencies required.
"""

from collections import defaultdict
from datetime import datetime, timezone
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class MetricsCollector:
    """
    Thread-safe operational and MLOps metrics collector.
    Tracks HTTP, WebSocket, Gemini API, RAG, and system resource metrics.
    """

    def __init__(self, service_name: str = "riva-voice-gateway"):
        self.service_name = service_name
        self._lock = threading.Lock()
        self.start_time = time.time()

        # HTTP counters: (method, route, status) -> count
        self.http_requests_total: Dict[Tuple[str, str, int], int] = defaultdict(int)

        # HTTP Latencies: (method, route) -> list of recent latencies in ms
        self.http_latencies_ms: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        # WebSocket metrics
        self.active_websocket_sessions: int = 0
        self.ws_sessions_total: Dict[Tuple[str, str], int] = defaultdict(int)  # (voice, language) -> count
        self.ws_session_durations_s: List[float] = []

        # Gemini Call metrics: (model, operation, status) -> count
        self.gemini_calls_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.gemini_latencies_ms: List[float] = []

        # RAG retrieval metrics
        self.rag_queries_total: int = 0
        self.rag_latencies_ms: List[float] = []

        # Error counters: (error_class, route) -> count
        self.errors_total: Dict[Tuple[str, str], int] = defaultdict(int)

    def record_request(self, method: str, route: str, status_code: int, duration_ms: float) -> None:
        """Records an HTTP request outcome and latency."""
        with self._lock:
            key = (method.upper(), route, int(status_code))
            self.http_requests_total[key] += 1

            latency_key = (method.upper(), route)
            hist = self.http_latencies_ms[latency_key]
            hist.append(round(duration_ms, 2))
            # Keep sliding window of last 1000 measurements
            if len(hist) > 1000:
                hist.pop(0)

    def record_ws_session_start(self, voice: str = "default", language: str = "auto") -> None:
        """Records a new WebSocket connection."""
        with self._lock:
            self.active_websocket_sessions += 1
            self.ws_sessions_total[(voice, language)] += 1

    def record_ws_session_end(self, duration_s: float) -> None:
        """Records WebSocket disconnection and session duration."""
        with self._lock:
            self.active_websocket_sessions = max(0, self.active_websocket_sessions - 1)
            self.ws_session_durations_s.append(round(duration_s, 2))
            if len(self.ws_session_durations_s) > 1000:
                self.ws_session_durations_s.pop(0)

    def record_gemini_call(
        self,
        model: str,
        operation: str = "live_stream",
        success: bool = True,
        duration_ms: Optional[float] = None,
        error_class: Optional[str] = None,
    ) -> None:
        """Records a Gemini API call outcome and latency."""
        with self._lock:
            status = "success" if success else (error_class or "error")
            self.gemini_calls_total[(model, operation, status)] += 1
            if duration_ms is not None:
                self.gemini_latencies_ms.append(round(duration_ms, 2))
                if len(self.gemini_latencies_ms) > 1000:
                    self.gemini_latencies_ms.pop(0)

    def record_rag_query(self, duration_ms: float, result_count: int = 0) -> None:
        """Records RAG retrieval metrics."""
        with self._lock:
            self.rag_queries_total += 1
            self.rag_latencies_ms.append(round(duration_ms, 2))
            if len(self.rag_latencies_ms) > 1000:
                self.rag_latencies_ms.pop(0)

    def record_error(self, error_class: str, route: str = "general") -> None:
        """Increments error counter by class and route."""
        with self._lock:
            self.errors_total[(error_class, route)] += 1

    def _get_system_info(self) -> Dict[str, Any]:
        """Collects host CPU and RAM metrics without throwing exceptions."""
        info = {
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "cpu_percent": None,
            "memory_percent": None,
        }
        try:
            import psutil
            info["cpu_percent"] = psutil.cpu_percent(interval=None)
            info["memory_percent"] = psutil.virtual_memory().percent
        except Exception:
            pass
        return info

    def get_metrics_json(self) -> Dict[str, Any]:
        """Returns all collected metrics in structured JSON format."""
        with self._lock:
            http_summary = []
            for (method, route, status), count in self.http_requests_total.items():
                latencies = self.http_latencies_ms.get((method, route), [])
                avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
                http_summary.append({
                    "method": method,
                    "route": route,
                    "status_code": status,
                    "count": count,
                    "avg_latency_ms": avg_lat,
                })

            ws_summary = {
                "active_sessions": self.active_websocket_sessions,
                "total_sessions": sum(self.ws_sessions_total.values()),
                "sessions_by_voice_lang": [
                    {"voice": v, "language": l, "count": c}
                    for (v, l), c in self.ws_sessions_total.items()
                ],
                "avg_session_duration_s": (
                    round(sum(self.ws_session_durations_s) / len(self.ws_session_durations_s), 2)
                    if self.ws_session_durations_s else 0.0
                ),
            }

            gemini_summary = [
                {"model": m, "operation": op, "status": st, "count": cnt}
                for (m, op, st), cnt in self.gemini_calls_total.items()
            ]

            errors_summary = [
                {"error_class": ec, "route": rt, "count": cnt}
                for (ec, rt), cnt in self.errors_total.items()
            ]

            return {
                "service": self.service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": self._get_system_info(),
                "http_requests": http_summary,
                "websocket": ws_summary,
                "gemini": gemini_summary,
                "errors": errors_summary,
            }

    def get_prometheus_metrics(self) -> str:
        """Formats metrics according to the Prometheus text exposition standard."""
        lines: List[str] = [
            f"# HELP riva_service_uptime_seconds Total running uptime in seconds",
            f"# TYPE riva_service_uptime_seconds gauge",
            f'riva_service_uptime_seconds{{service="{self.service_name}"}} {round(time.time() - self.start_time, 1)}',
        ]

        sys_info = self._get_system_info()
        if sys_info.get("cpu_percent") is not None:
            lines.extend([
                "# HELP riva_system_cpu_percent Host CPU usage percentage",
                "# TYPE riva_system_cpu_percent gauge",
                f'riva_system_cpu_percent{{service="{self.service_name}"}} {sys_info["cpu_percent"]}',
            ])
        if sys_info.get("memory_percent") is not None:
            lines.extend([
                "# HELP riva_system_memory_percent Host memory usage percentage",
                "# TYPE riva_system_memory_percent gauge",
                f'riva_system_memory_percent{{service="{self.service_name}"}} {sys_info["memory_percent"]}',
            ])

        with self._lock:
            # HTTP Requests Total
            lines.append("# HELP riva_http_requests_total Total number of HTTP requests processed")
            lines.append("# TYPE riva_http_requests_total counter")
            for (method, route, status), count in self.http_requests_total.items():
                lines.append(
                    f'riva_http_requests_total{{service="{self.service_name}",method="{method}",route="{route}",status="{status}"}} {count}'
                )

            # Active WebSocket Sessions
            lines.append("# HELP riva_active_websocket_sessions Current active WebSocket voice sessions")
            lines.append("# TYPE riva_active_websocket_sessions gauge")
            lines.append(
                f'riva_active_websocket_sessions{{service="{self.service_name}"}} {self.active_websocket_sessions}'
            )

            # Total WebSocket Sessions
            lines.append("# HELP riva_websocket_sessions_total Total completed WebSocket voice sessions")
            lines.append("# TYPE riva_websocket_sessions_total counter")
            for (voice, lang), count in self.ws_sessions_total.items():
                lines.append(
                    f'riva_websocket_sessions_total{{service="{self.service_name}",voice="{voice}",language="{lang}"}} {count}'
                )

            # Gemini API Calls
            lines.append("# HELP riva_gemini_api_calls_total Total calls made to Gemini Live/Generative API")
            lines.append("# TYPE riva_gemini_api_calls_total counter")
            for (model, op, status), count in self.gemini_calls_total.items():
                lines.append(
                    f'riva_gemini_api_calls_total{{service="{self.service_name}",model="{model}",operation="{op}",status="{status}"}} {count}'
                )

            # Errors Total
            lines.append("# HELP riva_errors_total Total errors encountered by type and route")
            lines.append("# TYPE riva_errors_total counter")
            for (err_cls, rt), count in self.errors_total.items():
                lines.append(
                    f'riva_errors_total{{service="{self.service_name}",error_class="{err_cls}",route="{rt}"}} {count}'
                )

        return "\n".join(lines) + "\n"


# Singleton metrics instance
metrics = MetricsCollector(service_name=os.getenv("SERVICE_NAME", "riva-voice-gateway"))
