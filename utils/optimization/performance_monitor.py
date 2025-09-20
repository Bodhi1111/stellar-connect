"""
Performance monitoring and bottleneck identification system.
Comprehensive monitoring of system performance with real-time analytics.
"""

import time
import threading
import asyncio
import psutil
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging
import json
from datetime import datetime, timedelta
import functools
import inspect


class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CUSTOM = "custom"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    name: str
    metric_type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckInfo:
    component: str
    metric_type: MetricType
    severity: AlertLevel
    description: str
    current_value: float
    threshold: float
    impact_score: float
    recommendations: List[str]
    timestamp: float


@dataclass
class PerformanceAlert:
    alert_id: str
    component: str
    alert_level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


class PerformanceThresholds:
    """Performance threshold configuration."""

    def __init__(self):
        self.thresholds = {
            MetricType.LATENCY: {
                AlertLevel.WARNING: 1000,  # ms
                AlertLevel.ERROR: 5000,
                AlertLevel.CRITICAL: 10000
            },
            MetricType.CPU_USAGE: {
                AlertLevel.WARNING: 70,  # %
                AlertLevel.ERROR: 85,
                AlertLevel.CRITICAL: 95
            },
            MetricType.MEMORY_USAGE: {
                AlertLevel.WARNING: 75,  # %
                AlertLevel.ERROR: 85,
                AlertLevel.CRITICAL: 95
            },
            MetricType.ERROR_RATE: {
                AlertLevel.WARNING: 5,  # %
                AlertLevel.ERROR: 10,
                AlertLevel.CRITICAL: 20
            },
            MetricType.THROUGHPUT: {
                AlertLevel.WARNING: 10,  # requests/sec (minimum)
                AlertLevel.ERROR: 5,
                AlertLevel.CRITICAL: 1
            }
        }

    def get_threshold(self, metric_type: MetricType, alert_level: AlertLevel) -> Optional[float]:
        """Get threshold value for metric type and alert level."""
        return self.thresholds.get(metric_type, {}).get(alert_level)

    def set_threshold(self, metric_type: MetricType, alert_level: AlertLevel, value: float):
        """Set custom threshold value."""
        if metric_type not in self.thresholds:
            self.thresholds[metric_type] = {}
        self.thresholds[metric_type][alert_level] = value


class PerformanceMonitor:
    """Comprehensive performance monitoring and bottleneck identification system."""

    def __init__(self,
                 buffer_size: int = 10000,
                 monitoring_interval: float = 5.0,
                 alert_cooldown: float = 300.0):
        """
        Initialize performance monitor.

        Args:
            buffer_size: Maximum number of metrics to keep in memory
            monitoring_interval: Interval in seconds between system metric collections
            alert_cooldown: Minimum seconds between duplicate alerts
        """
        self.buffer_size = buffer_size
        self.monitoring_interval = monitoring_interval
        self.alert_cooldown = alert_cooldown

        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.function_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.bottlenecks: List[BottleneckInfo] = []
        self.alerts: List[PerformanceAlert] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}

        # Configuration
        self.thresholds = PerformanceThresholds()
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        # Component tracking
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.dependency_map: Dict[str, List[str]] = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("Performance Monitor initialized")

    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="performance_monitor",
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._detect_bottlenecks()
                self._update_component_health()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        timestamp = time.time()

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu_usage", MetricType.CPU_USAGE, cpu_percent, timestamp)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_usage", MetricType.MEMORY_USAGE, memory.percent, timestamp)
            self.record_metric("system.memory_available", MetricType.MEMORY_USAGE, memory.available / (1024**3), timestamp, {"unit": "GB"})

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.record_metric("system.disk_read_bytes", MetricType.DISK_IO, disk_io.read_bytes, timestamp)
                self.record_metric("system.disk_write_bytes", MetricType.DISK_IO, disk_io.write_bytes, timestamp)

            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if network_io:
                self.record_metric("system.network_sent_bytes", MetricType.NETWORK_IO, network_io.bytes_sent, timestamp)
                self.record_metric("system.network_recv_bytes", MetricType.NETWORK_IO, network_io.bytes_recv, timestamp)

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            self.record_metric("process.memory_usage", MetricType.MEMORY_USAGE, process_memory, timestamp, {"unit": "MB"})

            cpu_times = process.cpu_times()
            self.record_metric("process.cpu_user_time", MetricType.CPU_USAGE, cpu_times.user, timestamp)
            self.record_metric("process.cpu_system_time", MetricType.CPU_USAGE, cpu_times.system, timestamp)

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def record_metric(self,
                     name: str,
                     metric_type: MetricType,
                     value: float,
                     timestamp: Optional[float] = None,
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = time.time()

        metric = PerformanceMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            tags=tags or {},
            metadata=metadata or {}
        )

        with self.lock:
            self.metrics[name].append(metric)
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    "metric_type": metric_type,
                    "first_recorded": timestamp,
                    "count": 0
                }
            self.metric_metadata[name]["count"] += 1
            self.metric_metadata[name]["last_recorded"] = timestamp

        # Check for alerts
        self._check_metric_thresholds(name, metric)

    def _check_metric_thresholds(self, metric_name: str, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and generate alerts."""
        for alert_level in [AlertLevel.CRITICAL, AlertLevel.ERROR, AlertLevel.WARNING]:
            threshold = self.thresholds.get_threshold(metric.metric_type, alert_level)
            if threshold is None:
                continue

            # Different comparison logic for different metric types
            exceeds_threshold = False
            if metric.metric_type in [MetricType.LATENCY, MetricType.CPU_USAGE, MetricType.MEMORY_USAGE, MetricType.ERROR_RATE]:
                exceeds_threshold = metric.value > threshold
            elif metric.metric_type == MetricType.THROUGHPUT:
                exceeds_threshold = metric.value < threshold

            if exceeds_threshold:
                self._generate_alert(metric_name, alert_level, metric, threshold)
                break  # Only generate highest severity alert

    def _generate_alert(self,
                       metric_name: str,
                       alert_level: AlertLevel,
                       metric: PerformanceMetric,
                       threshold: float):
        """Generate performance alert."""
        alert_key = f"{metric_name}_{alert_level.value}"

        # Check cooldown
        if alert_key in self.active_alerts:
            last_alert_time = self.active_alerts[alert_key].timestamp
            if time.time() - last_alert_time < self.alert_cooldown:
                return

        # Create alert
        alert = PerformanceAlert(
            alert_id=f"alert_{int(time.time())}_{hash(alert_key) % 10000}",
            component=metric_name.split('.')[0] if '.' in metric_name else "system",
            alert_level=alert_level,
            message=f"{metric_name} {alert_level.value}: {metric.value:.2f} exceeds threshold {threshold}",
            metric_name=metric_name,
            current_value=metric.value,
            threshold=threshold,
            timestamp=metric.timestamp
        )

        with self.lock:
            self.alerts.append(alert)
            self.active_alerts[alert_key] = alert

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        self.logger.warning(f"Performance alert: {alert.message}")

    def _detect_bottlenecks(self):
        """Detect system bottlenecks based on metrics."""
        timestamp = time.time()
        detected_bottlenecks = []

        # Analyze recent metrics for bottleneck patterns
        with self.lock:
            for metric_name, metric_queue in self.metrics.items():
                if len(metric_queue) < 10:  # Need sufficient data
                    continue

                recent_metrics = list(metric_queue)[-10:]  # Last 10 measurements
                values = [m.value for m in recent_metrics]

                if not values:
                    continue

                metric_type = recent_metrics[0].metric_type
                avg_value = statistics.mean(values)
                max_value = max(values)
                trend = self._calculate_trend(values)

                # Detect specific bottleneck patterns
                bottleneck = self._analyze_bottleneck_pattern(
                    metric_name, metric_type, avg_value, max_value, trend, timestamp
                )
                if bottleneck:
                    detected_bottlenecks.append(bottleneck)

        # Update bottleneck list
        with self.lock:
            # Remove old bottlenecks (older than 5 minutes)
            cutoff_time = timestamp - 300
            self.bottlenecks = [b for b in self.bottlenecks if b.timestamp > cutoff_time]

            # Add new bottlenecks
            for bottleneck in detected_bottlenecks:
                self.bottlenecks.append(bottleneck)

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction of values (-1 to 1)."""
        if len(values) < 2:
            return 0.0

        # Simple linear trend calculation
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))

        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Normalize slope to [-1, 1] range
        max_possible_slope = max(values) - min(values) if max(values) != min(values) else 1
        return max(-1, min(1, slope / max_possible_slope)) if max_possible_slope > 0 else 0

    def _analyze_bottleneck_pattern(self,
                                   metric_name: str,
                                   metric_type: MetricType,
                                   avg_value: float,
                                   max_value: float,
                                   trend: float,
                                   timestamp: float) -> Optional[BottleneckInfo]:
        """Analyze metric pattern for bottleneck indicators."""
        component = metric_name.split('.')[0] if '.' in metric_name else "system"

        # CPU bottleneck detection
        if metric_type == MetricType.CPU_USAGE and avg_value > 80:
            severity = AlertLevel.CRITICAL if avg_value > 95 else AlertLevel.ERROR if avg_value > 85 else AlertLevel.WARNING
            return BottleneckInfo(
                component=component,
                metric_type=metric_type,
                severity=severity,
                description=f"High CPU usage detected: {avg_value:.1f}%",
                current_value=avg_value,
                threshold=80,
                impact_score=min(1.0, avg_value / 100),
                recommendations=[
                    "Check for CPU-intensive processes",
                    "Consider scaling horizontally",
                    "Optimize algorithm efficiency",
                    "Implement caching strategies"
                ],
                timestamp=timestamp
            )

        # Memory bottleneck detection
        elif metric_type == MetricType.MEMORY_USAGE and avg_value > 85:
            severity = AlertLevel.CRITICAL if avg_value > 95 else AlertLevel.ERROR
            return BottleneckInfo(
                component=component,
                metric_type=metric_type,
                severity=severity,
                description=f"High memory usage detected: {avg_value:.1f}%",
                current_value=avg_value,
                threshold=85,
                impact_score=min(1.0, avg_value / 100),
                recommendations=[
                    "Implement memory optimization",
                    "Check for memory leaks",
                    "Use memory pooling",
                    "Consider increasing memory limits"
                ],
                timestamp=timestamp
            )

        # Latency bottleneck detection
        elif metric_type == MetricType.LATENCY and avg_value > 1000:
            severity = AlertLevel.CRITICAL if avg_value > 10000 else AlertLevel.ERROR if avg_value > 5000 else AlertLevel.WARNING
            return BottleneckInfo(
                component=component,
                metric_type=metric_type,
                severity=severity,
                description=f"High latency detected: {avg_value:.1f}ms",
                current_value=avg_value,
                threshold=1000,
                impact_score=min(1.0, avg_value / 10000),
                recommendations=[
                    "Optimize database queries",
                    "Implement caching",
                    "Check network connectivity",
                    "Review algorithm complexity"
                ],
                timestamp=timestamp
            )

        # Throughput bottleneck detection
        elif metric_type == MetricType.THROUGHPUT and avg_value < 10 and trend < -0.5:
            severity = AlertLevel.CRITICAL if avg_value < 1 else AlertLevel.ERROR if avg_value < 5 else AlertLevel.WARNING
            return BottleneckInfo(
                component=component,
                metric_type=metric_type,
                severity=severity,
                description=f"Low throughput detected: {avg_value:.1f} ops/sec",
                current_value=avg_value,
                threshold=10,
                impact_score=max(0.0, 1.0 - avg_value / 10),
                recommendations=[
                    "Scale up processing capacity",
                    "Optimize processing pipeline",
                    "Check for blocking operations",
                    "Implement parallel processing"
                ],
                timestamp=timestamp
            )

        return None

    def _update_component_health(self):
        """Update overall health status of system components."""
        timestamp = time.time()

        with self.lock:
            # Identify unique components
            components = set()
            for metric_name in self.metrics.keys():
                component = metric_name.split('.')[0] if '.' in metric_name else "system"
                components.add(component)

            # Calculate health for each component
            for component in components:
                component_metrics = {
                    name: queue for name, queue in self.metrics.items()
                    if name.startswith(f"{component}.")
                }

                if not component_metrics:
                    continue

                # Calculate component health score
                health_score = self._calculate_component_health_score(component_metrics)

                # Count recent alerts for this component
                recent_alerts = [
                    alert for alert in self.alerts
                    if alert.component == component and timestamp - alert.timestamp < 300
                ]

                # Determine health status
                if health_score > 0.8 and len(recent_alerts) == 0:
                    status = "healthy"
                elif health_score > 0.6 and len(recent_alerts) < 3:
                    status = "warning"
                elif health_score > 0.4:
                    status = "degraded"
                else:
                    status = "critical"

                self.component_health[component] = {
                    "status": status,
                    "health_score": health_score,
                    "recent_alerts": len(recent_alerts),
                    "last_updated": timestamp,
                    "metrics_count": sum(len(queue) for queue in component_metrics.values())
                }

    def _calculate_component_health_score(self, component_metrics: Dict[str, deque]) -> float:
        """Calculate health score for a component based on its metrics."""
        if not component_metrics:
            return 1.0

        health_scores = []

        for metric_name, metric_queue in component_metrics.items():
            if len(metric_queue) == 0:
                continue

            recent_metrics = list(metric_queue)[-5:]  # Last 5 measurements
            metric_type = recent_metrics[0].metric_type

            # Calculate metric health based on type
            values = [m.value for m in recent_metrics]
            avg_value = statistics.mean(values)

            # Score based on thresholds
            metric_score = 1.0
            for alert_level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
                threshold = self.thresholds.get_threshold(metric_type, alert_level)
                if threshold is None:
                    continue

                if metric_type in [MetricType.LATENCY, MetricType.CPU_USAGE, MetricType.MEMORY_USAGE, MetricType.ERROR_RATE]:
                    if avg_value > threshold:
                        metric_score = max(0.0, 1.0 - (avg_value / threshold - 1.0))
                        break
                elif metric_type == MetricType.THROUGHPUT:
                    if avg_value < threshold:
                        metric_score = max(0.0, avg_value / threshold)
                        break

            health_scores.append(metric_score)

        return statistics.mean(health_scores) if health_scores else 1.0

    def monitor_function(self, func_name: Optional[str] = None):
        """Decorator to monitor function performance."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                error_occurred = False

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_occurred = True
                    raise
                finally:
                    end_time = time.time()
                    duration = (end_time - start_time) * 1000  # Convert to milliseconds

                    # Record timing
                    self.function_timings[name].append({
                        'duration': duration,
                        'timestamp': end_time,
                        'error': error_occurred
                    })

                    # Record metrics
                    self.record_metric(
                        f"function.{name}.latency",
                        MetricType.LATENCY,
                        duration,
                        end_time,
                        {"function": name}
                    )

                    if error_occurred:
                        self.record_metric(
                            f"function.{name}.error_rate",
                            MetricType.ERROR_RATE,
                            1.0,
                            end_time,
                            {"function": name}
                        )

            return wrapper
        return decorator

    def monitor_async_function(self, func_name: Optional[str] = None):
        """Decorator to monitor async function performance."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                error_occurred = False

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_occurred = True
                    raise
                finally:
                    end_time = time.time()
                    duration = (end_time - start_time) * 1000

                    # Record timing
                    self.function_timings[name].append({
                        'duration': duration,
                        'timestamp': end_time,
                        'error': error_occurred
                    })

                    # Record metrics
                    self.record_metric(
                        f"function.{name}.latency",
                        MetricType.LATENCY,
                        duration,
                        end_time,
                        {"function": name}
                    )

                    if error_occurred:
                        self.record_metric(
                            f"function.{name}.error_rate",
                            MetricType.ERROR_RATE,
                            1.0,
                            end_time,
                            {"function": name}
                        )

            return wrapper
        return decorator

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)

    def get_metrics_summary(self, component: Optional[str] = None, time_window: float = 300) -> Dict[str, Any]:
        """Get summary of metrics for a component or entire system."""
        cutoff_time = time.time() - time_window
        summary = {
            "time_window_seconds": time_window,
            "metrics": {},
            "component_health": {},
            "active_alerts": len(self.active_alerts),
            "recent_bottlenecks": []
        }

        with self.lock:
            # Filter metrics by component if specified
            filtered_metrics = self.metrics
            if component:
                filtered_metrics = {
                    name: queue for name, queue in self.metrics.items()
                    if name.startswith(f"{component}.")
                }

            # Summarize metrics
            for metric_name, metric_queue in filtered_metrics.items():
                recent_metrics = [m for m in metric_queue if m.timestamp > cutoff_time]
                if not recent_metrics:
                    continue

                values = [m.value for m in recent_metrics]
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else None,
                    "metric_type": recent_metrics[0].metric_type.value
                }

                if len(values) > 1:
                    summary["metrics"][metric_name]["std_dev"] = statistics.stdev(values)

            # Add component health
            if component:
                summary["component_health"] = self.component_health.get(component, {})
            else:
                summary["component_health"] = self.component_health.copy()

            # Add recent bottlenecks
            recent_bottlenecks = [
                {
                    "component": b.component,
                    "severity": b.severity.value,
                    "description": b.description,
                    "impact_score": b.impact_score,
                    "timestamp": b.timestamp
                }
                for b in self.bottlenecks
                if b.timestamp > cutoff_time
            ]
            summary["recent_bottlenecks"] = recent_bottlenecks

        return summary

    def get_function_performance_summary(self, time_window: float = 300) -> Dict[str, Any]:
        """Get performance summary for monitored functions."""
        cutoff_time = time.time() - time_window
        summary = {}

        with self.lock:
            for func_name, timings in self.function_timings.items():
                recent_timings = [t for t in timings if t['timestamp'] > cutoff_time]
                if not recent_timings:
                    continue

                durations = [t['duration'] for t in recent_timings]
                error_count = sum(1 for t in recent_timings if t['error'])

                summary[func_name] = {
                    "call_count": len(recent_timings),
                    "average_latency_ms": statistics.mean(durations),
                    "min_latency_ms": min(durations),
                    "max_latency_ms": max(durations),
                    "error_count": error_count,
                    "error_rate": error_count / len(recent_timings) if recent_timings else 0,
                    "calls_per_second": len(recent_timings) / time_window
                }

                if len(durations) > 1:
                    summary[func_name]["latency_std_dev"] = statistics.stdev(durations)

                # Calculate percentiles
                if len(durations) >= 10:
                    sorted_durations = sorted(durations)
                    summary[func_name]["p50_latency_ms"] = sorted_durations[len(durations) // 2]
                    summary[func_name]["p95_latency_ms"] = sorted_durations[int(len(durations) * 0.95)]
                    summary[func_name]["p99_latency_ms"] = sorted_durations[int(len(durations) * 0.99)]

        return summary

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        summary = self.get_metrics_summary()
        function_summary = self.get_function_performance_summary()

        report = []
        report.append("# Performance Monitoring Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # System Overview
        report.append("## System Overview")
        report.append(f"- Active Alerts: {summary['active_alerts']}")
        report.append(f"- Recent Bottlenecks: {len(summary['recent_bottlenecks'])}")
        report.append(f"- Monitored Components: {len(summary['component_health'])}")
        report.append("")

        # Component Health
        if summary['component_health']:
            report.append("## Component Health")
            for component, health in summary['component_health'].items():
                status_emoji = {
                    "healthy": "✅",
                    "warning": "⚠️",
                    "degraded": "🔸",
                    "critical": "🔴"
                }.get(health.get("status", "unknown"), "❓")

                report.append(f"### {status_emoji} {component}")
                report.append(f"- Status: {health.get('status', 'unknown')}")
                report.append(f"- Health Score: {health.get('health_score', 0):.2f}")
                report.append(f"- Recent Alerts: {health.get('recent_alerts', 0)}")
                report.append("")

        # Key Metrics
        if summary['metrics']:
            report.append("## Key Performance Metrics")
            for metric_name, metric_data in summary['metrics'].items():
                if 'system' in metric_name or 'process' in metric_name:
                    report.append(f"### {metric_name}")
                    report.append(f"- Current: {metric_data['latest']:.2f}")
                    report.append(f"- Average: {metric_data['average']:.2f}")
                    report.append(f"- Peak: {metric_data['max']:.2f}")
                    report.append("")

        # Bottlenecks
        if summary['recent_bottlenecks']:
            report.append("## Detected Bottlenecks")
            for bottleneck in summary['recent_bottlenecks']:
                severity_emoji = {
                    "warning": "⚠️",
                    "error": "❌",
                    "critical": "🔴"
                }.get(bottleneck['severity'], "❓")

                report.append(f"### {severity_emoji} {bottleneck['component']}")
                report.append(f"- **Severity**: {bottleneck['severity']}")
                report.append(f"- **Issue**: {bottleneck['description']}")
                report.append(f"- **Impact Score**: {bottleneck['impact_score']:.2f}")
                report.append("")

        # Function Performance
        if function_summary:
            report.append("## Function Performance")
            # Sort by average latency (highest first)
            sorted_functions = sorted(
                function_summary.items(),
                key=lambda x: x[1]['average_latency_ms'],
                reverse=True
            )

            for func_name, func_data in sorted_functions[:10]:  # Top 10 slowest
                report.append(f"### {func_name}")
                report.append(f"- Calls: {func_data['call_count']}")
                report.append(f"- Avg Latency: {func_data['average_latency_ms']:.1f}ms")
                report.append(f"- Error Rate: {func_data['error_rate']*100:.1f}%")

                if 'p95_latency_ms' in func_data:
                    report.append(f"- P95 Latency: {func_data['p95_latency_ms']:.1f}ms")
                report.append("")

        # Recommendations
        report.append("## Recommendations")
        if summary['recent_bottlenecks']:
            critical_bottlenecks = [b for b in summary['recent_bottlenecks'] if b['severity'] == 'critical']
            if critical_bottlenecks:
                report.append("### Critical Issues")
                for bottleneck in critical_bottlenecks:
                    report.append(f"- **{bottleneck['component']}**: {bottleneck['description']}")
        else:
            report.append("✅ No critical performance issues detected")

        report.append("\n### General Recommendations")
        report.append("- Monitor trends over time for capacity planning")
        report.append("- Set up automated alerts for critical thresholds")
        report.append("- Review function performance regularly for optimization opportunities")
        report.append("- Consider implementing performance budgets for key operations")

        return "\n".join(report)

    def export_metrics(self, format_type: str = "json", time_window: float = 3600) -> str:
        """Export metrics data."""
        if format_type.lower() == "json":
            cutoff_time = time.time() - time_window

            exportable = {
                "timestamp": time.time(),
                "time_window_seconds": time_window,
                "metrics": {},
                "alerts": [],
                "bottlenecks": [],
                "component_health": self.component_health.copy()
            }

            with self.lock:
                # Export metrics
                for metric_name, metric_queue in self.metrics.items():
                    recent_metrics = [
                        {
                            "value": m.value,
                            "timestamp": m.timestamp,
                            "tags": m.tags,
                            "metadata": m.metadata
                        }
                        for m in metric_queue if m.timestamp > cutoff_time
                    ]
                    if recent_metrics:
                        exportable["metrics"][metric_name] = recent_metrics

                # Export recent alerts
                exportable["alerts"] = [
                    {
                        "alert_id": alert.alert_id,
                        "component": alert.component,
                        "level": alert.alert_level.value,
                        "message": alert.message,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp,
                        "resolved": alert.resolved
                    }
                    for alert in self.alerts if alert.timestamp > cutoff_time
                ]

                # Export bottlenecks
                exportable["bottlenecks"] = [
                    {
                        "component": b.component,
                        "metric_type": b.metric_type.value,
                        "severity": b.severity.value,
                        "description": b.description,
                        "current_value": b.current_value,
                        "threshold": b.threshold,
                        "impact_score": b.impact_score,
                        "recommendations": b.recommendations,
                        "timestamp": b.timestamp
                    }
                    for b in self.bottlenecks if b.timestamp > cutoff_time
                ]

            return json.dumps(exportable, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
        _global_performance_monitor.start_monitoring()
    return _global_performance_monitor


def monitor_performance(func_name: Optional[str] = None):
    """Decorator to monitor function performance using global monitor."""
    monitor = get_performance_monitor()
    return monitor.monitor_function(func_name)


def monitor_async_performance(func_name: Optional[str] = None):
    """Decorator to monitor async function performance using global monitor."""
    monitor = get_performance_monitor()
    return monitor.monitor_async_function(func_name)