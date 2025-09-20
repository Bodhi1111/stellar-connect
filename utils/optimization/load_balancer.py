"""
Load balancing system for specialist agents.
Intelligent distribution of workload across specialist agents with health monitoring.
"""

import asyncio
import time
import threading
import random
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging
import json
from datetime import datetime
import statistics
import weakref


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    INTELLIGENT = "intelligent"
    FAILOVER = "failover"


class AgentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class AgentMetrics:
    agent_id: str
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    queue_size: int = 0
    last_health_check: float = 0.0
    status: AgentStatus = AgentStatus.HEALTHY
    error_rate: float = 0.0
    throughput: float = 0.0  # requests per second
    capacity_utilization: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def load_score(self) -> float:
        """Calculate overall load score (0-1, lower is better)."""
        # Weighted combination of various factors
        connection_score = min(1.0, self.active_connections / 100)  # Assume max 100 connections
        response_time_score = min(1.0, self.average_response_time / 5000)  # 5s max
        error_score = min(1.0, self.error_rate)
        resource_score = max(self.cpu_usage, self.memory_usage) / 100
        queue_score = min(1.0, self.queue_size / 50)  # Assume max 50 queue size

        return (connection_score * 0.2 +
                response_time_score * 0.3 +
                error_score * 0.2 +
                resource_score * 0.2 +
                queue_score * 0.1)


@dataclass
class AgentConfiguration:
    agent_id: str
    agent_type: str
    endpoint: str
    weight: float = 1.0
    max_connections: int = 100
    max_queue_size: int = 50
    health_check_interval: float = 30.0
    timeout: float = 30.0
    retry_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class LoadBalancingResult:
    selected_agent: str
    strategy_used: LoadBalancingStrategy
    selection_time: float
    total_candidates: int
    selection_reasoning: str
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Health checking system for agents."""

    def __init__(self, health_check_func: Optional[Callable] = None):
        self.health_check_func = health_check_func or self._default_health_check
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.lock = threading.RLock()

    async def check_agent_health(self, agent_config: AgentConfiguration) -> Tuple[bool, Dict[str, Any]]:
        """Check health of a specific agent."""
        try:
            start_time = time.time()
            health_data = await self.health_check_func(agent_config)
            response_time = (time.time() - start_time) * 1000  # ms

            is_healthy = health_data.get('healthy', False)
            health_details = {
                'healthy': is_healthy,
                'response_time': response_time,
                'timestamp': time.time(),
                **health_data
            }

            # Record health history
            with self.lock:
                self.health_history[agent_config.agent_id].append(health_details)

            return is_healthy, health_details

        except Exception as e:
            health_details = {
                'healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }

            with self.lock:
                self.health_history[agent_config.agent_id].append(health_details)

            return False, health_details

    async def _default_health_check(self, agent_config: AgentConfiguration) -> Dict[str, Any]:
        """Default health check implementation."""
        # This would be replaced with actual health check logic
        # For now, simulate a basic health check
        await asyncio.sleep(0.1)  # Simulate network call
        return {
            'healthy': True,
            'cpu_usage': random.uniform(10, 80),
            'memory_usage': random.uniform(20, 70),
            'queue_size': random.randint(0, 20)
        }

    def get_health_trend(self, agent_id: str, window_seconds: float = 300) -> Dict[str, Any]:
        """Get health trend for an agent."""
        with self.lock:
            if agent_id not in self.health_history:
                return {}

            current_time = time.time()
            recent_checks = [
                check for check in self.health_history[agent_id]
                if current_time - check['timestamp'] <= window_seconds
            ]

            if not recent_checks:
                return {}

            healthy_count = sum(1 for check in recent_checks if check['healthy'])
            avg_response_time = statistics.mean([
                check.get('response_time', 0) for check in recent_checks
            ])

            return {
                'total_checks': len(recent_checks),
                'healthy_checks': healthy_count,
                'health_percentage': (healthy_count / len(recent_checks)) * 100,
                'average_response_time': avg_response_time,
                'trend': 'improving' if healthy_count > len(recent_checks) * 0.8 else 'degrading'
            }


class LoadBalancer:
    """Intelligent load balancer for specialist agents."""

    def __init__(self,
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT,
                 health_check_interval: float = 30.0,
                 enable_health_monitoring: bool = True):
        """
        Initialize load balancer.

        Args:
            strategy: Default load balancing strategy
            health_check_interval: Interval for health checks in seconds
            enable_health_monitoring: Whether to enable continuous health monitoring
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.enable_health_monitoring = enable_health_monitoring

        # Agent management
        self.agents: Dict[str, AgentConfiguration] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.agent_pools: Dict[str, List[str]] = defaultdict(list)  # Type -> agent IDs

        # Load balancing state
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.connection_counts: Dict[str, int] = defaultdict(int)

        # Health monitoring
        self.health_checker = HealthChecker()
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.monitoring_active = False

        # Performance tracking
        self.request_history: deque = deque(maxlen=10000)
        self.strategy_performance: Dict[LoadBalancingStrategy, Dict[str, float]] = defaultdict(
            lambda: {'total_requests': 0, 'total_response_time': 0, 'errors': 0}
        )

        # Thread safety
        self.lock = asyncio.Lock()

        self.logger = logging.getLogger(__name__)
        self.logger.info("Load balancer initialized")

    async def register_agent(self, agent_config: AgentConfiguration):
        """Register a new agent."""
        async with self.lock:
            self.agents[agent_config.agent_id] = agent_config
            self.agent_metrics[agent_config.agent_id] = AgentMetrics(agent_id=agent_config.agent_id)

            # Add to appropriate pools
            self.agent_pools[agent_config.agent_type].append(agent_config.agent_id)

            self.logger.info(f"Registered agent {agent_config.agent_id} of type {agent_config.agent_type}")

            # Perform initial health check
            if self.enable_health_monitoring:
                await self._perform_health_check(agent_config.agent_id)

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        async with self.lock:
            if agent_id in self.agents:
                agent_config = self.agents[agent_id]

                # Remove from pools
                if agent_id in self.agent_pools[agent_config.agent_type]:
                    self.agent_pools[agent_config.agent_type].remove(agent_id)

                # Clean up
                del self.agents[agent_id]
                del self.agent_metrics[agent_id]

                self.logger.info(f"Unregistered agent {agent_id}")

    async def select_agent(self,
                          agent_type: Optional[str] = None,
                          requirements: Optional[Dict[str, Any]] = None,
                          strategy: Optional[LoadBalancingStrategy] = None) -> Optional[LoadBalancingResult]:
        """
        Select an agent for handling a request.

        Args:
            agent_type: Type of agent required
            requirements: Specific requirements for agent selection
            strategy: Load balancing strategy to use (overrides default)

        Returns:
            LoadBalancingResult with selected agent information
        """
        start_time = time.time()
        strategy = strategy or self.strategy

        async with self.lock:
            # Get candidate agents
            candidates = self._get_candidate_agents(agent_type, requirements)

            if not candidates:
                return None

            # Filter healthy agents
            healthy_candidates = [
                agent_id for agent_id in candidates
                if self.agent_metrics[agent_id].status in [AgentStatus.HEALTHY, AgentStatus.DEGRADED]
            ]

            if not healthy_candidates:
                # Fallback to any available agents if no healthy ones
                if candidates:
                    self.logger.warning("No healthy agents available, using degraded agents")
                    healthy_candidates = candidates
                else:
                    return None

            # Select agent based on strategy
            selected_agent = await self._apply_selection_strategy(strategy, healthy_candidates, requirements)

            if selected_agent:
                # Update connection count
                self.connection_counts[selected_agent] += 1
                self.agent_metrics[selected_agent].active_connections += 1

                selection_time = (time.time() - start_time) * 1000  # ms

                result = LoadBalancingResult(
                    selected_agent=selected_agent,
                    strategy_used=strategy,
                    selection_time=selection_time,
                    total_candidates=len(candidates),
                    selection_reasoning=self._get_selection_reasoning(strategy, selected_agent, healthy_candidates),
                    metrics_snapshot={
                        agent_id: {
                            'load_score': metrics.load_score,
                            'active_connections': metrics.active_connections,
                            'response_time': metrics.average_response_time
                        }
                        for agent_id, metrics in self.agent_metrics.items()
                        if agent_id in healthy_candidates
                    }
                )

                return result

            return None

    def _get_candidate_agents(self,
                            agent_type: Optional[str],
                            requirements: Optional[Dict[str, Any]]) -> List[str]:
        """Get list of candidate agents based on type and requirements."""
        if agent_type:
            candidates = self.agent_pools.get(agent_type, [])
        else:
            candidates = list(self.agents.keys())

        # Filter by requirements
        if requirements:
            filtered_candidates = []
            for agent_id in candidates:
                agent_config = self.agents[agent_id]
                agent_metrics = self.agent_metrics[agent_id]

                # Check capabilities
                if 'capabilities' in requirements:
                    required_caps = set(requirements['capabilities'])
                    agent_caps = set(agent_config.capabilities)
                    if not required_caps.issubset(agent_caps):
                        continue

                # Check resource requirements
                if 'max_response_time' in requirements:
                    if agent_metrics.average_response_time > requirements['max_response_time']:
                        continue

                if 'min_success_rate' in requirements:
                    if agent_metrics.success_rate < requirements['min_success_rate']:
                        continue

                filtered_candidates.append(agent_id)

            candidates = filtered_candidates

        return candidates

    async def _apply_selection_strategy(self,
                                      strategy: LoadBalancingStrategy,
                                      candidates: List[str],
                                      requirements: Optional[Dict[str, Any]]) -> Optional[str]:
        """Apply the specified load balancing strategy."""
        if not candidates:
            return None

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(candidates)

        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(candidates)

        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(candidates)

        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(candidates)

        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(candidates)

        elif strategy == LoadBalancingStrategy.INTELLIGENT:
            return await self._intelligent_selection(candidates, requirements)

        elif strategy == LoadBalancingStrategy.FAILOVER:
            return self._failover_selection(candidates)

        else:
            # Default to round robin
            return self._round_robin_selection(candidates)

    def _round_robin_selection(self, candidates: List[str]) -> str:
        """Round-robin selection."""
        pool_key = "_".join(sorted(candidates))
        index = self.round_robin_counters[pool_key] % len(candidates)
        self.round_robin_counters[pool_key] += 1
        return candidates[index]

    def _weighted_round_robin_selection(self, candidates: List[str]) -> str:
        """Weighted round-robin selection based on agent weights."""
        # Create weighted list
        weighted_candidates = []
        for agent_id in candidates:
            weight = int(self.agents[agent_id].weight * 10)  # Scale up for integer operations
            weighted_candidates.extend([agent_id] * weight)

        if not weighted_candidates:
            return candidates[0]

        pool_key = f"wrr_{'_'.join(sorted(candidates))}"
        index = self.round_robin_counters[pool_key] % len(weighted_candidates)
        self.round_robin_counters[pool_key] += 1
        return weighted_candidates[index]

    def _least_connections_selection(self, candidates: List[str]) -> str:
        """Select agent with least active connections."""
        return min(candidates, key=lambda x: self.agent_metrics[x].active_connections)

    def _least_response_time_selection(self, candidates: List[str]) -> str:
        """Select agent with lowest average response time."""
        return min(candidates, key=lambda x: self.agent_metrics[x].average_response_time)

    def _resource_based_selection(self, candidates: List[str]) -> str:
        """Select agent based on resource utilization."""
        def resource_score(agent_id):
            metrics = self.agent_metrics[agent_id]
            return (metrics.cpu_usage + metrics.memory_usage + metrics.capacity_utilization) / 3

        return min(candidates, key=resource_score)

    async def _intelligent_selection(self,
                                   candidates: List[str],
                                   requirements: Optional[Dict[str, Any]]) -> str:
        """Intelligent selection combining multiple factors."""
        scores = {}

        for agent_id in candidates:
            metrics = self.agent_metrics[agent_id]
            config = self.agents[agent_id]

            # Base score from load score
            score = metrics.load_score

            # Adjust for health trend
            health_trend = self.health_checker.get_health_trend(agent_id)
            if health_trend:
                if health_trend.get('trend') == 'improving':
                    score *= 0.9  # Prefer improving agents
                elif health_trend.get('trend') == 'degrading':
                    score *= 1.1  # Penalize degrading agents

            # Adjust for agent weight
            score /= config.weight

            # Adjust for specific requirements
            if requirements:
                if 'priority' in requirements:
                    priority = requirements['priority']
                    if priority == 'low_latency' and metrics.average_response_time > 1000:
                        score *= 1.2
                    elif priority == 'high_throughput' and metrics.throughput < 10:
                        score *= 1.2

            scores[agent_id] = score

        # Select agent with lowest score (best)
        return min(scores.keys(), key=lambda x: scores[x])

    def _failover_selection(self, candidates: List[str]) -> str:
        """Failover selection - prefer primary, fallback to secondary."""
        # Sort by status priority: HEALTHY > DEGRADED > others
        status_priority = {
            AgentStatus.HEALTHY: 0,
            AgentStatus.DEGRADED: 1,
            AgentStatus.OVERLOADED: 2,
            AgentStatus.FAILED: 3,
            AgentStatus.MAINTENANCE: 4
        }

        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                status_priority.get(self.agent_metrics[x].status, 5),
                self.agent_metrics[x].load_score
            )
        )

        return sorted_candidates[0]

    def _get_selection_reasoning(self,
                               strategy: LoadBalancingStrategy,
                               selected_agent: str,
                               candidates: List[str]) -> str:
        """Generate reasoning for agent selection."""
        metrics = self.agent_metrics[selected_agent]

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return f"Round-robin selection from {len(candidates)} candidates"

        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return f"Selected for lowest connections ({metrics.active_connections})"

        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return f"Selected for lowest response time ({metrics.average_response_time:.1f}ms)"

        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return f"Selected for optimal resource usage (CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%)"

        elif strategy == LoadBalancingStrategy.INTELLIGENT:
            return f"Intelligent selection based on load score ({metrics.load_score:.3f})"

        elif strategy == LoadBalancingStrategy.FAILOVER:
            return f"Failover selection - status: {metrics.status.value}"

        else:
            return f"Selected using {strategy.value} strategy"

    async def release_agent(self, agent_id: str, response_time: float, success: bool):
        """Release an agent after request completion."""
        async with self.lock:
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]

                # Update connection count
                if agent_id in self.connection_counts:
                    self.connection_counts[agent_id] = max(0, self.connection_counts[agent_id] - 1)
                metrics.active_connections = max(0, metrics.active_connections - 1)

                # Update metrics
                metrics.total_requests += 1
                metrics.last_response_time = response_time

                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1

                # Update average response time (exponential moving average)
                alpha = 0.1  # Smoothing factor
                if metrics.average_response_time == 0:
                    metrics.average_response_time = response_time
                else:
                    metrics.average_response_time = (
                        alpha * response_time + (1 - alpha) * metrics.average_response_time
                    )

                # Update error rate
                metrics.error_rate = metrics.failed_requests / metrics.total_requests

                # Calculate throughput (requests per second over last minute)
                current_time = time.time()
                recent_requests = [
                    req for req in self.request_history
                    if current_time - req.get('timestamp', 0) <= 60
                ]
                metrics.throughput = len(recent_requests) / 60

                # Record request
                self.request_history.append({
                    'agent_id': agent_id,
                    'response_time': response_time,
                    'success': success,
                    'timestamp': current_time
                })

    async def start_health_monitoring(self):
        """Start background health monitoring."""
        if not self.enable_health_monitoring or self.monitoring_active:
            return

        self.monitoring_active = True
        self.health_monitor_task = asyncio.create_task(self._health_monitoring_loop())
        self.logger.info("Health monitoring started")

    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        self.monitoring_active = False
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")

    async def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self.monitoring_active:
            try:
                # Check health of all agents
                health_tasks = [
                    self._perform_health_check(agent_id)
                    for agent_id in self.agents.keys()
                ]

                if health_tasks:
                    await asyncio.gather(*health_tasks, return_exceptions=True)

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _perform_health_check(self, agent_id: str):
        """Perform health check for a specific agent."""
        if agent_id not in self.agents:
            return

        agent_config = self.agents[agent_id]
        metrics = self.agent_metrics[agent_id]

        try:
            is_healthy, health_data = await self.health_checker.check_agent_health(agent_config)

            # Update metrics from health check
            if 'cpu_usage' in health_data:
                metrics.cpu_usage = health_data['cpu_usage']
            if 'memory_usage' in health_data:
                metrics.memory_usage = health_data['memory_usage']
            if 'queue_size' in health_data:
                metrics.queue_size = health_data['queue_size']

            metrics.last_health_check = time.time()

            # Update status based on health and metrics
            if not is_healthy:
                metrics.status = AgentStatus.FAILED
            elif metrics.error_rate > 0.5:
                metrics.status = AgentStatus.FAILED
            elif metrics.error_rate > 0.2 or metrics.average_response_time > 5000:
                metrics.status = AgentStatus.DEGRADED
            elif metrics.cpu_usage > 90 or metrics.memory_usage > 90:
                metrics.status = AgentStatus.OVERLOADED
            else:
                metrics.status = AgentStatus.HEALTHY

        except Exception as e:
            self.logger.error(f"Health check failed for agent {agent_id}: {e}")
            metrics.status = AgentStatus.FAILED

    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific agent."""
        if agent_id not in self.agents:
            return None

        config = self.agents[agent_id]
        metrics = self.agent_metrics[agent_id]
        health_trend = self.health_checker.get_health_trend(agent_id)

        return {
            'agent_id': agent_id,
            'agent_type': config.agent_type,
            'status': metrics.status.value,
            'metrics': {
                'active_connections': metrics.active_connections,
                'total_requests': metrics.total_requests,
                'success_rate': metrics.success_rate,
                'error_rate': metrics.error_rate,
                'average_response_time': metrics.average_response_time,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'queue_size': metrics.queue_size,
                'throughput': metrics.throughput,
                'load_score': metrics.load_score
            },
            'health_trend': health_trend,
            'configuration': {
                'weight': config.weight,
                'max_connections': config.max_connections,
                'capabilities': config.capabilities
            }
        }

    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        async with self.lock:
            total_agents = len(self.agents)
            healthy_agents = sum(
                1 for metrics in self.agent_metrics.values()
                if metrics.status == AgentStatus.HEALTHY
            )

            total_requests = sum(metrics.total_requests for metrics in self.agent_metrics.values())
            total_errors = sum(metrics.failed_requests for metrics in self.agent_metrics.values())

            agent_stats = {}
            for agent_id, metrics in self.agent_metrics.items():
                agent_stats[agent_id] = {
                    'status': metrics.status.value,
                    'load_score': metrics.load_score,
                    'active_connections': metrics.active_connections,
                    'success_rate': metrics.success_rate,
                    'response_time': metrics.average_response_time
                }

            return {
                'total_agents': total_agents,
                'healthy_agents': healthy_agents,
                'agent_utilization': healthy_agents / total_agents if total_agents > 0 else 0,
                'total_requests': total_requests,
                'overall_error_rate': total_errors / total_requests if total_requests > 0 else 0,
                'default_strategy': self.strategy.value,
                'health_monitoring_active': self.monitoring_active,
                'agent_pools': {
                    pool_type: len(agents) for pool_type, agents in self.agent_pools.items()
                },
                'agent_stats': agent_stats
            }

    def generate_load_balancer_report(self) -> str:
        """Generate comprehensive load balancer report."""
        # This would need to be called in an async context
        import asyncio
        stats = asyncio.create_task(self.get_load_balancer_stats())

        # For synchronous report generation, we'll use current data
        report = []
        report.append("# Load Balancer Performance Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overview
        total_agents = len(self.agents)
        healthy_agents = sum(
            1 for metrics in self.agent_metrics.values()
            if metrics.status == AgentStatus.HEALTHY
        )

        report.append("## Overview")
        report.append(f"- Total Agents: {total_agents}")
        report.append(f"- Healthy Agents: {healthy_agents}")
        report.append(f"- Default Strategy: {self.strategy.value}")
        report.append(f"- Health Monitoring: {'Active' if self.monitoring_active else 'Inactive'}")
        report.append("")

        # Agent Status
        if self.agent_metrics:
            report.append("## Agent Status")
            for agent_id, metrics in self.agent_metrics.items():
                status_emoji = {
                    AgentStatus.HEALTHY: "✅",
                    AgentStatus.DEGRADED: "⚠️",
                    AgentStatus.OVERLOADED: "🔶",
                    AgentStatus.FAILED: "❌",
                    AgentStatus.MAINTENANCE: "🔧"
                }.get(metrics.status, "❓")

                report.append(f"### {status_emoji} {agent_id}")
                report.append(f"- Status: {metrics.status.value}")
                report.append(f"- Load Score: {metrics.load_score:.3f}")
                report.append(f"- Active Connections: {metrics.active_connections}")
                report.append(f"- Success Rate: {metrics.success_rate*100:.1f}%")
                report.append(f"- Avg Response Time: {metrics.average_response_time:.1f}ms")
                report.append("")

        # Pool Distribution
        if self.agent_pools:
            report.append("## Agent Pool Distribution")
            for pool_type, agents in self.agent_pools.items():
                report.append(f"- **{pool_type}**: {len(agents)} agents")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        if healthy_agents < total_agents * 0.8:
            report.append("⚠️ **WARNING**: Less than 80% of agents are healthy")
        if total_agents == 0:
            report.append("❌ **CRITICAL**: No agents registered")
        else:
            avg_load = statistics.mean([m.load_score for m in self.agent_metrics.values()])
            if avg_load > 0.8:
                report.append("⚠️ **HIGH LOAD**: Consider adding more agents")
            elif avg_load < 0.2:
                report.append("ℹ️ **LOW LOAD**: System has spare capacity")

        return "\n".join(report)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_health_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_health_monitoring()


# Convenience classes and functions
class AgentProxy:
    """Proxy for making requests through the load balancer."""

    def __init__(self, load_balancer: LoadBalancer, agent_type: str):
        self.load_balancer = load_balancer
        self.agent_type = agent_type

    async def execute(self,
                     request_func: Callable,
                     *args,
                     requirements: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Any:
        """Execute a request through the load balancer."""
        # Select agent
        result = await self.load_balancer.select_agent(self.agent_type, requirements)
        if not result:
            raise RuntimeError(f"No available agents of type {self.agent_type}")

        agent_id = result.selected_agent
        start_time = time.time()
        success = True
        response = None

        try:
            # Execute request (this would be implemented based on specific agent interface)
            response = await request_func(agent_id, *args, **kwargs)
            return response

        except Exception as e:
            success = False
            raise

        finally:
            # Release agent
            response_time = (time.time() - start_time) * 1000  # ms
            await self.load_balancer.release_agent(agent_id, response_time, success)


# Global load balancer instance
_global_load_balancer: Optional[LoadBalancer] = None


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer()
    return _global_load_balancer


async def create_agent_proxy(agent_type: str) -> AgentProxy:
    """Create an agent proxy for a specific agent type."""
    load_balancer = get_load_balancer()
    return AgentProxy(load_balancer, agent_type)