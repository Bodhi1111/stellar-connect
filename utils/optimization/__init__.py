"""
Performance Optimization Framework.

This module provides comprehensive performance optimization tools including:
- Memory management and optimization
- Performance monitoring and bottleneck identification
- Intelligent multi-tier caching
- Load balancing across specialist agents
"""

from .memory_manager import (
    MemoryManager,
    MemoryPool,
    MemoryPriority,
    MemoryStrategy,
    MemoryUsage,
    LazyLoadingProxy,
    memory_optimized,
    memory_pool_manager,
    get_memory_manager,
    cleanup_memory,
    emergency_memory_cleanup
)

from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    BottleneckInfo,
    PerformanceAlert,
    MetricType,
    AlertLevel,
    PerformanceThresholds,
    get_performance_monitor,
    monitor_performance,
    monitor_async_performance
)

from .caching_layer import (
    IntelligentCache,
    CacheEntry,
    CacheLevel,
    EvictionPolicy,
    CacheEntryStatus,
    AccessPattern,
    PatternAnalyzer,
    CacheStats,
    cached,
    async_cached,
    get_cache,
    cache_clear
)

from .load_balancer import (
    LoadBalancer,
    AgentConfiguration,
    AgentMetrics,
    LoadBalancingStrategy,
    AgentStatus,
    LoadBalancingResult,
    HealthChecker,
    AgentProxy,
    get_load_balancer,
    create_agent_proxy
)

__version__ = "1.0.0"
__author__ = "Stellar Connect Performance Team"

__all__ = [
    # Memory Management
    "MemoryManager",
    "MemoryPool",
    "MemoryPriority",
    "MemoryStrategy",
    "MemoryUsage",
    "LazyLoadingProxy",
    "memory_optimized",
    "memory_pool_manager",
    "get_memory_manager",
    "cleanup_memory",
    "emergency_memory_cleanup",

    # Performance Monitoring
    "PerformanceMonitor",
    "PerformanceMetric",
    "BottleneckInfo",
    "PerformanceAlert",
    "MetricType",
    "AlertLevel",
    "PerformanceThresholds",
    "get_performance_monitor",
    "monitor_performance",
    "monitor_async_performance",

    # Caching
    "IntelligentCache",
    "CacheEntry",
    "CacheLevel",
    "EvictionPolicy",
    "CacheEntryStatus",
    "AccessPattern",
    "PatternAnalyzer",
    "CacheStats",
    "cached",
    "async_cached",
    "get_cache",
    "cache_clear",

    # Load Balancing
    "LoadBalancer",
    "AgentConfiguration",
    "AgentMetrics",
    "LoadBalancingStrategy",
    "AgentStatus",
    "LoadBalancingResult",
    "HealthChecker",
    "AgentProxy",
    "get_load_balancer",
    "create_agent_proxy"
]