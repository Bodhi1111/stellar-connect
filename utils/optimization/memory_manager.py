"""
Memory optimization system for enhanced processing.
Manages memory usage across the enhanced query processing pipeline.
"""

import gc
import psutil
import threading
import time
import weakref
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor


class MemoryPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CLEANUP_CANDIDATE = "cleanup_candidate"


class MemoryStrategy(Enum):
    LAZY_LOADING = "lazy_loading"
    STREAMING = "streaming"
    CHUNKED_PROCESSING = "chunked_processing"
    MEMORY_MAPPING = "memory_mapping"
    COMPRESSION = "compression"


@dataclass
class MemoryUsage:
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    process_memory: float
    timestamp: float


@dataclass
class MemoryPool:
    name: str
    max_size: int
    current_size: int = 0
    objects: Dict[str, Any] = field(default_factory=dict)
    priorities: Dict[str, MemoryPriority] = field(default_factory=dict)
    access_times: Dict[str, float] = field(default_factory=dict)
    cleanup_callbacks: Dict[str, Callable] = field(default_factory=dict)


class MemoryManager:
    """Advanced memory management system for enhanced query processing."""

    def __init__(self,
                 max_memory_percent: float = 80.0,
                 cleanup_threshold: float = 90.0,
                 monitoring_interval: float = 30.0):
        """
        Initialize memory manager.

        Args:
            max_memory_percent: Maximum memory usage percentage before optimization
            cleanup_threshold: Memory usage threshold to trigger aggressive cleanup
            monitoring_interval: Interval in seconds for memory monitoring
        """
        self.max_memory_percent = max_memory_percent
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval

        # Memory pools for different data types
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.global_cache: OrderedDict = OrderedDict()
        self.weak_references: Set[weakref.ref] = set()

        # Memory tracking
        self.memory_history: List[MemoryUsage] = []
        self.memory_alerts: List[Dict[str, Any]] = []

        # Threading and async support
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory_mgr")

        # Strategy implementations
        self.strategies = self._initialize_strategies()

        self.logger = logging.getLogger(__name__)
        self.logger.info("Memory Manager initialized")

    def _initialize_strategies(self) -> Dict[MemoryStrategy, Callable]:
        """Initialize memory optimization strategies."""
        return {
            MemoryStrategy.LAZY_LOADING: self._lazy_loading_strategy,
            MemoryStrategy.STREAMING: self._streaming_strategy,
            MemoryStrategy.CHUNKED_PROCESSING: self._chunked_processing_strategy,
            MemoryStrategy.MEMORY_MAPPING: self._memory_mapping_strategy,
            MemoryStrategy.COMPRESSION: self._compression_strategy
        }

    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            name="memory_monitor",
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Memory monitoring stopped")

    def _memory_monitor_loop(self):
        """Main memory monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_memory_stats()
                self._check_memory_thresholds()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")

    def _collect_memory_stats(self):
        """Collect current memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB

            usage = MemoryUsage(
                total_memory=memory.total / (1024 * 1024),  # MB
                available_memory=memory.available / (1024 * 1024),  # MB
                used_memory=memory.used / (1024 * 1024),  # MB
                memory_percent=memory.percent,
                process_memory=process_memory,
                timestamp=time.time()
            )

            with self.lock:
                self.memory_history.append(usage)
                # Keep only last 100 entries
                if len(self.memory_history) > 100:
                    self.memory_history.pop(0)

        except Exception as e:
            self.logger.error(f"Error collecting memory stats: {e}")

    def _check_memory_thresholds(self):
        """Check memory usage against thresholds and take action."""
        if not self.memory_history:
            return

        current_usage = self.memory_history[-1]

        # Check if cleanup is needed
        if current_usage.memory_percent >= self.cleanup_threshold:
            self.logger.warning(f"Memory usage critical: {current_usage.memory_percent:.1f}%")
            self._emergency_cleanup()
        elif current_usage.memory_percent >= self.max_memory_percent:
            self.logger.info(f"Memory usage high: {current_usage.memory_percent:.1f}%")
            self._gentle_cleanup()

    def create_memory_pool(self,
                          name: str,
                          max_size: int,
                          cleanup_callback: Optional[Callable] = None) -> MemoryPool:
        """Create a new memory pool."""
        with self.lock:
            if name in self.memory_pools:
                raise ValueError(f"Memory pool '{name}' already exists")

            pool = MemoryPool(name=name, max_size=max_size)
            self.memory_pools[name] = pool
            self.logger.info(f"Created memory pool '{name}' with max size {max_size}")
            return pool

    def allocate_to_pool(self,
                        pool_name: str,
                        obj_id: str,
                        obj: Any,
                        priority: MemoryPriority = MemoryPriority.MEDIUM,
                        cleanup_callback: Optional[Callable] = None) -> bool:
        """Allocate object to memory pool."""
        with self.lock:
            if pool_name not in self.memory_pools:
                raise ValueError(f"Memory pool '{pool_name}' not found")

            pool = self.memory_pools[pool_name]

            # Check if pool is full
            if pool.current_size >= pool.max_size:
                if not self._make_space_in_pool(pool):
                    return False

            # Add object to pool
            pool.objects[obj_id] = obj
            pool.priorities[obj_id] = priority
            pool.access_times[obj_id] = time.time()
            pool.current_size += 1

            if cleanup_callback:
                pool.cleanup_callbacks[obj_id] = cleanup_callback

            return True

    def get_from_pool(self, pool_name: str, obj_id: str) -> Optional[Any]:
        """Retrieve object from memory pool."""
        with self.lock:
            if pool_name not in self.memory_pools:
                return None

            pool = self.memory_pools[pool_name]
            if obj_id not in pool.objects:
                return None

            # Update access time
            pool.access_times[obj_id] = time.time()
            return pool.objects[obj_id]

    def remove_from_pool(self, pool_name: str, obj_id: str) -> bool:
        """Remove object from memory pool."""
        with self.lock:
            if pool_name not in self.memory_pools:
                return False

            pool = self.memory_pools[pool_name]
            if obj_id not in pool.objects:
                return False

            # Call cleanup callback if exists
            if obj_id in pool.cleanup_callbacks:
                try:
                    pool.cleanup_callbacks[obj_id](pool.objects[obj_id])
                except Exception as e:
                    self.logger.error(f"Error in cleanup callback for {obj_id}: {e}")

            # Remove object
            del pool.objects[obj_id]
            del pool.priorities[obj_id]
            del pool.access_times[obj_id]
            if obj_id in pool.cleanup_callbacks:
                del pool.cleanup_callbacks[obj_id]

            pool.current_size -= 1
            return True

    def _make_space_in_pool(self, pool: MemoryPool) -> bool:
        """Make space in memory pool by removing least important items."""
        if not pool.objects:
            return False

        # Sort by priority and access time
        candidates = []
        for obj_id in pool.objects.keys():
            priority = pool.priorities[obj_id]
            access_time = pool.access_times[obj_id]

            # Priority scoring (lower is better for removal)
            priority_score = {
                MemoryPriority.CRITICAL: 1000,
                MemoryPriority.HIGH: 100,
                MemoryPriority.MEDIUM: 10,
                MemoryPriority.LOW: 1,
                MemoryPriority.CLEANUP_CANDIDATE: 0
            }[priority]

            # Age scoring (older = better candidate for removal)
            age_score = time.time() - access_time

            total_score = age_score - priority_score
            candidates.append((total_score, obj_id))

        # Sort by score (highest first)
        candidates.sort(reverse=True)

        # Remove items until we have space
        items_to_remove = max(1, pool.current_size // 4)  # Remove 25% or at least 1
        removed = 0

        for score, obj_id in candidates:
            if removed >= items_to_remove:
                break

            # Don't remove critical items unless absolutely necessary
            if pool.priorities[obj_id] == MemoryPriority.CRITICAL and removed > 0:
                continue

            self.remove_from_pool(pool.name, obj_id)
            removed += 1

        return removed > 0

    def register_weak_reference(self, obj: Any, cleanup_callback: Optional[Callable] = None) -> weakref.ref:
        """Register weak reference for automatic cleanup."""
        def cleanup_wrapper(ref):
            self.weak_references.discard(ref)
            if cleanup_callback:
                try:
                    cleanup_callback()
                except Exception as e:
                    self.logger.error(f"Error in weak reference cleanup: {e}")

        weak_ref = weakref.ref(obj, cleanup_wrapper)
        self.weak_references.add(weak_ref)
        return weak_ref

    def _gentle_cleanup(self):
        """Perform gentle memory cleanup."""
        self.logger.info("Performing gentle memory cleanup")

        # Clean up weak references
        dead_refs = [ref for ref in self.weak_references if ref() is None]
        for ref in dead_refs:
            self.weak_references.discard(ref)

        # Clean up global cache
        if len(self.global_cache) > 1000:
            # Remove oldest 25% of items
            items_to_remove = len(self.global_cache) // 4
            for _ in range(items_to_remove):
                self.global_cache.popitem(last=False)

        # Clean up memory pools
        for pool in self.memory_pools.values():
            if pool.current_size > pool.max_size * 0.8:
                self._make_space_in_pool(pool)

        # Force garbage collection
        collected = gc.collect()
        self.logger.info(f"Gentle cleanup completed, collected {collected} objects")

    def _emergency_cleanup(self):
        """Perform aggressive memory cleanup."""
        self.logger.warning("Performing emergency memory cleanup")

        # Clear all cleanup candidates
        for pool in self.memory_pools.values():
            candidates = [
                obj_id for obj_id, priority in pool.priorities.items()
                if priority == MemoryPriority.CLEANUP_CANDIDATE
            ]
            for obj_id in candidates:
                self.remove_from_pool(pool.name, obj_id)

        # Aggressive cache cleanup
        cache_items_to_keep = min(100, len(self.global_cache) // 4)
        while len(self.global_cache) > cache_items_to_keep:
            self.global_cache.popitem(last=False)

        # Clean up low priority items
        for pool in self.memory_pools.values():
            low_priority_items = [
                obj_id for obj_id, priority in pool.priorities.items()
                if priority == MemoryPriority.LOW
            ]
            for obj_id in low_priority_items[:len(low_priority_items)//2]:
                self.remove_from_pool(pool.name, obj_id)

        # Force multiple garbage collection cycles
        for _ in range(3):
            collected = gc.collect()

        self.logger.warning("Emergency cleanup completed")

    def optimize_for_strategy(self, strategy: MemoryStrategy, context: Dict[str, Any] = None):
        """Apply specific memory optimization strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown memory strategy: {strategy}")

        try:
            self.strategies[strategy](context or {})
            self.logger.info(f"Applied memory strategy: {strategy.value}")
        except Exception as e:
            self.logger.error(f"Error applying strategy {strategy.value}: {e}")

    def _lazy_loading_strategy(self, context: Dict[str, Any]):
        """Implement lazy loading strategy."""
        # Mark non-critical objects as lazy loading candidates
        for pool in self.memory_pools.values():
            for obj_id, priority in pool.priorities.items():
                if priority in [MemoryPriority.LOW, MemoryPriority.MEDIUM]:
                    # Convert to lazy loading proxy if possible
                    # This would be implemented based on specific object types
                    pass

    def _streaming_strategy(self, context: Dict[str, Any]):
        """Implement streaming processing strategy."""
        chunk_size = context.get('chunk_size', 1000)
        # Implementation would depend on specific data processing needs
        self.logger.info(f"Configured streaming with chunk size: {chunk_size}")

    def _chunked_processing_strategy(self, context: Dict[str, Any]):
        """Implement chunked processing strategy."""
        max_chunk_size = context.get('max_chunk_size', 10000)
        # Implementation for breaking large datasets into chunks
        self.logger.info(f"Configured chunked processing with max size: {max_chunk_size}")

    def _memory_mapping_strategy(self, context: Dict[str, Any]):
        """Implement memory mapping strategy."""
        # Use memory mapping for large files
        # Implementation would use mmap for file operations
        self.logger.info("Configured memory mapping strategy")

    def _compression_strategy(self, context: Dict[str, Any]):
        """Implement compression strategy."""
        compression_ratio = context.get('compression_ratio', 0.5)
        # Compress large objects in memory
        # Implementation would use compression libraries
        self.logger.info(f"Configured compression with target ratio: {compression_ratio}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_history:
            return {}

        current = self.memory_history[-1]

        stats = {
            "current_usage": {
                "memory_percent": current.memory_percent,
                "process_memory_mb": current.process_memory,
                "available_memory_mb": current.available_memory
            },
            "memory_pools": {},
            "global_cache_size": len(self.global_cache),
            "weak_references_count": len(self.weak_references),
            "monitoring_active": self.monitoring_active
        }

        # Pool statistics
        for name, pool in self.memory_pools.items():
            stats["memory_pools"][name] = {
                "current_size": pool.current_size,
                "max_size": pool.max_size,
                "utilization": pool.current_size / pool.max_size if pool.max_size > 0 else 0,
                "priority_distribution": {}
            }

            # Count by priority
            for priority in pool.priorities.values():
                priority_name = priority.value
                stats["memory_pools"][name]["priority_distribution"][priority_name] = \
                    stats["memory_pools"][name]["priority_distribution"].get(priority_name, 0) + 1

        # Memory trends
        if len(self.memory_history) > 1:
            recent_usage = [m.memory_percent for m in self.memory_history[-10:]]
            stats["trends"] = {
                "average_usage_last_10": sum(recent_usage) / len(recent_usage),
                "max_usage_last_10": max(recent_usage),
                "min_usage_last_10": min(recent_usage)
            }

        return stats

    def generate_memory_report(self) -> str:
        """Generate comprehensive memory usage report."""
        stats = self.get_memory_stats()

        if not stats:
            return "No memory statistics available"

        report = []
        report.append("# Memory Usage Report")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Current usage
        current = stats["current_usage"]
        report.append("## Current Memory Usage")
        report.append(f"- System Memory Usage: {current['memory_percent']:.1f}%")
        report.append(f"- Process Memory: {current['process_memory_mb']:.1f} MB")
        report.append(f"- Available Memory: {current['available_memory_mb']:.1f} MB")
        report.append("")

        # Memory pools
        if stats["memory_pools"]:
            report.append("## Memory Pools")
            for name, pool_stats in stats["memory_pools"].items():
                utilization = pool_stats["utilization"] * 100
                report.append(f"### {name}")
                report.append(f"- Size: {pool_stats['current_size']}/{pool_stats['max_size']} ({utilization:.1f}%)")

                if pool_stats["priority_distribution"]:
                    report.append("- Priority Distribution:")
                    for priority, count in pool_stats["priority_distribution"].items():
                        report.append(f"  * {priority}: {count}")
                report.append("")

        # Global statistics
        report.append("## Global Statistics")
        report.append(f"- Global Cache Size: {stats['global_cache_size']}")
        report.append(f"- Weak References: {stats['weak_references_count']}")
        report.append(f"- Monitoring Active: {stats['monitoring_active']}")

        # Trends
        if "trends" in stats:
            trends = stats["trends"]
            report.append("\n## Recent Trends (Last 10 Measurements)")
            report.append(f"- Average Usage: {trends['average_usage_last_10']:.1f}%")
            report.append(f"- Peak Usage: {trends['max_usage_last_10']:.1f}%")
            report.append(f"- Minimum Usage: {trends['min_usage_last_10']:.1f}%")

        return "\n".join(report)

    async def async_cleanup(self):
        """Perform asynchronous cleanup operations."""
        def cleanup_task():
            self._gentle_cleanup()

        await asyncio.get_event_loop().run_in_executor(self.executor, cleanup_task)

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop_monitoring()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass


# Memory optimization decorators and utilities
def memory_optimized(strategy: MemoryStrategy = MemoryStrategy.LAZY_LOADING):
    """Decorator for memory-optimized functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create memory manager
            manager = getattr(wrapper, '_memory_manager', None)
            if manager is None:
                manager = MemoryManager()
                wrapper._memory_manager = manager

            # Apply optimization strategy
            manager.optimize_for_strategy(strategy)

            try:
                return func(*args, **kwargs)
            finally:
                # Cleanup after execution
                if manager.memory_history and manager.memory_history[-1].memory_percent > 85:
                    manager._gentle_cleanup()

        return wrapper
    return decorator


def memory_pool_manager(pool_name: str, max_size: int = 1000):
    """Decorator for functions that use memory pools."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = getattr(wrapper, '_memory_manager', None)
            if manager is None:
                manager = MemoryManager()
                wrapper._memory_manager = manager

            # Create pool if it doesn't exist
            if pool_name not in manager.memory_pools:
                manager.create_memory_pool(pool_name, max_size)

            return func(*args, **kwargs)

        wrapper._memory_manager = None
        return wrapper
    return decorator


class LazyLoadingProxy:
    """Proxy object for lazy loading of large data structures."""

    def __init__(self, loader_func: Callable, *args, **kwargs):
        self._loader_func = loader_func
        self._loader_args = args
        self._loader_kwargs = kwargs
        self._loaded_object = None
        self._is_loaded = False

    def _load(self):
        """Load the actual object."""
        if not self._is_loaded:
            self._loaded_object = self._loader_func(*self._loader_args, **self._loader_kwargs)
            self._is_loaded = True

    def __getattr__(self, name):
        """Delegate attribute access to loaded object."""
        self._load()
        return getattr(self._loaded_object, name)

    def __call__(self, *args, **kwargs):
        """Make proxy callable if the underlying object is callable."""
        self._load()
        return self._loaded_object(*args, **kwargs)


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
        _global_memory_manager.start_monitoring()
    return _global_memory_manager


def cleanup_memory():
    """Global memory cleanup function."""
    manager = get_memory_manager()
    manager._gentle_cleanup()


def emergency_memory_cleanup():
    """Global emergency memory cleanup function."""
    manager = get_memory_manager()
    manager._emergency_cleanup()