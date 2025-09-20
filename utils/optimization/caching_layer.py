"""
Intelligent caching layer for frequently accessed patterns.
Multi-tier caching system with adaptive algorithms and pattern recognition.
"""

import time
import threading
import hashlib
import pickle
import json
import gzip
import weakref
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import asyncio
import logging
from datetime import datetime, timedelta
import statistics


class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_COMPRESSED = "l2_compressed"
    L3_DISK = "l3_disk"
    L4_REMOTE = "l4_remote"


class EvictionPolicy(Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive replacement
    PATTERN_BASED = "pattern_based"  # Based on access patterns


class CacheEntryStatus(Enum):
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    COMPUTING = "computing"


@dataclass
class CacheEntry:
    key: str
    value: Any
    level: CacheLevel
    size_bytes: int
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def status(self) -> CacheEntryStatus:
        """Get current status of cache entry."""
        if self.is_expired:
            return CacheEntryStatus.EXPIRED
        elif self.age > (self.ttl or float('inf')) * 0.8:
            return CacheEntryStatus.STALE
        else:
            return CacheEntryStatus.FRESH


@dataclass
class AccessPattern:
    key: str
    access_times: List[float] = field(default_factory=list)
    access_intervals: List[float] = field(default_factory=list)
    prediction_confidence: float = 0.0
    next_predicted_access: Optional[float] = None
    pattern_type: str = "unknown"  # periodic, burst, random, etc.


class PatternAnalyzer:
    """Analyzes access patterns to optimize caching decisions."""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.patterns: Dict[str, AccessPattern] = {}
        self.lock = threading.RLock()

    def record_access(self, key: str, timestamp: Optional[float] = None):
        """Record an access for pattern analysis."""
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            if key not in self.patterns:
                self.patterns[key] = AccessPattern(key=key)

            pattern = self.patterns[key]
            pattern.access_times.append(timestamp)

            # Keep only recent history
            if len(pattern.access_times) > self.max_history:
                pattern.access_times.pop(0)

            # Calculate intervals
            if len(pattern.access_times) >= 2:
                intervals = [
                    pattern.access_times[i] - pattern.access_times[i-1]
                    for i in range(1, len(pattern.access_times))
                ]
                pattern.access_intervals = intervals[-self.max_history:]

            # Analyze pattern
            self._analyze_pattern(pattern)

    def _analyze_pattern(self, pattern: AccessPattern):
        """Analyze access pattern and predict next access."""
        if len(pattern.access_intervals) < 3:
            return

        intervals = pattern.access_intervals
        avg_interval = statistics.mean(intervals)
        std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0

        # Determine pattern type
        if std_dev < avg_interval * 0.1:  # Very regular
            pattern.pattern_type = "periodic"
            pattern.prediction_confidence = 0.9
            pattern.next_predicted_access = pattern.access_times[-1] + avg_interval
        elif std_dev < avg_interval * 0.3:  # Somewhat regular
            pattern.pattern_type = "semi_periodic"
            pattern.prediction_confidence = 0.6
            pattern.next_predicted_access = pattern.access_times[-1] + avg_interval
        elif len(intervals) >= 5 and self._detect_burst_pattern(intervals):
            pattern.pattern_type = "burst"
            pattern.prediction_confidence = 0.4
            # Predict next burst
            pattern.next_predicted_access = pattern.access_times[-1] + avg_interval * 2
        else:
            pattern.pattern_type = "random"
            pattern.prediction_confidence = 0.1
            pattern.next_predicted_access = None

    def _detect_burst_pattern(self, intervals: List[float]) -> bool:
        """Detect if access pattern shows burst behavior."""
        if len(intervals) < 5:
            return False

        # Look for clusters of short intervals followed by longer gaps
        short_intervals = [i for i in intervals if i < statistics.mean(intervals) * 0.5]
        return len(short_intervals) >= len(intervals) * 0.3

    def predict_access_probability(self, key: str, future_time: float) -> float:
        """Predict probability of access at future time."""
        with self.lock:
            if key not in self.patterns:
                return 0.1  # Default low probability

            pattern = self.patterns[key]
            if pattern.next_predicted_access is None:
                return 0.1

            time_diff = abs(future_time - pattern.next_predicted_access)
            base_confidence = pattern.prediction_confidence

            # Decay confidence based on time distance from prediction
            if pattern.pattern_type == "periodic":
                avg_interval = statistics.mean(pattern.access_intervals) if pattern.access_intervals else 3600
                decay_factor = max(0, 1 - (time_diff / avg_interval))
            else:
                decay_factor = max(0, 1 - (time_diff / 3600))  # 1 hour decay

            return base_confidence * decay_factor

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all access patterns."""
        with self.lock:
            summary = {
                "total_keys": len(self.patterns),
                "pattern_types": defaultdict(int),
                "high_confidence_patterns": 0
            }

            for pattern in self.patterns.values():
                summary["pattern_types"][pattern.pattern_type] += 1
                if pattern.prediction_confidence > 0.7:
                    summary["high_confidence_patterns"] += 1

            return dict(summary)


class CacheStats:
    """Cache performance statistics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.writes = 0
        self.total_size_bytes = 0
        self.level_stats: Dict[CacheLevel, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.operation_times: List[float] = []
        self.lock = threading.RLock()

    def record_hit(self, level: CacheLevel):
        """Record cache hit."""
        with self.lock:
            self.hits += 1
            self.level_stats[level]["hits"] += 1

    def record_miss(self):
        """Record cache miss."""
        with self.lock:
            self.misses += 1

    def record_eviction(self, level: CacheLevel):
        """Record cache eviction."""
        with self.lock:
            self.evictions += 1
            self.level_stats[level]["evictions"] += 1

    def record_write(self, level: CacheLevel):
        """Record cache write."""
        with self.lock:
            self.writes += 1
            self.level_stats[level]["writes"] += 1

    def record_operation_time(self, duration: float):
        """Record operation duration."""
        with self.lock:
            self.operation_times.append(duration)
            # Keep only recent times
            if len(self.operation_times) > 1000:
                self.operation_times.pop(0)

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def average_operation_time(self) -> float:
        """Calculate average operation time."""
        with self.lock:
            return statistics.mean(self.operation_times) if self.operation_times else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        with self.lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "writes": self.writes,
                "hit_rate": self.hit_rate,
                "total_requests": self.hits + self.misses,
                "average_operation_time_ms": self.average_operation_time * 1000,
                "level_stats": dict(self.level_stats)
            }


class IntelligentCache:
    """Multi-tier intelligent caching system with pattern recognition."""

    def __init__(self,
                 l1_size: int = 1000,
                 l2_size: int = 5000,
                 default_ttl: float = 3600,
                 eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
                 enable_compression: bool = True,
                 compression_threshold: int = 1024):
        """
        Initialize intelligent cache.

        Args:
            l1_size: Maximum number of entries in L1 cache
            l2_size: Maximum number of entries in L2 cache
            default_ttl: Default time-to-live in seconds
            eviction_policy: Cache eviction policy
            enable_compression: Whether to enable compression in L2
            compression_threshold: Minimum size for compression (bytes)
        """
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold

        # Cache levels
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Access frequency tracking for LFU
        self.access_frequencies: Dict[str, int] = defaultdict(int)

        # Pattern analysis
        self.pattern_analyzer = PatternAnalyzer()

        # Statistics
        self.stats = CacheStats()

        # Computing cache (to prevent duplicate computation)
        self.computing: Dict[str, asyncio.Event] = {}
        self.computation_results: Dict[str, Any] = {}

        # Thread safety
        self.lock = threading.RLock()
        self.async_lock = asyncio.Lock()

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None

        self.logger = logging.getLogger(__name__)
        self.logger.info("Intelligent cache initialized")

    def _generate_key(self, key: Union[str, Tuple, Dict]) -> str:
        """Generate normalized cache key."""
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            return hashlib.md5(str(key).encode()).hexdigest()
        elif isinstance(key, dict):
            # Sort dict for consistent hashing
            sorted_items = sorted(key.items())
            return hashlib.md5(str(sorted_items).encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)

    def _compress_data(self, data: bytes) -> Tuple[bytes, float]:
        """Compress data and return compression ratio."""
        if len(data) < self.compression_threshold:
            return data, 1.0

        compressed = gzip.compress(data)
        ratio = len(compressed) / len(data)
        return compressed, ratio

    def _decompress_data(self, data: bytes, compression_ratio: float) -> bytes:
        """Decompress data if it was compressed."""
        if compression_ratio >= 1.0:
            return data
        return gzip.decompress(data)

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(self._serialize_value(value))
        except:
            # Fallback estimation
            return len(str(value)) * 2

    def get(self, key: Union[str, Tuple, Dict], default: Any = None) -> Any:
        """Get value from cache."""
        start_time = time.time()
        normalized_key = self._generate_key(key)

        try:
            with self.lock:
                # Check L1 cache first
                if normalized_key in self.l1_cache:
                    entry = self.l1_cache[normalized_key]
                    if not entry.is_expired:
                        # Move to end (LRU)
                        self.l1_cache.move_to_end(normalized_key)
                        entry.accessed_at = time.time()
                        entry.access_count += 1
                        self.access_frequencies[normalized_key] += 1
                        self.stats.record_hit(CacheLevel.L1_MEMORY)
                        self.pattern_analyzer.record_access(normalized_key)
                        return entry.value

                # Check L2 cache
                if normalized_key in self.l2_cache:
                    entry = self.l2_cache[normalized_key]
                    if not entry.is_expired:
                        # Decompress if needed
                        if entry.compression_ratio < 1.0:
                            data = self._decompress_data(entry.value, entry.compression_ratio)
                            value = self._deserialize_value(data)
                        else:
                            value = entry.value

                        # Promote to L1
                        self._promote_to_l1(normalized_key, value, entry)
                        self.stats.record_hit(CacheLevel.L2_COMPRESSED)
                        self.pattern_analyzer.record_access(normalized_key)
                        return value

                # Cache miss
                self.stats.record_miss()
                return default

        finally:
            self.stats.record_operation_time(time.time() - start_time)

    def put(self,
            key: Union[str, Tuple, Dict],
            value: Any,
            ttl: Optional[float] = None,
            tags: Optional[Set[str]] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put value in cache."""
        start_time = time.time()
        normalized_key = self._generate_key(key)

        try:
            with self.lock:
                current_time = time.time()
                size_bytes = self._calculate_size(value)
                entry_ttl = ttl or self.default_ttl

                # Create cache entry
                entry = CacheEntry(
                    key=normalized_key,
                    value=value,
                    level=CacheLevel.L1_MEMORY,
                    size_bytes=size_bytes,
                    created_at=current_time,
                    accessed_at=current_time,
                    access_count=1,
                    ttl=entry_ttl,
                    metadata=metadata or {},
                    tags=tags or set()
                )

                # Try to add to L1 first
                if self._can_fit_in_l1(size_bytes):
                    self._add_to_l1(normalized_key, entry)
                    self.stats.record_write(CacheLevel.L1_MEMORY)
                else:
                    # Add to L2 with compression
                    self._add_to_l2(normalized_key, entry)
                    self.stats.record_write(CacheLevel.L2_COMPRESSED)

                self.pattern_analyzer.record_access(normalized_key)
                return True

        except Exception as e:
            self.logger.error(f"Error putting value in cache: {e}")
            return False
        finally:
            self.stats.record_operation_time(time.time() - start_time)

    def _can_fit_in_l1(self, size_bytes: int) -> bool:
        """Check if entry can fit in L1 cache."""
        return len(self.l1_cache) < self.l1_size

    def _add_to_l1(self, key: str, entry: CacheEntry):
        """Add entry to L1 cache."""
        # Evict if necessary
        while len(self.l1_cache) >= self.l1_size:
            self._evict_from_l1()

        entry.level = CacheLevel.L1_MEMORY
        self.l1_cache[key] = entry

    def _add_to_l2(self, key: str, entry: CacheEntry):
        """Add entry to L2 cache with compression."""
        # Compress if enabled and beneficial
        if self.enable_compression and entry.size_bytes >= self.compression_threshold:
            serialized = self._serialize_value(entry.value)
            compressed, ratio = self._compress_data(serialized)
            entry.value = compressed
            entry.compression_ratio = ratio
            entry.size_bytes = len(compressed)

        # Evict if necessary
        while len(self.l2_cache) >= self.l2_size:
            self._evict_from_l2()

        entry.level = CacheLevel.L2_COMPRESSED
        self.l2_cache[key] = entry

    def _promote_to_l1(self, key: str, value: Any, l2_entry: CacheEntry):
        """Promote entry from L2 to L1."""
        if self._can_fit_in_l1(self._calculate_size(value)):
            # Remove from L2
            del self.l2_cache[key]

            # Create new L1 entry
            entry = CacheEntry(
                key=key,
                value=value,
                level=CacheLevel.L1_MEMORY,
                size_bytes=self._calculate_size(value),
                created_at=l2_entry.created_at,
                accessed_at=time.time(),
                access_count=l2_entry.access_count + 1,
                ttl=l2_entry.ttl,
                metadata=l2_entry.metadata,
                tags=l2_entry.tags
            )

            self._add_to_l1(key, entry)
            self.access_frequencies[key] += 1

    def _evict_from_l1(self):
        """Evict entry from L1 cache based on eviction policy."""
        if not self.l1_cache:
            return

        if self.eviction_policy == EvictionPolicy.LRU:
            key, entry = self.l1_cache.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Find least frequently used
            min_freq = min(self.access_frequencies.get(k, 0) for k in self.l1_cache.keys())
            for key in self.l1_cache:
                if self.access_frequencies.get(key, 0) == min_freq:
                    entry = self.l1_cache.pop(key)
                    break
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Find expired or oldest entry
            oldest_key = None
            oldest_time = float('inf')
            for key, entry in self.l1_cache.items():
                if entry.is_expired:
                    oldest_key = key
                    break
                if entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    oldest_key = key
            if oldest_key:
                entry = self.l1_cache.pop(oldest_key)
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # Use pattern analysis for adaptive eviction
            key, entry = self._adaptive_eviction_l1()
        else:  # Default to LRU
            key, entry = self.l1_cache.popitem(last=False)

        # Move to L2 if valuable
        if self._is_worth_keeping(key, entry):
            self._add_to_l2(key, entry)
        else:
            self.stats.record_eviction(CacheLevel.L1_MEMORY)

    def _evict_from_l2(self):
        """Evict entry from L2 cache."""
        if not self.l2_cache:
            return

        # Use similar logic but always evict completely
        if self.eviction_policy == EvictionPolicy.LRU:
            key, entry = self.l2_cache.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.LFU:
            min_freq = min(self.access_frequencies.get(k, 0) for k in self.l2_cache.keys())
            for key in self.l2_cache:
                if self.access_frequencies.get(key, 0) == min_freq:
                    entry = self.l2_cache.pop(key)
                    break
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            key, entry = self._adaptive_eviction_l2()
        else:
            key, entry = self.l2_cache.popitem(last=False)

        self.stats.record_eviction(CacheLevel.L2_COMPRESSED)

    def _adaptive_eviction_l1(self) -> Tuple[str, CacheEntry]:
        """Adaptive eviction for L1 based on access patterns."""
        candidates = []

        for key, entry in self.l1_cache.items():
            # Calculate eviction score
            age_score = entry.age / (self.default_ttl or 3600)
            frequency_score = 1.0 / (self.access_frequencies.get(key, 1))

            # Pattern-based score
            access_prob = self.pattern_analyzer.predict_access_probability(key, time.time() + 300)  # 5 min
            pattern_score = 1.0 - access_prob

            # Combined score (higher = better candidate for eviction)
            total_score = age_score * 0.4 + frequency_score * 0.3 + pattern_score * 0.3
            candidates.append((total_score, key, entry))

        # Sort by score (highest first)
        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][2]

    def _adaptive_eviction_l2(self) -> Tuple[str, CacheEntry]:
        """Adaptive eviction for L2 based on access patterns."""
        # Similar to L1 but with different weights
        candidates = []

        for key, entry in self.l2_cache.items():
            age_score = entry.age / (self.default_ttl or 3600)
            frequency_score = 1.0 / (self.access_frequencies.get(key, 1))
            access_prob = self.pattern_analyzer.predict_access_probability(key, time.time() + 900)  # 15 min
            pattern_score = 1.0 - access_prob

            # L2 weights favor frequency less (since they're already demoted)
            total_score = age_score * 0.5 + frequency_score * 0.2 + pattern_score * 0.3
            candidates.append((total_score, key, entry))

        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][2]

    def _is_worth_keeping(self, key: str, entry: CacheEntry) -> bool:
        """Determine if entry is worth keeping in L2."""
        # Consider access frequency and pattern
        frequency = self.access_frequencies.get(key, 0)
        future_access_prob = self.pattern_analyzer.predict_access_probability(key, time.time() + 1800)

        return frequency >= 2 or future_access_prob > 0.3

    def invalidate(self, key: Union[str, Tuple, Dict]) -> bool:
        """Invalidate cache entry."""
        normalized_key = self._generate_key(key)

        with self.lock:
            removed = False
            if normalized_key in self.l1_cache:
                del self.l1_cache[normalized_key]
                removed = True
            if normalized_key in self.l2_cache:
                del self.l2_cache[normalized_key]
                removed = True
            if normalized_key in self.access_frequencies:
                del self.access_frequencies[normalized_key]

            return removed

    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with any of the specified tags."""
        count = 0

        with self.lock:
            # Check L1
            keys_to_remove = []
            for key, entry in self.l1_cache.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.l1_cache[key]
                count += 1

            # Check L2
            keys_to_remove = []
            for key, entry in self.l2_cache.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.l2_cache[key]
                count += 1

        return count

    def clear(self):
        """Clear all cache levels."""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.access_frequencies.clear()
            self.computing.clear()
            self.computation_results.clear()

    async def get_or_compute(self,
                           key: Union[str, Tuple, Dict],
                           compute_func: Callable,
                           *args,
                           ttl: Optional[float] = None,
                           tags: Optional[Set[str]] = None,
                           **kwargs) -> Any:
        """Get value from cache or compute if not present."""
        normalized_key = self._generate_key(key)

        # Try to get from cache first
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        # Check if already computing
        async with self.async_lock:
            if normalized_key in self.computing:
                # Wait for computation to complete
                await self.computing[normalized_key].wait()
                return self.computation_results.get(normalized_key)

            # Start computation
            self.computing[normalized_key] = asyncio.Event()

        try:
            # Compute value
            if asyncio.iscoroutinefunction(compute_func):
                result = await compute_func(*args, **kwargs)
            else:
                result = compute_func(*args, **kwargs)

            # Cache result
            self.put(key, result, ttl=ttl, tags=tags)

            # Store result temporarily
            self.computation_results[normalized_key] = result

            return result

        finally:
            # Signal completion
            async with self.async_lock:
                if normalized_key in self.computing:
                    self.computing[normalized_key].set()
                    del self.computing[normalized_key]

                # Clean up temporary result after a delay
                asyncio.create_task(self._cleanup_computation_result(normalized_key))

    async def _cleanup_computation_result(self, key: str):
        """Clean up temporary computation result."""
        await asyncio.sleep(60)  # Keep for 1 minute
        if key in self.computation_results:
            del self.computation_results[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            l1_entries = len(self.l1_cache)
            l2_entries = len(self.l2_cache)
            total_entries = l1_entries + l2_entries

            l1_size_bytes = sum(entry.size_bytes for entry in self.l1_cache.values())
            l2_size_bytes = sum(entry.size_bytes for entry in self.l2_cache.values())
            total_size_bytes = l1_size_bytes + l2_size_bytes

            stats = self.stats.get_summary()
            pattern_summary = self.pattern_analyzer.get_pattern_summary()

            return {
                **stats,
                "cache_levels": {
                    "l1": {
                        "entries": l1_entries,
                        "max_entries": self.l1_size,
                        "utilization": l1_entries / self.l1_size,
                        "size_bytes": l1_size_bytes
                    },
                    "l2": {
                        "entries": l2_entries,
                        "max_entries": self.l2_size,
                        "utilization": l2_entries / self.l2_size,
                        "size_bytes": l2_size_bytes
                    }
                },
                "total_entries": total_entries,
                "total_size_bytes": total_size_bytes,
                "eviction_policy": self.eviction_policy.value,
                "access_patterns": pattern_summary,
                "computing_entries": len(self.computing)
            }

    def generate_cache_report(self) -> str:
        """Generate comprehensive cache performance report."""
        stats = self.get_stats()

        report = []
        report.append("# Intelligent Cache Performance Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overview
        report.append("## Cache Overview")
        report.append(f"- Hit Rate: {stats['hit_rate']*100:.1f}%")
        report.append(f"- Total Requests: {stats['total_requests']:,}")
        report.append(f"- Total Entries: {stats['total_entries']:,}")
        report.append(f"- Total Size: {stats['total_size_bytes'] / (1024*1024):.1f} MB")
        report.append(f"- Average Operation Time: {stats['average_operation_time_ms']:.2f} ms")
        report.append("")

        # Cache Levels
        report.append("## Cache Level Performance")
        for level, level_stats in stats['cache_levels'].items():
            utilization = level_stats['utilization'] * 100
            size_mb = level_stats['size_bytes'] / (1024*1024)
            report.append(f"### {level.upper()}")
            report.append(f"- Entries: {level_stats['entries']:,}/{level_stats['max_entries']:,} ({utilization:.1f}%)")
            report.append(f"- Size: {size_mb:.1f} MB")
            report.append("")

        # Access Patterns
        if stats['access_patterns']:
            patterns = stats['access_patterns']
            report.append("## Access Pattern Analysis")
            report.append(f"- Monitored Keys: {patterns['total_keys']:,}")
            report.append(f"- High Confidence Predictions: {patterns['high_confidence_patterns']:,}")
            report.append("- Pattern Distribution:")
            for pattern_type, count in patterns['pattern_types'].items():
                report.append(f"  * {pattern_type}: {count}")
            report.append("")

        # Performance Insights
        report.append("## Performance Insights")
        if stats['hit_rate'] > 0.8:
            report.append("✅ Excellent cache hit rate")
        elif stats['hit_rate'] > 0.6:
            report.append("⚠️ Good cache hit rate, room for improvement")
        else:
            report.append("❌ Low cache hit rate, optimization needed")

        if stats['average_operation_time_ms'] < 1.0:
            report.append("✅ Excellent cache response times")
        elif stats['average_operation_time_ms'] < 5.0:
            report.append("⚠️ Acceptable cache response times")
        else:
            report.append("❌ Slow cache response times")

        # Recommendations
        report.append("\n## Recommendations")
        l1_util = stats['cache_levels']['l1']['utilization']
        l2_util = stats['cache_levels']['l2']['utilization']

        if l1_util > 0.9:
            report.append("- Consider increasing L1 cache size")
        if l2_util > 0.9:
            report.append("- Consider increasing L2 cache size")
        if stats['hit_rate'] < 0.7:
            report.append("- Review cache TTL settings")
            report.append("- Analyze access patterns for optimization")
        if stats['average_operation_time_ms'] > 2.0:
            report.append("- Investigate cache performance bottlenecks")

        return "\n".join(report)

    def cleanup_expired(self):
        """Clean up expired entries."""
        current_time = time.time()

        with self.lock:
            # Clean L1
            expired_l1 = [
                key for key, entry in self.l1_cache.items()
                if entry.is_expired
            ]
            for key in expired_l1:
                del self.l1_cache[key]
                self.stats.record_eviction(CacheLevel.L1_MEMORY)

            # Clean L2
            expired_l2 = [
                key for key, entry in self.l2_cache.items()
                if entry.is_expired
            ]
            for key in expired_l2:
                del self.l2_cache[key]
                self.stats.record_eviction(CacheLevel.L2_COMPRESSED)

        return len(expired_l1) + len(expired_l2)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_expired()


# Cache decorators
def cached(cache: IntelligentCache,
          ttl: Optional[float] = None,
          key_func: Optional[Callable] = None,
          tags: Optional[Set[str]] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl, tags=tags)
            return result

        return wrapper
    return decorator


def async_cached(cache: IntelligentCache,
                ttl: Optional[float] = None,
                key_func: Optional[Callable] = None,
                tags: Optional[Set[str]] = None):
    """Decorator for caching async function results."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))

            # Use get_or_compute for async functions
            return await cache.get_or_compute(
                cache_key,
                func,
                *args,
                ttl=ttl,
                tags=tags,
                **kwargs
            )

        return wrapper
    return decorator


# Global cache instance
_global_cache: Optional[IntelligentCache] = None


def get_cache() -> IntelligentCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache


def cache_clear():
    """Clear global cache."""
    cache = get_cache()
    cache.clear()