"""
FencerAI Performance Profiling Utilities
=======================================
Version: 1.0 | Last Updated: 2026-03-27

Performance monitoring for latency and memory profiling.
Designed for edge deployment validation (<150ms target).

Usage:
    from src.utils.profiling import LatencyProfiler, MemoryProfiler

    with LatencyProfiler("pose_estimation") as profiler:
        poses = estimator.estimate(frame)

    print(profiler.report())  # Shows mean, p50, p95, p99 latency
"""

from __future__ import annotations

import gc
import time
import psutil
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from functools import wraps

import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    values_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        return np.percentile(self.values_ms, 50) if self.values_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return np.percentile(self.values_ms, 95) if self.values_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return np.percentile(self.values_ms, 99) if self.values_ms else 0.0

    def add(self, latency_ms: float) -> None:
        """Add a latency measurement."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.values_ms.append(latency_ms)

    def report(self) -> str:
        """Generate a report string."""
        if self.count == 0:
            return f"{self.name}: No measurements"

        return (
            f"{self.name}:\n"
            f"  Count: {self.count}\n"
            f"  Mean:  {self.mean_ms:.2f}ms\n"
            f"  Min:   {self.min_ms:.2f}ms\n"
            f"  Max:   {self.max_ms:.2f}ms\n"
            f"  P50:   {self.p50_ms:.2f}ms\n"
            f"  P95:   {self.p95_ms:.2f}ms\n"
            f"  P99:   {self.p99_ms:.2f}ms"
        )


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size


# =============================================================================
# Latency Profiler
# =============================================================================

class LatencyProfiler:
    """
    Context manager for profiling latency of code blocks.

    Example:
        profiler = LatencyProfiler()

        with profiler.profile("pose_estimation"):
            poses = estimator.estimate(frame)

        with profiler.profile("tracking"):
            tracked = tracker.update(poses)

        print(profiler.report())
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.stats: Dict[str, LatencyStats] = {}
        self._active: Dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def profile(self, operation: str):
        """Context manager for profiling an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            with self._lock:
                if operation not in self.stats:
                    self.stats[operation] = LatencyStats(name=operation)
                self.stats[operation].add(elapsed)

    def get_stats(self, operation: str) -> LatencyStats:
        """Get stats for a specific operation."""
        return self.stats[operation]

    def report(self) -> str:
        """Generate a full report."""
        lines = [f"Latency Profiler: {self.name}", "=" * 50]
        for op, stats in sorted(self.stats.items()):
            lines.append(stats.report())
            lines.append("-" * 50)
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self.stats.clear()
        self._active.clear()

    def total_latency_ms(self) -> float:
        """Get total latency across all operations."""
        return sum(s.total_ms for s in self.stats.values())

    def latency_by_operation(self) -> Dict[str, float]:
        """Get mean latency by operation."""
        return {op: stats.mean_ms for op, stats in self.stats.items()}


# =============================================================================
# Memory Profiler
# =============================================================================

class MemoryProfiler:
    """
    Profile memory usage of code blocks.

    Example:
        profiler = MemoryProfiler()

        with profiler.profile("frame_processing"):
            process_frame(frame)

        print(profiler.report())
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.snapshots: List[MemorySnapshot] = []
        self._sampling = False
        self._sample_thread: Optional[threading.Thread] = None
        self._sample_interval_s: float = 0.1
        self._peak_rss_mb: float = 0.0
        self._lock = threading.Lock()

    def _get_memory_mb(self) -> tuple[float, float]:
        """Get current memory usage in MB."""
        process = psutil.Process()
        mem_info = process.memory_info()
        return (
            mem_info.rss / (1024 * 1024),  # RSS in MB
            mem_info.vms / (1024 * 1024),  # VMS in MB
        )

    @contextmanager
    def profile(self, operation: str = "default"):
        """Context manager for profiling memory during an operation."""
        gc.collect()  # Clean up before measuring
        rss_start, vms_start = self._get_memory_mb()

        # Start background sampling if not already
        self._start_sampling()

        try:
            yield
        finally:
            gc.collect()
            rss_end, vms_end = self._get_memory_mb()

            with self._lock:
                snapshot = MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=rss_end - rss_start,
                    vms_mb=vms_end - vms_start,
                )
                self.snapshots.append(snapshot)
                self._peak_rss_mb = max(self._peak_rss_mb, rss_end)

    def _start_sampling(self):
        """Start background memory sampling."""
        if self._sampling:
            return

        self._sampling = True
        self._sample_thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._sample_thread.start()

    def _sample_loop(self):
        """Background sampling loop."""
        while self._sampling:
            rss, vms = self._get_memory_mb()
            with self._lock:
                self.snapshots.append(MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=rss,
                    vms_mb=vms,
                ))
            time.sleep(self._sample_interval_s)

    def stop_sampling(self):
        """Stop background sampling."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
            self._sample_thread = None

    def report(self) -> str:
        """Generate a memory report."""
        if not self.snapshots:
            return f"Memory Profiler: {self.name} - No measurements"

        rss_values = [s.rss_mb for s in self.snapshots]
        vms_values = [s.vms_mb for s in self.snapshots]

        lines = [
            f"Memory Profiler: {self.name}",
            "=" * 50,
            f"Peak RSS: {self._peak_rss_mb:.2f} MB",
            f"Samples:  {len(self.snapshots)}",
            f"Mean RSS: {np.mean(rss_values):.2f} MB",
            f"Max RSS:  {np.max(rss_values):.2f} MB",
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self.snapshots.clear()
        self._peak_rss_mb = 0.0

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_sampling()


# =============================================================================
# Decorators
# =============================================================================

def profile_latency(operation: str, profiler: Optional[LatencyProfiler] = None):
    """
    Decorator to profile function execution time.

    Args:
        operation: Name of the operation for reporting
        profiler: Optional shared profiler instance

    Example:
        @profile_latency("pose_estimation")
        def estimate_poses(frame):
            return estimator.estimate(frame)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                if profiler is not None:
                    profiler.stats[operation].add(elapsed)
        return wrapper
    return decorator


# =============================================================================
# Pipeline Performance Monitor
# =============================================================================

class PipelineMonitor:
    """
    Monitor performance of the full perception pipeline.

    Tracks per-stage latency and memory usage.

    Example:
        monitor = PipelineMonitor()

        monitor.start()
        # ... process frames ...
        monitor.stop()

        print(monitor.summary())
    """

    def __init__(self):
        self.latency_profiler = LatencyProfiler("pipeline")
        self.memory_profiler = MemoryProfiler("pipeline")
        self._enabled = False

    def start(self):
        """Start monitoring."""
        self._enabled = True
        self.latency_profiler.reset()
        self.memory_profiler.reset()

    def stop(self):
        """Stop monitoring."""
        self._enabled = False
        self.memory_profiler.stop_sampling()

    @contextmanager
    def stage(self, name: str):
        """Profile a pipeline stage."""
        if not self._enabled:
            yield
            return

        with self.latency_profiler.profile(name):
            with self.memory_profiler.profile(name):
                yield

    def summary(self) -> str:
        """Get performance summary."""
        latency_report = self.latency_profiler.report()
        memory_report = self.memory_profiler.report()

        # Calculate total pipeline latency
        total = self.latency_profiler.total_latency_ms()
        by_op = self.latency_profiler.latency_by_operation()

        summary_lines = [
            "=" * 60,
            "PIPELINE PERFORMANCE SUMMARY",
            "=" * 60,
            f"Total Pipeline Latency: {total:.2f}ms",
            "",
            "Latency by Stage:",
        ]

        for op, ms in sorted(by_op.items(), key=lambda x: -x[1]):
            pct = (ms / total * 100) if total > 0 else 0
            summary_lines.append(f"  {op:30s}: {ms:7.2f}ms ({pct:5.1f}%)")

        summary_lines.extend([
            "",
            "-" * 60,
            memory_report,
            "=" * 60,
        ])

        return "\n".join(summary_lines)

    def meets_target(self, target_ms: float = 150.0) -> bool:
        """Check if total latency meets target."""
        return self.latency_profiler.total_latency_ms() <= target_ms


# =============================================================================
# Utility Functions
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information for profiling context."""
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_percent": psutil.virtual_memory().percent,
    }


def format_latency_table(results: Dict[str, List[float]]) -> str:
    """
    Format latency results as a table.

    Args:
        results: Dict mapping operation names to list of measurements

    Returns:
        Formatted table string
    """
    if not results:
        return "No results"

    lines = [
        "Operation".ljust(30) + "Mean".rjust(10) + "P50".rjust(10) + "P95".rjust(10) + "P99".rjust(10),
        "-" * 70,
    ]

    for name, values in sorted(results.items()):
        if not values:
            continue
        mean = np.mean(values)
        p50 = np.percentile(values, 50)
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)
        lines.append(
            f"{name[:30]:30s}"
            f"{mean:9.2f}ms"
            f"{p50:9.2f}ms"
            f"{p95:9.2f}ms"
            f"{p99:9.2f}ms"
        )

    return "\n".join(lines)


# =============================================================================
# Pipeline Health Monitor
# =============================================================================

@dataclass
class HealthMetrics:
    """Health metrics for a single frame."""
    frame_id: int
    timestamp: float
    n_detections: int
    n_tracked: int
    mean_confidence: float
    min_confidence: float
    processing_time_ms: float
    is_healthy: bool
    issues: List[str]


class HealthMonitor:
    """
    Monitor pipeline health metrics.

    Tracks:
    - Detection consistency (are both fencers detected?)
    - Confidence scores (are they healthy?)
    - Processing time (are we keeping up with real-time?)
    - Error rates

    Example:
        monitor = HealthMonitor()

        for frame in video:
            monitor.start_frame(frame_id, timestamp)
            # ... process frame ...
            issues = monitor.end_frame(
                n_detections=2,
                confidences=[0.9, 0.8],
                processing_time_ms=120.0
            )
            if issues:
                logger.warning(f"Health issues: {issues}")

        print(monitor.summary())
    """

    def __init__(
        self,
        min_detections: int = 2,
        min_confidence: float = 0.3,
        max_processing_time_ms: float = 500.0,
    ):
        """
        Initialize health monitor.

        Args:
            min_detections: Minimum expected detections per frame
            min_confidence: Minimum acceptable confidence score
            max_processing_time_ms: Maximum acceptable processing time per frame
        """
        self.min_detections = min_detections
        self.min_confidence = min_confidence
        self.max_processing_time_ms = max_processing_time_ms

        self._metrics: List[HealthMetrics] = []
        self._frame_start_times: Dict[int, float] = {}
        self._current_frame_id: Optional[int] = None

    def start_frame(self, frame_id: int, timestamp: float) -> None:
        """Mark the start of a frame processing."""
        self._current_frame_id = frame_id
        self._frame_start_times[frame_id] = time.perf_counter()

    def end_frame(
        self,
        n_detections: int,
        confidences: List[float],
        processing_time_ms: Optional[float] = None,
    ) -> List[str]:
        """
        Mark the end of frame processing and record health metrics.

        Args:
            n_detections: Number of fencers detected
            confidences: List of confidence scores for each detection
            processing_time_ms: Actual processing time (if None, calculated)

        Returns:
            List of health issues detected
        """
        if self._current_frame_id is None:
            return ["No frame started"]

        frame_id = self._current_frame_id
        timestamp = self._frame_start_times.get(frame_id, 0.0)

        if processing_time_ms is None:
            processing_time_ms = (time.perf_counter() - timestamp) * 1000

        issues: List[str] = []

        # Check detection count
        if n_detections < self.min_detections:
            issues.append(f"Low detections: {n_detections}/{self.min_detections}")

        # Check confidence scores
        if confidences:
            mean_conf = np.mean(confidences)
            min_conf = np.min(confidences)

            if mean_conf < self.min_confidence:
                issues.append(f"Low mean confidence: {mean_conf:.2f}")

            if min_conf < self.min_confidence:
                issues.append(f"Low min confidence: {min_conf:.2f}")
        elif n_detections > 0:
            issues.append("No confidence scores provided")

        # Check processing time
        if processing_time_ms > self.max_processing_time_ms:
            issues.append(f"Slow processing: {processing_time_ms:.0f}ms")

        # Determine overall health
        is_healthy = len(issues) == 0

        # Record metrics
        metrics = HealthMetrics(
            frame_id=frame_id,
            timestamp=timestamp,
            n_detections=n_detections,
            n_tracked=min(n_detections, 2),  # Assume tracked = detected for now
            mean_confidence=np.mean(confidences) if confidences else 0.0,
            min_confidence=np.min(confidences) if confidences else 0.0,
            processing_time_ms=processing_time_ms,
            is_healthy=is_healthy,
            issues=issues,
        )
        self._metrics.append(metrics)

        self._current_frame_id = None
        return issues

    def record_skip(self, frame_id: int, reason: str = "skip") -> None:
        """Record a skipped frame."""
        metrics = HealthMetrics(
            frame_id=frame_id,
            timestamp=0.0,
            n_detections=0,
            n_tracked=0,
            mean_confidence=0.0,
            min_confidence=0.0,
            processing_time_ms=0.0,
            is_healthy=False,
            issues=[f"Frame skipped: {reason}"],
        )
        self._metrics.append(metrics)

    def summary(self) -> str:
        """Get health summary report."""
        if not self._metrics:
            return "Health Monitor: No metrics recorded"

        total_frames = len(self._metrics)
        healthy_frames = sum(1 for m in self._metrics if m.is_healthy)
        unhealthy_frames = total_frames - healthy_frames

        issues_count: Dict[str, int] = defaultdict(int)
        for m in self._metrics:
            for issue in m.issues:
                # Normalize issue string for counting
                normalized = issue.split(":")[0]  # Take first part before colon
                issues_count[normalized] += 1

        # Detection stats
        detection_rates = [m.n_detections for m in self._metrics]
        avg_detections = np.mean(detection_rates)

        # Confidence stats
        confidences = [m.mean_confidence for m in self._metrics if m.mean_confidence > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Processing time stats
        processing_times = [m.processing_time_ms for m in self._metrics if m.processing_time_ms > 0]
        avg_processing = np.mean(processing_times) if processing_times else 0.0

        lines = [
            "=" * 60,
            "PIPELINE HEALTH SUMMARY",
            "=" * 60,
            f"Total Frames:    {total_frames}",
            f"Healthy Frames:  {healthy_frames} ({healthy_frames/total_frames*100:.1f}%)",
            f"Unhealthy:       {unhealthy_frames} ({unhealthy_frames/total_frames*100:.1f}%)",
            "",
            "Detection Quality:",
            f"  Avg Detections/Frame: {avg_detections:.1f}",
            f"  Avg Confidence:      {avg_confidence:.2f}",
            "",
            "Processing Performance:",
            f"  Avg Processing Time:  {avg_processing:.1f}ms",
            f"  Real-time Capable:  {'Yes' if avg_processing < 33.33 else 'No'} (33.33ms = 30fps)",
            "",
            "Top Issues:",
        ]

        if issues_count:
            for issue, count in sorted(issues_count.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {issue}: {count}")
        else:
            lines.append("  None")

        lines.append("=" * 60)

        return "\n".join(lines)

    def is_healthy(self) -> bool:
        """Check if pipeline is currently healthy."""
        if not self._metrics:
            return True
        # Consider healthy if 90%+ frames are healthy
        recent = self._metrics[-100:]  # Last 100 frames
        healthy_count = sum(1 for m in recent if m.is_healthy)
        return healthy_count / len(recent) >= 0.9 if recent else True

    def get_metrics(self) -> List[HealthMetrics]:
        """Get all recorded metrics."""
        return self._metrics.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._frame_start_times.clear()
        self._current_frame_id = None
