"""
Real-time Performance Monitor for SpeedVQA.

Provides inference time statistics, throughput calculation, GPU memory monitoring,
and real-time metrics updates.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Real-time performance monitor for SpeedVQA inference.
    
    Tracks:
    - Inference time statistics
    - Throughput calculation
    - GPU memory monitoring
    - Real-time metrics updates
    """
    
    def __init__(
        self,
        window_size: int = 100,
        enable_gpu_monitoring: bool = True,
    ):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Size of sliding window for metrics calculation
            enable_gpu_monitoring: Whether to enable GPU memory monitoring
        """
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # Inference time tracking
        self.inference_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # GPU monitoring
        self.enable_gpu_monitoring = enable_gpu_monitoring and HAS_PYNVML
        if self.enable_gpu_monitoring:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.enable_gpu_monitoring = False
        
        # Metrics storage
        self.metrics_history = []
        self.start_time = time.time()
        self.total_samples_processed = 0
    
    def record_inference(
        self,
        inference_time_ms: float,
        batch_size: int = 1,
    ) -> None:
        """
        Record an inference measurement.
        
        Args:
            inference_time_ms: Inference time in milliseconds
            batch_size: Batch size for this inference
        """
        current_time = time.time()
        self.inference_times.append(inference_time_ms)
        self.batch_sizes.append(batch_size)
        self.timestamps.append(current_time)
        self.total_samples_processed += batch_size
    
    def get_inference_time_stats(self) -> Dict[str, float]:
        """
        Get inference time statistics.
        
        Returns:
            Dictionary with inference time statistics
        """
        if not self.inference_times:
            return {
                'mean_ms': 0.0,
                'median_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'std_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'count': 0,
            }
        
        times = np.array(list(self.inference_times))
        
        return {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'std_ms': float(np.std(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'count': len(times),
        }
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """
        Get throughput statistics.
        
        Returns:
            Dictionary with throughput metrics
        """
        if len(self.timestamps) < 2:
            return {
                'samples_per_sec': 0.0,
                'inferences_per_sec': 0.0,
                'total_samples': 0,
                'total_inferences': 0,
            }
        
        # Calculate throughput from sliding window
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span == 0:
            time_span = 1e-6
        
        total_samples = sum(self.batch_sizes)
        total_inferences = len(self.inference_times)
        
        samples_per_sec = total_samples / time_span
        inferences_per_sec = total_inferences / time_span
        
        return {
            'samples_per_sec': float(samples_per_sec),
            'inferences_per_sec': float(inferences_per_sec),
            'total_samples': int(total_samples),
            'total_inferences': int(total_inferences),
        }
    
    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """
        Get GPU memory statistics.
        
        Returns:
            Dictionary with GPU memory metrics
        """
        if not self.enable_gpu_monitoring:
            return {
                'used_mb': 0.0,
                'total_mb': 0.0,
                'free_mb': 0.0,
                'utilization_percent': 0.0,
            }
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_mb = mem_info.used / (1024 * 1024)
            total_mb = mem_info.total / (1024 * 1024)
            free_mb = mem_info.free / (1024 * 1024)
            utilization = (mem_info.used / mem_info.total) * 100
            
            return {
                'used_mb': float(used_mb),
                'total_mb': float(total_mb),
                'free_mb': float(free_mb),
                'utilization_percent': float(utilization),
            }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory stats: {e}")
            return {
                'used_mb': 0.0,
                'total_mb': 0.0,
                'free_mb': 0.0,
                'utilization_percent': 0.0,
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all current metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        elapsed_time = time.time() - self.start_time
        
        metrics = {
            'timestamp': time.time(),
            'elapsed_time_sec': float(elapsed_time),
            'total_samples_processed': int(self.total_samples_processed),
            'inference_time_stats': self.get_inference_time_stats(),
            'throughput_stats': self.get_throughput_stats(),
            'gpu_memory_stats': self.get_gpu_memory_stats(),
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_metrics_summary(self) -> str:
        """
        Get formatted metrics summary.
        
        Returns:
            Formatted string with metrics summary
        """
        metrics = self.get_all_metrics()
        
        inference_stats = metrics['inference_time_stats']
        throughput_stats = metrics['throughput_stats']
        gpu_stats = metrics['gpu_memory_stats']
        
        summary = (
            f"\n{'='*60}\n"
            f"Performance Metrics Summary\n"
            f"{'='*60}\n"
            f"Elapsed Time: {metrics['elapsed_time_sec']:.2f}s\n"
            f"Total Samples: {metrics['total_samples_processed']}\n"
            f"\nInference Time Statistics:\n"
            f"  Mean: {inference_stats['mean_ms']:.2f}ms\n"
            f"  Median: {inference_stats['median_ms']:.2f}ms\n"
            f"  P95: {inference_stats['p95_ms']:.2f}ms\n"
            f"  P99: {inference_stats['p99_ms']:.2f}ms\n"
            f"  Std Dev: {inference_stats['std_ms']:.2f}ms\n"
            f"\nThroughput:\n"
            f"  Samples/sec: {throughput_stats['samples_per_sec']:.1f}\n"
            f"  Inferences/sec: {throughput_stats['inferences_per_sec']:.1f}\n"
        )
        
        if self.enable_gpu_monitoring:
            summary += (
                f"\nGPU Memory:\n"
                f"  Used: {gpu_stats['used_mb']:.1f}MB\n"
                f"  Total: {gpu_stats['total_mb']:.1f}MB\n"
                f"  Utilization: {gpu_stats['utilization_percent']:.1f}%\n"
            )
        
        summary += f"{'='*60}\n"
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.inference_times.clear()
        self.batch_sizes.clear()
        self.timestamps.clear()
        self.metrics_history.clear()
        self.start_time = time.time()
        self.total_samples_processed = 0
        self.logger.info("Performance monitor reset")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Get metrics history.
        
        Returns:
            List of historical metrics
        """
        return self.metrics_history.copy()
    
    def verify_performance_target(
        self,
        target_latency_ms: float = 50.0,
        target_throughput_sps: Optional[float] = None,
    ) -> Dict[str, bool]:
        """
        Verify if performance targets are met.
        
        Args:
            target_latency_ms: Target latency in milliseconds
            target_throughput_sps: Target throughput in samples per second
            
        Returns:
            Dictionary with verification results
        """
        inference_stats = self.get_inference_time_stats()
        throughput_stats = self.get_throughput_stats()
        
        results = {
            'latency_target_met': inference_stats['mean_ms'] < target_latency_ms,
            'mean_latency_ms': inference_stats['mean_ms'],
            'target_latency_ms': target_latency_ms,
        }
        
        if target_throughput_sps is not None:
            results['throughput_target_met'] = (
                throughput_stats['samples_per_sec'] >= target_throughput_sps
            )
            results['actual_throughput_sps'] = throughput_stats['samples_per_sec']
            results['target_throughput_sps'] = target_throughput_sps
        
        return results
