"""
Property-based tests for Performance Monitor.

Tests monitoring accuracy, real-time metric updates, and consistency.
"""

import pytest
import time
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from speedvqa.monitoring.performance_monitor import PerformanceMonitor


@pytest.fixture
def monitor():
    """Create a performance monitor instance."""
    return PerformanceMonitor(window_size=100)


class TestPerformanceMonitorInitialization:
    """Tests for PerformanceMonitor initialization."""
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        monitor = PerformanceMonitor(window_size=50)
        assert monitor.window_size == 50
        assert monitor.total_samples_processed == 0
    
    def test_monitor_with_gpu_monitoring(self):
        """Test monitor with GPU monitoring enabled."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=True)
        assert monitor.enable_gpu_monitoring is not None
    
    def test_monitor_without_gpu_monitoring(self):
        """Test monitor with GPU monitoring disabled."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        assert not monitor.enable_gpu_monitoring


class TestInferenceRecording:
    """Tests for recording inference measurements."""
    
    def test_record_single_inference(self, monitor):
        """Test recording a single inference."""
        monitor.record_inference(10.5, batch_size=1)
        assert monitor.total_samples_processed == 1
    
    def test_record_multiple_inferences(self, monitor):
        """Test recording multiple inferences."""
        for i in range(5):
            monitor.record_inference(10.0 + i, batch_size=1)
        assert monitor.total_samples_processed == 5
    
    def test_record_batch_inference(self, monitor):
        """Test recording batch inference."""
        monitor.record_inference(20.0, batch_size=4)
        assert monitor.total_samples_processed == 4
    
    def test_record_multiple_batch_inferences(self, monitor):
        """Test recording multiple batch inferences."""
        monitor.record_inference(20.0, batch_size=4)
        monitor.record_inference(25.0, batch_size=8)
        assert monitor.total_samples_processed == 12
    
    def test_window_size_limit(self):
        """Test that window size limits stored measurements."""
        monitor = PerformanceMonitor(window_size=10)
        
        for i in range(20):
            monitor.record_inference(10.0, batch_size=1)
        
        # Should only keep last 10 measurements
        assert len(monitor.inference_times) == 10
        assert monitor.total_samples_processed == 20


class TestInferenceTimeStatistics:
    """Tests for inference time statistics."""
    
    def test_inference_time_stats_structure(self, monitor):
        """Test that inference time stats have correct structure."""
        monitor.record_inference(10.0, batch_size=1)
        monitor.record_inference(12.0, batch_size=1)
        monitor.record_inference(11.0, batch_size=1)
        
        stats = monitor.get_inference_time_stats()
        
        assert 'mean_ms' in stats
        assert 'median_ms' in stats
        assert 'min_ms' in stats
        assert 'max_ms' in stats
        assert 'std_ms' in stats
        assert 'p95_ms' in stats
        assert 'p99_ms' in stats
        assert 'count' in stats
    
    def test_inference_time_stats_values(self, monitor):
        """Test that inference time stats have correct values."""
        times = [10.0, 12.0, 11.0, 13.0, 9.0]
        for t in times:
            monitor.record_inference(t, batch_size=1)
        
        stats = monitor.get_inference_time_stats()
        
        assert stats['mean_ms'] == pytest.approx(11.0)
        assert stats['median_ms'] == 11.0
        assert stats['min_ms'] == 9.0
        assert stats['max_ms'] == 13.0
        assert stats['count'] == 5
    
    def test_inference_time_stats_empty(self, monitor):
        """Test inference time stats when no data."""
        stats = monitor.get_inference_time_stats()
        
        assert stats['mean_ms'] == 0.0
        assert stats['count'] == 0
    
    def test_percentile_calculations(self, monitor):
        """Test percentile calculations."""
        # Create 100 measurements
        for i in range(100):
            monitor.record_inference(float(i), batch_size=1)
        
        stats = monitor.get_inference_time_stats()
        
        # P95 should be around 95
        assert 90 < stats['p95_ms'] < 100
        # P99 should be around 99
        assert 95 < stats['p99_ms'] < 100


class TestThroughputStatistics:
    """Tests for throughput statistics."""
    
    def test_throughput_stats_structure(self, monitor):
        """Test that throughput stats have correct structure."""
        monitor.record_inference(10.0, batch_size=1)
        time.sleep(0.01)
        monitor.record_inference(10.0, batch_size=1)
        
        stats = monitor.get_throughput_stats()
        
        assert 'samples_per_sec' in stats
        assert 'inferences_per_sec' in stats
        assert 'total_samples' in stats
        assert 'total_inferences' in stats
    
    def test_throughput_calculation(self, monitor):
        """Test throughput calculation."""
        # Record 10 inferences with batch size 1
        for _ in range(10):
            monitor.record_inference(10.0, batch_size=1)
        
        stats = monitor.get_throughput_stats()
        
        assert stats['total_samples'] == 10
        assert stats['total_inferences'] == 10
        assert stats['samples_per_sec'] > 0
        assert stats['inferences_per_sec'] > 0
    
    def test_batch_throughput_calculation(self, monitor):
        """Test throughput with batch processing."""
        monitor.record_inference(20.0, batch_size=4)
        monitor.record_inference(20.0, batch_size=4)
        
        stats = monitor.get_throughput_stats()
        
        assert stats['total_samples'] == 8
        assert stats['total_inferences'] == 2
    
    def test_throughput_empty(self, monitor):
        """Test throughput stats when no data."""
        stats = monitor.get_throughput_stats()
        
        assert stats['samples_per_sec'] == 0.0
        assert stats['total_samples'] == 0


class TestGPUMemoryMonitoring:
    """Tests for GPU memory monitoring."""
    
    def test_gpu_memory_stats_structure(self, monitor):
        """Test that GPU memory stats have correct structure."""
        stats = monitor.get_gpu_memory_stats()
        
        assert 'used_mb' in stats
        assert 'total_mb' in stats
        assert 'free_mb' in stats
        assert 'utilization_percent' in stats
    
    def test_gpu_memory_stats_values_valid(self, monitor):
        """Test that GPU memory stats have valid values."""
        stats = monitor.get_gpu_memory_stats()
        
        # All values should be non-negative
        assert stats['used_mb'] >= 0
        assert stats['total_mb'] >= 0
        assert stats['free_mb'] >= 0
        assert 0 <= stats['utilization_percent'] <= 100


class TestAllMetrics:
    """Tests for getting all metrics."""
    
    def test_get_all_metrics_structure(self, monitor):
        """Test that all metrics have correct structure."""
        monitor.record_inference(10.0, batch_size=1)
        
        metrics = monitor.get_all_metrics()
        
        assert 'timestamp' in metrics
        assert 'elapsed_time_sec' in metrics
        assert 'total_samples_processed' in metrics
        assert 'inference_time_stats' in metrics
        assert 'throughput_stats' in metrics
        assert 'gpu_memory_stats' in metrics
    
    def test_metrics_history_tracking(self, monitor):
        """Test that metrics history is tracked."""
        monitor.record_inference(10.0, batch_size=1)
        metrics1 = monitor.get_all_metrics()
        
        time.sleep(0.01)
        
        monitor.record_inference(12.0, batch_size=1)
        metrics2 = monitor.get_all_metrics()
        
        history = monitor.get_metrics_history()
        assert len(history) == 2
        assert history[0]['timestamp'] <= history[1]['timestamp']
    
    def test_elapsed_time_tracking(self, monitor):
        """Test that elapsed time is tracked correctly."""
        start_time = time.time()
        monitor.record_inference(10.0, batch_size=1)
        metrics = monitor.get_all_metrics()
        
        elapsed = time.time() - start_time
        
        # Elapsed time should be close to actual elapsed time
        assert metrics['elapsed_time_sec'] >= 0
        assert metrics['elapsed_time_sec'] <= elapsed + 0.1


class TestMetricsSummary:
    """Tests for metrics summary."""
    
    def test_metrics_summary_format(self, monitor):
        """Test that metrics summary is properly formatted."""
        monitor.record_inference(10.0, batch_size=1)
        monitor.record_inference(12.0, batch_size=1)
        
        summary = monitor.get_metrics_summary()
        
        assert isinstance(summary, str)
        assert 'Performance Metrics Summary' in summary
        assert 'Inference Time Statistics' in summary
        assert 'Throughput' in summary
    
    def test_metrics_summary_contains_values(self, monitor):
        """Test that metrics summary contains actual values."""
        monitor.record_inference(10.0, batch_size=1)
        monitor.record_inference(12.0, batch_size=1)
        
        summary = monitor.get_metrics_summary()
        
        # Should contain numeric values
        assert 'ms' in summary
        assert 'sec' in summary


class TestMonitorReset:
    """Tests for monitor reset functionality."""
    
    def test_reset_clears_data(self, monitor):
        """Test that reset clears all data."""
        monitor.record_inference(10.0, batch_size=1)
        monitor.record_inference(12.0, batch_size=1)
        
        assert monitor.total_samples_processed == 2
        
        monitor.reset()
        
        assert monitor.total_samples_processed == 0
        assert len(monitor.inference_times) == 0
        assert len(monitor.metrics_history) == 0
    
    def test_reset_allows_new_measurements(self, monitor):
        """Test that reset allows new measurements."""
        monitor.record_inference(10.0, batch_size=1)
        monitor.reset()
        
        monitor.record_inference(20.0, batch_size=1)
        
        stats = monitor.get_inference_time_stats()
        assert stats['mean_ms'] == 20.0


class TestPerformanceTargetVerification:
    """Tests for performance target verification."""
    
    def test_latency_target_verification(self, monitor):
        """Test latency target verification."""
        monitor.record_inference(30.0, batch_size=1)
        monitor.record_inference(35.0, batch_size=1)
        
        results = monitor.verify_performance_target(target_latency_ms=50.0)
        
        assert 'latency_target_met' in results
        assert 'mean_latency_ms' in results
        assert 'target_latency_ms' in results
        assert results['latency_target_met']
    
    def test_latency_target_not_met(self, monitor):
        """Test when latency target is not met."""
        monitor.record_inference(60.0, batch_size=1)
        monitor.record_inference(65.0, batch_size=1)
        
        results = monitor.verify_performance_target(target_latency_ms=50.0)
        
        assert not results['latency_target_met']
    
    def test_throughput_target_verification(self, monitor):
        """Test throughput target verification."""
        for _ in range(10):
            monitor.record_inference(10.0, batch_size=1)
        
        results = monitor.verify_performance_target(
            target_latency_ms=50.0,
            target_throughput_sps=50.0
        )
        
        assert 'throughput_target_met' in results
        assert 'actual_throughput_sps' in results
        assert 'target_throughput_sps' in results


class TestPerformanceMonitorProperties:
    """Property-based tests for PerformanceMonitor."""
    
    @given(
        num_inferences=st.integers(min_value=1, max_value=100),
        batch_size=st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_total_samples_consistency(self, num_inferences, batch_size):
        """
        Property 13: Total samples should equal sum of batch sizes.
        
        For any sequence of inferences with different batch sizes,
        total_samples_processed should equal the sum of all batch sizes.
        """
        monitor = PerformanceMonitor()
        
        for _ in range(num_inferences):
            monitor.record_inference(10.0, batch_size=batch_size)
        
        assert monitor.total_samples_processed == num_inferences * batch_size
    
    @given(
        latencies=st.lists(
            st.floats(min_value=1.0, max_value=100.0),
            min_size=1,
            max_size=50
        ),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_statistics_validity(self, latencies):
        """
        Property: Statistics should always be valid and consistent.
        
        For any sequence of latency measurements, statistics should be
        valid and maintain proper ordering relationships.
        """
        monitor = PerformanceMonitor()
        
        for latency in latencies:
            monitor.record_inference(latency, batch_size=1)
        
        stats = monitor.get_inference_time_stats()
        
        # All statistics should be positive
        assert stats['mean_ms'] > 0
        assert stats['median_ms'] > 0
        assert stats['min_ms'] > 0
        assert stats['max_ms'] > 0
        
        # Ordering should be correct
        assert stats['min_ms'] <= stats['median_ms'] <= stats['max_ms']
        assert stats['p95_ms'] <= stats['p99_ms']
        assert stats['count'] == len(latencies)
    
    @given(
        num_measurements=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_metrics_history_consistency(self, num_measurements):
        """
        Property: Metrics history should be consistent with current state.
        
        For any number of measurements, metrics history should contain
        all recorded metrics and be consistent with current state.
        """
        monitor = PerformanceMonitor()
        
        for i in range(num_measurements):
            monitor.record_inference(10.0 + i, batch_size=1)
            monitor.get_all_metrics()
        
        history = monitor.get_metrics_history()
        
        # History should contain all metrics calls
        assert len(history) == num_measurements
        
        # Each entry should have required fields
        for entry in history:
            assert 'timestamp' in entry
            assert 'elapsed_time_sec' in entry
            assert 'total_samples_processed' in entry
    
    @given(
        window_size=st.integers(min_value=5, max_value=100),
        num_records=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_window_size_enforcement(self, window_size, num_records):
        """
        Property: Window size should limit stored measurements.
        
        For any window size and number of records, the number of stored
        measurements should not exceed the window size.
        """
        monitor = PerformanceMonitor(window_size=window_size)
        
        for _ in range(num_records):
            monitor.record_inference(10.0, batch_size=1)
        
        # Stored measurements should not exceed window size
        assert len(monitor.inference_times) <= window_size
        assert len(monitor.batch_sizes) <= window_size
        assert len(monitor.timestamps) <= window_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
