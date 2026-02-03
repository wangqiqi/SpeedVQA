"""
T4 GPU Performance Benchmark Suite for SpeedVQA.

Provides comprehensive benchmarking for inference latency, throughput, and memory usage
on NVIDIA T4 GPUs.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)


class T4Benchmark:
    """
    T4 GPU performance benchmark suite for SpeedVQA.
    
    Provides:
    - Single inference latency testing
    - Batch inference throughput testing
    - GPU memory monitoring
    - Performance metrics collection
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0', warmup_runs: int = 10):
        """
        Initialize T4 benchmark.
        
        Args:
            model: PyTorch model to benchmark
            device: Device to use for benchmarking
            warmup_runs: Number of warmup runs before measurement
        """
        self.model = model
        self.device = device
        self.warmup_runs = warmup_runs
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVIDIA GPU monitoring if available
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_gpu_monitoring = True
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.has_gpu_monitoring = False
        else:
            self.has_gpu_monitoring = False
        
        self.benchmark_results = {}
    
    def _warmup(self, sample_input: Tuple[torch.Tensor, ...]) -> None:
        """
        Warmup GPU with sample inputs.
        
        Args:
            sample_input: Sample input tensors
        """
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(*sample_input)
        torch.cuda.synchronize()
    
    def _get_gpu_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory usage in MB
        """
        if not self.has_gpu_monitoring:
            return {'used_mb': 0, 'total_mb': 0, 'free_mb': 0}
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                'used_mb': mem_info.used / (1024 * 1024),
                'total_mb': mem_info.total / (1024 * 1024),
                'free_mb': mem_info.free / (1024 * 1024),
            }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory info: {e}")
            return {'used_mb': 0, 'total_mb': 0, 'free_mb': 0}
    
    def benchmark_single_inference(
        self,
        sample_input: Tuple[torch.Tensor, ...],
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """
        Benchmark single inference latency.
        
        Args:
            sample_input: Sample input tensors
            num_runs: Number of inference runs for measurement
            
        Returns:
            Dictionary with latency statistics
        """
        self.logger.info(f"Benchmarking single inference latency ({num_runs} runs)...")
        
        # Warmup
        self._warmup(sample_input)
        
        # Measure latencies
        latencies = []
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                _ = self.model(*sample_input)
                
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        latencies = np.array(latencies)
        
        results = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'num_runs': num_runs,
        }
        
        self.logger.info(f"Single inference latency: {results['mean_latency_ms']:.2f}ms "
                        f"(median: {results['median_latency_ms']:.2f}ms, "
                        f"p95: {results['p95_latency_ms']:.2f}ms)")
        
        return results
    
    def benchmark_batch_throughput(
        self,
        sample_input: Tuple[torch.Tensor, ...],
        batch_sizes: Optional[List[int]] = None,
        num_runs: int = 100,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Benchmark batch inference throughput.
        
        Args:
            sample_input: Sample input tensors (batch_size=1)
            batch_sizes: List of batch sizes to test
            num_runs: Number of inference runs per batch size
            
        Returns:
            Dictionary with throughput results for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        self.logger.info(f"Benchmarking batch throughput for batch sizes: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create batched input
            batched_input = tuple(
                torch.cat([inp] * batch_size, dim=0).to(self.device)
                for inp in sample_input
            )
            
            # Warmup
            self._warmup(batched_input)
            
            # Measure throughput
            latencies = []
            self.model.eval()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    
                    _ = self.model(*batched_input)
                    
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
            
            latencies = np.array(latencies)
            
            # Calculate throughput (samples per second)
            throughput_sps = (batch_size * 1000) / np.mean(latencies)
            
            results[batch_size] = {
                'batch_size': batch_size,
                'mean_latency_ms': float(np.mean(latencies)),
                'median_latency_ms': float(np.median(latencies)),
                'throughput_samples_per_sec': float(throughput_sps),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
            }
            
            self.logger.info(f"Batch size {batch_size}: {results[batch_size]['mean_latency_ms']:.2f}ms, "
                           f"throughput: {throughput_sps:.1f} samples/sec")
        
        return results
    
    def benchmark_memory_usage(
        self,
        sample_input: Tuple[torch.Tensor, ...],
        batch_sizes: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Benchmark GPU memory usage for different batch sizes.
        
        Args:
            sample_input: Sample input tensors (batch_size=1)
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with memory usage for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        self.logger.info(f"Benchmarking memory usage for batch sizes: {batch_sizes}")
        
        results = {}
        
        # Get baseline memory
        torch.cuda.empty_cache()
        baseline_memory = self._get_gpu_memory_usage()
        
        for batch_size in batch_sizes:
            # Create batched input
            batched_input = tuple(
                torch.cat([inp] * batch_size, dim=0).to(self.device)
                for inp in sample_input
            )
            
            # Measure memory
            self.model.eval()
            with torch.no_grad():
                _ = self.model(*batched_input)
            
            torch.cuda.synchronize()
            memory_usage = self._get_gpu_memory_usage()
            
            results[batch_size] = {
                'batch_size': batch_size,
                'used_mb': memory_usage['used_mb'],
                'total_mb': memory_usage['total_mb'],
                'free_mb': memory_usage['free_mb'],
                'allocated_mb': memory_usage['used_mb'] - baseline_memory['used_mb'],
            }
            
            self.logger.info(f"Batch size {batch_size}: {results[batch_size]['allocated_mb']:.1f}MB allocated")
        
        return results
    
    def run_full_benchmark(
        self,
        sample_input: Tuple[torch.Tensor, ...],
        batch_sizes: Optional[List[int]] = None,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            sample_input: Sample input tensors
            batch_sizes: List of batch sizes to test
            num_runs: Number of inference runs per test
            
        Returns:
            Dictionary with all benchmark results
        """
        self.logger.info("Starting full benchmark suite...")
        
        results = {
            'single_inference': self.benchmark_single_inference(sample_input, num_runs),
            'batch_throughput': self.benchmark_batch_throughput(sample_input, batch_sizes, num_runs),
            'memory_usage': self.benchmark_memory_usage(sample_input, batch_sizes),
        }
        
        self.benchmark_results = results
        
        # Log summary
        single_latency = results['single_inference']['mean_latency_ms']
        self.logger.info(f"\n=== Benchmark Summary ===")
        self.logger.info(f"Single inference latency: {single_latency:.2f}ms")
        self.logger.info(f"Target (<50ms): {'✓ PASS' if single_latency < 50 else '✗ FAIL'}")
        
        return results
    
    def get_benchmark_results(self) -> Dict[str, Any]:
        """
        Get stored benchmark results.
        
        Returns:
            Dictionary with benchmark results
        """
        return self.benchmark_results.copy()
    
    def verify_performance_target(self, target_latency_ms: float = 50.0) -> bool:
        """
        Verify if performance target is met.
        
        Args:
            target_latency_ms: Target latency in milliseconds
            
        Returns:
            True if target is met, False otherwise
        """
        if not self.benchmark_results or 'single_inference' not in self.benchmark_results:
            self.logger.warning("No benchmark results available")
            return False
        
        mean_latency = self.benchmark_results['single_inference']['mean_latency_ms']
        target_met = mean_latency < target_latency_ms
        
        self.logger.info(f"Performance target verification: {mean_latency:.2f}ms < {target_latency_ms}ms: "
                        f"{'✓ PASS' if target_met else '✗ FAIL'}")
        
        return target_met
