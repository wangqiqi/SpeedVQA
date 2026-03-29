"""
TensorRT Optimization Pipeline for SpeedVQA.

Provides FP16 precision optimization, dynamic batch processing, and memory optimization
for efficient inference on NVIDIA GPUs (especially T4).
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """
    TensorRT optimization pipeline for SpeedVQA models.
    
    Supports:
    - FP16 precision optimization
    - Dynamic batch processing
    - Memory optimization strategies
    - Model conversion and optimization
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        """
        Initialize TensorRT optimizer.
        
        Args:
            model: PyTorch model to optimize
            device: Device to use for optimization (default: 'cuda:0')
        """
        if not HAS_TENSORRT:
            raise ImportError("TensorRT is not installed. Please install tensorrt package.")
        
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.optimization_config = {
            'fp16_enabled': False,
            'dynamic_batch': False,
            'max_batch_size': 1,
            'memory_optimized': False,
        }
    
    def enable_fp16(self) -> 'TensorRTOptimizer':
        """
        Enable FP16 precision optimization.
        
        Returns:
            Self for method chaining
        """
        self.optimization_config['fp16_enabled'] = True
        self.logger.info("FP16 precision optimization enabled")
        return self
    
    def enable_dynamic_batch(self, max_batch_size: int = 32) -> 'TensorRTOptimizer':
        """
        Enable dynamic batch processing support.
        
        Args:
            max_batch_size: Maximum batch size for dynamic batching
            
        Returns:
            Self for method chaining
        """
        self.optimization_config['dynamic_batch'] = True
        self.optimization_config['max_batch_size'] = max_batch_size
        self.logger.info(f"Dynamic batch processing enabled (max_batch_size={max_batch_size})")
        return self
    
    def enable_memory_optimization(self) -> 'TensorRTOptimizer':
        """
        Enable memory optimization strategies.
        
        Returns:
            Self for method chaining
        """
        self.optimization_config['memory_optimized'] = True
        self.logger.info("Memory optimization enabled")
        return self
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get current optimization configuration.
        
        Returns:
            Dictionary containing optimization settings
        """
        return self.optimization_config.copy()
    
    def convert_to_onnx(
        self,
        output_path: str,
        sample_input: Optional[Tuple[torch.Tensor, ...]] = None,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            sample_input: Sample input tensors for tracing
            input_names: Names of input tensors
            output_names: Names of output tensors
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Path to saved ONNX model
        """
        self.model.eval()
        
        # Default sample input if not provided
        if sample_input is None:
            sample_input = (torch.randn(1, 3, 224, 224).to(self.device),
                           torch.randn(1, 512).to(self.device))
        
        # Default names if not provided
        if input_names is None:
            input_names = ['image', 'text_embedding']
        if output_names is None:
            output_names = ['output']
        
        # Default dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {
                'image': {0: 'batch_size'},
                'text_embedding': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        try:
            torch.onnx.export(
                self.model,
                sample_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                verbose=False,
            )
            self.logger.info(f"Model successfully converted to ONNX: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to convert model to ONNX: {e}")
            raise
    
    def build_tensorrt_engine(
        self,
        onnx_path: str,
        engine_path: str,
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,  # 1GB
    ) -> str:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            max_batch_size: Maximum batch size
            workspace_size: GPU workspace size in bytes
            
        Returns:
            Path to saved TensorRT engine
        """
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            # Enable FP16 if configured
            if self.optimization_config['fp16_enabled']:
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info("FP16 optimization enabled in TensorRT engine")
            
            # Set max batch size
            config.max_batch_size = max_batch_size
            
            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"TensorRT engine successfully built: {engine_path}")
            return engine_path
            
        except Exception as e:
            self.logger.error(f"Failed to build TensorRT engine: {e}")
            raise
    
    def optimize_model(
        self,
        output_dir: str,
        model_name: str = 'speedvqa',
        max_batch_size: int = 32,
        workspace_size: int = 1 << 30,
    ) -> Dict[str, str]:
        """
        Complete optimization pipeline: PyTorch -> ONNX -> TensorRT.
        
        Args:
            output_dir: Directory to save optimized models
            model_name: Base name for output files
            max_batch_size: Maximum batch size for optimization
            workspace_size: GPU workspace size in bytes
            
        Returns:
            Dictionary with paths to optimized models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Convert to ONNX
        onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
        self.logger.info("Step 1: Converting to ONNX...")
        self.convert_to_onnx(onnx_path)
        
        # Step 2: Build TensorRT engine
        engine_path = os.path.join(output_dir, f'{model_name}.engine')
        self.logger.info("Step 2: Building TensorRT engine...")
        self.build_tensorrt_engine(
            onnx_path,
            engine_path,
            max_batch_size=max_batch_size,
            workspace_size=workspace_size,
        )
        
        result = {
            'pytorch': None,  # Original model path
            'onnx': onnx_path,
            'tensorrt': engine_path,
        }
        
        self.logger.info(f"Optimization complete. Results: {result}")
        return result
    
    def estimate_memory_usage(self, batch_size: int = 1) -> Dict[str, float]:
        """
        Estimate memory usage for different precisions.
        
        Args:
            batch_size: Batch size for estimation
            
        Returns:
            Dictionary with memory estimates in MB
        """
        self.model.eval()
        
        # Create sample input
        sample_input = (
            torch.randn(batch_size, 3, 224, 224).to(self.device),
            torch.randn(batch_size, 512).to(self.device)
        )
        
        # Estimate FP32 memory
        fp32_memory = 0
        for param in self.model.parameters():
            fp32_memory += param.numel() * 4  # 4 bytes per FP32
        
        # Add activation memory (rough estimate)
        with torch.no_grad():
            output = self.model(*sample_input)
            activation_memory = output.numel() * 4
        
        fp32_total = (fp32_memory + activation_memory) / (1024 * 1024)  # Convert to MB
        
        return {
            'fp32_model_mb': fp32_memory / (1024 * 1024),
            'fp32_activation_mb': activation_memory / (1024 * 1024),
            'fp32_total_mb': fp32_total,
            'fp16_total_mb': fp32_total / 2,  # FP16 uses half the memory
            'int8_total_mb': fp32_total / 4,  # INT8 uses quarter the memory
        }
    
    def validate_optimization(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        sample_input: Tuple[torch.Tensor, ...],
        tolerance: float = 1e-3,
    ) -> bool:
        """
        Validate that optimization preserves model correctness.
        
        Args:
            original_model: Original PyTorch model
            optimized_model: Optimized model
            sample_input: Sample input for validation
            tolerance: Tolerance for output difference
            
        Returns:
            True if validation passes, False otherwise
        """
        original_model.eval()
        optimized_model.eval()
        
        with torch.no_grad():
            original_output = original_model(*sample_input)
            optimized_output = optimized_model(*sample_input)
        
        # Compare outputs
        diff = torch.abs(original_output - optimized_output).max().item()
        
        if diff <= tolerance:
            self.logger.info(f"Optimization validation passed (max diff: {diff:.6f})")
            return True
        else:
            self.logger.warning(f"Optimization validation failed (max diff: {diff:.6f})")
            return False
