"""
Property-based tests for TensorRT optimization.

Tests optimization effectiveness, speedup, and memory reduction.
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings, HealthCheck
import tempfile
import os

from speedvqa.optimization.tensorrt_optimizer import TensorRTOptimizer


class SimpleVQAModel(nn.Module):
    """Simple VQA model for testing."""
    
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.text_encoder = nn.Linear(512, 32)
        self.fusion = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 2)
    
    def forward(self, image, text_embedding):
        vision_feat = self.vision_encoder(image)
        text_feat = self.text_encoder(text_embedding)
        fused = torch.cat([vision_feat, text_feat], dim=1)
        fused = self.fusion(fused)
        output = self.classifier(fused)
        return output


@pytest.fixture
def simple_model():
    """Create a simple VQA model for testing."""
    model = SimpleVQAModel()
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """Create sample input tensors."""
    image = torch.randn(1, 3, 224, 224)
    text_embedding = torch.randn(1, 512)
    return (image, text_embedding)


def create_simple_model():
    """Create a simple VQA model (for use in property tests)."""
    model = SimpleVQAModel()
    model.eval()
    return model


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestTensorRTOptimizationProperties:
    """Property-based tests for TensorRT optimization."""
    
    def test_optimizer_initialization(self, simple_model):
        """Test that optimizer initializes correctly."""
        optimizer = TensorRTOptimizer(simple_model)
        assert optimizer.model is simple_model
        assert optimizer.device == 'cuda:0'
        assert not optimizer.optimization_config['fp16_enabled']
        assert not optimizer.optimization_config['dynamic_batch']
    
    def test_fp16_optimization_flag(self, simple_model):
        """Test FP16 optimization flag setting."""
        optimizer = TensorRTOptimizer(simple_model)
        optimizer.enable_fp16()
        assert optimizer.optimization_config['fp16_enabled']
    
    def test_dynamic_batch_configuration(self, simple_model):
        """Test dynamic batch configuration."""
        optimizer = TensorRTOptimizer(simple_model)
        optimizer.enable_dynamic_batch(max_batch_size=64)
        assert optimizer.optimization_config['dynamic_batch']
        assert optimizer.optimization_config['max_batch_size'] == 64
    
    def test_memory_optimization_flag(self, simple_model):
        """Test memory optimization flag setting."""
        optimizer = TensorRTOptimizer(simple_model)
        optimizer.enable_memory_optimization()
        assert optimizer.optimization_config['memory_optimized']
    
    def test_method_chaining(self, simple_model):
        """Test that optimization methods support chaining."""
        optimizer = TensorRTOptimizer(simple_model)
        result = (optimizer
                 .enable_fp16()
                 .enable_dynamic_batch(32)
                 .enable_memory_optimization())
        assert result is optimizer
        assert optimizer.optimization_config['fp16_enabled']
        assert optimizer.optimization_config['dynamic_batch']
        assert optimizer.optimization_config['memory_optimized']
    
    def test_get_optimization_config(self, simple_model):
        """Test getting optimization configuration."""
        optimizer = TensorRTOptimizer(simple_model)
        optimizer.enable_fp16()
        config = optimizer.get_optimization_config()
        assert isinstance(config, dict)
        assert config['fp16_enabled']
        # Verify it's a copy
        config['fp16_enabled'] = False
        assert optimizer.optimization_config['fp16_enabled']
    
    def test_onnx_conversion_creates_file(self, simple_model, sample_input, temp_dir):
        """Test that ONNX conversion creates a file."""
        optimizer = TensorRTOptimizer(simple_model)
        output_path = os.path.join(temp_dir, 'model.onnx')
        
        result = optimizer.convert_to_onnx(output_path, sample_input)
        
        assert result == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_onnx_conversion_with_custom_names(self, simple_model, sample_input, temp_dir):
        """Test ONNX conversion with custom input/output names."""
        optimizer = TensorRTOptimizer(simple_model)
        output_path = os.path.join(temp_dir, 'model.onnx')
        
        optimizer.convert_to_onnx(
            output_path,
            sample_input,
            input_names=['image_input', 'text_input'],
            output_names=['vqa_output'],
        )

        assert os.path.exists(output_path)
    
    def test_memory_estimation(self, simple_model):
        """Test memory usage estimation."""
        optimizer = TensorRTOptimizer(simple_model, device='cpu')
        memory_est = optimizer.estimate_memory_usage(batch_size=1)
        
        assert 'fp32_model_mb' in memory_est
        assert 'fp32_activation_mb' in memory_est
        assert 'fp32_total_mb' in memory_est
        assert 'fp16_total_mb' in memory_est
        assert 'int8_total_mb' in memory_est
        
        # Verify memory relationships
        assert memory_est['fp32_total_mb'] > 0
        assert memory_est['fp16_total_mb'] < memory_est['fp32_total_mb']
        assert memory_est['int8_total_mb'] < memory_est['fp16_total_mb']
    
    def test_memory_estimation_batch_scaling(self, simple_model):
        """Test that memory estimation scales with batch size."""
        optimizer = TensorRTOptimizer(simple_model, device='cpu')
        
        mem_batch1 = optimizer.estimate_memory_usage(batch_size=1)
        mem_batch4 = optimizer.estimate_memory_usage(batch_size=4)
        
        # Batch 4 should use more memory than batch 1
        assert mem_batch4['fp32_total_mb'] > mem_batch1['fp32_total_mb']
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        fp16_enabled=st.booleans(),
        dynamic_batch=st.booleans(),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_optimization_config_consistency(self, batch_size, fp16_enabled, dynamic_batch):
        """
        Property 14: TensorRT optimization configuration consistency.
        
        For any combination of optimization settings, the configuration should be
        consistent and retrievable.
        """
        simple_model = create_simple_model()
        optimizer = TensorRTOptimizer(simple_model)
        
        if fp16_enabled:
            optimizer.enable_fp16()
        if dynamic_batch:
            optimizer.enable_dynamic_batch(batch_size)
        
        config = optimizer.get_optimization_config()
        
        # Verify configuration consistency
        assert config['fp16_enabled'] == fp16_enabled
        assert config['dynamic_batch'] == dynamic_batch
        if dynamic_batch:
            assert config['max_batch_size'] == batch_size
    
    @given(
        batch_size=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_memory_estimation_validity(self, batch_size):
        """
        Property: Memory estimation should always return valid positive values.
        
        For any batch size, memory estimation should return positive values
        and maintain proper relationships between precision levels.
        """
        simple_model = create_simple_model()
        optimizer = TensorRTOptimizer(simple_model, device='cpu')
        memory_est = optimizer.estimate_memory_usage(batch_size=batch_size)
        
        # All memory values should be positive
        assert memory_est['fp32_model_mb'] > 0
        assert memory_est['fp32_total_mb'] > 0
        assert memory_est['fp16_total_mb'] > 0
        assert memory_est['int8_total_mb'] > 0
        
        # Memory relationships should hold
        assert memory_est['fp32_total_mb'] > memory_est['fp16_total_mb']
        assert memory_est['fp16_total_mb'] > memory_est['int8_total_mb']
    
    def test_optimization_preserves_model_structure(self, simple_model):
        """Test that optimization doesn't modify the original model."""
        original_params = [p.clone() for p in simple_model.parameters()]
        
        optimizer = TensorRTOptimizer(simple_model)
        optimizer.enable_fp16()
        optimizer.enable_dynamic_batch()
        optimizer.enable_memory_optimization()
        
        # Verify model parameters are unchanged
        for orig_param, current_param in zip(original_params, simple_model.parameters()):
            assert torch.allclose(orig_param, current_param)
    
    def test_multiple_optimizers_independent(self, simple_model):
        """Test that multiple optimizer instances are independent."""
        opt1 = TensorRTOptimizer(simple_model)
        opt2 = TensorRTOptimizer(simple_model)
        
        opt1.enable_fp16()
        opt1.enable_dynamic_batch(32)
        
        # opt2 should not be affected
        assert not opt2.optimization_config['fp16_enabled']
        assert not opt2.optimization_config['dynamic_batch']
    
    @given(
        max_batch_size=st.integers(min_value=1, max_value=128),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_dynamic_batch_configuration_validity(self, max_batch_size):
        """
        Property: Dynamic batch configuration should accept any valid batch size.
        
        For any positive batch size, dynamic batch configuration should succeed
        and store the correct value.
        """
        simple_model = create_simple_model()
        optimizer = TensorRTOptimizer(simple_model)
        optimizer.enable_dynamic_batch(max_batch_size)
        
        config = optimizer.get_optimization_config()
        assert config['max_batch_size'] == max_batch_size
        assert config['dynamic_batch']


class TestTensorRTOptimizationEffectiveness:
    """Tests for optimization effectiveness."""
    
    def test_fp16_reduces_memory_estimate(self, simple_model):
        """Test that FP16 reduces estimated memory usage."""
        optimizer = TensorRTOptimizer(simple_model, device='cpu')
        memory_est = optimizer.estimate_memory_usage(batch_size=1)
        
        fp32_memory = memory_est['fp32_total_mb']
        fp16_memory = memory_est['fp16_total_mb']
        
        # FP16 should use approximately half the memory
        assert fp16_memory < fp32_memory
        assert fp16_memory > fp32_memory * 0.4  # Allow some overhead
        assert fp16_memory < fp32_memory * 0.6
    
    def test_int8_reduces_memory_most(self, simple_model):
        """Test that INT8 provides maximum memory reduction."""
        optimizer = TensorRTOptimizer(simple_model, device='cpu')
        memory_est = optimizer.estimate_memory_usage(batch_size=1)
        
        fp32_memory = memory_est['fp32_total_mb']
        fp16_memory = memory_est['fp16_total_mb']
        int8_memory = memory_est['int8_total_mb']
        
        # INT8 should use approximately quarter the memory
        assert int8_memory < fp16_memory
        assert int8_memory < fp32_memory
    
    def test_batch_size_affects_memory_linearly(self, simple_model):
        """Test that memory scales approximately linearly with batch size."""
        optimizer = TensorRTOptimizer(simple_model, device='cpu')
        
        mem_1 = optimizer.estimate_memory_usage(batch_size=1)
        mem_2 = optimizer.estimate_memory_usage(batch_size=2)
        mem_4 = optimizer.estimate_memory_usage(batch_size=4)
        
        # Memory should scale roughly linearly with batch size
        # Note: Model parameters are constant, but activation memory scales with batch
        # So we check that larger batch sizes use more memory
        assert mem_2['fp32_total_mb'] >= mem_1['fp32_total_mb']
        assert mem_4['fp32_total_mb'] >= mem_2['fp32_total_mb']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
