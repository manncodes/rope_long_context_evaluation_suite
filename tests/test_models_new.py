"""Tests for models module."""

import pytest
import torch
from unittest.mock import Mock, patch

from src.rope_long_context_evaluation_suite.models.base import BaseRoPEExtension
from src.rope_long_context_evaluation_suite.models.linear_interpolation import LinearInterpolationRoPE
from src.rope_long_context_evaluation_suite.models.ntk_aware import NTKAwareRoPE
from src.rope_long_context_evaluation_suite.models.yarn import YaRNRoPE
from src.rope_long_context_evaluation_suite.models.factory import get_rope_extension, register_rope_extension


class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.config = Mock()
        self.config.model_type = "llama"
        self.config.max_position_embeddings = 4096
        
        # Mock layers
        self.model = Mock()
        self.model.layers = [Mock() for _ in range(4)]
        
        for layer in self.model.layers:
            layer.self_attn = Mock()
            layer.self_attn.rotary_emb = Mock()
            layer.self_attn.rotary_emb.inv_freq = torch.tensor([1.0, 0.5, 0.25])


class TestBaseRoPEExtension:
    """Test BaseRoPEExtension class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        config = {"test": True}
        
        class TestExtension(BaseRoPEExtension):
            pass
        
        extension = TestExtension(config)
        
        with pytest.raises(TypeError):
            # Should fail to instantiate because of abstract methods
            pass
    
    def test_freq_cis_computation(self):
        """Test frequency computation for complex exponentials."""
        dim = 128
        max_pos = 2048
        theta = 10000.0
        
        freqs_cis = BaseRoPEExtension.compute_freq_cis(dim, max_pos, theta)
        
        assert freqs_cis.shape == (max_pos, dim // 2)
        assert freqs_cis.dtype == torch.complex64
    
    def test_freq_cis_with_scaling(self):
        """Test frequency computation with scaling factor."""
        dim = 128
        max_pos = 2048
        scaling_factor = 2.0
        
        freqs_cis = BaseRoPEExtension.compute_freq_cis(dim, max_pos, scaling_factor=scaling_factor)
        freqs_cis_no_scale = BaseRoPEExtension.compute_freq_cis(dim, max_pos)
        
        # Scaling should change the frequencies
        assert not torch.allclose(freqs_cis, freqs_cis_no_scale)
    
    def test_apply_rotary_pos_emb(self):
        """Test application of rotary position embeddings."""
        batch_size, seq_len, heads, head_dim = 2, 512, 8, 64
        
        q = torch.randn(batch_size, heads, seq_len, head_dim)
        k = torch.randn(batch_size, heads, seq_len, head_dim)
        freqs_cis = BaseRoPEExtension.compute_freq_cis(head_dim, seq_len)
        
        # Expand for batch and heads
        freqs_cis = freqs_cis[None, None, :, :].expand(batch_size, heads, -1, -1)
        
        q_rot, k_rot = BaseRoPEExtension.apply_rotary_pos_emb(q, k, freqs_cis)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.allclose(q_rot, q)  # Should be different after rotation
        assert not torch.allclose(k_rot, k)


class TestLinearInterpolationRoPE:
    """Test LinearInterpolationRoPE class."""
    
    def test_initialization(self):
        """Test initialization with different configs."""
        config = {"scaling_factor": 8.0}
        extension = LinearInterpolationRoPE(config)
        
        assert extension.scaling_factor == 8.0
        
    def test_default_scaling_factor(self):
        """Test default scaling factor."""
        config = {}
        extension = LinearInterpolationRoPE(config)
        
        assert extension.scaling_factor == 4.0
    
    def test_compute_rope_scaling(self):
        """Test RoPE scaling computation."""
        config = {"scaling_factor": 4.0}
        extension = LinearInterpolationRoPE(config)
        
        scaling_config = extension.compute_rope_scaling(16384, 4096)
        
        assert scaling_config["type"] == "linear"
        assert scaling_config["factor"] == 4.0
    
    def test_get_scaling_info(self):
        """Test scaling info retrieval."""
        config = {"scaling_factor": 2.0}
        extension = LinearInterpolationRoPE(config)
        extension.original_max_position_embeddings = 4096
        
        info = extension.get_scaling_info()
        
        assert info["method"] == "linear_interpolation"
        assert info["scaling_factor"] == 2.0
        assert info["original_max_position_embeddings"] == 4096
    
    def test_apply(self):
        """Test application to mock model."""
        config = {"scaling_factor": 2.0}
        extension = LinearInterpolationRoPE(config)
        model = MockModel()
        
        modified_model = extension.apply(model)
        
        assert modified_model.config.max_position_embeddings == 8192
        assert extension.original_max_position_embeddings == 4096


class TestNTKAwareRoPE:
    """Test NTKAwareRoPE class."""
    
    def test_initialization(self):
        """Test initialization."""
        config = {"alpha": 2.0, "beta": 64.0}
        extension = NTKAwareRoPE(config)
        
        assert extension.alpha == 2.0
        assert extension.beta == 64.0
        assert extension.base_theta == 10000.0
    
    def test_compute_rope_scaling(self):
        """Test NTK-aware scaling computation."""
        config = {"alpha": 1.0, "beta": 32.0}
        extension = NTKAwareRoPE(config)
        
        scaling_config = extension.compute_rope_scaling(32768, 4096)
        
        assert scaling_config["type"] == "ntk_aware"
        assert scaling_config["alpha"] == 1.0
        assert scaling_config["beta"] == 32.0
        assert "adjusted_theta" in scaling_config
    
    def test_compute_ntk_alpha(self):
        """Test NTK alpha computation."""
        alpha = NTKAwareRoPE.compute_ntk_alpha(scale_factor=4.0, base_alpha=1.0)
        
        assert alpha > 1.0  # Should increase with scale factor


class TestYaRNRoPE:
    """Test YaRNRoPE class."""
    
    def test_initialization(self):
        """Test YaRN initialization."""
        config = {
            "s": 8.0,
            "alpha": 1.5,
            "beta": 16.0,
            "attention_factor": 0.2
        }
        extension = YaRNRoPE(config)
        
        assert extension.s == 8.0
        assert extension.alpha == 1.5
        assert extension.beta == 16.0
        assert extension.attention_factor == 0.2
    
    def test_compute_rope_scaling(self):
        """Test YaRN scaling computation."""
        config = {"s": 16.0, "alpha": 1.0, "beta": 32.0}
        extension = YaRNRoPE(config)
        
        scaling_config = extension.compute_rope_scaling(65536, 4096)
        
        assert scaling_config["type"] == "yarn"
        assert scaling_config["s"] == 16.0
        assert scaling_config["alpha"] == 1.0
        assert scaling_config["beta"] == 32.0
    
    def test_attention_temperature(self):
        """Test attention temperature computation."""
        temp = YaRNRoPE.compute_attention_temperature(scale_factor=4.0, attention_factor=0.1)
        
        assert temp > 1.0  # Should be greater than 1 for scaling > 1


class TestFactory:
    """Test factory functions."""
    
    def test_get_rope_extension_linear(self):
        """Test getting linear interpolation extension."""
        config = {"scaling_factor": 2.0}
        extension = get_rope_extension("linear_interpolation", config)
        
        assert isinstance(extension, LinearInterpolationRoPE)
        assert extension.scaling_factor == 2.0
    
    def test_get_rope_extension_ntk(self):
        """Test getting NTK-aware extension."""
        config = {"alpha": 1.5}
        extension = get_rope_extension("ntk_aware", config)
        
        assert isinstance(extension, NTKAwareRoPE)
        assert extension.alpha == 1.5
    
    def test_get_rope_extension_yarn(self):
        """Test getting YaRN extension."""
        config = {"s": 8.0}
        extension = get_rope_extension("yarn", config)
        
        assert isinstance(extension, YaRNRoPE)
        assert extension.s == 8.0
    
    def test_get_rope_extension_aliases(self):
        """Test extension aliases."""
        config = {"scaling_factor": 2.0}
        
        extension1 = get_rope_extension("linear_interpolation", config)
        extension2 = get_rope_extension("linear", config)
        
        assert type(extension1) == type(extension2)
    
    def test_unknown_extension_error(self):
        """Test error for unknown extension."""
        with pytest.raises(ValueError, match="Unknown RoPE extension method"):
            get_rope_extension("unknown_method", {})
    
    def test_register_custom_extension(self):
        """Test registering custom extension."""
        class CustomRoPE(BaseRoPEExtension):
            def apply(self, model):
                return model
            
            def compute_rope_scaling(self, seq_len, original_max_len):
                return {"type": "custom"}
            
            def get_scaling_info(self):
                return {"method": "custom"}
        
        register_rope_extension("custom", CustomRoPE)
        
        config = {}
        extension = get_rope_extension("custom", config)
        
        assert isinstance(extension, CustomRoPE)
    
    def test_register_invalid_extension(self):
        """Test error when registering invalid extension."""
        class InvalidExtension:
            pass
        
        with pytest.raises(TypeError, match="must inherit from BaseRoPEExtension"):
            register_rope_extension("invalid", InvalidExtension)