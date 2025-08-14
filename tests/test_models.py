"""Tests for model loading and RoPE extensions."""


import pytest

import torch
from unittest.mock import MagicMock, patch

from rope_long_context_evaluation_suite.models import ModelLoader, get_rope_extension
from rope_long_context_evaluation_suite.models.rope_extensions import (
    LinearInterpolation,
    NTKAwareInterpolation, 
    YaRNExtension,
)


class TestModelLoader:
    """Test cases for ModelLoader class."""
    
    def test_get_device_cpu(self):
        """Test device selection when CUDA is not available."""
        config = MagicMock()
        config.model.device_map = "auto"
        
        with patch('torch.cuda.is_available', return_value=False):
            loader = ModelLoader(config)
            assert loader._get_device() == "cpu"
    
    def test_get_device_cuda(self):
        """Test device selection when CUDA is available.""" 
        config = MagicMock()
        config.model.device_map = "auto"
        
        with patch('torch.cuda.is_available', return_value=True):
            loader = ModelLoader(config)
            assert loader._get_device() == "cuda"
    
    def test_get_torch_dtype_auto(self):
        """Test automatic torch dtype selection."""
        config = MagicMock()
        config.model.get.return_value = "auto"
        
        with patch('torch.cuda.is_available', return_value=True):
            loader = ModelLoader(config)
            assert loader._get_torch_dtype() == torch.float16
    
    def test_get_torch_dtype_explicit(self):
        """Test explicit torch dtype setting."""
        config = MagicMock()
        config.model.get.return_value = "float32"
        
        loader = ModelLoader(config)
        assert loader._get_torch_dtype() == torch.float32


class TestRoPEExtensions:
    """Test cases for RoPE extension methods."""
    
    def test_linear_interpolation(self):
        """Test LinearInterpolation extension."""
        config = {"scaling_factor": 4}
        extension = LinearInterpolation(config)
        
        info = extension.get_scaling_info()
        assert info["method"] == "linear_interpolation"
        assert info["scaling_factor"] == 4
    
    def test_ntk_aware_interpolation(self):
        """Test NTKAwareInterpolation extension."""
        config = {"alpha": 1.0, "beta": 32.0}
        extension = NTKAwareInterpolation(config)
        
        info = extension.get_scaling_info()
        assert info["method"] == "ntk_aware"
        assert info["alpha"] == 1.0
        assert info["beta"] == 32.0
    
    def test_yarn_extension(self):
        """Test YaRNExtension."""
        config = {
            "s": 16,
            "alpha": 1.0,
            "beta": 32.0,
            "attention_factor": 0.1,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }
        extension = YaRNExtension(config)
        
        info = extension.get_scaling_info()
        assert info["method"] == "yarn"
        assert info["s"] == 16
        assert info["scaling_type"] == "adaptive_ramp"
    
    def test_get_rope_extension_factory(self):
        """Test RoPE extension factory function."""
        config = {"scaling_factor": 2}
        extension = get_rope_extension("linear_interpolation", config)
        
        assert isinstance(extension, LinearInterpolation)
        assert extension.scaling_factor == 2
    
    def test_get_rope_extension_invalid(self):
        """Test factory function with invalid method."""
        config = {}
        
        
        with pytest.raises(ValueError, match="Unsupported RoPE extension method"):
            get_rope_extension("invalid_method", config)
        