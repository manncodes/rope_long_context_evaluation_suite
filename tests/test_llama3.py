"""Tests for Llama3 RoPE scaling."""

import pytest
import torch
from unittest.mock import Mock, patch

from rope_long_context_evaluation_suite.models.llama3 import Llama3RoPE


class TestLlama3RoPE:
    """Test cases for Llama3RoPE."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        config = {}
        llama3_rope = Llama3RoPE(config)
        
        assert llama3_rope.factor == 8.0
        assert llama3_rope.low_freq_factor == 1.0
        assert llama3_rope.high_freq_factor == 4.0
        assert llama3_rope.original_max_position_embeddings == 8192
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        config = {
            "factor": 4.0,
            "low_freq_factor": 0.5,
            "high_freq_factor": 2.0,
            "original_max_position_embeddings": 4096
        }
        llama3_rope = Llama3RoPE(config)
        
        assert llama3_rope.factor == 4.0
        assert llama3_rope.low_freq_factor == 0.5
        assert llama3_rope.high_freq_factor == 2.0
        assert llama3_rope.original_max_position_embeddings == 4096
    
    def test_compute_rope_scaling(self):
        """Test rope scaling computation."""
        config = {
            "factor": 4.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
        llama3_rope = Llama3RoPE(config)
        
        scaling = llama3_rope.compute_rope_scaling(
            seq_len=32768, 
            original_max_len=8192
        )
        
        expected = {
            "rope_type": "llama3",
            "factor": 4.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
        
        assert scaling == expected
    
    def test_apply(self):
        """Test applying Llama3 RoPE to a model."""
        config = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
        llama3_rope = Llama3RoPE(config)
        
        # Mock model
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.max_position_embeddings = 8192
        
        result_model = llama3_rope.apply(mock_model)
        
        # Check that rope_scaling was set
        expected_scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
        assert result_model.config.rope_scaling == expected_scaling
        
        # Check that max_position_embeddings was updated
        assert result_model.config.max_position_embeddings == 65536  # 8192 * 8
    
    def test_compute_frequency_scaling(self):
        """Test frequency scaling computation."""
        config = {
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0
        }
        llama3_rope = Llama3RoPE(config)
        
        scaling_factors = llama3_rope._compute_frequency_scaling(dim=128)
        
        # Should return tensor with same length as dim/2
        assert len(scaling_factors) == 64
        
        # First element should be close to low_freq_factor
        assert abs(scaling_factors[0].item() - 1.0) < 0.1
        
        # Last element should be close to high_freq_factor
        assert abs(scaling_factors[-1].item() - 4.0) < 0.1
        
        # Should be monotonically increasing
        assert torch.all(scaling_factors[1:] >= scaling_factors[:-1])
    
    def test_get_config(self):
        """Test getting configuration."""
        config = {
            "factor": 4.0,
            "low_freq_factor": 0.5,
            "high_freq_factor": 2.0,
            "original_max_position_embeddings": 4096
        }
        llama3_rope = Llama3RoPE(config)
        
        result_config = llama3_rope.get_config()
        
        expected = {
            "type": "llama3",
            "factor": 4.0,
            "low_freq_factor": 0.5,
            "high_freq_factor": 2.0,
            "original_max_position_embeddings": 4096
        }
        
        assert result_config == expected
    
    def test_repr(self):
        """Test string representation."""
        config = {
            "factor": 4.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 2.0,
            "original_max_position_embeddings": 4096
        }
        llama3_rope = Llama3RoPE(config)
        
        repr_str = repr(llama3_rope)
        expected = (
            "Llama3RoPE(factor=4.0, low_freq_factor=1.0, "
            "high_freq_factor=2.0, original_max_len=4096)"
        )
        
        assert repr_str == expected
    
    def test_frequency_interpolation_bounds(self):
        """Test that frequency interpolation stays within bounds."""
        config = {
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0
        }
        llama3_rope = Llama3RoPE(config)
        
        scaling_factors = llama3_rope._compute_frequency_scaling(dim=64)
        
        # All scaling factors should be between low and high freq factors
        assert torch.all(scaling_factors >= 1.0)
        assert torch.all(scaling_factors <= 4.0)
    
    def test_different_dimensions(self):
        """Test frequency scaling with different dimensions."""
        config = {
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0
        }
        llama3_rope = Llama3RoPE(config)
        
        for dim in [64, 128, 256, 512]:
            scaling_factors = llama3_rope._compute_frequency_scaling(dim=dim)
            assert len(scaling_factors) == dim // 2
            assert torch.all(scaling_factors >= 1.0)
            assert torch.all(scaling_factors <= 4.0)


if __name__ == "__main__":
    pytest.main([__file__])