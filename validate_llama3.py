"""Validation script for Llama3 RoPE implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rope_long_context_evaluation_suite.models.llama3 import Llama3RoPE
from rope_long_context_evaluation_suite.models.factory import get_rope_extension
from unittest.mock import Mock

def test_llama3_basic():
    """Test basic Llama3 functionality."""
    print("Testing Llama3RoPE basic functionality...")
    
    # Test initialization
    config = {
        "factor": 4.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    }
    
    llama3_rope = Llama3RoPE(config)
    print(f"✓ Initialized: {llama3_rope}")
    
    # Test rope scaling computation
    scaling = llama3_rope.compute_rope_scaling(32768, 8192)
    expected = {
        "rope_type": "llama3",
        "factor": 4.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    }
    assert scaling == expected
    print("✓ RoPE scaling computation correct")
    
    # Test frequency scaling
    freq_scaling = llama3_rope._compute_frequency_scaling(dim=128)
    assert len(freq_scaling) == 64
    assert freq_scaling[0] <= freq_scaling[-1]  # Should be monotonic
    print("✓ Frequency scaling computation works")
    
    # Test model application
    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.config.max_position_embeddings = 8192
    
    result = llama3_rope.apply(mock_model)
    assert result.config.rope_scaling == expected
    assert result.config.max_position_embeddings == 32768  # 8192 * 4
    print("✓ Model application works")
    
    print("✓ All Llama3RoPE tests passed!")

def test_factory_integration():
    """Test factory integration."""
    print("\nTesting factory integration...")
    
    config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0
    }
    
    # Test creating via factory
    llama3_rope = get_rope_extension("llama3", config)
    assert isinstance(llama3_rope, Llama3RoPE)
    assert llama3_rope.factor == 8.0
    print("✓ Factory creation works")
    
    # Test listing available extensions
    from rope_long_context_evaluation_suite.models.factory import list_available_extensions
    available = list_available_extensions()
    assert "llama3" in available
    print("✓ Llama3 listed in available extensions")
    
    print("✓ Factory integration tests passed!")

if __name__ == "__main__":
    try:
        test_llama3_basic()
        test_factory_integration()
        print("\n🎉 All validation tests passed!")
        print("\nNow we support ALL transformers RoPE scaling methods:")
        print("- linear (Linear Interpolation)")
        print("- dynamic (Dynamic NTK)")
        print("- yarn (YaRN)")
        print("- longrope (LongRoPE)")
        print("- llama3 (Llama3 frequency-based)")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)