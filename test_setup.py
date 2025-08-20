#!/usr/bin/env python3
"""Test script to validate the setup with Llama 3.2 1B configuration."""

import sys
import os
sys.path.append('src')

import yaml
from omegaconf import OmegaConf
from rope_long_context_evaluation_suite.core import RoPEEvaluator
from rope_long_context_evaluation_suite.models import get_rope_extension


def test_configuration():
    """Test loading and validating the configuration."""
    print("🧪 Testing configuration loading...")
    
    with open("test_llama32_1b.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config_obj = OmegaConf.create(config)
    
    print(f"✅ Model: {config_obj.model.name}")
    print(f"✅ RoPE method: {config_obj.rope_extension.method}")
    print(f"✅ Benchmarks enabled: {[k for k, v in config_obj.benchmarks.items() if v.enabled]}")
    
    return config_obj


def test_rope_extension():
    """Test RoPE extension creation."""
    print("\n🔧 Testing RoPE extension creation...")
    
    try:
        rope_ext = get_rope_extension("linear_interpolation", {"scaling_factor": 2.0})
        print(f"✅ RoPE extension created: {type(rope_ext).__name__}")
        
        # Test scaling info
        info = rope_ext.get_scaling_info()
        print(f"✅ Scaling info: {info}")
        
    except Exception as e:
        print(f"❌ RoPE extension test failed: {e}")
        return False
    
    return True


def test_evaluator_initialization():
    """Test evaluator initialization without model loading."""
    print("\n🚀 Testing evaluator initialization...")
    
    try:
        config = test_configuration()
        
        # Create evaluator 
        evaluator = RoPEEvaluator(config)
        print("✅ RoPEEvaluator initialized successfully")
        
        # Test that we can access the model loader
        model_info = evaluator.model_loader.get_model_info()
        print(f"✅ Model info: {model_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧩 RoPE Long Context Evaluation Suite - Setup Test\n")
    
    tests = [
        ("Configuration Loading", test_configuration),
        ("RoPE Extension", test_rope_extension), 
        ("Evaluator Initialization", test_evaluator_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! The setup is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())