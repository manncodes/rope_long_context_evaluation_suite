#!/usr/bin/env python3
"""Test script to validate comprehensive configuration works with RoPEEvaluator."""

import sys
import os
sys.path.append('src')

import yaml
from omegaconf import OmegaConf
from rope_long_context_evaluation_suite.core import RoPEEvaluator
from rope_long_context_evaluation_suite.models import get_rope_extension


def test_comprehensive_config():
    """Test the fixed comprehensive configuration."""
    print("🧪 Testing Comprehensive Configuration...")
    
    # Load the fixed config
    with open("comprehensive_config_fixed.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config_obj = OmegaConf.create(config)
    
    print(f"✅ Model: {config_obj.model.name}")
    print(f"✅ Model Path: {config_obj.model.path}")
    print(f"✅ RoPE method: {config_obj.rope_extension.method}")
    print(f"✅ Output dir: {config_obj.data.output_dir}")
    
    # Test benchmarks configuration
    enabled_benchmarks = [name for name, bench in config_obj.benchmarks.items() if bench.enabled]
    print(f"✅ Enabled benchmarks: {enabled_benchmarks}")
    
    return config_obj


def test_rope_extension_creation(config_obj):
    """Test RoPE extension creation from config."""
    print("\n🔧 Testing RoPE Extension Creation...")
    
    try:
        rope_method = config_obj.rope_extension.method
        rope_config = dict(config_obj.rope_extension[rope_method])
        
        rope_ext = get_rope_extension(rope_method, rope_config)
        print(f"✅ RoPE extension created: {type(rope_ext).__name__}")
        
        info = rope_ext.get_scaling_info()
        print(f"✅ Scaling info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ RoPE extension creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator_initialization(config_obj):
    """Test RoPEEvaluator initialization with the config."""
    print("\n🚀 Testing RoPEEvaluator Initialization...")
    
    try:
        evaluator = RoPEEvaluator(config_obj)
        print("✅ RoPEEvaluator initialized successfully")
        
        model_info = evaluator.model_loader.get_model_info()
        print(f"✅ Model info: {model_info}")
        
        # Test that output directory is created
        output_dir = evaluator.output_dir
        print(f"✅ Output directory: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ RoPEEvaluator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧩 Testing Comprehensive Configuration Compatibility\n")
    
    tests = [
        ("Configuration Loading", test_comprehensive_config),
    ]
    
    passed = 0
    total = len(tests)
    config_obj = None
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Testing: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            if result is not None:
                config_obj = result
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Run additional tests if config loaded successfully
    if config_obj is not None:
        additional_tests = [
            ("RoPE Extension Creation", lambda: test_rope_extension_creation(config_obj)),
            ("Evaluator Initialization", lambda: test_evaluator_initialization(config_obj))
        ]
        
        for test_name, test_func in additional_tests:
            print(f"\n{'='*50}")
            print(f"Testing: {test_name}")
            print(f"{'='*50}")
            
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
        
        total += len(additional_tests)
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! The comprehensive config is ready.")
        print("\nTo run the evaluation:")
        print("python run_comprehensive.py --config comprehensive_config_fixed.yaml")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())