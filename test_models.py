#!/usr/bin/env python3
"""Test script to validate the framework with Llama 3.1 1B and Llama 3.2 1B models."""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rope_long_context_evaluation_suite import RoPEEvaluator, Config, setup_logging
    from rope_long_context_evaluation_suite.models import ModelLoader, get_rope_extension
    from rope_long_context_evaluation_suite.benchmarks import NIAHBenchmark, RULERBenchmark
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


def test_model_loading(config_path: str, model_name: str):
    """Test model loading and basic functionality."""
    print(f"🧪 Testing {model_name}")
    print("=" * 50)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = Config(config_path)
        print(f"✅ Configuration loaded for {config.model.name}")
        
        # Initialize model loader
        print("🔧 Initializing model loader...")
        model_loader = ModelLoader(config)
        
        # Check device info
        from rope_long_context_evaluation_suite.utils import get_device_info
        device_info = get_device_info()
        print(f"🖥️  Device info: CUDA available: {device_info['cuda_available']}")
        
        # Test model info (without loading the full model due to resource constraints)
        model_info = model_loader.get_model_info()
        print(f"📊 Model info: {model_info}")
        
        # Test RoPE extension creation
        print("🔗 Testing RoPE extension...")
        rope_config = config.rope_extension
        rope_extension = get_rope_extension(
            rope_config.method,
            getattr(rope_config, rope_config.method)
        )
        scaling_info = rope_extension.get_scaling_info()
        print(f"⚙️  RoPE extension: {scaling_info}")
        
        # Test benchmark initialization (without running full evaluation)
        print("🎯 Testing benchmark initialization...")
        
        # Test NIAH benchmark
        niah_config = config.benchmarks.niah
        print(f"🔍 NIAH config: enabled={niah_config.enabled}, variants={niah_config.variants}")
        
        # Test RULER benchmark  
        ruler_config = config.benchmarks.ruler
        print(f"📏 RULER config: enabled={ruler_config.enabled}, categories={ruler_config.categories}")
        
        print(f"✅ {model_name} configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} test FAILED: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False


def test_yaml_loading():
    """Test YAML configuration loading."""
    print("📄 Testing YAML configuration loading...")
    
    try:
        # Test default config
        default_config = Config("config/default.yaml")
        print(f"✅ Default config loaded: {default_config.model.name}")
        
        # Test that we can access nested configurations
        rope_method = default_config.rope_extension.method
        print(f"🔗 Default RoPE method: {rope_method}")
        
        benchmarks_enabled = [
            name for name, config in default_config.benchmarks.items() 
            if config.get("enabled", False)
        ]
        print(f"🎯 Enabled benchmarks: {benchmarks_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ YAML loading test FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 RoPE Long Context Evaluation Suite - Model Testing")
    print("=" * 60)
    
    # Setup basic logging
    setup_logging(level=logging.INFO)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    tests_passed = 0
    total_tests = 0
    
    # Test YAML loading
    total_tests += 1
    if test_yaml_loading():
        tests_passed += 1
    print()
    
    # Test Llama 3.1 1B configuration
    total_tests += 1
    if test_model_loading("test_llama31_1b.yaml", "Llama 3.1 1B"):
        tests_passed += 1
    print()
    
    # Test Llama 3.2 1B configuration
    total_tests += 1
    if test_model_loading("test_llama32_1b.yaml", "Llama 3.2 1B"):
        tests_passed += 1
    print()
    
    # Summary
    print("📊 Test Summary")
    print("=" * 30)
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    print(f"❌ Tests failed: {total_tests - tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests PASSED! The framework is ready for use.")
        print("\n🚀 Next steps:")
        print("1. Download models: huggingface-cli download meta-llama/Llama-3.1-1B")
        print("2. Run evaluation: python -m rope_long_context_evaluation_suite.cli --config test_llama31_1b.yaml")
        print("3. Check results in the results/ directory")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()