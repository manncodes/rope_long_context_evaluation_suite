#!/usr/bin/env python3
"""Validation script to test all imports are working correctly."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all module imports."""
    print("Testing module imports...")
    
    try:
        # Test basic imports
        print("✓ Testing basic imports...")
        from rope_long_context_evaluation_suite.utils import Config, setup_logging, save_results
        print("  ✓ utils module")
        
        # Test models module
        print("✓ Testing models module...")
        from rope_long_context_evaluation_suite.models import (
            BaseRoPEExtension, LinearInterpolationRoPE, NTKAwareRoPE,
            YaRNRoPE, LongRoPE, DynamicNTKRoPE, ModelLoader, get_rope_extension
        )
        print("  ✓ models module")
        
        # Test metrics module
        print("✓ Testing metrics module...")
        from rope_long_context_evaluation_suite.metrics import (
            BaseMetric, PerplexityMetric, SlidingWindowPerplexity,
            PasskeyRetrievalMetric, MultiNeedleRetrievalMetric, LongPPLMetric
        )
        print("  ✓ metrics module")
        
        # Test sweep module
        print("✓ Testing sweep module...")
        from rope_long_context_evaluation_suite.sweep import (
            SweepConfig, SweepRunner, ParallelSweepRunner,
            SweepAnalyzer, SweepVisualizer
        )
        print("  ✓ sweep module")
        
        # Test core module
        print("✓ Testing core module...")
        from rope_long_context_evaluation_suite.core import RoPEEvaluator
        print("  ✓ core module")
        
        # Test main package import
        print("✓ Testing main package import...")
        import rope_long_context_evaluation_suite
        print(f"  ✓ Package version: {rope_long_context_evaluation_suite.__version__}")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print(f"   Error details: {e.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test RoPE extension creation
        print("✓ Testing RoPE extension creation...")
        from rope_long_context_evaluation_suite.models import get_rope_extension
        
        linear_config = {"scaling_factor": 4.0}
        linear_rope = get_rope_extension("linear_interpolation", linear_config)
        print(f"  ✓ Created LinearInterpolationRoPE: {linear_rope.scaling_factor}")
        
        yarn_config = {"s": 16.0, "alpha": 1.0, "beta": 32.0}
        yarn_rope = get_rope_extension("yarn", yarn_config)
        print(f"  ✓ Created YaRNRoPE: s={yarn_rope.s}, alpha={yarn_rope.alpha}")
        
        # Test sweep config creation
        print("✓ Testing sweep config creation...")
        from rope_long_context_evaluation_suite.sweep import SweepConfig
        
        config = SweepConfig(
            model_name="test-model",
            rope_methods=["linear_interpolation", "yarn"],
            context_lengths=[2048, 4096],
            metrics=["perplexity"]
        )
        print(f"  ✓ Created SweepConfig: {config.get_total_experiments()} experiments")
        
        # Test metric creation
        print("✓ Testing metric creation...")
        from rope_long_context_evaluation_suite.metrics import PerplexityMetric
        
        metric = PerplexityMetric()
        print(f"  ✓ Created PerplexityMetric")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    print("RoPE Long Context Evaluation Suite - Import Validation")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ All validation tests passed!")
            print("\nNext steps:")
            print("1. Install dependencies: pip install -e .")
            print("2. Run example: python examples/run_sweep_example.py --config examples/sweep_configs/quick_comparison_sweep.yaml")
            return 0
        else:
            print("\n❌ Functionality tests failed!")
            return 1
    else:
        print("\n❌ Import tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())