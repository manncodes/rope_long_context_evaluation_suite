#!/usr/bin/env python3
"""Simple test runner for the RoPE evaluation framework."""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rope_long_context_evaluation_suite import RoPEEvaluator, Config, setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def run_simple_test():
    """Run a simple test with GPT-2 to validate the framework."""
    print("🧪 Running simple framework test with GPT-2...")
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Load test configuration
    config = Config("test_gpt2.yaml")
    print(f"✅ Configuration loaded: {config.model.name}")
    
    # Initialize evaluator
    evaluator = RoPEEvaluator(config)
    print("✅ Evaluator initialized")
    
    try:
        # Load model
        print("📦 Loading model...")
        evaluator.load_model()
        print("✅ Model loaded successfully")
        
        # Run evaluation
        print("🚀 Starting evaluation...")
        results = evaluator.evaluate()
        print("✅ Evaluation completed!")
        
        # Print summary
        if "summary" in results:
            summary = results["summary"]
            print(f"📊 Overall average score: {summary.get('overall_average', 0.0):.3f}")
            print(f"🎯 Benchmarks run: {', '.join(summary.get('benchmarks_run', []))}")
        
        print(f"💾 Results saved to: {config.data.output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🔬 RoPE Long Context Evaluation Suite - Test Runner")
    print("=" * 55)
    
    success = run_simple_test()
    
    if success:
        print("\n🎉 Framework test PASSED!")
        print("✅ Ready to run full evaluations with larger models")
    else:
        print("\n❌ Framework test FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()