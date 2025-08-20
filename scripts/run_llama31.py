#!/usr/bin/env python3
"""Run Llama 3.1 1B evaluation with YaRN."""

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


def run_llama31_evaluation():
    """Run Llama 3.1 1B evaluation with YaRN."""
    print("🦙 Running Llama 3.1 1B evaluation with YaRN...")
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Load test configuration
    config = Config("test_llama31_1b.yaml")
    print(f"✅ Configuration loaded: {config.model.name}")
    print(f"🔗 RoPE method: {config.rope_extension.method}")
    
    # Initialize evaluator
    evaluator = RoPEEvaluator(config)
    print("✅ Evaluator initialized")
    
    try:
        # Load model
        print("📦 Loading Llama 3.1 1B model...")
        evaluator.load_model()
        print("✅ Model loaded successfully")
        print(f"🎛️  RoPE info: {evaluator.rope_info}")
        
        # Run evaluation
        print("🚀 Starting evaluation...")
        results = evaluator.evaluate()
        print("✅ Evaluation completed!")
        
        # Print detailed results
        if "summary" in results:
            summary = results["summary"]
            print(f"📊 Overall average score: {summary.get('overall_average', 0.0):.3f}")
            print(f"🎯 Benchmarks run: {', '.join(summary.get('benchmarks_run', []))}")
            print(f"📈 Number of benchmarks: {summary.get('num_benchmarks', 0)}")
        
        # Print individual benchmark results
        if "benchmarks" in results:
            print("\n📋 Individual Benchmark Results:")
            for bench_name, bench_results in results["benchmarks"].items():
                if "average_score" in bench_results:
                    score = bench_results["average_score"]
                    num_samples = bench_results.get("num_samples", 0)
                    print(f"  • {bench_name.upper()}: {score:.3f} (on {num_samples} samples)")
        
        print(f"\n💾 Results saved to: {config.data.output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main evaluation function."""
    print("🔬 RoPE Long Context Evaluation Suite - Llama 3.1 1B Test")
    print("=" * 60)
    
    success = run_llama31_evaluation()
    
    if success:
        print("\n🎉 Llama 3.1 1B evaluation COMPLETED!")
        print("✅ YaRN RoPE extension tested successfully")
    else:
        print("\n❌ Llama 3.1 1B evaluation FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()