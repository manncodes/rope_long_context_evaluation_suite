#!/usr/bin/env python3
"""Quick start example for RoPE Long Context Evaluation Suite."""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rope_long_context_evaluation_suite import RoPEEvaluator, Config


def main():
    """Run a quick evaluation example."""
    print("🚀 RoPE Long Context Evaluation Suite - Quick Start")
    print("=" * 60)
    
    # Create a simple configuration
    config_dict = {
        "model": {
            "type": "hf_local",
            "name": "gpt2",  # Use GPT-2 for quick testing
            "path": "gpt2",
            "max_length": 1024,
            "device_map": "auto",
        },
        "rope_extension": {
            "method": "linear_interpolation",
            "linear_interpolation": {
                "scaling_factor": 2
            }
        },
        "benchmarks": {
            "niah": {
                "enabled": True,
                "variants": ["standard"],
                "context_lengths": [512, 1024],
            },
            "ruler": {
                "enabled": False,
            },
            "longbench": {
                "enabled": False,
            },
            "longbench_v2": {
                "enabled": False,
            }
        },
        "evaluation": {
            "batch_size": 1,
            "save_predictions": True,
            "generation": {
                "max_new_tokens": 50,
                "temperature": 0.0,
            }
        },
        "data": {
            "output_dir": "./results/quick_start/",
        },
        "logging": {
            "level": "INFO",
        },
        "seed": 42,
    }
    
    # Initialize evaluator
    print("📦 Initializing evaluator...")
    config = Config(config_dict)
    evaluator = RoPEEvaluator(config)
    
    print("🤖 Loading model and applying RoPE extension...")
    evaluator.load_model()
    
    print("🧪 Running evaluation...")
    results = evaluator.evaluate()
    
    print("✅ Evaluation completed!")
    print(f"📊 Results saved to: {config.data.output_dir}")
    
    # Print summary
    if "summary" in results:
        summary = results["summary"]
        print(f"📈 Overall average score: {summary.get('overall_average', 0.0):.3f}")
        print(f"🎯 Benchmarks run: {', '.join(summary.get('benchmarks_run', []))}")
    
    print("🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Check the results in the output directory")
    print("2. Modify the configuration for your specific model and tasks")
    print("3. Run full evaluation with: rope-eval --config config/default.yaml")


if __name__ == "__main__":
    main()