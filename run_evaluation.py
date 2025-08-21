#!/usr/bin/env python3
"""Model-agnostic evaluation script using the unified infrastructure."""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rope_long_context_evaluation_suite.core import RoPEEvaluator
from rope_long_context_evaluation_suite.utils import setup_logging
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description="Run RoPE evaluation with any model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="comprehensive_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model path/name (e.g., 'meta-llama/Llama-3.2-1B', 'mistralai/Mistral-7B-v0.1')"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name for results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["niah", "ruler", "longbench", "longbench_v2"],
        help="Specify which benchmarks to run"
    )
    parser.add_argument(
        "--rope-method",
        type=str,
        choices=["none", "linear", "ntk_aware", "yarn", "longrope", "dynamic_ntk", "llama3"],
        default="none",
        help="RoPE scaling method to use"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per benchmark"
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        help="Context lengths to test (e.g., 2048 4096 8192)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        help="Override torch dtype"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable model caching"
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    config = OmegaConf.load(args.config)
    
    # Apply command-line overrides
    if args.model:
        config.model.path = args.model
        # Also update tokenizer path to match
        config.model.tokenizer_path = args.model
        print(f"📦 Using model: {args.model}")
    
    if args.model_name:
        config.model.name = args.model_name
    elif args.model:
        # Auto-generate name from model path
        config.model.name = args.model.split("/")[-1].lower()
    
    if args.output_dir:
        config.data.output_dir = args.output_dir
    
    if args.benchmarks:
        # Disable all benchmarks first
        for bench in ["niah", "ruler", "longbench", "longbench_v2"]:
            if bench in config.benchmarks:
                config.benchmarks[bench].enabled = False
        # Enable only specified benchmarks
        for bench in args.benchmarks:
            if bench in config.benchmarks:
                config.benchmarks[bench].enabled = True
                print(f"✅ Enabled benchmark: {bench}")
    
    if args.rope_method:
        config.rope_extension.method = args.rope_method
        print(f"🔄 Using RoPE method: {args.rope_method}")
    
    if args.max_samples:
        for bench in ["niah", "ruler", "longbench", "longbench_v2"]:
            if bench in config.benchmarks:
                config.benchmarks[bench].max_samples = args.max_samples
    
    if args.context_lengths:
        # Update context lengths for benchmarks that support it
        if "niah" in config.benchmarks:
            config.benchmarks.niah.context_lengths = args.context_lengths
        if "ruler" in config.benchmarks:
            config.benchmarks.ruler.max_length = max(args.context_lengths)
    
    if args.device:
        config.model.device_map = args.device
    
    if args.dtype:
        config.model.torch_dtype = args.dtype
    
    if args.no_cache:
        config.evaluation.use_cache = False
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level, log_file=config.logging.file)
    
    # Print configuration summary
    print("\n" + "="*60)
    print("🚀 RoPE LONG CONTEXT EVALUATION")
    print("="*60)
    print(f"Model: {config.model.path}")
    print(f"Model Name: {config.model.name}")
    print(f"RoPE Method: {config.rope_extension.method}")
    print(f"Output Directory: {config.data.output_dir}")
    print(f"Device: {config.model.device_map}")
    print(f"Dtype: {config.model.torch_dtype}")
    
    print("\nEnabled Benchmarks:")
    for bench in ["niah", "ruler", "longbench", "longbench_v2"]:
        if bench in config.benchmarks and config.benchmarks[bench].enabled:
            print(f"  - {bench.upper()}")
    
    print("\n" + "="*60)
    
    # Run evaluation
    try:
        evaluator = RoPEEvaluator(config)
        results = evaluator.evaluate()
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE")
        print("="*60)
        
        # Print summary if available
        if "benchmarks" in results:
            for bench_name, bench_results in results["benchmarks"].items():
                if isinstance(bench_results, dict) and "average_score" in bench_results:
                    print(f"{bench_name.upper()}: {bench_results['average_score']:.3f}")
        
        print(f"\n📁 Results saved to: {config.data.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()