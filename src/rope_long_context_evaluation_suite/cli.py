"""Command-line interface for RoPE Long Context Evaluation Suite."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from .core import RoPEEvaluator
from .utils import Config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RoPE Long Context Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=[],
        help="Configuration overrides (key=value format)",
    )
    
    # Model settings
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name or path to override config",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["hf_local", "hf_hub", "openai", "anthropic"],
        help="Model type to override config",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum sequence length to override config",
    )
    
    # RoPE method
    parser.add_argument(
        "--rope-method",
        type=str,
        choices=["linear_interpolation", "ntk_aware", "yarn", "longrope", "dynamic_ntk"],
        help="RoPE extension method to override config",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        help="Scaling factor for linear interpolation",
    )
    
    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="*",
        choices=["niah", "ruler", "longbench", "longbench_v2"],
        help="Benchmarks to run (overrides config)",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to evaluate per benchmark",
    )
    
    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        help="GPU IDs to use for evaluation",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output",
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running evaluation",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume evaluation from checkpoint directory",
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """Apply command-line argument overrides to configuration."""
    # Model overrides
    if args.model_name is not None:
        config.model.name = args.model_name
        if args.model_type == "hf_local":
            config.model.path = args.model_name
    
    if args.model_type is not None:
        config.model.type = args.model_type
    
    if args.max_length is not None:
        config.model.max_length = args.max_length
    
    # RoPE method overrides
    if args.rope_method is not None:
        config.rope_extension.method = args.rope_method
    
    if args.scaling_factor is not None:
        config.rope_extension.linear_interpolation.scaling_factor = args.scaling_factor
    
    # Benchmark overrides
    if args.benchmarks is not None:
        # Disable all benchmarks first
        for benchmark in ["niah", "ruler", "longbench", "longbench_v2"]:
            if benchmark in config.benchmarks:
                config.benchmarks[benchmark].enabled = False
        
        # Enable selected benchmarks
        for benchmark in args.benchmarks:
            if benchmark in config.benchmarks:
                config.benchmarks[benchmark].enabled = True
    
    # Evaluation overrides
    if args.output_dir is not None:
        config.data.output_dir = args.output_dir
    
    if args.batch_size is not None:
        config.evaluation.batch_size = args.batch_size
    
    if args.num_samples is not None:
        for benchmark in config.benchmarks.values():
            if hasattr(benchmark, "max_samples"):
                benchmark.max_samples = args.num_samples
    
    # Hardware overrides
    if args.device is not None:
        config.model.device_map = args.device
    
    if args.gpu_ids is not None:
        config.hardware.gpu_ids = args.gpu_ids
    
    # Other overrides
    if args.seed is not None:
        config.seed = args.seed
    
    return config


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file {config_path} not found", file=sys.stderr)
            return 1
        
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))
        
        # Apply overrides from command line
        for override in args.overrides:
            if "=" not in override:
                print(f"Error: Invalid override format '{override}'. Use key=value format.", file=sys.stderr)
                return 1
            key, value = override.split("=", 1)
            OmegaConf.set(config, key, yaml.safe_load(value))
        
        # Apply CLI argument overrides
        config = apply_cli_overrides(config, args)
        
        # Setup logging
        log_level = args.log_level if not args.quiet else "ERROR"
        setup_logging(
            level=getattr(logging, log_level),
            log_file=config.logging.get("file"),
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting RoPE Long Context Evaluation Suite v0.1.0")
        
        # Dry run mode
        if args.dry_run:
            print("Configuration:")
            print(OmegaConf.to_yaml(config))
            return 0
        
        # Initialize evaluator
        evaluator = RoPEEvaluator(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            evaluator.resume_from_checkpoint(args.resume)
        
        # Run evaluation
        results = evaluator.evaluate()
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Results saved to {config.data.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\nEvaluation interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        logging.getLogger(__name__).exception("Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())