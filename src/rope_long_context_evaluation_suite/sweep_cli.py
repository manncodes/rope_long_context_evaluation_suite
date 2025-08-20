#!/usr/bin/env python3
"""Command-line interface for RoPE hyperparameter sweeps."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .sweep import SweepConfig, SweepRunner, ParallelSweepRunner, SweepAnalyzer, SweepVisualizer
from .utils import setup_logging


def create_sweep_config_cli():
    """CLI command to create a new sweep configuration."""
    parser = argparse.ArgumentParser(description="Create new sweep configuration")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output YAML file path")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--model-type", choices=["hf_local", "hf_hub", "openai", "anthropic"],
                       default="hf_local", help="Model type")
    parser.add_argument("--methods", nargs="+", 
                       choices=["linear_interpolation", "ntk_aware", "yarn", "longrope", "dynamic_ntk"],
                       default=["linear_interpolation", "yarn"],
                       help="RoPE methods to evaluate")
    parser.add_argument("--context-lengths", nargs="+", type=int,
                       default=[2048, 4096, 8192, 16384, 32768],
                       help="Context lengths to test")
    parser.add_argument("--metrics", nargs="+",
                       choices=["perplexity", "passkey_retrieval", "longppl"],
                       default=["perplexity", "passkey_retrieval"],
                       help="Metrics to evaluate")
    parser.add_argument("--parallel-jobs", type=int, default=1,
                       help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    # Create sweep configuration
    config = SweepConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        rope_methods=args.methods,
        context_lengths=args.context_lengths,
        metrics=args.metrics,
        parallel_jobs=args.parallel_jobs
    )
    
    # Save configuration
    config.to_yaml(args.output)
    print(f"Sweep configuration saved to {args.output}")
    print(f"Total experiments: {config.get_total_experiments()}")


def run_sweep_cli():
    """CLI command to run a hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Run RoPE hyperparameter sweep")
    parser.add_argument("config", type=str, help="Path to sweep configuration YAML")
    parser.add_argument("--parallel", action="store_true", help="Use parallel runner")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = SweepConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration: {config.get_total_experiments()} total experiments")
    
    # Choose runner
    runner_class = ParallelSweepRunner if args.parallel else SweepRunner
    runner = runner_class(config)
    
    # Run sweep
    logger.info("Starting sweep...")
    results = runner.run()
    
    # Report results
    successful = sum(1 for r in results if r['status'] == 'completed')
    logger.info(f"Sweep completed: {successful}/{len(results)} experiments successful")
    logger.info(f"Results saved to {config.output_dir}")


def analyze_sweep_cli():
    """CLI command to analyze sweep results."""
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("results", type=str, help="Path to sweep results JSON")
    parser.add_argument("--output", "-o", type=str, help="Output analysis JSON path")
    parser.add_argument("--metric", type=str, default="perplexity",
                       help="Primary metric for analysis")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top configurations to show")
    
    args = parser.parse_args()
    
    # Load and analyze results
    analyzer = SweepAnalyzer()
    analyzer.load_results(args.results)
    
    # Generate report
    output_path = args.output or Path(args.results).with_suffix('.analysis.json')
    report = analyzer.generate_analysis_report(output_path)
    
    # Print summary
    print(f"Analysis completed. Report saved to {output_path}")
    
    # Show best configurations
    best_configs = analyzer.find_best_configurations(args.metric, args.top_k)
    print(f"\nTop {len(best_configs)} configurations for {args.metric}:")
    for i, config in enumerate(best_configs, 1):
        print(f"{i:2d}. {config['rope_method']:15s} @ {config['context_length']:6d} "
              f"= {config['metric_value']:8.4f}")


def visualize_sweep_cli():
    """CLI command to visualize sweep results."""
    parser = argparse.ArgumentParser(description="Visualize sweep results")
    parser.add_argument("results", type=str, help="Path to sweep results JSON")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for plots")
    parser.add_argument("--contours", action="store_true", help="Generate contour plots")
    parser.add_argument("--3d", action="store_true", help="Generate 3D surface plots")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive plots")
    
    args = parser.parse_args()
    
    # Load results and create visualizer
    visualizer = SweepVisualizer()
    visualizer.load_results(args.results)
    
    # Set output directory
    output_dir = args.output_dir or Path(args.results).parent / "visualizations"
    
    # Generate plots
    generated_files = visualizer.generate_report(
        output_dir=output_dir,
        include_contours=args.contours,
        include_3d=args.contours and args.__dict__.get('3d', False),
        include_interactive=args.interactive
    )
    
    print(f"Generated {len(generated_files)} visualization files in {output_dir}")
    for name, path in generated_files.items():
        print(f"  {name}: {path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RoPE Hyperparameter Sweep CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config subcommand
    create_parser = subparsers.add_parser('create-config', help='Create sweep configuration')
    create_parser.set_defaults(func=create_sweep_config_cli)
    
    # Run sweep subcommand  
    run_parser = subparsers.add_parser('run', help='Run hyperparameter sweep')
    run_parser.set_defaults(func=run_sweep_cli)
    
    # Analyze results subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze sweep results')
    analyze_parser.set_defaults(func=analyze_sweep_cli)
    
    # Visualize results subcommand
    viz_parser = subparsers.add_parser('visualize', help='Visualize sweep results')
    viz_parser.set_defaults(func=visualize_sweep_cli)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()