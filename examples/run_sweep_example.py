#!/usr/bin/env python3
"""
Example script demonstrating hyperparameter sweep usage.

This script shows how to:
1. Load and configure a hyperparameter sweep
2. Run the sweep with different runners
3. Analyze and visualize results
4. Generate comprehensive reports

Usage:
    python examples/run_sweep_example.py --config examples/sweep_configs/quick_comparison_sweep.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rope_long_context_evaluation_suite.sweep import (
    SweepConfig, SweepRunner, ParallelSweepRunner, 
    SweepAnalyzer, SweepVisualizer
)
from rope_long_context_evaluation_suite.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run RoPE hyperparameter sweep")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to sweep configuration YAML file")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel sweep runner")
    parser.add_argument("--analyze-only", type=str,
                       help="Skip sweep and only analyze existing results at this path")
    parser.add_argument("--visualize-only", type=str,
                       help="Skip sweep and only visualize existing results at this path") 
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RoPE hyperparameter sweep example")
    
    # Handle analysis-only mode
    if args.analyze_only:
        logger.info(f"Analyzing existing results from {args.analyze_only}")
        analyzer = SweepAnalyzer()
        analyzer.load_results(args.analyze_only)
        
        # Generate analysis report
        report = analyzer.generate_analysis_report(
            output_path=Path(args.analyze_only).parent / "analysis_report.json"
        )
        
        logger.info("Analysis complete. Check analysis_report.json for results.")
        return
    
    # Handle visualization-only mode  
    if args.visualize_only:
        logger.info(f"Visualizing existing results from {args.visualize_only}")
        visualizer = SweepVisualizer()
        visualizer.load_results(args.visualize_only)
        
        # Generate visualization report
        output_dir = Path(args.visualize_only).parent / "visualizations"
        files = visualizer.generate_report(
            output_dir=output_dir,
            include_contours=True,
            include_3d=True,
            include_interactive=True
        )
        
        logger.info(f"Visualization complete. Generated {len(files)} files in {output_dir}")
        return
    
    # Load sweep configuration
    logger.info(f"Loading sweep configuration from {args.config}")
    sweep_config = SweepConfig.from_yaml(args.config)
    
    logger.info(f"Sweep configuration loaded:")
    logger.info(f"  Model: {sweep_config.model_name}")
    logger.info(f"  Methods: {sweep_config.rope_methods}")
    logger.info(f"  Context lengths: {sweep_config.context_lengths}")
    logger.info(f"  Metrics: {sweep_config.metrics}")
    logger.info(f"  Total experiments: {sweep_config.get_total_experiments()}")
    
    # Choose runner type
    if args.parallel:
        logger.info(f"Using parallel runner with {sweep_config.parallel_jobs} jobs")
        runner = ParallelSweepRunner(sweep_config)
    else:
        logger.info("Using sequential runner")
        runner = SweepRunner(sweep_config)
    
    # Run the sweep
    logger.info("Starting hyperparameter sweep...")
    try:
        results = runner.run()
        logger.info(f"Sweep completed! Generated {len(results)} results")
        
        # Count successful vs failed experiments
        successful = sum(1 for r in results if r['status'] == 'completed')
        failed = len(results) - successful
        logger.info(f"Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
        
        if failed > 0:
            logger.warning(f"{failed} experiments failed")
            
    except Exception as e:
        logger.error(f"Sweep failed with error: {e}")
        return 1
    
    # Analyze results
    logger.info("Analyzing results...")
    analyzer = SweepAnalyzer(results)
    
    # Generate summary statistics
    stats = analyzer.compute_summary_statistics()
    logger.info("Summary statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Find best configurations
    for metric in sweep_config.metrics:
        logger.info(f"\nTop 3 configurations for {metric}:")
        best_configs = analyzer.find_best_configurations(metric, top_k=3)
        for i, config in enumerate(best_configs, 1):
            logger.info(f"  {i}. Method: {config['rope_method']}, "
                       f"Context: {config['context_length']}, "
                       f"Value: {config['metric_value']:.4f}")
    
    # Analyze parameter sensitivity
    for metric in sweep_config.metrics:
        sensitivity = analyzer.analyze_parameter_sensitivity(metric)
        if sensitivity:
            logger.info(f"\nParameter sensitivity for {metric}:")
            for param, sens in sensitivity.items():
                logger.info(f"  {param}: correlation={sens['pearson_correlation']:.3f}, "
                           f"variance_explained={sens['variance_explained']:.3f}")
    
    # Generate comprehensive analysis report
    report_path = Path(sweep_config.output_dir) / "analysis_report.json"
    report = analyzer.generate_analysis_report(report_path)
    logger.info(f"Detailed analysis saved to {report_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = SweepVisualizer(results)
    
    viz_dir = Path(sweep_config.output_dir) / "visualizations"
    generated_files = visualizer.generate_report(
        output_dir=viz_dir,
        include_contours=True,
        include_3d=True,
        include_interactive=True
    )
    
    logger.info(f"Generated {len(generated_files)} visualization files:")
    for name, path in generated_files.items():
        logger.info(f"  {name}: {path}")
    
    # Example: Create specific contour plot
    if len(sweep_config.rope_methods) > 0:
        method = sweep_config.rope_methods[0]
        try:
            # Get parameter columns from the first method
            analyzer_df = analyzer.df
            param_cols = [col.replace('param_', '') for col in analyzer_df.columns 
                         if col.startswith('param_')]
            
            if len(param_cols) >= 2 and not analyzer_df.empty:
                fig = visualizer.plot_contour(
                    param_cols[0], param_cols[1], 'perplexity',
                    rope_method=method
                )
                contour_path = viz_dir / f"example_contour_{method}.png"
                fig.savefig(contour_path, dpi=300, bbox_inches='tight')
                logger.info(f"Example contour plot saved to {contour_path}")
                
        except Exception as e:
            logger.warning(f"Could not generate example contour plot: {e}")
    
    # Performance vs context length plot
    try:
        fig = visualizer.plot_performance_vs_context_length('perplexity')
        perf_path = viz_dir / "performance_vs_context_length.png"
        fig.savefig(perf_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance vs context length plot saved to {perf_path}")
    except Exception as e:
        logger.warning(f"Could not generate performance plot: {e}")
    
    logger.info("Sweep example completed successfully!")
    logger.info(f"Results available in: {sweep_config.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())