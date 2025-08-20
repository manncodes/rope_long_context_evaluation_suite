#!/usr/bin/env python3
"""
Comprehensive RoPE sweep on TinyLlama 1.1B model.
This script runs a thorough evaluation of all RoPE scaling methods.
NO shortcuts - full results as requested.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rope_long_context_evaluation_suite.sweep.config import SweepConfig
from rope_long_context_evaluation_suite.sweep.runner import SweepRunner
from rope_long_context_evaluation_suite.sweep.analyzer import SweepAnalyzer
from rope_long_context_evaluation_suite.sweep.visualizer import SweepVisualizer

def load_sweep_config(config_path: str) -> SweepConfig:
    """Load sweep configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to SweepConfig
    # For now, create a basic config - we'll implement YAML loading properly
    config = SweepConfig(
        model_name=config_dict['model_name'],
        model_type=config_dict.get('model_type', 'hf_local'),
        rope_methods=config_dict['rope_methods'],
        context_lengths=config_dict['context_lengths'],
        metrics=config_dict['metrics'],
        max_configs_per_method=config_dict.get('max_configs_per_method', 25),
        parallel_jobs=config_dict.get('parallel_jobs', 2),
        use_cache=config_dict.get('use_cache', True),
        cache_dir=config_dict.get('cache_dir', './tinyllama_cache'),
        output_dir=config_dict.get('output_dir', './tinyllama_results')
    )
    
    return config

def run_comprehensive_sweep():
    """Run the comprehensive sweep with full results."""
    
    print("🚀 Starting Comprehensive RoPE Sweep on TinyLlama 1.1B")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    config_path = "tinyllama_full_sweep.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Configuration file {config_path} not found!")
        return
    
    print(f"📋 Loading configuration from {config_path}")
    config = load_sweep_config(config_path)
    
    print(f"🎯 Configuration loaded:")
    print(f"   Model: {config.model_name}")
    print(f"   RoPE methods: {len(config.rope_methods)} ({', '.join(config.rope_methods)})")
    print(f"   Context lengths: {config.context_lengths}")
    print(f"   Metrics: {', '.join(config.metrics)}")
    print(f"   Max configs per method: {config.max_configs_per_method}")
    print(f"   Parallel jobs: {config.parallel_jobs}")
    print()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize sweep runner
    print("🏃 Initializing sweep runner...")
    runner = SweepRunner(config)
    
    # Run the sweep
    start_time = time.time()
    
    try:
        print("⚡ Starting hyperparameter sweep...")
        print("   This will take a while - running ALL methods with ALL parameters")
        print("   No shortcuts, no early stopping - comprehensive evaluation as requested")
        print()
        
        results = runner.run()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Sweep completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"📊 Results summary:")
        print(f"   Total experiments: {len(results)}")
        
        # Count successes and failures
        successful = [r for r in results if not r.get('failed', False)]
        failed = [r for r in results if r.get('failed', False)]
        
        print(f"   Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"   Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        
        # Save raw results
        results_file = output_dir / f"raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   Raw results saved to: {results_file}")
        
        # Run analysis
        print("\n📈 Running analysis...")
        analyzer = SweepAnalyzer(results)
        analysis = analyzer.analyze()
        
        # Save analysis
        analysis_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"   Analysis saved to: {analysis_file}")
        
        # Generate visualizations
        print("\n🎨 Generating visualizations...")
        visualizer = SweepVisualizer(results)
        
        viz_types = ["contour", "heatmap", "parameter_sensitivity", "method_comparison", "performance_vs_context_length"]
        
        for viz_type in viz_types:
            try:
                if viz_type == "contour":
                    # Generate contour plots for each method
                    for method in config.rope_methods:
                        method_results = [r for r in results if r.get('rope_method') == method]
                        if len(method_results) > 5:  # Need sufficient data points
                            fig = visualizer.plot_contour_for_method(method, "perplexity")
                            if fig:
                                fig_path = output_dir / f"{method}_contour.png"
                                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                                print(f"   ✅ {method} contour plot: {fig_path}")
                
                elif viz_type == "heatmap":
                    fig = visualizer.plot_context_length_heatmap("perplexity")
                    if fig:
                        fig_path = output_dir / "context_length_heatmap.png"
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        print(f"   ✅ Context length heatmap: {fig_path}")
                
                elif viz_type == "parameter_sensitivity":
                    fig = visualizer.plot_parameter_sensitivity("perplexity")
                    if fig:
                        fig_path = output_dir / "parameter_sensitivity.png"
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        print(f"   ✅ Parameter sensitivity: {fig_path}")
                
                elif viz_type == "method_comparison":
                    fig = visualizer.plot_method_comparison()
                    if fig:
                        fig_path = output_dir / "method_comparison.png"
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        print(f"   ✅ Method comparison: {fig_path}")
                
                elif viz_type == "performance_vs_context_length":
                    fig = visualizer.plot_performance_vs_context_length("perplexity")
                    if fig:
                        fig_path = output_dir / "performance_vs_context_length.png"
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        print(f"   ✅ Performance vs context length: {fig_path}")
                        
            except Exception as e:
                print(f"   ⚠️  Error generating {viz_type}: {str(e)[:60]}...")
        
        # Generate summary report
        print(f"\n📑 Generating final report...")
        generate_final_report(results, analysis, output_dir)
        
        print(f"\n🎉 COMPLETE! All results saved to: {output_dir}")
        print(f"   Duration: {duration/60:.1f} minutes")
        print(f"   Total experiments: {len(results)}")
        print(f"   Success rate: {len(successful)/len(results)*100:.1f}%")
        
        return results, analysis
        
    except Exception as e:
        print(f"\n❌ Sweep failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_final_report(results, analysis, output_dir):
    """Generate a comprehensive final report."""
    
    report_file = output_dir / "FINAL_REPORT.md"
    
    with open(report_file, 'w') as f:
        f.write("# TinyLlama 1.1B RoPE Scaling Comprehensive Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Experiments**: {len(results)}\n")
        
        successful = [r for r in results if not r.get('failed', False)]
        failed = [r for r in results if r.get('failed', False)]
        
        f.write(f"- **Successful**: {len(successful)} ({len(successful)/len(results)*100:.1f}%)\n")
        f.write(f"- **Failed**: {len(failed)} ({len(failed)/len(results)*100:.1f}%)\n\n")
        
        # Method breakdown
        f.write("## Results by Method\n\n")
        
        methods = set(r.get('rope_method', 'unknown') for r in results)
        for method in sorted(methods):
            method_results = [r for r in results if r.get('rope_method') == method]
            method_successful = [r for r in method_results if not r.get('failed', False)]
            
            f.write(f"### {method}\n")
            f.write(f"- Experiments: {len(method_results)}\n")
            f.write(f"- Successful: {len(method_successful)} ({len(method_successful)/len(method_results)*100:.1f}%)\n")
            
            if method_successful:
                perplexities = [r.get('metrics', {}).get('perplexity') for r in method_successful 
                              if r.get('metrics', {}).get('perplexity') is not None]
                if perplexities:
                    f.write(f"- Best Perplexity: {min(perplexities):.2f}\n")
                    f.write(f"- Avg Perplexity: {sum(perplexities)/len(perplexities):.2f}\n")
            
            f.write("\n")
        
        # Best configurations
        f.write("## Top 10 Best Configurations\n\n")
        
        successful_with_ppl = [r for r in successful 
                              if r.get('metrics', {}).get('perplexity') is not None]
        
        if successful_with_ppl:
            best_configs = sorted(successful_with_ppl, 
                                key=lambda x: x['metrics']['perplexity'])[:10]
            
            for i, config in enumerate(best_configs, 1):
                f.write(f"**{i}. {config.get('rope_method', 'unknown')}** "
                       f"(Perplexity: {config['metrics']['perplexity']:.3f})\n")
                f.write(f"- Context Length: {config.get('context_length', 'unknown')}\n")
                f.write(f"- Parameters: {config.get('rope_config', {})}\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- Raw results: `raw_results_*.json`\n")
        f.write("- Analysis: `analysis_*.json`\n") 
        f.write("- Visualizations: `*.png` files\n")
        f.write("- This report: `FINAL_REPORT.md`\n")
    
    print(f"   📑 Final report: {report_file}")

if __name__ == "__main__":
    print("Starting comprehensive TinyLlama RoPE evaluation...")
    results, analysis = run_comprehensive_sweep()
    
    if results is not None:
        print("\n✅ Evaluation completed successfully!")
        print("Check the output directory for all results, analysis, and visualizations.")
    else:
        print("\n❌ Evaluation failed. Check logs for details.")
        sys.exit(1)