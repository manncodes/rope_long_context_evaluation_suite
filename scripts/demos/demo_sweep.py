#!/usr/bin/env python3
"""
Demonstration script showing hyperparameter sweep results.
Uses mock data to demonstrate functionality without requiring large models.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rope_long_context_evaluation_suite.sweep import SweepConfig, SweepAnalyzer, SweepVisualizer

def generate_mock_results():
    """Generate realistic mock results for demonstration."""
    
    # Create realistic experimental results
    results = []
    
    methods = ["linear_interpolation", "ntk_aware", "yarn"]
    context_lengths = [4096, 8192, 16384, 32768]
    
    for method in methods:
        for context_length in context_lengths:
            if method == "linear_interpolation":
                for scaling_factor in [2, 4, 8]:
                    # Simulate that performance degrades with higher scaling factors
                    base_perplexity = 25.0 + scaling_factor * 2.0 + (context_length / 4096) * 5.0
                    base_accuracy = max(0.1, 0.95 - scaling_factor * 0.1 - (context_length / 32768) * 0.3)
                    
                    result = {
                        "experiment_config": {
                            "rope_method": method,
                            "context_length": context_length,
                            "parameters": {"scaling_factor": scaling_factor},
                            "model_name": "demo-model",
                            "model_type": "hf_local"
                        },
                        "metrics": {
                            "perplexity": {
                                "perplexity": base_perplexity + np.random.normal(0, 2.0),
                                "loss": np.log(base_perplexity + np.random.normal(0, 2.0)),
                                "total_tokens": 10000
                            },
                            "passkey_retrieval": {
                                "passkey_accuracy": max(0.0, min(1.0, base_accuracy + np.random.normal(0, 0.05))),
                                "correct_retrievals": int(10 * base_accuracy),
                                "total_samples": 10
                            }
                        },
                        "execution_time": 120.0 + np.random.normal(0, 20.0),
                        "status": "completed"
                    }
                    results.append(result)
                    
            elif method == "ntk_aware":
                for alpha in [1.0, 2.0]:
                    for beta in [16, 32]:
                        # NTK performs better at longer contexts
                        base_perplexity = 20.0 + alpha * 3.0 + max(0, (16384 - context_length) / 4096) * 2.0
                        base_accuracy = min(0.98, 0.7 + (context_length / 32768) * 0.2 + alpha * 0.05)
                        
                        result = {
                            "experiment_config": {
                                "rope_method": method,
                                "context_length": context_length,
                                "parameters": {"alpha": alpha, "beta": beta},
                                "model_name": "demo-model",
                                "model_type": "hf_local"
                            },
                            "metrics": {
                                "perplexity": {
                                    "perplexity": base_perplexity + np.random.normal(0, 1.5),
                                    "loss": np.log(base_perplexity + np.random.normal(0, 1.5)),
                                    "total_tokens": 10000
                                },
                                "passkey_retrieval": {
                                    "passkey_accuracy": max(0.0, min(1.0, base_accuracy + np.random.normal(0, 0.03))),
                                    "correct_retrievals": int(10 * base_accuracy),
                                    "total_samples": 10
                                }
                            },
                            "execution_time": 140.0 + np.random.normal(0, 25.0),
                            "status": "completed"
                        }
                        results.append(result)
                        
            elif method == "yarn":
                for s in [8, 16]:
                    for alpha in [1.0, 2.0]:
                        # YaRN performs best overall
                        base_perplexity = 18.0 + (s / 16) * 2.0 + (context_length / 32768) * 3.0
                        base_accuracy = min(0.99, 0.85 + (s / 16) * 0.05 + alpha * 0.03)
                        
                        result = {
                            "experiment_config": {
                                "rope_method": method,
                                "context_length": context_length,
                                "parameters": {
                                    "s": s, 
                                    "alpha": alpha, 
                                    "beta": 32,
                                    "attention_factor": 0.1,
                                    "beta_fast": 32,
                                    "beta_slow": 1.0
                                },
                                "model_name": "demo-model",
                                "model_type": "hf_local"
                            },
                            "metrics": {
                                "perplexity": {
                                    "perplexity": base_perplexity + np.random.normal(0, 1.0),
                                    "loss": np.log(base_perplexity + np.random.normal(0, 1.0)),
                                    "total_tokens": 10000
                                },
                                "passkey_retrieval": {
                                    "passkey_accuracy": max(0.0, min(1.0, base_accuracy + np.random.normal(0, 0.02))),
                                    "correct_retrievals": int(10 * base_accuracy),
                                    "total_samples": 10
                                }
                            },
                            "execution_time": 150.0 + np.random.normal(0, 30.0),
                            "status": "completed"
                        }
                        results.append(result)
    
    # Add a few failed experiments for realism
    for i in range(3):
        failed_result = {
            "experiment_config": {
                "rope_method": "linear_interpolation",
                "context_length": 32768,
                "parameters": {"scaling_factor": 16},
                "model_name": "demo-model",
                "model_type": "hf_local"
            },
            "metrics": {},
            "execution_time": 0,
            "status": "failed",
            "error": "CUDA out of memory"
        }
        results.append(failed_result)
    
    return results

def demonstrate_analysis(results):
    """Demonstrate analysis capabilities."""
    print("🔬 ANALYSIS RESULTS")
    print("=" * 50)
    
    analyzer = SweepAnalyzer(results)
    
    # Summary statistics
    stats = analyzer.compute_summary_statistics()
    print(f"📊 Total experiments: {stats['total_experiments']}")
    print(f"📊 Methods tested: {stats['methods_tested']}")
    print(f"📊 Context lengths: {stats['context_lengths_tested']}")
    
    if 'perplexity_stats' in stats:
        ppl_stats = stats['perplexity_stats']
        print(f"📊 Perplexity range: {ppl_stats['min']:.2f} - {ppl_stats['max']:.2f}")
        print(f"📊 Mean perplexity: {ppl_stats['mean']:.2f} ± {ppl_stats['std']:.2f}")
    
    # Best configurations
    print("\n🏆 TOP 5 CONFIGURATIONS BY PERPLEXITY:")
    best_configs = analyzer.find_best_configurations("perplexity", top_k=5)
    for i, config in enumerate(best_configs, 1):
        print(f"{i:2d}. {config['rope_method']:18s} @ {config['context_length']:5d} "
              f"= {config['metric_value']:6.2f} PPL")
        
    print("\n🏆 TOP 5 CONFIGURATIONS BY PASSKEY ACCURACY:")
    best_configs = analyzer.find_best_configurations("passkey_retrieval", top_k=5)
    for i, config in enumerate(best_configs, 1):
        print(f"{i:2d}. {config['rope_method']:18s} @ {config['context_length']:5d} "
              f"= {config['metric_value']:6.3f} ACC")
    
    # Parameter sensitivity
    print("\n🔍 PARAMETER SENSITIVITY (Perplexity):")
    sensitivity = analyzer.analyze_parameter_sensitivity("perplexity")
    for param, sens in sensitivity.items():
        correlation = sens['pearson_correlation']
        significance = "***" if sens['pearson_p_value'] < 0.001 else "**" if sens['pearson_p_value'] < 0.01 else "*" if sens['pearson_p_value'] < 0.05 else ""
        print(f"   {param:15s}: r = {correlation:6.3f} {significance:3s} "
              f"(explains {sens['variance_explained']*100:4.1f}% variance)")
    
    # Method comparison
    print("\n⚔️  METHOD COMPARISON:")
    comparison = analyzer.compare_methods("perplexity")
    for comparison_key, comp in comparison.get('pairwise_comparisons', {}).items():
        method1, method2 = comparison_key.split('_vs_')
        significance = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
        print(f"   {method1:12s} vs {method2:12s}: "
              f"p = {comp['p_value']:6.4f} {significance:3s} "
              f"(effect size: {comp['effect_size']:.3f})")
    
    return analyzer

def demonstrate_visualization(results):
    """Demonstrate visualization capabilities."""
    print("\n🎨 VISUALIZATION CAPABILITIES")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        
        visualizer = SweepVisualizer(results)
        
        # Performance vs context length
        print("📈 Creating performance vs context length plot...")
        fig = visualizer.plot_performance_vs_context_length("perplexity")
        plt.close(fig)
        print("   ✓ Performance vs context length plot created")
        
        # Try to create a contour plot
        print("📈 Creating contour plot...")
        try:
            fig = visualizer.plot_contour("alpha", "beta", "perplexity", rope_method="ntk_aware")
            plt.close(fig)
            print("   ✓ Contour plot created (alpha vs beta for NTK-aware)")
        except Exception as e:
            print(f"   ⚠️  Contour plot failed: {e}")
        
        # Heatmap grid
        print("📈 Creating parameter analysis grid...")
        try:
            fig = visualizer.plot_heatmap_grid("perplexity")
            plt.close(fig)
            print("   ✓ Parameter analysis grid created")
        except Exception as e:
            print(f"   ⚠️  Parameter grid failed: {e}")
        
        print("   📊 All plots would be saved to files in a real sweep")
        
    except ImportError:
        print("   ⚠️  Matplotlib not available for visualization demo")

def main():
    """Run the demonstration."""
    print("🚀 RoPE HYPERPARAMETER SWEEP DEMONSTRATION")
    print("=" * 60)
    print("This demo shows what results look like from a hyperparameter sweep.")
    print("Using mock data to demonstrate analysis without requiring large models.\n")
    
    # Generate mock results
    print("📋 Generating mock sweep results...")
    results = generate_mock_results()
    
    successful = sum(1 for r in results if r['status'] == 'completed')
    failed = len(results) - successful
    print(f"   Generated {len(results)} total experiments")
    print(f"   Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    
    # Save results to show JSON structure
    output_file = "demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to {output_file}")
    
    # Demonstrate analysis
    analyzer = demonstrate_analysis(results)
    
    # Demonstrate visualization
    demonstrate_visualization(results)
    
    print(f"\n📁 RESULT FILES STRUCTURE:")
    print("In a real sweep, you would see:")
    print("   sweep_results/")
    print("   ├── sweep_results.json         # Detailed results")
    print("   ├── sweep_summary.json         # Summary statistics")
    print("   ├── analysis_report.json       # Statistical analysis")
    print("   └── visualizations/")
    print("       ├── performance_vs_context_length.png")
    print("       ├── contour_yarn_perplexity.png")
    print("       ├── parameter_analysis_grid.png")
    print("       └── interactive_dashboard.html")
    
    print(f"\n✨ NEXT STEPS:")
    print("1. Install the package: pip install -e .")
    print("2. Run a real sweep: python examples/run_sweep_example.py --config examples/sweep_configs/quick_comparison_sweep.yaml")
    print("3. Analyze results with the CLI tools")
    print("4. Create custom sweep configurations for your models")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())