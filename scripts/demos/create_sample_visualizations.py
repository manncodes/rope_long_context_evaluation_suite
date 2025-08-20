#!/usr/bin/env python3
"""Create sample visualizations to demonstrate the system capabilities."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rope_long_context_evaluation_suite.sweep import SweepVisualizer

def create_sample_plots():
    """Create sample plots from demo results."""
    
    # Load demo results
    with open("demo_results.json", "r") as f:
        results = json.load(f)
    
    print("📊 Creating sample visualizations...")
    
    # Create visualizer
    visualizer = SweepVisualizer(results)
    
    # Create output directory
    output_dir = Path("sample_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Performance vs Context Length
    print("   Creating performance vs context length plot...")
    fig = visualizer.plot_performance_vs_context_length("perplexity", figsize=(12, 8))
    fig.suptitle("RoPE Performance vs Context Length", fontsize=16, fontweight='bold')
    
    # Add annotations
    ax = fig.gca()
    ax.annotate("YaRN performs best\nacross all context lengths", 
                xy=(16384, 20), xytext=(8192, 35),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_context_length.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Contour Plot for NTK-Aware method
    print("   Creating NTK-aware parameter contour plot...")
    try:
        fig = visualizer.plot_contour("alpha", "beta", "perplexity", 
                                     rope_method="ntk_aware", figsize=(10, 8))
        fig.suptitle("NTK-Aware: Parameter Optimization Landscape", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "ntk_aware_contour.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"     ⚠️ Contour plot failed: {e}")
    
    # 3. Method Comparison Box Plot
    print("   Creating method comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for each method
    methods = ["linear_interpolation", "ntk_aware", "yarn"]
    perplexities = {method: [] for method in methods}
    accuracies = {method: [] for method in methods}
    
    for result in results:
        if result['status'] == 'completed':
            method = result['experiment_config']['rope_method']
            if 'perplexity' in result['metrics']:
                perplexities[method].append(result['metrics']['perplexity']['perplexity'])
            if 'passkey_retrieval' in result['metrics']:
                accuracies[method].append(result['metrics']['passkey_retrieval']['passkey_accuracy'])
    
    # Box plot for perplexity
    ax1.boxplot([perplexities[method] for method in methods], labels=[m.replace('_', '\n') for m in methods])
    ax1.set_title("Perplexity Distribution", fontweight='bold')
    ax1.set_ylabel("Perplexity")
    ax1.grid(True, alpha=0.3)
    
    # Box plot for accuracy
    ax2.boxplot([accuracies[method] for method in methods], labels=[m.replace('_', '\n') for m in methods])
    ax2.set_title("Passkey Accuracy Distribution", fontweight='bold')
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle("RoPE Method Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Parameter Sensitivity Plot
    print("   Creating parameter sensitivity plot...")
    
    # Calculate correlations manually for visualization
    from scipy.stats import pearsonr
    
    # Prepare data
    df_data = []
    for result in results:
        if result['status'] == 'completed' and 'perplexity' in result['metrics']:
            row = {
                'perplexity': result['metrics']['perplexity']['perplexity'],
                'method': result['experiment_config']['rope_method']
            }
            for param, value in result['experiment_config']['parameters'].items():
                if isinstance(value, (int, float)):
                    row[param] = value
            df_data.append(row)
    
    # Create sensitivity plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = ['scaling_factor', 'alpha', 'beta', 's']
    correlations = []
    param_names = []
    
    for param in params:
        param_values = [row[param] for row in df_data if param in row]
        perplexity_values = [row['perplexity'] for row in df_data if param in row]
        
        if len(param_values) > 1 and len(set(param_values)) > 1:
            corr, p_value = pearsonr(param_values, perplexity_values)
            correlations.append(abs(corr))  # Use absolute correlation
            param_names.append(param.replace('_', '\n'))
    
    if correlations:
        bars = ax.bar(param_names, correlations, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(correlations)])
        ax.set_title("Parameter Sensitivity Analysis", fontweight='bold', fontsize=14)
        ax.set_ylabel("Absolute Correlation with Perplexity")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Context Length Scaling Heatmap
    print("   Creating context length scaling heatmap...")
    
    # Create heatmap data
    context_lengths = sorted(set(r['experiment_config']['context_length'] for r in results if r['status'] == 'completed'))
    methods = sorted(set(r['experiment_config']['rope_method'] for r in results if r['status'] == 'completed'))
    
    heatmap_data = np.full((len(methods), len(context_lengths)), np.nan)
    
    for i, method in enumerate(methods):
        for j, context_len in enumerate(context_lengths):
            method_results = [r for r in results 
                            if r['status'] == 'completed' 
                            and r['experiment_config']['rope_method'] == method
                            and r['experiment_config']['context_length'] == context_len
                            and 'perplexity' in r['metrics']]
            if method_results:
                perplexities = [r['metrics']['perplexity']['perplexity'] for r in method_results]
                heatmap_data[i, j] = np.mean(perplexities)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(context_lengths)))
    ax.set_xticklabels([f"{cl//1024}K" for cl in context_lengths])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace('_', ' ').title() for m in methods])
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Perplexity', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(context_lengths)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                             ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title("Perplexity Across Methods and Context Lengths", fontweight='bold', fontsize=14)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("RoPE Method")
    
    plt.tight_layout()
    plt.savefig(output_dir / "context_length_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Sample visualizations created in {output_dir}/")
    print("   Generated files:")
    for file_path in output_dir.glob("*.png"):
        print(f"     📊 {file_path.name}")
    
    return output_dir

def main():
    """Create sample visualizations."""
    print("🎨 CREATING SAMPLE VISUALIZATIONS")
    print("=" * 50)
    
    # First run demo to generate results if needed
    if not Path("demo_results.json").exists():
        print("Running demo first to generate results...")
        import subprocess
        subprocess.run([sys.executable, "demo_sweep.py"], check=True)
    
    # Create visualizations
    output_dir = create_sample_plots()
    
    print(f"\n📁 VISUALIZATION OUTPUTS:")
    print(f"   All plots saved to: {output_dir.absolute()}")
    print(f"   View them to see the analysis capabilities!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())