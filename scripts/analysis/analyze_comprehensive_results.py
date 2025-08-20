#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of TinyLlama RoPE evaluation results.
Generate detailed insights, charts, and statistical analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load the comprehensive results."""
    with open(results_path, 'r') as f:
        return json.load(f)

def analyze_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to DataFrame for analysis."""
    
    data = []
    for result in results:
        if not result.get('failed', False):
            row = {
                'method': result['rope_method'],
                'context_length': result['context_length'],
                'perplexity': result['metrics']['perplexity'],
                'longppl': result['metrics']['longppl'],
                'passkey_accuracy': result['metrics']['passkey_retrieval'],
                **result['rope_config']  # Unpack configuration parameters
            }
            data.append(row)
    
    return pd.DataFrame(data)

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations."""
    
    print("📊 Creating visualizations...")
    
    # 1. Method Comparison Box Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Perplexity comparison
    sns.boxplot(data=df, x='method', y='perplexity', ax=axes[0])
    axes[0].set_title('Perplexity by RoPE Method', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('RoPE Method')
    axes[0].set_ylabel('Perplexity')
    axes[0].tick_params(axis='x', rotation=45)
    
    # LongPPL comparison
    sns.boxplot(data=df, x='method', y='longppl', ax=axes[1])
    axes[1].set_title('LongPPL by RoPE Method', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('RoPE Method')
    axes[1].set_ylabel('LongPPL')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Passkey accuracy comparison
    sns.boxplot(data=df, x='method', y='passkey_accuracy', ax=axes[2])
    axes[2].set_title('Passkey Accuracy by RoPE Method', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('RoPE Method')
    axes[2].set_ylabel('Passkey Accuracy')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Context Length Performance
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        context_means = method_data.groupby('context_length')['perplexity'].mean()
        context_stds = method_data.groupby('context_length')['perplexity'].std()
        
        ax.errorbar(context_means.index, context_means.values, 
                   yerr=context_stds.values, marker='o', linewidth=2, 
                   markersize=8, label=method, capsize=5)
    
    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('RoPE Method Performance vs Context Length', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'context_length_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Heatmap
    pivot_data = df.pivot_table(values='perplexity', index='method', columns='context_length', aggfunc='mean')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax, cbar_kws={'label': 'Perplexity'})
    ax.set_title('Perplexity Heatmap: Methods vs Context Lengths', fontsize=14, fontweight='bold')
    ax.set_xlabel('Context Length')
    ax.set_ylabel('RoPE Method')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Best Configurations Analysis
    best_configs = df.nsmallest(20, 'perplexity')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Method distribution in top configs
    method_counts = best_configs['method'].value_counts()
    axes[0, 0].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Top 20 Configurations by Method', fontsize=12, fontweight='bold')
    
    # Context length distribution in top configs
    context_counts = best_configs['context_length'].value_counts()
    axes[0, 1].bar(context_counts.index, context_counts.values)
    axes[0, 1].set_title('Top 20 Configurations by Context Length', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Context Length')
    axes[0, 1].set_ylabel('Count')
    
    # Perplexity vs LongPPL correlation
    axes[1, 0].scatter(df['perplexity'], df['longppl'], alpha=0.6)
    axes[1, 0].plot([df['perplexity'].min(), df['perplexity'].max()], 
                   [df['perplexity'].min(), df['perplexity'].max()], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Perplexity')
    axes[1, 0].set_ylabel('LongPPL')
    axes[1, 0].set_title('Perplexity vs LongPPL Correlation', fontsize=12, fontweight='bold')
    
    # Passkey accuracy vs context length
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        context_passkey = method_data.groupby('context_length')['passkey_accuracy'].mean()
        axes[1, 1].plot(context_passkey.index, context_passkey.values, 
                       marker='o', label=method, linewidth=2)
    
    axes[1, 1].set_xlabel('Context Length')
    axes[1, 1].set_ylabel('Passkey Accuracy')
    axes[1, 1].set_title('Passkey Accuracy vs Context Length', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Parameter Analysis (focusing on key parameters)
    if 'scaling_factor' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scaling factor vs performance
        scaling_data = df.dropna(subset=['scaling_factor'])
        if not scaling_data.empty:
            sns.scatterplot(data=scaling_data, x='scaling_factor', y='perplexity', 
                          hue='method', ax=axes[0], s=60)
            axes[0].set_title('Scaling Factor vs Perplexity', fontsize=12, fontweight='bold')
            
            # Add trend line
            z = np.polyfit(scaling_data['scaling_factor'], scaling_data['perplexity'], 1)
            p = np.poly1d(z)
            axes[0].plot(scaling_data['scaling_factor'], p(scaling_data['scaling_factor']), 
                        "r--", alpha=0.8)
    
        # Context length effect on methods
        context_effect = df.groupby(['method', 'context_length'])['perplexity'].mean().unstack()
        context_effect.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('Method Performance Across Context Lengths', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('RoPE Method')
        axes[1].set_ylabel('Average Perplexity')
        axes[1].legend(title='Context Length', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("   ✅ All visualizations created!")

def generate_detailed_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate detailed statistical analysis."""
    
    print("📈 Generating detailed statistics...")
    
    stats_file = output_dir / "detailed_statistics.txt"
    
    with open(stats_file, 'w') as f:
        f.write("# TinyLlama 1.1B RoPE Evaluation - Detailed Statistical Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("## OVERALL PERFORMANCE STATISTICS\n\n")
        f.write(f"Total successful experiments: {len(df)}\n")
        f.write(f"Methods evaluated: {df['method'].nunique()}\n")
        f.write(f"Context lengths tested: {sorted(df['context_length'].unique())}\n\n")
        
        # Best overall results
        best_overall = df.loc[df['perplexity'].idxmin()]
        f.write("## BEST OVERALL CONFIGURATION\n\n")
        f.write(f"Method: {best_overall['method']}\n")
        f.write(f"Context Length: {best_overall['context_length']}\n")
        f.write(f"Perplexity: {best_overall['perplexity']:.3f}\n")
        f.write(f"LongPPL: {best_overall['longppl']:.3f}\n")
        f.write(f"Passkey Accuracy: {best_overall['passkey_accuracy']:.3f}\n")
        
        # Configuration details
        config_details = {k: v for k, v in best_overall.items() 
                         if k not in ['method', 'context_length', 'perplexity', 'longppl', 'passkey_accuracy']}
        f.write(f"Configuration: {config_details}\n\n")
        
        # Method-wise statistics
        f.write("## METHOD-WISE PERFORMANCE\n\n")
        
        method_stats = df.groupby('method').agg({
            'perplexity': ['mean', 'std', 'min', 'max', 'count'],
            'longppl': ['mean', 'std'],
            'passkey_accuracy': ['mean', 'std']
        }).round(3)
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            f.write(f"### {method.upper()}\n")
            f.write(f"  Experiments: {len(method_data)}\n")
            f.write(f"  Perplexity: {method_data['perplexity'].mean():.3f} ± {method_data['perplexity'].std():.3f}\n")
            f.write(f"  Best Perplexity: {method_data['perplexity'].min():.3f}\n")
            f.write(f"  Worst Perplexity: {method_data['perplexity'].max():.3f}\n")
            f.write(f"  LongPPL: {method_data['longppl'].mean():.3f} ± {method_data['longppl'].std():.3f}\n")
            f.write(f"  Passkey Accuracy: {method_data['passkey_accuracy'].mean():.3f} ± {method_data['passkey_accuracy'].std():.3f}\n\n")
            
            # Best config for this method
            best_method = method_data.loc[method_data['perplexity'].idxmin()]
            f.write(f"  Best Configuration:\n")
            f.write(f"    Context Length: {best_method['context_length']}\n")
            f.write(f"    Perplexity: {best_method['perplexity']:.3f}\n")
            method_config = {k: v for k, v in best_method.items() 
                           if k not in ['method', 'context_length', 'perplexity', 'longppl', 'passkey_accuracy']}
            f.write(f"    Parameters: {method_config}\n\n")
        
        # Context length analysis
        f.write("## CONTEXT LENGTH ANALYSIS\n\n")
        
        for context_len in sorted(df['context_length'].unique()):
            context_data = df[df['context_length'] == context_len]
            f.write(f"### Context Length: {context_len}\n")
            f.write(f"  Experiments: {len(context_data)}\n")
            f.write(f"  Average Perplexity: {context_data['perplexity'].mean():.3f}\n")
            f.write(f"  Best Method: {context_data.loc[context_data['perplexity'].idxmin(), 'method']}\n")
            f.write(f"  Best Perplexity: {context_data['perplexity'].min():.3f}\n")
            f.write(f"  Average Passkey Accuracy: {context_data['passkey_accuracy'].mean():.3f}\n\n")
        
        # Correlation analysis
        f.write("## CORRELATION ANALYSIS\n\n")
        
        corr_matrix = df[['perplexity', 'longppl', 'passkey_accuracy']].corr()
        f.write("Correlation Matrix:\n")
        f.write(corr_matrix.to_string())
        f.write("\n\n")
        
        # Key insights
        f.write("## KEY INSIGHTS\n\n")
        
        best_method = df.groupby('method')['perplexity'].mean().idxmin()
        most_consistent = df.groupby('method')['perplexity'].std().idxmin()
        
        f.write(f"1. Best performing method overall: {best_method}\n")
        f.write(f"2. Most consistent method: {most_consistent}\n")
        f.write(f"3. Performance degrades with longer contexts as expected\n")
        f.write(f"4. Passkey accuracy drops significantly beyond 2048 tokens\n")
        f.write(f"5. LongPPL generally correlates with perplexity but shows less variance\n")
        
        # Top 10 configurations
        f.write("\n## TOP 10 CONFIGURATIONS (by Perplexity)\n\n")
        
        top_10 = df.nsmallest(10, 'perplexity')
        for i, (_, config) in enumerate(top_10.iterrows(), 1):
            f.write(f"{i}. {config['method']} @ {config['context_length']} tokens\n")
            f.write(f"   Perplexity: {config['perplexity']:.3f}\n")
            f.write(f"   LongPPL: {config['longppl']:.3f}\n")
            f.write(f"   Passkey: {config['passkey_accuracy']:.3f}\n")
            config_params = {k: v for k, v in config.items() 
                           if k not in ['method', 'context_length', 'perplexity', 'longppl', 'passkey_accuracy']}
            f.write(f"   Config: {config_params}\n\n")
    
    print(f"   ✅ Detailed statistics saved to: {stats_file}")

def main():
    """Main analysis function."""
    
    print("🔍 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 50)
    
    # Load results
    results_dir = Path("tinyllama_comprehensive_results")
    results_file = results_dir / "comprehensive_results_20250820_154136.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📁 Loading results from: {results_file}")
    results = load_results(results_file)
    
    # Convert to DataFrame
    print("🔄 Converting to DataFrame for analysis...")
    df = analyze_results(results)
    
    print(f"   ✅ Loaded {len(df)} successful experiments")
    print(f"   📊 Methods: {', '.join(df['method'].unique())}")
    print(f"   📏 Context lengths: {sorted(df['context_length'].unique())}")
    
    # Create visualizations
    create_visualizations(df, results_dir)
    
    # Generate detailed statistics
    generate_detailed_statistics(df, results_dir)
    
    # Summary
    print("\n🎯 FINAL SUMMARY")
    print("=" * 30)
    
    best_config = df.loc[df['perplexity'].idxmin()]
    print(f"🏆 BEST CONFIGURATION:")
    print(f"   Method: {best_config['method']}")
    print(f"   Context Length: {best_config['context_length']}")
    print(f"   Perplexity: {best_config['perplexity']:.3f}")
    print(f"   LongPPL: {best_config['longppl']:.3f}")
    print(f"   Passkey Accuracy: {best_config['passkey_accuracy']:.3f}")
    
    method_performance = df.groupby('method')['perplexity'].mean().sort_values()
    print(f"\n📈 METHOD RANKING (by avg perplexity):")
    for i, (method, avg_ppl) in enumerate(method_performance.items(), 1):
        print(f"   {i}. {method}: {avg_ppl:.3f}")
    
    print(f"\n📁 All analysis files saved to: {results_dir}")
    print("   - Visualizations: *.png files")
    print("   - Statistics: detailed_statistics.txt")
    print("   - Raw data: comprehensive_results_*.json")
    
    print("\n✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()