#!/usr/bin/env python3
"""Comprehensive visualization and analysis for parameter sweep results."""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class ComprehensiveSweepVisualizer:
    """Advanced visualizer for comprehensive parameter sweep results."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        """Initialize visualizer with sweep results.
        
        Args:
            results: List of sweep results with metadata
        """
        self.results = results
        self.df = self._create_analysis_dataframe()
        
    def _create_analysis_dataframe(self) -> pd.DataFrame:
        """Convert sweep results to analysis-ready DataFrame."""
        rows = []
        
        for result in self.results:
            if result.get('failed', False):
                continue
                
            metadata = result.get('sweep_metadata', {})
            if not metadata:
                continue
                
            base_row = {
                'rope_method': metadata.get('rope_method', 'unknown'),
                'context_length': metadata.get('context_length', 0),
                'run_name': metadata.get('run_name', ''),
                'timestamp': metadata.get('timestamp', ''),
            }
            
            # Add RoPE configuration parameters
            rope_config = metadata.get('rope_config', {})
            for param, value in rope_config.items():
                if isinstance(value, (int, float)):
                    base_row[f'rope_{param}'] = value
                elif isinstance(value, list) and len(value) <= 2:
                    # Handle short/long factors
                    for i, v in enumerate(value):
                        base_row[f'rope_{param}_{i}'] = v
            
            # Extract benchmark scores
            benchmarks = result.get('benchmarks', {})
            for bench_name, bench_result in benchmarks.items():
                if isinstance(bench_result, dict):
                    if 'average_score' in bench_result:
                        base_row[f'{bench_name}_score'] = bench_result['average_score']
                    
                    # Extract additional metrics
                    for metric_name, metric_value in bench_result.items():
                        if isinstance(metric_value, (int, float)) and metric_name != 'average_score':
                            base_row[f'{bench_name}_{metric_name}'] = metric_value
            
            # Overall performance
            if 'summary' in result and 'overall_average' in result['summary']:
                base_row['overall_score'] = result['summary']['overall_average']
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Created analysis DataFrame with {len(df)} valid results")
        return df
    
    def create_contour_plots(self, output_dir: Path) -> Dict[str, str]:
        """Create contour plots for parameter pairs vs benchmark scores.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of generated plot files
        """
        contour_dir = output_dir / "contour_plots"
        contour_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Get parameter columns (numerical only)
        param_cols = [col for col in self.df.columns 
                     if col.startswith('rope_') and self.df[col].dtype in ['float64', 'int64']]
        
        # Get benchmark score columns  
        score_cols = [col for col in self.df.columns if col.endswith('_score')]
        
        logger.info(f"Found {len(param_cols)} parameters and {len(score_cols)} benchmark scores")
        
        # Create contour plots for all parameter pairs vs all benchmarks
        for i, param1 in enumerate(param_cols):
            for j, param2 in enumerate(param_cols[i+1:], i+1):  # Avoid duplicates
                for score_col in score_cols:
                    
                    # Filter valid data
                    valid_mask = (~self.df[param1].isna() & 
                                 ~self.df[param2].isna() & 
                                 ~self.df[score_col].isna())
                    
                    if valid_mask.sum() < 4:  # Need at least 4 points
                        continue
                    
                    valid_df = self.df[valid_mask]
                    
                    try:
                        fig = self._create_contour_plot(
                            valid_df, param1, param2, score_col,
                            title_suffix=f"{param1.replace('rope_', '')} vs {param2.replace('rope_', '')}"
                        )
                        
                        filename = f"contour_{param1}_{param2}_{score_col}.png"
                        filepath = contour_dir / filename
                        fig.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        generated_files[f'contour_{param1}_{param2}_{score_col}'] = str(filepath)
                        
                    except Exception as e:
                        logger.warning(f"Failed to create contour plot {param1} vs {param2} for {score_col}: {e}")
        
        return generated_files
    
    def _create_contour_plot(self, df: pd.DataFrame, x_param: str, y_param: str, 
                           score_col: str, title_suffix: str = "") -> plt.Figure:
        """Create a single contour plot."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = df[x_param].values
        y = df[y_param].values
        z = df[score_col].values
        
        # Create interpolation grid
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate
        zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        
        # Create filled contour
        levels = np.linspace(z.min(), z.max(), 20)
        contour_filled = ax.contourf(xi_grid, yi_grid, zi, levels=levels, cmap='viridis', alpha=0.8)
        
        # Add contour lines
        contour_lines = ax.contour(xi_grid, yi_grid, zi, levels=levels, colors='black', alpha=0.4, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
        
        # Scatter plot of actual data points
        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=60, edgecolors='black', linewidth=0.7, alpha=0.9)
        
        # Add colorbar
        cbar = fig.colorbar(contour_filled, ax=ax)
        cbar.set_label(score_col.replace('_', ' ').title(), rotation=270, labelpad=15)
        
        # Labels and title
        ax.set_xlabel(x_param.replace('rope_', '').replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('rope_', '').replace('_', ' ').title())
        
        benchmark_name = score_col.split('_score')[0].upper()
        title = f'{benchmark_name} Performance Contour'
        if title_suffix:
            title += f' - {title_suffix}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_analysis_plots(self, output_dir: Path) -> Dict[str, str]:
        """Create comprehensive analysis plots."""
        generated_files = {}
        
        # 1. Overall performance comparison
        fig = self._plot_method_performance_comparison()
        filepath = output_dir / "method_performance_comparison.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        generated_files['method_comparison'] = str(filepath)
        
        # 2. Context length scaling analysis
        fig = self._plot_context_length_scaling()
        filepath = output_dir / "context_length_scaling.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        generated_files['context_scaling'] = str(filepath)
        
        # 3. Parameter correlation heatmap
        fig = self._plot_parameter_correlation_heatmap()
        filepath = output_dir / "parameter_correlation_heatmap.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        generated_files['correlation_heatmap'] = str(filepath)
        
        # 4. Benchmark-specific analysis
        fig = self._plot_benchmark_specific_analysis()
        filepath = output_dir / "benchmark_specific_analysis.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        generated_files['benchmark_analysis'] = str(filepath)
        
        # 5. Best configuration heatmap
        fig = self._plot_best_configuration_heatmap()
        filepath = output_dir / "best_configuration_heatmap.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        generated_files['best_config_heatmap'] = str(filepath)
        
        return generated_files
    
    def _plot_method_performance_comparison(self) -> plt.Figure:
        """Plot RoPE method performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall performance boxplot
        if 'overall_score' in self.df.columns:
            ax = axes[0, 0]
            self.df.boxplot(column='overall_score', by='rope_method', ax=ax)
            ax.set_title('Overall Performance by RoPE Method')
            ax.set_xlabel('RoPE Method')
            ax.set_ylabel('Overall Score')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Performance vs context length
        ax = axes[0, 1]
        for method in self.df['rope_method'].unique():
            method_data = self.df[self.df['rope_method'] == method]
            if 'overall_score' in method_data.columns:
                ax.plot(method_data['context_length'], method_data['overall_score'], 
                       'o-', label=method.upper(), alpha=0.7)
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Overall Score')
        ax.set_title('Performance vs Context Length')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Method ranking by average score
        ax = axes[1, 0]
        if 'overall_score' in self.df.columns:
            method_avg = self.df.groupby('rope_method')['overall_score'].mean().sort_values(ascending=True)
            method_avg.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title('Average Performance Ranking')
            ax.set_xlabel('Average Overall Score')
        
        # Parameter distribution
        ax = axes[1, 1]
        param_cols = [col for col in self.df.columns if col.startswith('rope_') and self.df[col].dtype in ['float64', 'int64']]
        if param_cols:
            # Plot distribution of first parameter
            param = param_cols[0]
            for method in self.df['rope_method'].unique():
                method_data = self.df[self.df['rope_method'] == method]
                if not method_data[param].isna().all():
                    ax.hist(method_data[param].dropna(), alpha=0.6, label=method.upper(), bins=10)
            ax.set_title(f'Distribution of {param.replace("rope_", "").title()}')
            ax.set_xlabel(param.replace('rope_', '').title())
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.suptitle('RoPE Method Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_context_length_scaling(self) -> plt.Figure:
        """Plot context length scaling analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        score_cols = [col for col in self.df.columns if col.endswith('_score')]
        
        for idx, score_col in enumerate(score_cols[:4]):  # Limit to 4 plots
            ax = axes[idx // 2, idx % 2]
            
            for method in self.df['rope_method'].unique():
                method_data = self.df[self.df['rope_method'] == method]
                if not method_data[score_col].isna().all():
                    # Group by context length and compute mean
                    grouped = method_data.groupby('context_length')[score_col].mean()
                    ax.plot(grouped.index, grouped.values, 'o-', label=method.upper(), alpha=0.8)
            
            benchmark_name = score_col.split('_score')[0].upper()
            ax.set_title(f'{benchmark_name} Scaling with Context Length')
            ax.set_xlabel('Context Length')
            ax.set_ylabel(f'{benchmark_name} Score')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Context Length Scaling Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_parameter_correlation_heatmap(self) -> plt.Figure:
        """Plot parameter correlation heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get numerical columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove non-parameter columns
        exclude_cols = ['context_length', 'overall_score'] + [col for col in numeric_cols if col.endswith('_score')]
        param_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(param_cols) > 1:
            corr_matrix = self.df[param_cols].corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
            
            ax.set_title('Parameter Correlation Heatmap', fontsize=14, fontweight='bold')
            
            # Clean up labels
            labels = [label.get_text().replace('rope_', '').replace('_', ' ').title() 
                     for label in ax.get_xticklabels()]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)
        else:
            ax.text(0.5, 0.5, 'Insufficient parameter data for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Parameter Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_benchmark_specific_analysis(self) -> plt.Figure:
        """Plot benchmark-specific analysis."""
        score_cols = [col for col in self.df.columns if col.endswith('_score')]
        
        if not score_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No benchmark scores available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Benchmark-Specific Analysis')
            return fig
        
        n_benchmarks = len(score_cols)
        n_cols = min(3, n_benchmarks)
        n_rows = (n_benchmarks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, score_col in enumerate(score_cols):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                continue
                
            benchmark_name = score_col.split('_score')[0].upper()
            
            # Create violin plot
            method_scores = []
            method_names = []
            
            for method in self.df['rope_method'].unique():
                method_data = self.df[self.df['rope_method'] == method]
                if not method_data[score_col].isna().all():
                    scores = method_data[score_col].dropna()
                    if len(scores) > 0:
                        method_scores.append(scores)
                        method_names.append(method.upper())
            
            if method_scores:
                parts = ax.violinplot(method_scores, positions=range(len(method_names)), 
                                    showmeans=True, showmedians=True)
                ax.set_xticks(range(len(method_names)))
                ax.set_xticklabels(method_names, rotation=45, ha='right')
                ax.set_ylabel(f'{benchmark_name} Score')
                ax.set_title(f'{benchmark_name} Score Distribution')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(score_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Benchmark-Specific Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_best_configuration_heatmap(self) -> plt.Figure:
        """Plot best configuration heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if 'overall_score' not in self.df.columns or self.df['overall_score'].isna().all():
            ax.text(0.5, 0.5, 'No overall score data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Best Configuration Analysis')
            return fig
        
        # Create pivot table for heatmap
        pivot_data = self.df.pivot_table(
            values='overall_score', 
            index='rope_method', 
            columns='context_length', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f', 
                   cbar_kws={'label': 'Overall Score'}, ax=ax)
        
        ax.set_title('Overall Performance Heatmap: RoPE Method vs Context Length', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Context Length')
        ax.set_ylabel('RoPE Method')
        
        # Improve labels
        method_labels = [label.get_text().upper() for label in ax.get_yticklabels()]
        ax.set_yticklabels(method_labels, rotation=0)
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive visualization report.
        
        Args:
            output_dir: Directory to save all outputs
            
        Returns:
            Report summary with file paths and statistics
        """
        logger.info(f"Generating comprehensive visualization report in {output_dir}")
        
        # Create subdirectories
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'output_directory': str(output_dir),
            'generated_files': {},
            'statistics': {},
            'summary': {}
        }
        
        # Generate main analysis plots
        logger.info("Creating comprehensive analysis plots...")
        analysis_files = self.create_comprehensive_analysis_plots(plots_dir)
        report['generated_files'].update(analysis_files)
        
        # Generate contour plots
        logger.info("Creating contour plots...")
        contour_files = self.create_contour_plots(plots_dir)
        report['generated_files'].update(contour_files)
        
        # Generate statistics
        report['statistics'] = self._generate_statistics()
        
        # Generate summary
        report['summary'] = self._generate_summary()
        
        # Save report
        report_file = output_dir / "visualization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated {len(report['generated_files'])} visualization files")
        logger.info(f"Report saved to {report_file}")
        
        return report
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        stats = {
            'total_experiments': len(self.df),
            'unique_methods': self.df['rope_method'].nunique(),
            'unique_context_lengths': self.df['context_length'].nunique(),
            'methods': self.df['rope_method'].unique().tolist(),
            'context_lengths': sorted(self.df['context_length'].unique().tolist()),
        }
        
        # Performance statistics
        if 'overall_score' in self.df.columns:
            overall_scores = self.df['overall_score'].dropna()
            if len(overall_scores) > 0:
                stats['performance'] = {
                    'mean_score': float(overall_scores.mean()),
                    'std_score': float(overall_scores.std()),
                    'min_score': float(overall_scores.min()),
                    'max_score': float(overall_scores.max()),
                    'best_config': self.df.loc[overall_scores.idxmax()].to_dict()
                }
        
        # Method-specific statistics
        stats['method_performance'] = {}
        for method in self.df['rope_method'].unique():
            method_data = self.df[self.df['rope_method'] == method]
            if 'overall_score' in method_data.columns:
                scores = method_data['overall_score'].dropna()
                if len(scores) > 0:
                    stats['method_performance'][method] = {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'count': len(scores)
                    }
        
        return stats
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary insights."""
        summary = {}
        
        if 'overall_score' in self.df.columns and not self.df['overall_score'].isna().all():
            # Best performing method
            method_performance = self.df.groupby('rope_method')['overall_score'].mean()
            best_method = method_performance.idxmax()
            best_score = method_performance.max()
            
            summary['best_method'] = {
                'method': best_method,
                'average_score': float(best_score)
            }
            
            # Best performing configuration
            best_idx = self.df['overall_score'].idxmax()
            best_config = self.df.loc[best_idx]
            
            summary['best_configuration'] = {
                'method': best_config['rope_method'],
                'context_length': int(best_config['context_length']),
                'score': float(best_config['overall_score']),
                'run_name': best_config.get('run_name', '')
            }
            
            # Context length insights
            context_performance = self.df.groupby('context_length')['overall_score'].mean()
            summary['context_length_insights'] = {
                'best_length': int(context_performance.idxmax()),
                'best_score': float(context_performance.max()),
                'performance_by_length': {
                    int(k): float(v) for k, v in context_performance.items()
                }
            }
        
        return summary


def enhance_sweep_results_with_visualization(sweep_results: List[Dict[str, Any]], 
                                           output_dir: Path) -> Dict[str, Any]:
    """Enhance sweep results with comprehensive visualization.
    
    Args:
        sweep_results: Results from comprehensive sweep
        output_dir: Directory to save visualization outputs
        
    Returns:
        Enhanced results with visualization report
    """
    logger.info("Creating comprehensive visualization analysis...")
    
    # Create visualizer
    visualizer = ComprehensiveSweepVisualizer(sweep_results)
    
    # Generate comprehensive report
    viz_report = visualizer.generate_comprehensive_report(output_dir)
    
    return viz_report