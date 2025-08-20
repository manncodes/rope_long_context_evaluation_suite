"""Visualization module for hyperparameter sweep results."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# Try importing plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive plots will be disabled.")

logger = logging.getLogger(__name__)


class SweepVisualizer:
    """Comprehensive visualization for hyperparameter sweep results."""
    
    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        """Initialize visualizer.
        
        Args:
            results: Sweep results to visualize
        """
        self.results = results or []
        self.df = None
        if self.results:
            self.df = self._create_dataframe(results)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self, results_path: Union[str, Path]):
        """Load results from JSON file.
        
        Args:
            results_path: Path to results JSON file
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        self.df = self._create_dataframe(self.results)
    
    def _create_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for easier manipulation.
        
        Args:
            results: Sweep results
            
        Returns:
            DataFrame with flattened results
        """
        rows = []
        
        for result in results:
            if result['status'] != 'completed':
                continue
                
            base_row = {
                'rope_method': result['experiment_config']['rope_method'],
                'context_length': result['experiment_config']['context_length'],
                'execution_time': result['execution_time']
            }
            
            # Add parameter values
            params = result['experiment_config']['parameters']
            for param_name, param_value in params.items():
                if isinstance(param_value, (list, tuple)):
                    # Convert lists to strings for now
                    base_row[f'param_{param_name}'] = str(param_value)
                else:
                    base_row[f'param_{param_name}'] = param_value
            
            # Add metric values
            for metric_name, metric_result in result['metrics'].items():
                if isinstance(metric_result, dict) and 'error' not in metric_result:
                    # Extract primary metric value
                    if metric_name == 'perplexity':
                        base_row[f'metric_{metric_name}'] = metric_result.get('perplexity', np.nan)
                    elif metric_name == 'passkey_retrieval':
                        base_row[f'metric_{metric_name}'] = metric_result.get('passkey_accuracy', np.nan)
                    elif metric_name == 'longppl':
                        base_row[f'metric_{metric_name}'] = metric_result.get('longppl', np.nan)
                    
                    # Add additional metric details
                    for key, value in metric_result.items():
                        if isinstance(value, (int, float)):
                            base_row[f'{metric_name}_{key}'] = value
            
            rows.append(base_row)
        
        return pd.DataFrame(rows)
    
    def plot_contour(self, x_param: str, y_param: str, metric: str,
                     rope_method: Optional[str] = None,
                     context_length: Optional[int] = None,
                     levels: int = 20,
                     figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Create contour plot for two parameters.
        
        Args:
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            metric: Metric to plot
            rope_method: Filter by specific RoPE method
            context_length: Filter by specific context length
            levels: Number of contour levels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.df is None or self.df.empty:
            raise ValueError("No results available for plotting")
        
        # Filter data
        df_filtered = self.df.copy()
        if rope_method:
            df_filtered = df_filtered[df_filtered['rope_method'] == rope_method]
        if context_length:
            df_filtered = df_filtered[df_filtered['context_length'] == context_length]
        
        if df_filtered.empty:
            raise ValueError("No data available after filtering")
        
        # Prepare data
        x_col = f'param_{x_param}' if f'param_{x_param}' in df_filtered.columns else x_param
        y_col = f'param_{y_param}' if f'param_{y_param}' in df_filtered.columns else y_param
        metric_col = f'metric_{metric}' if f'metric_{metric}' in df_filtered.columns else metric
        
        if not all(col in df_filtered.columns for col in [x_col, y_col, metric_col]):
            available_cols = df_filtered.columns.tolist()
            raise ValueError(f"Required columns not found. Available: {available_cols}")
        
        x = df_filtered[x_col].values
        y = df_filtered[y_col].values
        z = df_filtered[metric_col].values
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]
        
        if len(x) == 0:
            raise ValueError("No valid data points after removing NaN values")
        
        # Create grid for interpolation
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate
        zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filled contour plot
        contour_filled = ax.contourf(xi_grid, yi_grid, zi, levels=levels, cmap='viridis', alpha=0.8)
        
        # Contour lines
        contour_lines = ax.contour(xi_grid, yi_grid, zi, levels=levels, colors='black', alpha=0.4, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Scatter plot of actual data points
        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=50, edgecolors='black', linewidth=0.5, alpha=0.9)
        
        # Add colorbar
        cbar = fig.colorbar(contour_filled, ax=ax)
        cbar.set_label(metric, rotation=270, labelpad=15)
        
        # Labels and title
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        
        title = f'{metric.replace("_", " ").title()} Contour Plot'
        if rope_method:
            title += f' - {rope_method.upper()}'
        if context_length:
            title += f' - Context Length: {context_length}'
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_surface(self, x_param: str, y_param: str, metric: str,
                        rope_method: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 9)) -> plt.Figure:
        """Create 3D surface plot.
        
        Args:
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis  
            metric: Metric for z-axis
            rope_method: Filter by specific RoPE method
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.df is None or self.df.empty:
            raise ValueError("No results available for plotting")
        
        # Filter data
        df_filtered = self.df.copy()
        if rope_method:
            df_filtered = df_filtered[df_filtered['rope_method'] == rope_method]
        
        # Prepare data
        x_col = f'param_{x_param}' if f'param_{x_param}' in df_filtered.columns else x_param
        y_col = f'param_{y_param}' if f'param_{y_param}' in df_filtered.columns else y_param
        metric_col = f'metric_{metric}' if f'metric_{metric}' in df_filtered.columns else metric
        
        x = df_filtered[x_col].values
        y = df_filtered[y_col].values
        z = df_filtered[metric_col].values
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]
        
        # Create grid
        xi = np.linspace(x.min(), x.max(), 30)
        yi = np.linspace(y.min(), y.max(), 30)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surface = ax.plot_surface(xi_grid, yi_grid, zi, cmap='viridis', alpha=0.8)
        
        # Scatter plot of actual points
        ax.scatter(x, y, z, c=z, cmap='viridis', s=50)
        
        # Labels
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        ax.set_zlabel(metric.replace('_', ' ').title())
        
        title = f'{metric.replace("_", " ").title()} 3D Surface'
        if rope_method:
            title += f' - {rope_method.upper()}'
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surface)
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap_grid(self, metric: str = 'perplexity',
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create grid of heatmaps for all parameter combinations.
        
        Args:
            metric: Metric to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.df is None or self.df.empty:
            raise ValueError("No results available for plotting")
        
        # Get parameter columns
        param_cols = [col for col in self.df.columns if col.startswith('param_')]
        
        if len(param_cols) < 2:
            raise ValueError("Need at least 2 parameters for heatmap grid")
        
        # Create subplot grid
        n_params = len(param_cols)
        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
        
        for i, param1 in enumerate(param_cols):
            for j, param2 in enumerate(param_cols):
                ax = axes[i, j] if n_params > 1 else axes
                
                if i == j:
                    # Diagonal: histogram
                    self.df[param1].hist(ax=ax, bins=20, alpha=0.7)
                    ax.set_title(param1.replace('param_', '').replace('_', ' ').title())
                elif i > j:
                    # Lower triangle: contour plot
                    try:
                        self._plot_mini_contour(ax, param2, param1, metric)
                    except:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                else:
                    # Upper triangle: scatter plot with correlation
                    x = self.df[param2].values
                    y = self.df[param1].values
                    
                    # Remove NaN
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() > 1:
                        x_clean, y_clean = x[mask], y[mask]
                        ax.scatter(x_clean, y_clean, alpha=0.6, s=30)
                        
                        # Add correlation coefficient
                        corr, _ = pearsonr(x_clean, y_clean)
                        ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
                               verticalalignment='top', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                # Set labels only on edges
                if i == n_params - 1:
                    ax.set_xlabel(param2.replace('param_', '').replace('_', ' ').title())
                if j == 0:
                    ax.set_ylabel(param1.replace('param_', '').replace('_', ' ').title())
        
        plt.suptitle(f'Parameter Analysis - {metric.replace("_", " ").title()}', fontsize=16)
        plt.tight_layout()
        return fig
    
    def _plot_mini_contour(self, ax, x_param: str, y_param: str, metric: str):
        """Plot mini contour for heatmap grid."""
        metric_col = f'metric_{metric}' if f'metric_{metric}' in self.df.columns else metric
        
        x = self.df[x_param].values
        y = self.df[y_param].values
        z = self.df[metric_col].values
        
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]
        
        if len(x) > 3:  # Need at least 3 points for interpolation
            xi = np.linspace(x.min(), x.max(), 20)
            yi = np.linspace(y.min(), y.max(), 20)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            
            ax.contourf(xi_grid, yi_grid, zi, levels=10, cmap='viridis', alpha=0.8)
            ax.scatter(x, y, c=z, cmap='viridis', s=20, edgecolors='black', linewidth=0.3)
    
    def plot_performance_vs_context_length(self, metric: str = 'perplexity',
                                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot performance vs context length for different methods.
        
        Args:
            metric: Metric to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.df is None or self.df.empty:
            raise ValueError("No results available for plotting")
        
        metric_col = f'metric_{metric}' if f'metric_{metric}' in self.df.columns else metric
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each RoPE method
        for method in self.df['rope_method'].unique():
            method_data = self.df[self.df['rope_method'] == method]
            
            # Group by context length and compute statistics
            stats = method_data.groupby('context_length')[metric_col].agg(['mean', 'std']).reset_index()
            
            if not stats.empty:
                ax.errorbar(stats['context_length'], stats['mean'], yerr=stats['std'],
                           label=method.upper(), marker='o', capsize=5, capthick=2)
        
        ax.set_xlabel('Context Length')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Context Length')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, output_path: Optional[str] = None) -> Optional[str]:
        """Create interactive Plotly dashboard.
        
        Args:
            output_path: Path to save HTML file
            
        Returns:
            HTML string if output_path is None, otherwise None
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return None
        
        if self.df is None or self.df.empty:
            raise ValueError("No results available for plotting")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Performance vs Context Length', 'Parameter Correlation',
                           'Method Comparison', 'Execution Time Analysis'],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Plot 1: Performance vs Context Length
        for method in self.df['rope_method'].unique():
            method_data = self.df[self.df['rope_method'] == method]
            fig.add_trace(
                go.Scatter(
                    x=method_data['context_length'],
                    y=method_data['metric_perplexity'],
                    mode='lines+markers',
                    name=f'{method.upper()} - Perplexity',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Plot 2: Parameter correlation (example with alpha vs beta)
        if 'param_alpha' in self.df.columns and 'param_beta' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['param_alpha'],
                    y=self.df['param_beta'],
                    mode='markers',
                    marker=dict(
                        color=self.df['metric_perplexity'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Perplexity")
                    ),
                    name='Alpha vs Beta'
                ),
                row=1, col=2
            )
        
        # Plot 3: Method comparison boxplot
        methods = self.df['rope_method'].unique()
        for i, method in enumerate(methods):
            method_data = self.df[self.df['rope_method'] == method]
            fig.add_trace(
                go.Box(
                    y=method_data['metric_perplexity'],
                    name=method.upper(),
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=2, col=1
            )
        
        # Plot 4: Execution time
        fig.add_trace(
            go.Scatter(
                x=self.df['context_length'],
                y=self.df['execution_time'],
                mode='markers',
                marker=dict(
                    color=self.df['rope_method'].astype('category').cat.codes,
                    showscale=False
                ),
                name='Execution Time'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='RoPE Hyperparameter Sweep Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Context Length", row=1, col=1, type="log")
        fig.update_xaxes(title_text="Alpha", row=1, col=2)
        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_xaxes(title_text="Context Length", row=2, col=2, type="log")
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Perplexity", row=1, col=1)
        fig.update_yaxes(title_text="Beta", row=1, col=2)
        fig.update_yaxes(title_text="Perplexity", row=2, col=1)
        fig.update_yaxes(title_text="Execution Time (s)", row=2, col=2)
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Interactive dashboard saved to {output_path}")
            return None
        else:
            return fig.to_html()
    
    def generate_report(self, output_dir: Union[str, Path],
                       include_contours: bool = True,
                       include_3d: bool = True,
                       include_interactive: bool = True) -> Dict[str, str]:
        """Generate comprehensive visualization report.
        
        Args:
            output_dir: Directory to save plots
            include_contours: Whether to include contour plots
            include_3d: Whether to include 3D plots  
            include_interactive: Whether to include interactive plots
            
        Returns:
            Dictionary of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        if self.df is None or self.df.empty:
            logger.warning("No results available for report generation")
            return generated_files
        
        # Performance vs context length
        fig = self.plot_performance_vs_context_length()
        filepath = output_dir / "performance_vs_context_length.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        generated_files['performance_vs_context'] = str(filepath)
        
        # Parameter analysis grid
        try:
            fig = self.plot_heatmap_grid()
            filepath = output_dir / "parameter_analysis_grid.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            generated_files['parameter_grid'] = str(filepath)
        except Exception as e:
            logger.warning(f"Failed to generate parameter grid: {e}")
        
        # Contour plots for each method
        if include_contours:
            param_cols = [col.replace('param_', '') for col in self.df.columns if col.startswith('param_')]
            if len(param_cols) >= 2:
                for method in self.df['rope_method'].unique():
                    try:
                        fig = self.plot_contour(param_cols[0], param_cols[1], 'perplexity', rope_method=method)
                        filepath = output_dir / f"contour_{method}_perplexity.png"
                        fig.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        generated_files[f'contour_{method}'] = str(filepath)
                    except Exception as e:
                        logger.warning(f"Failed to generate contour for {method}: {e}")
        
        # 3D surface plots
        if include_3d:
            param_cols = [col.replace('param_', '') for col in self.df.columns if col.startswith('param_')]
            if len(param_cols) >= 2:
                try:
                    fig = self.plot_3d_surface(param_cols[0], param_cols[1], 'perplexity')
                    filepath = output_dir / "3d_surface_perplexity.png"
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    generated_files['3d_surface'] = str(filepath)
                except Exception as e:
                    logger.warning(f"Failed to generate 3D surface: {e}")
        
        # Interactive dashboard
        if include_interactive and PLOTLY_AVAILABLE:
            try:
                filepath = output_dir / "interactive_dashboard.html"
                self.create_interactive_dashboard(str(filepath))
                generated_files['interactive_dashboard'] = str(filepath)
            except Exception as e:
                logger.warning(f"Failed to generate interactive dashboard: {e}")
        
        logger.info(f"Generated {len(generated_files)} visualization files in {output_dir}")
        return generated_files