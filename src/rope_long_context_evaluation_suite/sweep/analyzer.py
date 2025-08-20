"""Analysis module for hyperparameter sweep results."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SweepAnalyzer:
    """Comprehensive analysis of hyperparameter sweep results."""
    
    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        """Initialize analyzer.
        
        Args:
            results: Sweep results to analyze
        """
        self.results = results or []
        self.df = None
        if self.results:
            self.df = self._create_dataframe(results)
    
    def load_results(self, results_path: Union[str, Path]):
        """Load results from JSON file.
        
        Args:
            results_path: Path to results JSON file
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        self.df = self._create_dataframe(self.results)
    
    def _create_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame.
        
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
                    # For now, convert to string - could be improved
                    base_row[f'param_{param_name}'] = str(param_value)
                else:
                    base_row[f'param_{param_name}'] = param_value
            
            # Add metric values
            for metric_name, metric_result in result['metrics'].items():
                if isinstance(metric_result, dict) and 'error' not in metric_result:
                    # Primary metric value
                    if metric_name == 'perplexity':
                        base_row[f'metric_{metric_name}'] = metric_result.get('perplexity', np.nan)
                    elif metric_name == 'passkey_retrieval':
                        base_row[f'metric_{metric_name}'] = metric_result.get('passkey_accuracy', np.nan)
                    elif metric_name == 'longppl':
                        base_row[f'metric_{metric_name}'] = metric_result.get('longppl', np.nan)
                    
                    # Additional metrics
                    for key, value in metric_result.items():
                        if isinstance(value, (int, float)):
                            base_row[f'{metric_name}_{key}'] = value
            
            rows.append(base_row)
        
        return pd.DataFrame(rows)
    
    def compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the sweep results.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.df is None or self.df.empty:
            return {}
        
        stats = {
            'total_experiments': len(self.df),
            'methods_tested': self.df['rope_method'].nunique(),
            'context_lengths_tested': sorted(self.df['context_length'].unique().tolist()),
            'method_distribution': self.df['rope_method'].value_counts().to_dict()
        }
        
        # Metric statistics
        metric_cols = [col for col in self.df.columns if col.startswith('metric_')]
        for metric_col in metric_cols:
            metric_name = metric_col.replace('metric_', '')
            metric_data = self.df[metric_col].dropna()
            
            if not metric_data.empty:
                stats[f'{metric_name}_stats'] = {
                    'mean': float(metric_data.mean()),
                    'std': float(metric_data.std()),
                    'min': float(metric_data.min()),
                    'max': float(metric_data.max()),
                    'median': float(metric_data.median()),
                    'q25': float(metric_data.quantile(0.25)),
                    'q75': float(metric_data.quantile(0.75))
                }
        
        # Parameter statistics
        param_cols = [col for col in self.df.columns if col.startswith('param_')]
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            try:
                param_data = pd.to_numeric(self.df[param_col], errors='coerce').dropna()
                if not param_data.empty:
                    stats[f'{param_name}_param_stats'] = {
                        'mean': float(param_data.mean()),
                        'std': float(param_data.std()),
                        'min': float(param_data.min()),
                        'max': float(param_data.max()),
                        'unique_values': int(param_data.nunique())
                    }
            except:
                # Non-numeric parameter
                stats[f'{param_name}_param_stats'] = {
                    'unique_values': int(self.df[param_col].nunique()),
                    'most_common': self.df[param_col].mode().iloc[0] if not self.df[param_col].mode().empty else None
                }
        
        return stats
    
    def find_best_configurations(self, metric: str = 'perplexity',
                               top_k: int = 10) -> List[Dict[str, Any]]:
        """Find the best performing configurations.
        
        Args:
            metric: Metric to optimize for
            top_k: Number of top configurations to return
            
        Returns:
            List of best configurations
        """
        if self.df is None or self.df.empty:
            return []
        
        metric_col = f'metric_{metric}' if f'metric_{metric}' in self.df.columns else metric
        
        if metric_col not in self.df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return []
        
        # Sort by metric (assuming lower is better for most metrics)
        # For accuracy metrics, we'd want higher is better
        if metric in ['passkey_retrieval', 'accuracy']:
            # Higher is better
            best_configs = self.df.nlargest(top_k, metric_col)
        else:
            # Lower is better (perplexity, loss, etc.)
            best_configs = self.df.nsmallest(top_k, metric_col)
        
        results = []
        for _, row in best_configs.iterrows():
            config = {
                'rank': len(results) + 1,
                'rope_method': row['rope_method'],
                'context_length': row['context_length'],
                'metric_value': row[metric_col],
                'parameters': {}
            }
            
            # Extract parameters
            for col in row.index:
                if col.startswith('param_'):
                    param_name = col.replace('param_', '')
                    config['parameters'][param_name] = row[col]
            
            # Add other metrics
            for col in row.index:
                if col.startswith('metric_') and col != metric_col:
                    other_metric = col.replace('metric_', '')
                    config[f'{other_metric}_value'] = row[col]
            
            results.append(config)
        
        return results
    
    def analyze_parameter_sensitivity(self, metric: str = 'perplexity') -> Dict[str, Dict[str, float]]:
        """Analyze sensitivity of performance to different parameters.
        
        Args:
            metric: Metric to analyze sensitivity for
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        if self.df is None or self.df.empty:
            return {}
        
        metric_col = f'metric_{metric}' if f'metric_{metric}' in self.df.columns else metric
        
        if metric_col not in self.df.columns:
            return {}
        
        param_cols = [col for col in self.df.columns if col.startswith('param_')]
        sensitivity_results = {}
        
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            
            try:
                # Convert to numeric if possible
                param_data = pd.to_numeric(self.df[param_col], errors='coerce')
                metric_data = self.df[metric_col]
                
                # Remove NaN values
                mask = ~(param_data.isna() | metric_data.isna())
                param_clean = param_data[mask]
                metric_clean = metric_data[mask]
                
                if len(param_clean) < 3:
                    continue
                
                # Compute correlations
                pearson_corr, pearson_p = pearsonr(param_clean, metric_clean)
                spearman_corr, spearman_p = spearmanr(param_clean, metric_clean)
                
                # Compute variance explained
                try:
                    # Fit simple linear model
                    from sklearn.linear_model import LinearRegression
                    X = param_clean.values.reshape(-1, 1)
                    y = metric_clean.values
                    
                    reg = LinearRegression().fit(X, y)
                    variance_explained = reg.score(X, y)
                except:
                    variance_explained = pearson_corr ** 2
                
                # Compute range effect (performance change across parameter range)
                param_min, param_max = param_clean.min(), param_clean.max()
                
                # Bin parameter values and compute mean metric per bin
                if param_clean.nunique() > 5:
                    bins = pd.qcut(param_clean, q=5, duplicates='drop')
                    binned_means = metric_clean.groupby(bins).mean()
                    range_effect = binned_means.max() - binned_means.min()
                else:
                    range_effect = metric_clean.max() - metric_clean.min()
                
                sensitivity_results[param_name] = {
                    'pearson_correlation': float(pearson_corr),
                    'pearson_p_value': float(pearson_p),
                    'spearman_correlation': float(spearman_corr),
                    'spearman_p_value': float(spearman_p),
                    'variance_explained': float(variance_explained),
                    'range_effect': float(range_effect),
                    'parameter_range': [float(param_min), float(param_max)]
                }
                
            except Exception as e:
                logger.warning(f"Failed to analyze sensitivity for {param_name}: {e}")
                continue
        
        return sensitivity_results
    
    def analyze_method_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance characteristics of different RoPE methods.
        
        Returns:
            Dictionary containing method performance analysis
        """
        if self.df is None or self.df.empty:
            return {}
        
        method_analysis = {}
        
        for method in self.df['rope_method'].unique():
            method_data = self.df[self.df['rope_method'] == method]
            
            analysis = {
                'sample_count': len(method_data),
                'context_lengths_tested': sorted(method_data['context_length'].unique().tolist()),
                'avg_execution_time': float(method_data['execution_time'].mean()),
                'metrics': {}
            }
            
            # Analyze each metric
            metric_cols = [col for col in method_data.columns if col.startswith('metric_')]
            for metric_col in metric_cols:
                metric_name = metric_col.replace('metric_', '')
                metric_data = method_data[metric_col].dropna()
                
                if not metric_data.empty:
                    analysis['metrics'][metric_name] = {
                        'mean': float(metric_data.mean()),
                        'std': float(metric_data.std()),
                        'min': float(metric_data.min()),
                        'max': float(metric_data.max()),
                        'median': float(metric_data.median())
                    }
                    
                    # Performance vs context length
                    context_analysis = method_data.groupby('context_length')[metric_col].agg(['mean', 'std']).reset_index()
                    analysis['metrics'][metric_name]['vs_context_length'] = {
                        'context_lengths': context_analysis['context_length'].tolist(),
                        'means': context_analysis['mean'].tolist(),
                        'stds': context_analysis['std'].tolist()
                    }
            
            method_analysis[method] = analysis
        
        return method_analysis
    
    def compare_methods(self, metric: str = 'perplexity',
                       statistical_test: str = 'ttest') -> Dict[str, Any]:
        """Compare performance between different RoPE methods.
        
        Args:
            metric: Metric to compare
            statistical_test: Statistical test to use ('ttest', 'mannwhitney', 'kruskal')
            
        Returns:
            Dictionary containing comparison results
        """
        if self.df is None or self.df.empty:
            return {}
        
        metric_col = f'metric_{metric}' if f'metric_{metric}' in self.df.columns else metric
        
        if metric_col not in self.df.columns:
            return {}
        
        methods = self.df['rope_method'].unique()
        if len(methods) < 2:
            return {'error': 'Need at least 2 methods for comparison'}
        
        # Statistical tests
        from scipy import stats
        
        comparison_results = {
            'methods': methods.tolist(),
            'pairwise_comparisons': {},
            'overall_test': {}
        }
        
        # Pairwise comparisons
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i >= j:
                    continue
                
                data1 = self.df[self.df['rope_method'] == method1][metric_col].dropna()
                data2 = self.df[self.df['rope_method'] == method2][metric_col].dropna()
                
                if len(data1) < 3 or len(data2) < 3:
                    continue
                
                comparison_key = f'{method1}_vs_{method2}'
                
                try:
                    if statistical_test == 'ttest':
                        stat, p_value = stats.ttest_ind(data1, data2)
                    elif statistical_test == 'mannwhitney':
                        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    elif statistical_test == 'kruskal':
                        stat, p_value = stats.kruskal(data1, data2)
                    else:
                        stat, p_value = stats.ttest_ind(data1, data2)
                    
                    comparison_results['pairwise_comparisons'][comparison_key] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'method1_mean': float(data1.mean()),
                        'method2_mean': float(data2.mean()),
                        'effect_size': float(abs(data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2))
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to compare {method1} vs {method2}: {e}")
        
        # Overall test (ANOVA or Kruskal-Wallis)
        method_data = [self.df[self.df['rope_method'] == method][metric_col].dropna().values 
                      for method in methods]
        method_data = [data for data in method_data if len(data) > 0]
        
        if len(method_data) >= 2:
            try:
                if statistical_test in ['ttest', 'anova']:
                    stat, p_value = stats.f_oneway(*method_data)
                    test_name = 'ANOVA'
                else:
                    stat, p_value = stats.kruskal(*method_data)
                    test_name = 'Kruskal-Wallis'
                
                comparison_results['overall_test'] = {
                    'test_name': test_name,
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
                
            except Exception as e:
                logger.warning(f"Failed to perform overall test: {e}")
        
        return comparison_results
    
    def predict_performance(self, metric: str = 'perplexity') -> Dict[str, Any]:
        """Build predictive model for performance based on parameters.
        
        Args:
            metric: Metric to predict
            
        Returns:
            Dictionary containing model performance and feature importance
        """
        if self.df is None or self.df.empty:
            return {}
        
        metric_col = f'metric_{metric}' if f'metric_{metric}' in self.df.columns else metric
        
        if metric_col not in self.df.columns:
            return {}
        
        # Prepare features
        feature_cols = []
        
        # Add numeric parameters
        param_cols = [col for col in self.df.columns if col.startswith('param_')]
        for param_col in param_cols:
            try:
                param_data = pd.to_numeric(self.df[param_col], errors='coerce')
                if not param_data.isna().all():
                    feature_cols.append(param_col)
            except:
                pass
        
        # Add context length
        feature_cols.append('context_length')
        
        # Add method as categorical (one-hot encoded)
        method_dummies = pd.get_dummies(self.df['rope_method'], prefix='method')
        
        if len(feature_cols) == 0 and method_dummies.shape[1] == 0:
            return {'error': 'No valid features found for prediction'}
        
        # Prepare data
        X_numeric = self.df[feature_cols].select_dtypes(include=[np.number])
        X = pd.concat([X_numeric, method_dummies], axis=1)
        y = self.df[metric_col]
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 10:
            return {'error': 'Insufficient data for prediction model'}
        
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Train Random Forest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y_clean)
            
            # Cross-validation
            cv_scores = cross_val_score(rf, X_scaled, y_clean, cv=5, scoring='r2')
            
            # Feature importance
            feature_importance = dict(zip(X_clean.columns, rf.feature_importances_))
            
            # Model performance
            y_pred = rf.predict(X_scaled)
            r2 = r2_score(y_clean, y_pred)
            mse = mean_squared_error(y_clean, y_pred)
            
            return {
                'model_performance': {
                    'r2_score': float(r2),
                    'mean_squared_error': float(mse),
                    'cv_mean_r2': float(cv_scores.mean()),
                    'cv_std_r2': float(cv_scores.std())
                },
                'feature_importance': {k: float(v) for k, v in feature_importance.items()},
                'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                'sample_size': len(X_clean),
                'feature_count': X_clean.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Failed to build prediction model: {e}")
            return {'error': str(e)}
    
    def generate_analysis_report(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Complete analysis report
        """
        report = {
            'summary_statistics': self.compute_summary_statistics(),
            'best_configurations': {},
            'parameter_sensitivity': {},
            'method_performance': self.analyze_method_performance(),
            'method_comparisons': {},
            'predictive_models': {}
        }
        
        # Analyze for each metric
        metric_cols = [col for col in (self.df.columns if self.df is not None else []) 
                      if col.startswith('metric_')]
        
        for metric_col in metric_cols:
            metric_name = metric_col.replace('metric_', '')
            
            # Best configurations
            report['best_configurations'][metric_name] = self.find_best_configurations(metric_name, top_k=5)
            
            # Parameter sensitivity
            report['parameter_sensitivity'][metric_name] = self.analyze_parameter_sensitivity(metric_name)
            
            # Method comparisons
            report['method_comparisons'][metric_name] = self.compare_methods(metric_name)
            
            # Predictive model
            report['predictive_models'][metric_name] = self.predict_performance(metric_name)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Analysis report saved to {output_path}")
        
        return report