"""Sweep configuration management."""

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import yaml
import numpy as np


@dataclass
class SweepParameter:
    """Definition of a parameter to sweep over."""
    
    name: str
    values: Union[List[Any], Dict[str, Any]]
    distribution: str = "grid"  # grid, random, log, linear
    num_samples: Optional[int] = None  # For random sampling
    
    def sample(self, n_samples: Optional[int] = None) -> List[Any]:
        """Sample values from the parameter distribution.
        
        Args:
            n_samples: Number of samples (for random distribution)
            
        Returns:
            List of parameter values
        """
        if self.distribution == "grid":
            return self.values if isinstance(self.values, list) else list(self.values.values())
            
        elif self.distribution == "random":
            n = n_samples or self.num_samples or 10
            if isinstance(self.values, dict):
                # Random sampling from range
                min_val = self.values.get('min', 0)
                max_val = self.values.get('max', 1)
                if self.values.get('type') == 'int':
                    return [random.randint(min_val, max_val) for _ in range(n)]
                else:
                    return [random.uniform(min_val, max_val) for _ in range(n)]
            else:
                # Random choice from list
                return random.choices(self.values, k=n)
                
        elif self.distribution == "log":
            if isinstance(self.values, dict):
                min_val = np.log10(self.values.get('min', 0.001))
                max_val = np.log10(self.values.get('max', 1000))
                n = n_samples or self.num_samples or 10
                return list(10 ** np.linspace(min_val, max_val, n))
            else:
                return self.values
                
        elif self.distribution == "linear":
            if isinstance(self.values, dict):
                min_val = self.values.get('min', 0)
                max_val = self.values.get('max', 1)
                n = n_samples or self.num_samples or 10
                return list(np.linspace(min_val, max_val, n))
            else:
                return self.values
                
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class ParameterGrid:
    """Grid of parameters for sweeping."""
    
    parameters: List[SweepParameter]
    
    def generate_configs(self, max_configs: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate all parameter configurations.
        
        Args:
            max_configs: Maximum number of configurations to generate
            
        Returns:
            List of parameter configuration dictionaries
        """
        # Get all parameter values
        param_values = {}
        for param in self.parameters:
            param_values[param.name] = param.sample()
        
        # Generate cartesian product
        param_names = list(param_values.keys())
        param_value_lists = [param_values[name] for name in param_names]
        
        configs = []
        for values in itertools.product(*param_value_lists):
            config = dict(zip(param_names, values))
            configs.append(config)
            
            if max_configs and len(configs) >= max_configs:
                break
                
        return configs


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep."""
    
    # Model configuration
    model_name: str
    model_type: str = "hf_local"
    model_path: Optional[str] = None
    
    # RoPE methods to evaluate
    rope_methods: List[str] = field(default_factory=lambda: ["linear_interpolation", "ntk_aware", "yarn", "longrope", "dynamic_ntk", "llama3"])
    
    # Context lengths to evaluate
    context_lengths: List[int] = field(default_factory=lambda: [2048, 4096, 8192, 16384, 32768])
    
    # Parameter grids for each method
    parameter_grids: Dict[str, ParameterGrid] = field(default_factory=dict)
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: ["perplexity", "passkey_retrieval"])
    
    # Sweep settings
    max_configs_per_method: Optional[int] = None
    parallel_jobs: int = 1
    use_cache: bool = True
    cache_dir: str = "./sweep_cache"
    output_dir: str = "./sweep_results"
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_metric: str = "perplexity"
    early_stopping_threshold: float = 100.0
    early_stopping_patience: int = 3
    
    # Resource management
    max_gpu_memory_gb: Optional[float] = None
    auto_batch_size: bool = True
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SweepConfig":
        """Load sweep configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            SweepConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse parameter grids
        if 'parameter_grids' in data:
            for method, grid_data in data['parameter_grids'].items():
                params = []
                for param_name, param_config in grid_data.items():
                    params.append(SweepParameter(
                        name=param_name,
                        values=param_config.get('values', []),
                        distribution=param_config.get('distribution', 'grid'),
                        num_samples=param_config.get('num_samples')
                    ))
                data['parameter_grids'][method] = ParameterGrid(params)
        
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]):
        """Save sweep configuration to YAML file.
        
        Args:
            path: Path to save YAML configuration
        """
        data = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_path': self.model_path,
            'rope_methods': self.rope_methods,
            'context_lengths': self.context_lengths,
            'metrics': self.metrics,
            'max_configs_per_method': self.max_configs_per_method,
            'parallel_jobs': self.parallel_jobs,
            'use_cache': self.use_cache,
            'cache_dir': self.cache_dir,
            'output_dir': self.output_dir,
            'early_stopping': self.early_stopping,
            'early_stopping_metric': self.early_stopping_metric,
            'early_stopping_threshold': self.early_stopping_threshold,
            'early_stopping_patience': self.early_stopping_patience,
            'max_gpu_memory_gb': self.max_gpu_memory_gb,
            'auto_batch_size': self.auto_batch_size,
        }
        
        # Convert parameter grids
        if self.parameter_grids:
            data['parameter_grids'] = {}
            for method, grid in self.parameter_grids.items():
                data['parameter_grids'][method] = {}
                for param in grid.parameters:
                    data['parameter_grids'][method][param.name] = {
                        'values': param.values,
                        'distribution': param.distribution,
                        'num_samples': param.num_samples
                    }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_total_experiments(self) -> int:
        """Calculate total number of experiments in the sweep.
        
        Returns:
            Total number of experiments
        """
        total = 0
        for method in self.rope_methods:
            if method in self.parameter_grids:
                grid = self.parameter_grids[method]
                configs = grid.generate_configs(self.max_configs_per_method)
                total += len(configs) * len(self.context_lengths)
            else:
                # Default single configuration
                total += len(self.context_lengths)
        return total
    
    def generate_experiment_configs(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations.
        
        Returns:
            List of experiment configuration dictionaries
        """
        experiments = []
        
        for method in self.rope_methods:
            if method in self.parameter_grids:
                grid = self.parameter_grids[method]
                param_configs = grid.generate_configs(self.max_configs_per_method)
            else:
                # Use default configuration
                param_configs = [self._get_default_params(method)]
            
            for param_config in param_configs:
                for context_length in self.context_lengths:
                    experiment = {
                        'rope_method': method,
                        'context_length': context_length,
                        'parameters': param_config,
                        'metrics': self.metrics,
                        'model_name': self.model_name,
                        'model_type': self.model_type,
                        'model_path': self.model_path
                    }
                    experiments.append(experiment)
        
        return experiments
    
    def _get_default_params(self, method: str) -> Dict[str, Any]:
        """Get default parameters for a RoPE method.
        
        Args:
            method: RoPE method name
            
        Returns:
            Default parameter dictionary
        """
        defaults = {
            'linear_interpolation': {'scaling_factor': 4.0},
            'ntk_aware': {'alpha': 1.0, 'beta': 32.0},
            'yarn': {
                's': 16.0,
                'alpha': 1.0,
                'beta': 32.0,
                'attention_factor': 0.1,
                'beta_fast': 32.0,
                'beta_slow': 1.0
            },
            'longrope': {
                'short_factor': [1.0] * 8,
                'long_factor': [1.0, 2.0, 4.0, 8.0, 8.0, 4.0, 2.0, 1.0],
                'factor_strategy': 'layer_wise'
            },
            'dynamic_ntk': {'alpha': 1.0}
        }
        return defaults.get(method, {})