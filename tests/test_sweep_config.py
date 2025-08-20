"""Tests for sweep configuration module."""

import pytest
import tempfile
from pathlib import Path

from src.rope_long_context_evaluation_suite.sweep.config import (
    SweepParameter, ParameterGrid, SweepConfig
)


class TestSweepParameter:
    """Test SweepParameter class."""
    
    def test_grid_sampling(self):
        """Test grid sampling."""
        param = SweepParameter(
            name="alpha",
            values=[1.0, 2.0, 4.0],
            distribution="grid"
        )
        
        samples = param.sample()
        assert samples == [1.0, 2.0, 4.0]
    
    def test_random_sampling(self):
        """Test random sampling."""
        param = SweepParameter(
            name="beta",
            values={'min': 0.0, 'max': 1.0},
            distribution="random",
            num_samples=10
        )
        
        samples = param.sample()
        assert len(samples) == 10
        assert all(0.0 <= s <= 1.0 for s in samples)
    
    def test_log_sampling(self):
        """Test logarithmic sampling."""
        param = SweepParameter(
            name="learning_rate",
            values={'min': 0.001, 'max': 1.0},
            distribution="log",
            num_samples=5
        )
        
        samples = param.sample()
        assert len(samples) == 5
        assert all(0.001 <= s <= 1.0 for s in samples)
    
    def test_linear_sampling(self):
        """Test linear sampling."""
        param = SweepParameter(
            name="temperature",
            values={'min': 0.1, 'max': 2.0},
            distribution="linear",
            num_samples=5
        )
        
        samples = param.sample()
        assert len(samples) == 5
        assert samples[0] == pytest.approx(0.1)
        assert samples[-1] == pytest.approx(2.0)


class TestParameterGrid:
    """Test ParameterGrid class."""
    
    def test_single_parameter_grid(self):
        """Test grid with single parameter."""
        param = SweepParameter("alpha", [1.0, 2.0])
        grid = ParameterGrid([param])
        
        configs = grid.generate_configs()
        expected = [{"alpha": 1.0}, {"alpha": 2.0}]
        assert configs == expected
    
    def test_multiple_parameter_grid(self):
        """Test grid with multiple parameters."""
        param1 = SweepParameter("alpha", [1.0, 2.0])
        param2 = SweepParameter("beta", [8, 16])
        grid = ParameterGrid([param1, param2])
        
        configs = grid.generate_configs()
        expected = [
            {"alpha": 1.0, "beta": 8},
            {"alpha": 1.0, "beta": 16},
            {"alpha": 2.0, "beta": 8},
            {"alpha": 2.0, "beta": 16}
        ]
        assert configs == expected
    
    def test_max_configs_limit(self):
        """Test maximum configurations limit."""
        param1 = SweepParameter("alpha", [1.0, 2.0, 3.0])
        param2 = SweepParameter("beta", [8, 16, 32])
        grid = ParameterGrid([param1, param2])
        
        configs = grid.generate_configs(max_configs=5)
        assert len(configs) == 5


class TestSweepConfig:
    """Test SweepConfig class."""
    
    def test_default_initialization(self):
        """Test default SweepConfig initialization."""
        config = SweepConfig(model_name="test_model")
        
        assert config.model_name == "test_model"
        assert config.model_type == "hf_local"
        assert "linear_interpolation" in config.rope_methods
        assert 2048 in config.context_lengths
        assert "perplexity" in config.metrics
    
    def test_custom_initialization(self):
        """Test custom SweepConfig initialization."""
        config = SweepConfig(
            model_name="custom_model",
            model_type="hf_hub",
            rope_methods=["yarn", "ntk_aware"],
            context_lengths=[4096, 8192],
            metrics=["passkey_retrieval"]
        )
        
        assert config.model_name == "custom_model"
        assert config.model_type == "hf_hub"
        assert config.rope_methods == ["yarn", "ntk_aware"]
        assert config.context_lengths == [4096, 8192]
        assert config.metrics == ["passkey_retrieval"]
    
    def test_yaml_roundtrip(self):
        """Test YAML save/load roundtrip."""
        config = SweepConfig(
            model_name="test_model",
            rope_methods=["linear_interpolation"],
            context_lengths=[2048, 4096]
        )
        
        # Create parameter grid
        param = SweepParameter("scaling_factor", [2, 4, 8])
        grid = ParameterGrid([param])
        config.parameter_grids = {"linear_interpolation": grid}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            config.to_yaml(temp_path)
        
        try:
            loaded_config = SweepConfig.from_yaml(temp_path)
            
            assert loaded_config.model_name == config.model_name
            assert loaded_config.rope_methods == config.rope_methods
            assert loaded_config.context_lengths == config.context_lengths
            assert "linear_interpolation" in loaded_config.parameter_grids
        finally:
            Path(temp_path).unlink()  # Cleanup
    
    def test_total_experiments_calculation(self):
        """Test calculation of total experiments."""
        config = SweepConfig(
            model_name="test_model",
            rope_methods=["linear_interpolation", "yarn"],
            context_lengths=[2048, 4096]
        )
        
        # Add parameter grid for linear_interpolation
        param = SweepParameter("scaling_factor", [2, 4])
        grid = ParameterGrid([param])
        config.parameter_grids = {"linear_interpolation": grid}
        
        total = config.get_total_experiments()
        # linear_interpolation: 2 params * 2 context lengths = 4
        # yarn: 1 default config * 2 context lengths = 2
        # Total: 6
        assert total == 6
    
    def test_experiment_config_generation(self):
        """Test generation of experiment configurations."""
        config = SweepConfig(
            model_name="test_model",
            rope_methods=["linear_interpolation"],
            context_lengths=[2048]
        )
        
        param = SweepParameter("scaling_factor", [2, 4])
        grid = ParameterGrid([param])
        config.parameter_grids = {"linear_interpolation": grid}
        
        experiments = config.generate_experiment_configs()
        
        assert len(experiments) == 2
        assert all(exp['rope_method'] == 'linear_interpolation' for exp in experiments)
        assert all(exp['context_length'] == 2048 for exp in experiments)
        assert experiments[0]['parameters']['scaling_factor'] == 2
        assert experiments[1]['parameters']['scaling_factor'] == 4
    
    def test_default_parameters(self):
        """Test default parameter generation."""
        config = SweepConfig(
            model_name="test_model",
            rope_methods=["yarn"]
        )
        
        default_params = config._get_default_params("yarn")
        expected_keys = ['s', 'alpha', 'beta', 'attention_factor', 'beta_fast', 'beta_slow']
        
        for key in expected_keys:
            assert key in default_params
        
        assert default_params['s'] == 16.0
        assert default_params['alpha'] == 1.0