# Installation and Setup Guide

This guide will help you install and set up the RoPE Long Context Evaluation Suite with hyperparameter sweeping capabilities.

## Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (32GB+ recommended for large models)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rope_long_context_evaluation_suite
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n rope-eval python=3.11
conda activate rope-eval

# Or using venv
python -m venv rope-eval
source rope-eval/bin/activate  # On Windows: rope-eval\Scripts\activate
```

### 3. Install the Package

```bash
# Install in development mode with all dependencies
pip install -e .

# Or install specific dependency groups
pip install -e ".[dev,evaluation]"
```

### 4. Verify Installation

```bash
# Run the validation script
python validate_imports.py

# Test CLI
python -m rope_long_context_evaluation_suite.sweep_cli --help
```

Expected output:
```
✅ All validation tests passed!

Next steps:
1. Install dependencies: pip install -e .
2. Run example: python examples/run_sweep_example.py --config examples/sweep_configs/quick_comparison_sweep.yaml
```

## Quick Test

### 1. Run Example Sweep

```bash
# Quick comparison sweep (small model, limited parameters)
python examples/run_sweep_example.py \
  --config examples/sweep_configs/quick_comparison_sweep.yaml \
  --parallel

# View results
ls sweep_results/quick_comparison/
```

### 2. Analyze Results

```bash
# Generate analysis report
python -m rope_long_context_evaluation_suite.sweep_cli analyze \
  sweep_results/quick_comparison/sweep_results.json

# Create visualizations
python -m rope_long_context_evaluation_suite.sweep_cli visualize \
  sweep_results/quick_comparison/sweep_results.json \
  --contours --interactive
```

## Module Structure Verification

The package should have the following structure:

```
src/rope_long_context_evaluation_suite/
├── __init__.py                 # Main package imports
├── core.py                     # RoPEEvaluator
├── utils.py                    # Utility functions
├── cli.py                      # Original CLI
├── sweep_cli.py               # Sweep CLI
├── models/                     # RoPE extensions
│   ├── __init__.py
│   ├── base.py                # BaseRoPEExtension
│   ├── linear_interpolation.py
│   ├── ntk_aware.py
│   ├── yarn.py
│   ├── longrope.py
│   ├── dynamic_ntk.py
│   ├── model_loader.py
│   └── factory.py
├── metrics/                    # Evaluation metrics
│   ├── __init__.py
│   ├── base.py
│   ├── perplexity.py
│   ├── passkey.py
│   └── longppl.py
├── sweep/                      # Hyperparameter sweeping
│   ├── __init__.py
│   ├── config.py
│   ├── runner.py
│   ├── analyzer.py
│   └── visualizer.py
└── benchmarks/                 # Evaluation benchmarks
    ├── __init__.py
    ├── base.py
    ├── niah.py
    ├── ruler.py
    ├── longbench.py
    └── longbench_v2.py
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'rope_long_context_evaluation_suite'`

**Solution**:
```bash
# Ensure you're in the correct directory
cd rope_long_context_evaluation_suite

# Install in development mode
pip install -e .

# Check if src path is correct
python validate_imports.py
```

### Missing Dependencies

**Problem**: `ImportError: No module named 'torch'` or similar

**Solution**:
```bash
# Install all dependencies
pip install -e .

# Or install specific missing packages
pip install torch transformers matplotlib seaborn plotly scikit-learn
```

### CUDA Issues

**Problem**: GPU not detected or CUDA errors

**Solution**:
```bash
# Check CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

**Problem**: Out of memory during sweeps

**Solution**:
```yaml
# In your sweep config
max_gpu_memory_gb: 8.0      # Reduce limit
auto_batch_size: true       # Enable automatic sizing
parallel_jobs: 1           # Reduce parallelization
```

### Package Version Mismatch

**Problem**: Old version or import inconsistencies

**Solution**:
```bash
# Reinstall in development mode
pip uninstall rope_long_context_evaluation_suite
pip install -e .

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
```

## Development Setup

### Additional Development Dependencies

```bash
# Install development tools
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rope_long_context_evaluation_suite

# Run specific test file
pytest tests/test_models_new.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Next Steps

1. **Explore Examples**: Check `examples/sweep_configs/` for different sweep configurations
2. **Read Documentation**: See `HYPERPARAMETER_SWEEP.md` for detailed usage guide
3. **Customize**: Create your own sweep configurations for your models
4. **Contribute**: Add new RoPE methods or metrics following the established patterns

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Run `python validate_imports.py` to diagnose problems
3. Review the example configurations in `examples/sweep_configs/`
4. Check logs in the `logs/` directory for detailed error messages

## Environment Information

You can check your environment setup:

```bash
python -c "
import rope_long_context_evaluation_suite
from rope_long_context_evaluation_suite.utils import get_device_info
print(f'Package version: {rope_long_context_evaluation_suite.__version__}')
print(f'Device info: {get_device_info()}')
"
```

This should display your package version and available compute devices.