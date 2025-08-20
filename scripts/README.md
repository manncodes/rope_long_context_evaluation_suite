# Scripts Directory

This directory contains various utility and evaluation scripts organized by purpose.

## Structure

### `/analysis/`
- `analyze_comprehensive_benchmark_results.py` - Analyze comprehensive benchmark results
- `analyze_comprehensive_results.py` - General comprehensive analysis
- `analyze_llama32_results.py` - Llama 3.2 specific analysis

### `/demos/`
- `add_longbench_demo.py` - LongBench integration demonstration
- `create_comprehensive_benchmark_demo.py` - Comprehensive benchmark demo
- `create_sample_visualizations.py` - Sample visualization generation
- `demo_benchmark_comparison.py` - Benchmark comparison demo
- `describe_visualizations.py` - Visualization description tool

### `/validation/`
- `validate_all_methods.py` - Validate all RoPE methods
- `validate_imports.py` - Package import validation
- `validate_llama3.py` - Llama 3 specific validation
- `test_models.py` - Model loading tests
- `validate_datasets.py` - Dataset validation (from setup script)
- `quick_test.py` - Quick framework test (from setup script)

### Main Scripts (in `/scripts/`)
- `run_comprehensive_benchmark_evaluation.py` - Full benchmark suite
- `run_focused_benchmark_evaluation.py` - Focused evaluation
- `run_longbench_evaluation.py` - LongBench specific evaluation
- `run_llama31.py`, `run_llama32_evaluation.py` - Model-specific runs
- `run_mistral.py`, `run_tinyllama.py` - Other model evaluations
- `setup_data.py` - Data setup and download

## Usage

Most scripts can be run directly:
```bash
# Run validation
python scripts/validation/validate_imports.py

# Create visualizations
python scripts/demos/create_sample_visualizations.py

# Analyze results
python scripts/analysis/analyze_comprehensive_results.py results.json
```

For the main evaluation framework, use the root-level runner:
```bash
# Main comprehensive evaluation
python run_comprehensive.py --config comprehensive_config.yaml
```