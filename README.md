# RoPE Long Context Evaluation Suite

A comprehensive evaluation framework for RoPE (Rotary Position Embedding) extension methods with support for state-of-the-art benchmarks and hyperparameter optimization.

## Overview

This suite provides a unified framework for evaluating and comparing different RoPE scaling methods on long context tasks. It supports both synthetic benchmarks (NIAH, RULER) and real-world datasets (LongBench), with built-in hyperparameter sweeping and comprehensive result analysis.

## Key Features

- **6 RoPE Extension Methods**: Linear Interpolation, NTK-Aware, YaRN, LongRoPE, Dynamic NTK, Llama3 scaling
- **4 Benchmark Types**: Traditional retrieval, NIAH (Needle-in-Haystack), RULER synthetic tasks, LongBench real-world tasks
- **Advanced Evaluation**: Hyperparameter sweeping with grid/random search and parallel execution
- **Production Ready**: Offline dataset support, NFS integration, GPU cluster deployment
- **Rich Analytics**: Detailed metrics, visualizations, comparative analysis across methods

## Quick Start

### Prerequisites
- Python ≥3.9  
- CUDA-compatible GPU with ≥8GB VRAM
- PyTorch ≥2.0.0

### Installation

```bash
git clone https://github.com/manncodes/rope_long_context_evaluation_suite.git
cd rope_long_context_evaluation_suite

# Option 1: Direct installation (recommended)
./install_direct.sh

# Option 2: UV-based setup (for advanced users)  
./setup_comprehensive_evaluation.sh
```

### Configuration

Edit `comprehensive_config.yaml` for your setup:

```yaml
model:
  name: "llama-3.2-1b"
  path: "unsloth/Llama-3.2-1B"  # HuggingFace model or local path
  device: "cuda"
  torch_dtype: "bfloat16"

datasets:
  longbench:
    path: "data/longbench"  # Update for your dataset location
    tasks: ["narrativeqa", "qasper", "multifieldqa_en"]

rope_methods:
  - name: "yarn"
    config:
      scaling_factor: 2.0
      attention_factor: 0.1
  - name: "ntk_aware" 
    config:
      scaling_factor: 2.0
      alpha: 8.0
```

## Usage

### Single Model Evaluation

```bash
# Quick validation
python scripts/validation/validate_imports.py

# Single evaluation with specific parameters
python run_evaluation.py --model "meta-llama/Llama-2-7b-hf" --benchmarks niah ruler --rope-method yarn
```

### Comprehensive Parameter Sweep

For systematic evaluation across **all RoPE methods, context lengths, and benchmarks**:

```bash
# Quick test sweep (3 RoPE methods × 2 context lengths × 1 benchmark = 6 runs)
python run_comprehensive_sweep.py --config sweep_configs/quick_test_sweep.yaml

# Full comprehensive sweep (14 RoPE methods × 4 context lengths × 3 benchmarks = 168 runs)  
python run_comprehensive_sweep.py --config sweep_configs/full_sweep.yaml

# Custom filtered sweeps
python run_comprehensive_sweep.py --rope-methods none linear ntk_aware
python run_comprehensive_sweep.py --context-lengths 4000 8000
python run_comprehensive_sweep.py --benchmarks niah ruler
python run_comprehensive_sweep.py --max-runs 10  # Limit for testing
```

### Sweep Results Analysis

The comprehensive sweep generates:
- **Individual run results**: Detailed results for each parameter combination
- **Performance analysis**: Statistical comparison across all dimensions  
- **Best configurations**: Top-performing RoPE method + context length combinations
- **Method comparison**: Rankings and statistical significance tests
- **Context scaling**: Performance trends across different sequence lengths

Results saved in: `comprehensive_results/comprehensive_sweep_YYYYMMDD_HHMMSS/`

## Supported Methods

### RoPE Extensions
- **Linear Interpolation**: Simple position scaling
- **NTK-Aware**: Frequency-dependent scaling with alpha parameter
- **YaRN**: Adaptive interpolation with attention factor and beta parameters
- **LongRoPE**: Evolutionary search with short/long factors
- **Dynamic NTK**: Runtime adaptation based on sequence length
- **Llama3**: Official Llama 3 scaling method

### Benchmarks
- **Traditional Retrieval**: Synthetic passkey retrieval tasks
- **NIAH**: Needle In A Haystack with multi-needle and NoLiMa variants
- **RULER**: Synthetic benchmark with retrieval, multi-hop, aggregation, QA
- **LongBench**: Real-world long context tasks (12 core tasks)

## Results & Analysis

The suite generates comprehensive results with multiple output formats:

### Generated Outputs
- **JSON Results**: Detailed metrics with full configuration traces
- **CSV Summaries**: Tabular data for easy analysis and plotting  
- **Visualizations**: Performance heatmaps, method comparisons, context length scaling
- **Statistical Reports**: Detailed performance statistics and confidence intervals

### Sample Results Structure
```
comprehensive_results/
├── llama32_comprehensive_results_20250820.json    # Detailed JSON metrics
├── comprehensive_analysis.png                     # Overview visualization  
├── method_comparison_detailed.png                 # Side-by-side comparison
├── perplexity_heatmap.png                        # Context length analysis
└── detailed_statistics.txt                        # Statistical summary
```

### Performance Insights  
Based on extensive evaluations with TinyLlama 1.1B and Llama 3.2:
- **YaRN** consistently outperforms other methods across context lengths
- **NTK-Aware** provides good balance between performance and simplicity  
- **Linear Interpolation** works well for moderate context extensions
- Context length scaling varies significantly by method and model size

## Configuration

The framework uses YAML configuration for flexibility:

```yaml
# Hardware optimization
hardware:
  num_gpus: 1
  gpu_memory_fraction: 0.9
  mixed_precision: true

# Evaluation settings
evaluation:
  batch_size: 1
  max_context_length: 32768
  gradient_checkpointing: true
  use_cache: false

# Dataset configuration
datasets:
  retrieval:
    context_lengths: [4000, 8000, 16000, 32000]
    num_samples: 50
  
  longbench:
    path: "/nfs/datasets/longbench"
    tasks: ["narrativeqa", "qasper", "multifieldqa_en"]
```

## Offline Deployment

For GPU clusters with restricted internet access:

1. Set offline environment variables:
```bash
export HF_OFFLINE=1
export LONGBENCH_DATA_PATH=/nfs/datasets/longbench
export MODEL_CACHE_PATH=/nfs/models
```

2. Use pre-downloaded datasets and models with NFS paths in configuration

3. The framework automatically handles offline mode with local dataset loading

## Development

### Project Structure
```
rope_long_context_evaluation_suite/
├── src/rope_long_context_evaluation_suite/
│   ├── benchmarks/           # NIAH, RULER, LongBench implementations
│   ├── metrics/             # Perplexity, passkey, longppl metrics
│   ├── sweep/               # Hyperparameter optimization
│   └── core.py             # Main RoPEEvaluator class
├── scripts/
│   ├── analysis/           # Result analysis and plotting
│   ├── demos/             # Usage examples and demos
│   └── validation/        # Setup and import validation
├── examples/              # Configuration examples and tutorials
└── comprehensive_config.yaml    # Main configuration file
```

### Architecture Overview

- **RoPEEvaluator**: Main evaluation orchestrator handling model loading, RoPE application, and benchmark execution
- **Benchmark System**: Modular benchmark implementations with consistent interfaces
- **Sweep Framework**: Grid search and random search with parallel execution support
- **Configuration**: YAML-based configuration with comprehensive validation

## Performance & Optimization

### Framework Optimizations
- **Flash Attention 2**: Efficient attention computation for long sequences
- **Memory Management**: Gradient checkpointing and automatic batch sizing
- **Mixed Precision**: FP16/BF16 support for memory efficiency  
- **Parallel Execution**: Multi-GPU and multi-process sweep execution
- **Caching**: Intelligent model and dataset caching

### Resource Requirements

| Model Size | GPU Memory | Recommended GPU | Max Context Length |
|------------|------------|-----------------|-------------------|
| 1B models  | ~8GB      | RTX 3080/4070   | 32K tokens       |
| 3B models  | ~16GB     | RTX 4080/A100   | 32K tokens       |
| 7B models  | ~32GB     | A100 80GB       | 16K tokens       |
| 13B+ models| ~64GB     | Multi-GPU       | 8K tokens        |

### Performance Tips
- Use `bfloat16` for optimal memory/accuracy balance
- Enable gradient checkpointing for longer contexts  
- Set `use_cache: false` when evaluating very long sequences
- Consider model compilation with PyTorch 2.0+ for additional speedup

## License

MIT License

## Contributing

Contributions are welcome! Please see our contribution guidelines:

1. **Fork & Create Branch**: Fork the repo and create a feature branch
2. **Add Tests**: Include tests for new functionality
3. **Follow Style**: Use Black formatting and type hints
4. **Documentation**: Update docs for new features
5. **Submit PR**: Create a pull request with clear description

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rope_evaluation_suite,
  title = {RoPE Long Context Evaluation Suite},
  author = {Mann Patel},
  year = {2024},
  url = {https://github.com/manncodes/rope_long_context_evaluation_suite},
  note = {A comprehensive framework for evaluating RoPE scaling methods}
}
```

## Acknowledgments

- Original RoPE paper and implementations
- LongBench, NIAH, and RULER benchmark authors  
- PyTorch and Transformers library developers
- Flash Attention authors for efficient attention implementation