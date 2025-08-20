# RoPE Long Context Evaluation Suite

A comprehensive evaluation framework for RoPE (Rotary Position Embedding) extension methods with support for modern benchmarks including NIAH, RULER, and LongBench.

## Features

- **Multiple RoPE Extension Methods**: Linear, NTK-Aware, YaRN, LongRoPE, Dynamic NTK, Llama3
- **Comprehensive Benchmarks**: Traditional retrieval, NIAH, RULER, LongBench
- **Hyperparameter Sweeping**: Grid search, random search with parallel execution
- **GPU Cluster Deployment**: UV-based setup with offline/NFS dataset support
- **Advanced Metrics**: Passkey retrieval, perplexity, long perplexity

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup with UV
git clone https://github.com/manncodes/rope_long_context_evaluation_suite.git
cd rope_long_context_evaluation_suite
./setup_comprehensive_evaluation.sh
```

### 2. Configure Evaluation

Edit `comprehensive_config.yaml`:

```yaml
model:
  name: "llama-3.2-1b"
  path: "/path/to/your/model"  # Local path or HF model name
  device: "cuda"

datasets:
  longbench:
    path: "/nfs/datasets/longbench"  # Your dataset path
    tasks: ["narrativeqa", "qasper", "hotpotqa", ...]

rope_methods:
  - name: "yarn"
    config:
      scaling_factor: 4.0
      attention_factor: 0.1
```

### 3. Run Evaluation

```bash
# Validate setup
uv run scripts/validate_datasets.py
uv run scripts/quick_test.py

# Run comprehensive evaluation
uv run run_comprehensive_evaluation.py --config comprehensive_config.yaml

# Run hyperparameter sweep
uv run sweep_runner.py --config configs/sweep/rope_methods_sweep.yaml
```

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

## Results

The framework generates comprehensive results including:

- JSON metrics files with detailed scores
- CSV summaries for easy analysis  
- Performance visualizations and heatmaps
- Comparative analysis across methods and context lengths

Example results structure:
```
comprehensive_results/
├── llama32_comprehensive_results_20250820.json
├── comprehensive_analysis.png
├── method_comparison_detailed.png
└── perplexity_heatmap.png
```

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
src/rope_long_context_evaluation_suite/
├── benchmarks/          # Benchmark implementations
├── models/             # RoPE extensions and model loading  
├── metrics/            # Evaluation metrics
├── sweep/              # Hyperparameter sweeping
└── core.py            # Main evaluation framework
```

### Adding New Methods

1. Create new RoPE extension in `models/`
2. Inherit from `BaseRoPEExtension`
3. Implement `apply()` method
4. Register in factory

### Adding New Benchmarks

1. Create benchmark class in `benchmarks/`
2. Inherit from `BaseBenchmark`
3. Implement required methods
4. Add to configuration schema

## Performance

Framework optimizations:
- Flash Attention 2 support
- Gradient checkpointing for memory efficiency
- Mixed precision training
- Parallel sweep execution
- Automatic batch sizing

Memory requirements:
- 1B models: ~8GB GPU memory
- 3B models: ~16GB GPU memory  
- 7B models: ~32GB GPU memory

## License

MIT License

## Citation

```bibtex
@software{rope_evaluation_suite,
  title = {RoPE Long Context Evaluation Suite},
  author = {Mann Patel},
  year = {2024},
  url = {https://github.com/manncodes/rope_long_context_evaluation_suite}
}
```