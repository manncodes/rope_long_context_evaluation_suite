# RoPE Long Context Evaluation Suite

A comprehensive evaluation framework for long context RoPE extension methods

## Overview

This is a comprehensive evaluation framework for long context RoPE (Rotary Position Embedding) extension methods. The framework provides:

- **Multiple RoPE Extension Methods**: Linear Interpolation, NTK-Aware, YaRN, LongRoPE, Dynamic NTK
- **Comprehensive Benchmarks**: NIAH, RULER, LongBench, LongBench-V2
- **Flexible Model Loading**: Support for local HuggingFace models, HF Hub, and API models
- **YAML Configuration**: Easy-to-use configuration system with sensible defaults
- **CLI Interface**: Simple command-line interface for running evaluations

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rope_long_context_evaluation_suite
```

2. Install dependencies:
```bash
pip install -e .
```

### Basic Usage

1. **Configure your evaluation** by editing `config/default.yaml` or creating a custom config:

```yaml
model:
  type: "hf_local"
  name: "meta-llama/Llama-2-7b-hf"
  path: "./models/llama-2-7b-hf"
  max_length: 32768

rope_extension:
  method: "yarn"
  yarn:
    s: 16
    alpha: 1.0
```

2. **Run evaluation**:
```bash
rope-eval --config config/default.yaml
```

3. **View results** in the `results/` directory.

### Advanced Usage

#### Custom Configuration

```bash
# Override specific settings
rope-eval --config config/default.yaml --rope-method yarn --scaling-factor 8

# Run specific benchmarks
rope-eval --benchmarks niah ruler --max-length 65536

# Use different model
rope-eval --model-name microsoft/DialoGPT-medium --model-type hf_hub
```

#### Configuration File Examples

```yaml
# Example: Evaluate YaRN with multiple scaling factors
rope_extension:
  method: "yarn"
  yarn:
    s: 16
    alpha: 1.0
    beta: 32.0

benchmarks:
  niah:
    enabled: true
    context_lengths: [4000, 8000, 16000, 32000, 64000]
  ruler:
    enabled: true
    max_length: 64000
```

## Supported Methods

### RoPE Extension Methods

1. **Linear Interpolation (Position Interpolation)**
   - Simple scaling of rotary angles
   - Configuration: `scaling_factor`

2. **NTK-Aware Interpolation**
   - Frequency-dependent scaling
   - Configuration: `alpha`, `beta`

3. **YaRN (Yet another RoPE extensioN)**
   - Combines linear and NTK-aware with ramp function
   - Configuration: `s`, `alpha`, `beta`, `attention_factor`

4. **LongRoPE**
   - Evolutionary search-based method
   - Configuration: `short_factor`, `long_factor`

5. **Dynamic NTK**
   - Runtime sequence length adaptation
   - Configuration: `alpha`

### Evaluation Benchmarks

1. **NIAH (Needle in a Haystack)**
   - Standard single needle retrieval
   - Multi-needle variants
   - NoLiMa (Non-lexical matching)

2. **RULER**
   - Synthetic benchmark with 4 categories:
     - Retrieval tasks
     - Multi-hop tracing
     - Aggregation
     - Question answering

3. **LongBench**
   - Real-world long context tasks
   - Multiple domains and languages

4. **LongBench-V2**
   - More challenging version
   - Context lengths up to 2M tokens

## Configuration

The framework uses YAML configuration files with the following structure:

```yaml
model:           # Model settings
  type: "hf_local"
  name: "model_name"
  path: "./models/model_name"
  max_length: 32768

rope_extension: # RoPE extension settings
  method: "yarn"
  yarn:
    s: 16
    alpha: 1.0

benchmarks:     # Benchmark settings
  niah:
    enabled: true
    context_lengths: [4000, 8000, 16000]
  
evaluation:     # Evaluation settings
  batch_size: 1
  save_predictions: true
  
logging:        # Logging settings
  level: "INFO"
  wandb:
    enabled: false
```

## Results and Analysis

Results are saved in JSON format with the following structure:

```json
{
  "config": {...},           // Configuration used
  "model_info": {...},       // Model information
  "rope_info": {...},        // RoPE extension details
  "benchmarks": {            // Benchmark results
    "niah": {
      "average_score": 0.95,
      "results": [...]
    }
  },
  "summary": {...}           // Overall summary
}
```

### Visualization

The framework can generate plots and export results in multiple formats:

- JSON (detailed results)
- CSV (tabular format)
- Weights & Biases integration

## Development

### Project Structure

```
rope_long_context_evaluation_suite/
├── config/                 # Configuration files
├── src/rope_long_context_evaluation_suite/
│   ├── benchmarks/        # Benchmark implementations
│   ├── models/           # Model loading and RoPE extensions
│   ├── core.py           # Main evaluation framework
│   ├── cli.py            # Command-line interface
│   └── utils.py          # Utility functions
├── tests/                # Unit tests
├── data/                 # Data files
└── results/              # Evaluation results
```

### Adding New Benchmarks

1. Create a new benchmark class inheriting from `BaseBenchmark`
2. Implement required methods: `load_data`, `prepare_input`, `extract_answer`, `compute_score`
3. Register the benchmark in `core.py`

### Adding New RoPE Methods

1. Create a new class inheriting from `RoPEExtension`
2. Implement the `apply` method to modify model
3. Add to the factory function in `rope_extensions.py`

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rope_eval_suite,
  title = { RoPE Long Context Evaluation Suite },
  author = { Your Name },
  year = {2024},
  url = {https://github.com/Your Name/rope_long_context_evaluation_suite}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues for solutions

## Acknowledgments

This work builds upon research in long context modeling and RoPE extensions, including:

- YaRN: Efficient Context Window Extension of Large Language Models
- RULER: What's the Real Context Size of Your Long-Context Language Models?
- LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
- And many other contributions to the field

## Changelog

### v0.1.0
- Initial release
- Support for 5 RoPE extension methods
- 4 comprehensive benchmarks
- Flexible YAML configuration
- CLI interface with extensive options