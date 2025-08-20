# RoPE Hyperparameter Sweep System

A comprehensive system for hyperparameter optimization of RoPE (Rotary Position Embedding) extensions across different context lengths and metrics.

## Features

### 🔬 **RoPE Extension Methods**
- **Linear Interpolation**: Simple position scaling
- **NTK-Aware**: Neural Tangent Kernel scaling 
- **YaRN**: Yet another RoPE extensioN with attention temperature
- **LongRoPE**: Non-uniform scaling for extreme lengths (2M+ tokens)
- **Dynamic NTK**: Runtime adaptive scaling

### 📊 **Evaluation Metrics**
- **Perplexity**: Standard language modeling metric
- **LongPPL**: Context-aware perplexity focusing on key tokens
- **Passkey Retrieval**: Synthetic long context understanding
- **Multi-Needle Retrieval**: Advanced retrieval tasks

### 🎛️ **Sweep Configuration**
- **Grid Search**: Systematic parameter exploration
- **Random Search**: Efficient sampling-based optimization
- **Context Length Sweeping**: Performance across 2K-2M tokens
- **Early Stopping**: Automatic termination of poor performers

### ⚡ **Execution**
- **Parallel Processing**: Multi-GPU/CPU execution
- **Result Caching**: Skip duplicate experiments
- **Progress Monitoring**: Real-time sweep progress
- **Resource Management**: Automatic memory optimization

### 📈 **Analysis & Visualization**
- **Statistical Analysis**: Parameter sensitivity, method comparison
- **Contour Plots**: 2D parameter interaction visualization
- **3D Surface Plots**: Multi-dimensional relationship mapping
- **Interactive Dashboards**: Plotly-powered exploration
- **Performance Landscapes**: Context length scaling analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
pip install matplotlib seaborn plotly scikit-learn
```

### 2. Run a Quick Sweep

```bash
# Run the quick comparison example
python examples/run_sweep_example.py --config examples/sweep_configs/quick_comparison_sweep.yaml --parallel

# Or use the CLI
python -m rope_long_context_evaluation_suite.sweep_cli run examples/sweep_configs/quick_comparison_sweep.yaml --parallel
```

### 3. Analyze Results

```bash
# Analyze results from a sweep
python -m rope_long_context_evaluation_suite.sweep_cli analyze sweep_results/quick_comparison/sweep_results.json

# Generate visualizations  
python -m rope_long_context_evaluation_suite.sweep_cli visualize sweep_results/quick_comparison/sweep_results.json --contours --interactive
```

## Configuration Examples

### Basic Configuration

```yaml
model_name: "meta-llama/Llama-2-7b-hf"
model_type: "hf_local"
rope_methods: ["linear_interpolation", "yarn"]
context_lengths: [4096, 16384, 65536]
metrics: ["perplexity", "passkey_retrieval"]

parameter_grids:
  linear_interpolation:
    scaling_factor:
      values: [2, 4, 8, 16]
      distribution: "grid"
  yarn:
    s: 
      values: [8, 16, 32]
      distribution: "grid"
    alpha:
      values: [1.0, 2.0]
      distribution: "grid"
```

### Advanced Random Search

```yaml
parameter_grids:
  yarn:
    s:
      values: {min: 1.0, max: 128.0}
      distribution: "log"
      num_samples: 20
    alpha:
      values: {min: 0.1, max: 8.0}
      distribution: "log" 
      num_samples: 15
    attention_factor:
      values: {min: 0.01, max: 1.0}
      distribution: "log"
      num_samples: 10
```

## Usage Patterns

### 1. Method Comparison

Compare all RoPE methods with default parameters:

```python
from rope_long_context_evaluation_suite import SweepConfig, ParallelSweepRunner

config = SweepConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    rope_methods=["linear_interpolation", "ntk_aware", "yarn", "dynamic_ntk"],
    context_lengths=[4096, 16384, 65536],
    metrics=["perplexity", "passkey_retrieval"]
)

runner = ParallelSweepRunner(config)
results = runner.run()
```

### 2. Deep Parameter Exploration

Focus on one method with extensive parameter search:

```python
from rope_long_context_evaluation_suite.sweep import ParameterGrid, SweepParameter

# Create detailed YaRN parameter grid
yarn_params = [
    SweepParameter("s", values=list(range(4, 65, 4)), distribution="grid"),
    SweepParameter("alpha", {"min": 0.1, "max": 4.0}, "log", num_samples=15),
    SweepParameter("attention_factor", {"min": 0.01, "max": 0.5}, "log", num_samples=10)
]
yarn_grid = ParameterGrid(yarn_params)

config.parameter_grids = {"yarn": yarn_grid}
```

### 3. Context Length Analysis

Analyze performance scaling across context lengths:

```python
config = SweepConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    rope_methods=["yarn"],
    context_lengths=[2**i for i in range(12, 20)],  # 4K to 512K
    metrics=["perplexity", "passkey_retrieval"]
)
```

### 4. Results Analysis

```python
from rope_long_context_evaluation_suite import SweepAnalyzer, SweepVisualizer

# Load and analyze results
analyzer = SweepAnalyzer()
analyzer.load_results("sweep_results.json")

# Find best configurations
best_configs = analyzer.find_best_configurations("perplexity", top_k=5)
print("Top 5 configurations:", best_configs)

# Parameter sensitivity analysis
sensitivity = analyzer.analyze_parameter_sensitivity("perplexity")
print("Most sensitive parameters:", sensitivity)

# Statistical comparison between methods
comparison = analyzer.compare_methods("perplexity")
print("Method comparison:", comparison)

# Generate visualizations
visualizer = SweepVisualizer()
visualizer.load_results("sweep_results.json")

# Contour plot: alpha vs beta for YaRN
fig = visualizer.plot_contour("alpha", "beta", "perplexity", rope_method="yarn")
fig.show()

# Performance vs context length
fig = visualizer.plot_performance_vs_context_length("perplexity")
fig.show()

# Interactive dashboard
visualizer.create_interactive_dashboard("dashboard.html")
```

## Sweep Configuration Reference

### Model Configuration
```yaml
model_name: str          # Model name or HF model ID
model_type: str          # "hf_local", "hf_hub", "openai", "anthropic"  
model_path: str          # Local path (for hf_local)
```

### RoPE Methods
```yaml
rope_methods:            # List of methods to evaluate
  - "linear_interpolation"
  - "ntk_aware"
  - "yarn" 
  - "longrope"
  - "dynamic_ntk"
```

### Parameter Grid Types

**Grid Search:**
```yaml
parameter_name:
  values: [1, 2, 4, 8]
  distribution: "grid"
```

**Random Search:**
```yaml
parameter_name:
  values: {min: 0.1, max: 10.0, type: float}
  distribution: "random"
  num_samples: 20
```

**Logarithmic Sampling:**
```yaml
parameter_name:
  values: {min: 0.001, max: 1000.0}
  distribution: "log"
  num_samples: 15
```

**Linear Sampling:**
```yaml
parameter_name:
  values: {min: 0.0, max: 1.0}
  distribution: "linear"
  num_samples: 10
```

### Execution Settings
```yaml
parallel_jobs: 4                    # Number of parallel workers
max_configs_per_method: 50          # Limit experiments per method
use_cache: true                     # Enable result caching
cache_dir: "./sweep_cache"          # Cache directory
output_dir: "./sweep_results"       # Output directory

# Early stopping
early_stopping: true
early_stopping_metric: "perplexity"
early_stopping_threshold: 100.0
early_stopping_patience: 5

# Resource management  
max_gpu_memory_gb: 24.0
auto_batch_size: true
```

## Method-Specific Parameters

### Linear Interpolation
```yaml
linear_interpolation:
  scaling_factor: [2, 4, 8, 16, 32]
```

### NTK-Aware
```yaml
ntk_aware:
  alpha: [0.5, 1.0, 2.0, 4.0]       # Scaling strength
  beta: [8, 16, 32, 64]             # Target context multiplier
```

### YaRN
```yaml
yarn:
  s: [4, 8, 16, 32]                 # Scale ramp parameter
  alpha: [0.5, 1.0, 2.0]            # NTK interpolation factor
  beta: [16, 32, 64]                # Context length multiplier
  attention_factor: [0.05, 0.1, 0.2] # Attention temperature
  beta_fast: [16, 32, 64]           # Fast frequency scaling
  beta_slow: [0.5, 1.0, 2.0]        # Slow frequency scaling
```

### LongRoPE
```yaml
longrope:
  factor_strategy: ["uniform", "layer_wise", "dimension_wise"]
  short_factor: 
    - [1, 1, 1, 1, 1, 1, 1, 1]      # Uniform scaling
    - [1, 2, 4, 8, 8, 4, 2, 1]      # Pyramid scaling
  long_factor:
    - [1, 4, 8, 16, 16, 8, 4, 1]    # Long context scaling
```

### Dynamic NTK
```yaml
dynamic_ntk:
  alpha: [0.5, 1.0, 2.0, 4.0, 8.0]  # Dynamic scaling factor
```

## Best Practices

### 1. **Start Small**
- Use quick comparison configs first
- Test with smaller models before scaling up
- Validate setup with short context lengths

### 2. **Progressive Exploration**
- Begin with coarse grid search
- Focus on promising parameter regions
- Use random search for fine-tuning

### 3. **Context Length Strategy**
- Test powers of 2: [2K, 4K, 8K, 16K, 32K, 64K, 128K]
- Include original model's max context length
- Consider computational constraints

### 4. **Metric Selection**
- Always include perplexity as baseline
- Add passkey retrieval for long context validation
- Use LongPPL for research applications

### 5. **Resource Management**
- Monitor GPU memory usage
- Use caching to avoid redundant computation
- Enable early stopping for large sweeps

### 6. **Analysis Workflow**
- Generate statistical analysis reports
- Create visualization dashboards
- Compare methods systematically
- Document best configurations

## CLI Commands

### Create Configuration
```bash
python -m rope_long_context_evaluation_suite.sweep_cli create-config \
  --output config.yaml \
  --model-name meta-llama/Llama-2-7b-hf \
  --methods yarn ntk_aware \
  --context-lengths 4096 16384 65536 \
  --parallel-jobs 4
```

### Run Sweep
```bash
python -m rope_long_context_evaluation_suite.sweep_cli run config.yaml --parallel
```

### Analyze Results
```bash
python -m rope_long_context_evaluation_suite.sweep_cli analyze sweep_results.json --metric perplexity --top-k 10
```

### Generate Visualizations
```bash
python -m rope_long_context_evaluation_suite.sweep_cli visualize sweep_results.json \
  --output-dir ./plots --contours --3d --interactive
```

## Output Files

```
sweep_results/
├── sweep_results.json          # Detailed results
├── sweep_summary.json          # Summary statistics  
├── analysis_report.json        # Statistical analysis
└── visualizations/
    ├── performance_vs_context_length.png
    ├── contour_yarn_perplexity.png
    ├── 3d_surface_perplexity.png
    ├── parameter_analysis_grid.png
    └── interactive_dashboard.html
```

## Advanced Features

### Custom RoPE Extensions

```python
from rope_long_context_evaluation_suite.models import BaseRoPEExtension, register_rope_extension

class CustomRoPE(BaseRoPEExtension):
    def apply(self, model):
        # Your custom implementation
        return model
    
    def compute_rope_scaling(self, seq_len, original_max_len):
        return {"type": "custom", "factor": 2.0}
    
    def get_scaling_info(self):
        return {"method": "custom"}

# Register for use in sweeps
register_rope_extension("custom", CustomRoPE)
```

### Custom Metrics

```python
from rope_long_context_evaluation_suite.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def compute(self, model, tokenizer, text, **kwargs):
        # Your metric implementation
        return {"custom_score": 0.95}
    
    def aggregate(self, results):
        scores = [r["custom_score"] for r in results]
        return {"mean_custom_score": sum(scores) / len(scores)}
```

## Performance Tuning

### Memory Optimization
- Use gradient checkpointing
- Enable automatic batch size adjustment
- Set GPU memory limits
- Use model sharding for large models

### Speed Optimization  
- Enable result caching
- Use parallel execution
- Implement early stopping
- Limit parameter grid size

### Reproducibility
- Set random seeds
- Use deterministic algorithms
- Save configuration with results
- Version control sweep configs

## Troubleshooting

### Common Issues

**Out of Memory:**
```yaml
max_gpu_memory_gb: 16.0    # Reduce limit
auto_batch_size: true      # Enable automatic sizing
```

**Slow Execution:**
```yaml
parallel_jobs: 8           # Increase parallelization
early_stopping: true      # Enable early termination
```

**Poor Results:**
- Check parameter ranges
- Verify metric implementations
- Compare with baseline methods
- Review context length scaling

### Debug Mode
```bash
python examples/run_sweep_example.py --config config.yaml --log-level DEBUG
```

This comprehensive hyperparameter sweep system enables systematic exploration of RoPE extension methods, providing the tools needed to find optimal configurations for your specific use cases and computational constraints.