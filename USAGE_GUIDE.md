# Usage Guide for RoPE Long Context Evaluation Suite

## Quick Start on New PC

### 1. Clone Repository
```bash
git clone https://github.com/manncodes/rope_long_context_evaluation_suite.git
cd rope_long_context_evaluation_suite
```

### 2. Validate Setup
```bash
# Test imports and configuration
python test_comprehensive_config.py

# Test Llama 3.2 1B config (without model loading)
python test_llama32_quick.py
```

### 3. Run Evaluation

**Option A: Use Fixed Comprehensive Config (Recommended)**
```bash
python run_comprehensive.py --config comprehensive_config_fixed.yaml
```

**Option B: Use Test Config for Quick Testing**
```bash
python run_comprehensive_evaluation.py --config test_llama32_1b.yaml
```

### 4. Configuration Notes

**The FIXED config format** (`comprehensive_config_fixed.yaml`) uses:
- `model.type: "hf_hub"` - Required field
- `rope_extension` - Single RoPE method configuration
- `benchmarks` - With `enabled: true/false` flags
- `data.output_dir` - Output directory

**Original config** (`comprehensive_config.yaml`) had incompatible format and will cause errors.

### 5. Model Path Configuration

Update the model path in your config file:
```yaml
model:
  path: "unsloth/Llama-3.2-1B"  # HuggingFace model name
  # OR
  path: "/exp/model/Huggingface/meta-llama/Llama-3.2-1B"  # Local path
```

### 6. Expected Output

The evaluation will:
1. Load the model and apply RoPE extension
2. Run enabled benchmarks (NIAH, RULER, etc.)
3. Save results to `comprehensive_results/` directory
4. Generate JSON results and visualizations

### 7. Troubleshooting

**If you get import errors:**
- Check that the `models/` directory exists in `src/rope_long_context_evaluation_suite/`
- Run `python test_setup.py` to validate all imports

**If you get config validation errors:**
- Use `comprehensive_config_fixed.yaml` instead of `comprehensive_config.yaml`
- Run `python test_validation_fix.py` to test config validation
- The fixed config format uses:
  - `rope_extension` (single method) instead of `rope_methods` (array)
  - `benchmarks` instead of `datasets`
  - `data.output_dir` instead of `output.base_dir`
- Ensure model path is accessible
- Check that output directory is writable

**For memory issues:**
- Reduce `model.max_length` in config
- Set `evaluation.batch_size: 1`
- Enable `evaluation.gradient_checkpointing: true`

## Available RoPE Methods

- `none` - No scaling
- `linear` - Linear interpolation  
- `ntk_aware` - NTK-aware scaling
- `yarn` - YaRN method (recommended)
- `longrope` - LongRoPE evolutionary search
- `dynamic_ntk` - Dynamic NTK scaling
- `llama3` - Llama 3 official scaling

## Example Working Command

```bash
python run_comprehensive.py --config comprehensive_config_fixed.yaml --verbose
```

This should now work without the `unexpected keyword argument 'model_path'` error!