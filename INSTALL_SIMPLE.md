# Simple Installation Guide

If you're having UV workspace conflicts, use this direct installation method.

## Quick Install (Recommended)

```bash
# Run direct installer (bypasses UV workspace issues)
bash ./install_direct.sh
```

This script will:
- ✅ Install all dependencies directly with pip/uv
- ✅ Set up environment and directories  
- ✅ Test the installation
- ✅ Create a run script

## Manual Installation (Alternative)

If the script fails, install manually:

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install core dependencies
pip install transformers accelerate datasets tokenizers
pip install numpy pandas matplotlib seaborn plotly
pip install pyyaml tqdm pytest scipy scikit-learn omegaconf

# 3. Optional dependencies (may fail on some systems)
pip install flash-attn --no-build-isolation  # optional
pip install bitsandbytes  # optional
pip install wandb  # optional

# 4. Set Python path
export PYTHONPATH="${PYTHONPATH}:./src"

# 5. Test installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import sys; sys.path.insert(0, './src'); import rope_long_context_evaluation_suite; print('Package works!')"
```

## Running Evaluations

### Option 1: Use the run script (created by installer)
```bash
./run_evaluation.sh --config comprehensive_config.yaml
```

### Option 2: Direct execution
```bash
PYTHONPATH=./src python run_comprehensive.py --config comprehensive_config.yaml
```

### Option 3: With environment file
```bash
# Source environment
source .env

# Run evaluation  
python run_comprehensive.py --config comprehensive_config.yaml
```

## Configuration

Update `comprehensive_config.yaml`:

```yaml
model:
  path: "unsloth/Llama-3.2-1B"  # Or your model path

datasets:
  longbench:
    path: "data/longbench"  # Or your data path
```

## Validation

Test your setup:

```bash
# Test package imports
PYTHONPATH=./src python scripts/validation/validate_imports.py

# Test CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Quick framework test  
PYTHONPATH=./src python -c "
import sys
sys.path.insert(0, './src')
from rope_long_context_evaluation_suite.core import RoPEEvaluator
print('✅ Framework ready!')
"
```

## Troubleshooting

**Import errors**: Make sure `PYTHONPATH=./src` is set

**CUDA errors**: Install CPU version: `pip install torch torchvision torchaudio`

**Flash Attention fails**: Skip it, it's optional

**Transformers version issues**: Update: `pip install transformers>=4.30.0`

**Module not found**: Use: `python -m pip install <package>` instead of `pip install`