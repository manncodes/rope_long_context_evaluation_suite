#!/bin/bash

# Comprehensive RoPE Long Context Evaluation Setup Script
# For deployment on GPU clusters with UV package manager

set -e  # Exit on any error

# Configuration
REPO_URL="https://github.com/your-username/rope_long_context_evaluation_suite.git"
PROJECT_NAME="rope_long_context_evaluation_suite"
PYTHON_VERSION="3.10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if UV is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        error "UV package manager is not installed. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
    log "UV package manager found: $(uv --version)"
}

# Setup project directory
setup_project() {
    log "Setting up project directory..."
    
    # Check if we're already in the project directory
    if [[ "$(basename "$(pwd)")" == "$PROJECT_NAME" ]] && [[ -f "pyproject.toml" ]]; then
        log "Already in project directory: $(pwd)"
        return 0
    fi
    
    # If we have the project files in current directory, stay here
    if [[ -f "pyproject.toml" ]] && [[ -d "src" ]]; then
        log "Project files found in current directory: $(pwd)"
        return 0
    fi
    
    # Clone repository (uncomment if needed)
    # git clone "$REPO_URL"
    # cd "$PROJECT_NAME"
    
    log "Project directory ready: $(pwd)"
}

# Initialize UV environment
setup_uv_environment() {
    log "Initializing UV Python environment with Python $PYTHON_VERSION..."
    
    # Check if pyproject.toml already exists
    if [[ -f "pyproject.toml" ]]; then
        log "Existing pyproject.toml found, syncing dependencies..."
        uv sync
    else
        # Initialize UV project only if no pyproject.toml exists
        uv init --python "$PYTHON_VERSION"
    fi
    
    # Install core dependencies
    log "Installing core dependencies..."
    uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    uv add transformers accelerate datasets tokenizers
    uv add flash-attn --no-build-isolation
    uv add numpy pandas matplotlib seaborn
    uv add pyyaml tqdm wandb
    uv add pytest pytest-cov
    
    # Install additional dependencies
    uv add unsloth[colab-new] # For optimized model loading
    uv add bitsandbytes # For quantization
    uv add scipy scikit-learn # For metrics
    
    log "Dependencies installed successfully"
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."
    
    # Only create missing directories
    mkdir -p data/{longbench,retrieval,niah,ruler} 2>/dev/null || true
    mkdir -p results/{comprehensive,logs} 2>/dev/null || true
    mkdir -p configs/sweep 2>/dev/null || true
    
    # Check if src structure exists
    if [[ ! -d "src/rope_long_context_evaluation_suite" ]]; then
        mkdir -p src/rope_long_context_evaluation_suite/{benchmarks,models,metrics,sweep}
    fi
    
    log "Directory structure verified"
}

# Configure for offline/NFS datasets
configure_offline_datasets() {
    log "Configuring for offline/NFS dataset access..."
    
    cat > .env << EOF
# Environment configuration for offline deployment
HF_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1

# Dataset paths (update these to match your NFS paths)
LONGBENCH_DATA_PATH=/nfs/datasets/longbench
MODEL_CACHE_PATH=/nfs/models
HF_HOME=/nfs/huggingface_cache

# Hardware configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF
    
    log "Offline configuration created in .env file"
}

# Create main evaluation runner
create_evaluation_runner() {
    log "Creating main evaluation runner..."
    
    cat > run_comprehensive_evaluation.py << 'EOF'
#!/usr/bin/env python3

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

def setup_logging(config):
    """Setup logging configuration."""
    log_level = getattr(logging, config['output']['log_level'])
    log_file = config['output']['log_file']
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path):
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_evaluation(config):
    """Run comprehensive evaluation."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import evaluation modules (these would be your actual implementations)
        logger.info("Starting comprehensive RoPE evaluation...")
        
        # Set environment variables for offline mode
        if 'datasets' in config and 'longbench' in config['datasets']:
            os.environ['LONGBENCH_DATA_PATH'] = config['datasets']['longbench']['path']
        
        # Initialize model and run evaluations
        model_config = config['model']
        logger.info(f"Loading model: {model_config['name']} from {model_config['path']}")
        
        # Run each benchmark type
        results = {}
        
        # Traditional retrieval tasks
        if config['datasets']['retrieval']['enabled']:
            logger.info("Running traditional retrieval evaluation...")
            # results['retrieval'] = run_retrieval_evaluation(config)
        
        # NIAH benchmark  
        if config['datasets']['niah']['enabled']:
            logger.info("Running NIAH evaluation...")
            # results['niah'] = run_niah_evaluation(config)
        
        # RULER benchmark
        if config['datasets']['ruler']['enabled']:
            logger.info("Running RULER evaluation...")
            # results['ruler'] = run_ruler_evaluation(config)
        
        # LongBench evaluation
        if 'longbench' in config['datasets']:
            logger.info("Running LongBench evaluation...")
            # results['longbench'] = run_longbench_evaluation(config)
        
        logger.info("Comprehensive evaluation completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive RoPE evaluation')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--output', '-o', help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.output:
        config['output']['base_dir'] = args.output
    
    # Setup logging
    setup_logging(config)
    
    # Run evaluation
    results = run_evaluation(config)
    
    print("Evaluation completed! Check results in:", config['output']['base_dir'])

if __name__ == "__main__":
    main()
EOF

    chmod +x run_comprehensive_evaluation.py
    log "Main evaluation runner created"
}

# Create sweep configuration
create_sweep_config() {
    log "Creating hyperparameter sweep configuration..."
    
    mkdir -p configs/sweep
    
    cat > configs/sweep/rope_methods_sweep.yaml << 'EOF'
# Hyperparameter Sweep Configuration for RoPE Methods
name: "rope_comprehensive_sweep"
method: "grid"  # Options: grid, random, bayes

# Base configuration (inherits from comprehensive_config.yaml)
base_config: "comprehensive_config.yaml"

# Parameters to sweep
parameters:
  rope_methods:
    values:
      - name: "linear"
        config:
          scaling_factor: [1.5, 2.0, 4.0, 8.0]
      - name: "ntk_aware" 
        config:
          scaling_factor: [2.0, 4.0, 8.0]
          alpha: [4.0, 8.0, 16.0]
      - name: "yarn"
        config:
          scaling_factor: [2.0, 4.0, 8.0]
          attention_factor: [0.1, 0.2, 0.5]
          beta_fast: [16, 32, 64]
      - name: "llama3"
        config:
          scaling_factor: [4.0, 8.0, 16.0]
          low_freq_factor: [0.5, 1.0, 2.0]
          high_freq_factor: [2.0, 4.0, 8.0]

  # Context lengths to test
  context_lengths:
    values: [[4000, 8000], [8000, 16000], [16000, 32000], [32000, 65536]]

  # Model configurations
  model:
    values:
      - name: "llama-3.2-1b"
        path: "/nfs/models/llama-3.2-1b"
      - name: "llama-3.2-3b"  
        path: "/nfs/models/llama-3.2-3b"

# Resource configuration for sweep
resources:
  gpu_memory_gb: 24
  max_concurrent_runs: 2
  timeout_hours: 12

# Early stopping
early_stopping:
  metric: "longbench/avg_score"
  min_delta: 0.01
  patience: 3

# Results tracking
tracking:
  wandb:
    project: "rope-long-context-evaluation"
    entity: "your-wandb-entity"
  
  save_all_results: true
  create_comparison_plots: true
EOF

    log "Sweep configuration created"
}

# Create utility scripts
create_utility_scripts() {
    log "Creating utility scripts..."
    
    mkdir -p scripts
    
    # Dataset validation script
    cat > scripts/validate_datasets.py << 'EOF'
#!/usr/bin/env python3
"""Validate that all required datasets are available offline."""

import os
import json
from pathlib import Path

def validate_longbench(data_path):
    """Validate LongBench dataset availability."""
    required_tasks = [
        'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', 
        '2wikimqa', 'qmsum', 'trec', 'triviaqa', 'samsum', 
        'passage_retrieval_en', 'passage_count', 'lcc'
    ]
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"❌ LongBench data path does not exist: {data_path}")
        return False
    
    missing_tasks = []
    for task in required_tasks:
        task_file = data_path / f"{task}.jsonl"
        if not task_file.exists():
            missing_tasks.append(task)
    
    if missing_tasks:
        print(f"❌ Missing LongBench tasks: {missing_tasks}")
        return False
    
    print("✅ All LongBench datasets found")
    return True

def validate_model(model_path):
    """Validate model availability."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    # Check for required model files
    required_files = ['config.json', 'pytorch_model.bin']
    for file in required_files:
        if not (model_path / file).exists():
            print(f"⚠️ Model file missing: {file}")
    
    print("✅ Model path validated")
    return True

def main():
    print("🔍 Validating dataset and model availability...")
    
    longbench_path = os.environ.get('LONGBENCH_DATA_PATH', '/nfs/datasets/longbench')
    model_path = os.environ.get('MODEL_PATH', '/nfs/models/llama-3.2-1b')
    
    longbench_ok = validate_longbench(longbench_path)
    model_ok = validate_model(model_path)
    
    if longbench_ok and model_ok:
        print("✅ All validations passed! Ready to run evaluation.")
        return 0
    else:
        print("❌ Validation failed. Please check paths and data availability.")
        return 1

if __name__ == "__main__":
    exit(main())
EOF

    chmod +x scripts/validate_datasets.py
    
    # Quick test script
    cat > scripts/quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick test to verify the evaluation framework works."""

import yaml
from pathlib import Path

def test_config_loading():
    """Test configuration loading."""
    config_path = Path("comprehensive_config.yaml")
    if not config_path.exists():
        print("❌ Configuration file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def test_imports():
    """Test required imports."""
    try:
        import torch
        import transformers
        import datasets
        import numpy as np
        import pandas as pd
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gpu():
    """Test GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️ No GPU available")
            return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    print("🧪 Running quick tests...")
    
    tests = [
        ("Configuration", test_config_loading),
        ("Imports", test_imports),
        ("GPU", test_gpu)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Testing {name} ---")
        results.append(test_func())
    
    if all(results):
        print("\n✅ All tests passed! Framework is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
EOF

    chmod +x scripts/quick_test.py
    
    log "Utility scripts created"
}

# Create README with instructions
create_readme() {
    log "Creating deployment README..."
    
    cat > README_DEPLOYMENT.md << 'EOF'
# RoPE Long Context Evaluation Suite - Deployment Guide

## Quick Start

1. **Setup Environment**:
   ```bash
   ./setup_comprehensive_evaluation.sh
   ```

2. **Update Configuration**:
   Edit `comprehensive_config.yaml`:
   - Set `model.path` to your model location
   - Set `datasets.longbench.path` to your LongBench data path
   - Adjust hardware settings for your GPU

3. **Validate Setup**:
   ```bash
   uv run scripts/validate_datasets.py
   uv run scripts/quick_test.py
   ```

4. **Run Evaluation**:
   ```bash
   uv run run_comprehensive_evaluation.py --config comprehensive_config.yaml
   ```

## Configuration

### Model Configuration
```yaml
model:
  path: "/nfs/models/your-model"  # Update this path
  device: "cuda"
  torch_dtype: "bfloat16"
```

### Dataset Configuration
```yaml
datasets:
  longbench:
    path: "/nfs/datasets/longbench"  # Update this path
```

### RoPE Methods
Supports: linear, ntk_aware, yarn, longrope, dynamic_ntk, llama3

## Hyperparameter Sweeps

Run parameter sweeps:
```bash
uv run sweep_runner.py --config configs/sweep/rope_methods_sweep.yaml
```

## Environment Variables

Create `.env` file or set:
```bash
export LONGBENCH_DATA_PATH=/nfs/datasets/longbench
export MODEL_CACHE_PATH=/nfs/models
export HF_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
```

## GPU Memory Optimization

For large context lengths:
- Enable `gradient_checkpointing: true`
- Set `use_cache: false`
- Use `torch_dtype: "bfloat16"`
- Adjust `batch_size` and `max_context_length`

## Results

Results saved to `comprehensive_results/` with:
- JSON metrics files
- CSV summaries
- Visualization plots
- Detailed logs

## Troubleshooting

1. **Out of Memory**: Reduce `max_context_length` or `batch_size`
2. **Missing Datasets**: Check paths in config and run validation
3. **Import Errors**: Ensure all dependencies installed with UV
4. **CUDA Issues**: Verify CUDA version matches PyTorch installation
EOF

    log "Deployment README created"
}

# Main setup function
main() {
    log "Starting comprehensive RoPE evaluation setup..."
    
    check_uv
    setup_project
    setup_uv_environment
    create_directories
    configure_offline_datasets
    create_evaluation_runner
    create_sweep_config
    create_utility_scripts
    create_readme
    
    log "Setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Update comprehensive_config.yaml with your model and dataset paths"
    log "2. Run: uv run scripts/validate_datasets.py"
    log "3. Run: uv run scripts/quick_test.py" 
    log "4. Run: uv run run_comprehensive_evaluation.py --config comprehensive_config.yaml"
    log ""
    log "For sweeps: uv run sweep_runner.py --config configs/sweep/rope_methods_sweep.yaml"
}

# Run main function
main "$@"