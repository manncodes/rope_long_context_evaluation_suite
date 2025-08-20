#!/bin/bash

# Direct Installation Script for RoPE Evaluation Suite
# Bypasses UV workspace conflicts by using direct dependency installation

set -e  # Exit on any error

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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
    fi
    
    if ! command -v pip &> /dev/null && ! command -v uv &> /dev/null; then
        error "Neither pip nor uv package manager found"
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    log "Python version: $python_version"
    
    if command -v uv &> /dev/null; then
        log "UV found: $(uv --version)"
    else
        log "Using pip for installation"
    fi
}

# Install dependencies directly
install_dependencies() {
    log "Installing dependencies directly..."
    
    # Create a temporary requirements file
    cat > temp_requirements.txt << EOF
torch>=2.0.0
torchvision
torchaudio
transformers>=4.30.0
accelerate
datasets
tokenizers
numpy
pandas
matplotlib
seaborn
plotly
pyyaml
tqdm
pytest
pytest-cov
scipy
scikit-learn
omegaconf
EOF

    # Try UV first, fall back to pip
    if command -v uv &> /dev/null; then
        log "Installing with UV (bypassing workspace)..."
        
        # Install PyTorch with CUDA
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --system || \
        warn "CUDA PyTorch install failed, trying CPU version"
        
        # Install other dependencies
        uv pip install -r temp_requirements.txt --system || error "UV pip install failed"
        
        # Optional dependencies (may fail)
        uv pip install flash-attn --no-build-isolation --system 2>/dev/null || warn "Flash Attention skipped (optional)"
        uv pip install bitsandbytes --system 2>/dev/null || warn "bitsandbytes skipped (optional)"
        uv pip install unsloth --system 2>/dev/null || warn "unsloth skipped (optional)"
        uv pip install wandb --system 2>/dev/null || warn "wandb skipped (optional)"
        
    else
        log "Installing with pip..."
        
        # Install PyTorch with CUDA
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
        warn "CUDA PyTorch install failed, trying CPU version"
        
        # Install other dependencies
        pip install -r temp_requirements.txt || error "Pip install failed"
        
        # Optional dependencies (may fail)
        pip install flash-attn --no-build-isolation 2>/dev/null || warn "Flash Attention skipped (optional)"
        pip install bitsandbytes 2>/dev/null || warn "bitsandbytes skipped (optional)"
        pip install unsloth 2>/dev/null || warn "unsloth skipped (optional)"
        pip install wandb 2>/dev/null || warn "wandb skipped (optional)"
    fi
    
    # Clean up
    rm temp_requirements.txt
    
    log "Dependencies installation completed"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create directories
    mkdir -p data/{longbench,retrieval,niah,ruler} 2>/dev/null || true
    mkdir -p results/{comprehensive,logs} 2>/dev/null || true
    mkdir -p cache/{huggingface,transformers,torch} 2>/dev/null || true
    
    # Create environment file
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# RoPE Evaluation Environment Configuration
PYTHONPATH=\${PYTHONPATH}:./src
HF_HOME=./cache/huggingface
TRANSFORMERS_CACHE=./cache/transformers
TORCH_HOME=./cache/torch
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Dataset paths (update for your setup)
LONGBENCH_DATA_PATH=./data/longbench
MODEL_CACHE_PATH=./cache/models

# Uncomment for offline mode
# HF_OFFLINE=1
# TRANSFORMERS_OFFLINE=1
EOF
        log "Created .env file"
    fi
    
    log "Environment setup completed"
}

# Test installation
test_installation() {
    log "Testing installation..."
    
    # Set PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:./src"
    
    python3 -c "
import sys
sys.path.insert(0, './src')

try:
    # Test basic imports
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'✅ CUDA: {torch.cuda.get_device_name(0)}')
        print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('⚠️ CUDA not available (using CPU)')
    
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    
    import numpy as np
    import pandas as pd
    import matplotlib
    print('✅ Data science libraries working')
    
    # Test package imports
    try:
        import rope_long_context_evaluation_suite
        print('✅ Main package imports successfully')
        
        from rope_long_context_evaluation_suite.core import RoPEEvaluator
        print('✅ Core evaluator available')
        
        from rope_long_context_evaluation_suite.models.factory import get_available_rope_methods
        methods = get_available_rope_methods()
        print(f'✅ RoPE methods: {methods}')
        
    except ImportError as e:
        print(f'⚠️ Package import issue: {e}')
        print('   This is normal if package is not installed in editable mode')
    
    print('\\n🎉 Installation test completed successfully!')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || error "Installation test failed"
    
    log "Installation test passed!"
}

# Create run script
create_run_script() {
    log "Creating run script..."
    
    cat > run_evaluation.sh << 'EOF'
#!/bin/bash

# RoPE Evaluation Runner Script
# Sets up environment and runs evaluation

# Set Python path
export PYTHONPATH="${PYTHONPATH}:./src"

# Load environment variables
if [[ -f ".env" ]]; then
    source .env
fi

# Run the evaluation
python3 run_comprehensive.py "$@"
EOF
    
    chmod +x run_evaluation.sh
    
    log "Created run_evaluation.sh script"
}

# Main installation function
main() {
    log "Starting direct installation for RoPE Evaluation Suite..."
    
    check_prerequisites
    install_dependencies
    setup_environment
    test_installation
    create_run_script
    
    log "Installation completed successfully!"
    log ""
    log "🎉 Ready to use!"
    log ""
    log "Next steps:"
    log "1. Update comprehensive_config.yaml with your paths:"
    log "   - model.path: 'your-model-path'"
    log "   - datasets.longbench.path: 'your-data-path'"
    log ""
    log "2. Run evaluation:"
    log "   ./run_evaluation.sh --config comprehensive_config.yaml"
    log ""
    log "3. Or run directly:"
    log "   PYTHONPATH=./src python3 run_comprehensive.py --config comprehensive_config.yaml"
    log ""
    log "4. Test specific components:"
    log "   PYTHONPATH=./src python3 scripts/validation/validate_imports.py"
}

# Run main function
main "$@"