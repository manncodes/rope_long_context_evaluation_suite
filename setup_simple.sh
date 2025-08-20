#!/bin/bash

# Simplified RoPE Long Context Evaluation Setup Script
# For use in existing project directory with UV

set -e  # Exit on any error

# Configuration
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

# Verify project structure
verify_project() {
    log "Verifying project structure..."
    
    if [[ ! -f "pyproject.toml" ]]; then
        error "pyproject.toml not found. Please run this script from the project root directory."
    fi
    
    if [[ ! -d "src" ]]; then
        error "src/ directory not found. Please run this script from the project root directory."
    fi
    
    log "Project structure verified: $(pwd)"
}

# Install dependencies with UV
install_dependencies() {
    log "Installing dependencies with UV..."
    
    # Sync existing dependencies first
    if [[ -f "uv.lock" ]] || [[ -f "pyproject.toml" ]]; then
        log "Syncing existing dependencies..."
        uv sync --dev || warn "Sync failed, continuing with fresh install"
    fi
    
    # Install PyTorch with CUDA support
    log "Installing PyTorch with CUDA..."
    uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || warn "PyTorch install may have issues"
    
    # Install transformers ecosystem
    log "Installing transformers ecosystem..."
    uv add transformers accelerate datasets tokenizers
    
    # Install Flash Attention (may fail on some systems)
    log "Installing Flash Attention (optional)..."
    uv add flash-attn --no-build-isolation || warn "Flash Attention install failed (optional dependency)"
    
    # Install data science libraries
    log "Installing data science libraries..."
    uv add numpy pandas matplotlib seaborn plotly
    
    # Install utilities
    log "Installing utilities..."
    uv add pyyaml tqdm wandb
    
    # Install testing dependencies
    log "Installing testing dependencies..."
    uv add pytest pytest-cov
    
    # Install optimization libraries (optional)
    log "Installing optimization libraries (optional)..."
    uv add bitsandbytes || warn "bitsandbytes install failed (optional)"
    uv add unsloth || warn "unsloth install failed (optional)"
    uv add scipy scikit-learn || warn "scipy/sklearn install failed"
    
    log "Dependencies installation completed"
}

# Configure environment
configure_environment() {
    log "Configuring environment..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Environment configuration for RoPE evaluation
HF_HOME=./cache/huggingface
TRANSFORMERS_CACHE=./cache/transformers
TORCH_HOME=./cache/torch

# Hardware configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Dataset paths (update these to match your setup)
LONGBENCH_DATA_PATH=./data/longbench
MODEL_CACHE_PATH=./cache/models

# Offline mode (uncomment if needed)
# HF_OFFLINE=1
# TRANSFORMERS_OFFLINE=1
# HF_DATASETS_OFFLINE=1
EOF
        log "Created .env configuration file"
    else
        log "Using existing .env configuration"
    fi
    
    # Create cache directories
    mkdir -p cache/{huggingface,transformers,torch,models} 2>/dev/null || true
    
    log "Environment configuration completed"
}

# Create missing directories
create_missing_directories() {
    log "Creating missing directories..."
    
    mkdir -p data/{longbench,retrieval,niah,ruler} 2>/dev/null || true
    mkdir -p results/{comprehensive,logs} 2>/dev/null || true
    mkdir -p configs/sweep 2>/dev/null || true
    mkdir -p cache 2>/dev/null || true
    
    log "Directory structure completed"
}

# Quick validation test
run_quick_test() {
    log "Running quick validation test..."
    
    python -c "
import sys
sys.path.insert(0, 'src')

try:
    import rope_long_context_evaluation_suite
    print('✅ Package imports successfully')
    
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️ CUDA not available (CPU mode)')
    
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    
    print('✅ All core dependencies working!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
" || error "Quick test failed"
    
    log "Quick test passed!"
}

# Main setup function
main() {
    log "Starting simplified RoPE evaluation setup..."
    
    check_uv
    verify_project
    install_dependencies
    configure_environment
    create_missing_directories
    run_quick_test
    
    log "Setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Update comprehensive_config.yaml with your model and dataset paths"
    log "2. Run: uv run scripts/validation/validate_imports.py"
    log "3. Run: python run_comprehensive.py --config comprehensive_config.yaml"
    log ""
    log "Environment activated. Use 'uv run' to execute Python scripts."
}

# Run main function
main "$@"