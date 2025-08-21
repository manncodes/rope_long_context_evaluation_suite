#!/bin/bash
# Setup script for RoPE Long Context Evaluation Suite
# Initializes git submodules and installs dependencies

set -e

echo "🚀 Setting up RoPE Long Context Evaluation Suite..."

# Initialize and update git submodules
echo "📦 Initializing official benchmark repositories..."
git submodule update --init --recursive

# Verify submodules were cloned
echo "✅ Verifying benchmark repositories..."
if [ -d "third_party/LLMTest_NeedleInAHaystack/needlehaystack" ]; then
    echo "  ✓ NIAH (Needle-in-Haystack) - Official implementation by Greg Kamradt"
else
    echo "  ❌ NIAH repository not found"
    exit 1
fi

if [ -d "third_party/RULER/scripts" ]; then
    echo "  ✓ RULER - Official implementation by NVIDIA"
else
    echo "  ❌ RULER repository not found"
    exit 1
fi

if [ -d "third_party/LongBench" ]; then
    echo "  ✓ LongBench - Official implementation by THUDM"
else
    echo "  ❌ LongBench repository not found"
    exit 1
fi

# Install dependencies
echo "📋 Installing dependencies..."

# Check for pip vs pip3
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "❌ Neither pip nor pip3 found. Please install pip."
    exit 1
fi

# Install based on user choice
echo "Choose installation type:"
echo "1. Basic (local models only)"
echo "2. With benchmarks (recommended)"  
echo "3. With API providers (OpenAI, Anthropic, etc.)"
echo "4. Full installation"
read -p "Enter choice (1-4) [default: 2]: " choice
choice=${choice:-2}

case $choice in
    1)
        echo "Installing basic dependencies..."
        $PIP_CMD install -e .
        ;;
    2)
        echo "Installing with official benchmarks..."
        $PIP_CMD install -e .[benchmarks]
        ;;
    3)
        echo "Installing with API providers..."
        $PIP_CMD install -e .[api-providers]
        ;;
    4)
        echo "Installing full dependencies..."
        $PIP_CMD install -e .[benchmarks,api-providers]
        ;;
    *)
        echo "Invalid choice. Installing with benchmarks (default)..."
        $PIP_CMD install -e .[benchmarks]
        ;;
esac

# Test installation
echo "🧪 Testing installation..."
python3 -c "from src.rope_long_context_evaluation_suite.benchmarks import NIAHOfficialBenchmark; print('✅ NIAH import successful')" 2>/dev/null || echo "⚠️ NIAH import failed (dependencies may be missing)"
python3 -c "from src.rope_long_context_evaluation_suite.benchmarks import RULEROfficialBenchmark; print('✅ RULER import successful')" 2>/dev/null || echo "⚠️ RULER import failed (dependencies may be missing)"
python3 -c "from src.rope_long_context_evaluation_suite.benchmarks import LongBenchOfficialBenchmark; print('✅ LongBench import successful')" 2>/dev/null || echo "⚠️ LongBench import failed (dependencies may be missing)"

echo ""
echo "🎉 Setup complete! "
echo ""
echo "📚 Quick start:"
echo "  # Test with quick sweep (6 runs)"
echo "  python run_comprehensive_sweep.py --config sweep_configs/quick_test_sweep.yaml --max-runs 3"
echo ""
echo "  # Single evaluation"  
echo "  python run_evaluation.py --model meta-llama/Llama-2-7b-hf --benchmarks niah"
echo ""
echo "📖 See INSTALL.md for detailed usage instructions"