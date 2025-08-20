#!/usr/bin/env python3
"""Quick test for Llama 3.2 1B configuration without actual model loading."""

import sys
sys.path.append('src')

import yaml
from omegaconf import OmegaConf
from rope_long_context_evaluation_suite.core import RoPEEvaluator
from rope_long_context_evaluation_suite.models import get_rope_extension


def main():
    print("🦙 Testing Llama 3.2 1B Configuration")
    print("="*50)
    
    # Load configuration
    with open("test_llama32_1b.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config_obj = OmegaConf.create(config)
    print(f"✅ Configuration loaded")
    print(f"   Model: {config_obj.model.name}")
    print(f"   RoPE Method: {config_obj.rope_extension.method}")
    print(f"   Context Lengths: {config_obj.benchmarks.niah.context_lengths}")
    
    # Test RoPE extension creation
    rope_method = config_obj.rope_extension.method
    rope_config = config_obj.rope_extension[rope_method]
    
    rope_ext = get_rope_extension(rope_method, dict(rope_config))
    print(f"✅ RoPE extension created: {type(rope_ext).__name__}")
    print(f"   Scaling Factor: {rope_config.scaling_factor}")
    
    # Test evaluator initialization (without model loading)
    evaluator = RoPEEvaluator(config_obj)
    print(f"✅ RoPEEvaluator initialized")
    
    model_info = evaluator.model_loader.get_model_info()
    print(f"   Model Type: {model_info['type']}")
    print(f"   Max Length: {model_info['max_length']}")
    print(f"   Device: {model_info['device']}")
    
    print("\n🎉 All tests passed! The setup is ready for Llama 3.2 1B evaluation.")
    print("\nTo run actual evaluation (requires GPU and model download):")
    print("python run_comprehensive_evaluation.py --config test_llama32_1b.yaml")


if __name__ == "__main__":
    main()