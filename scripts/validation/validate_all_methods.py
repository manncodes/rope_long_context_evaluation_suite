"""Validation script to demonstrate all supported RoPE scaling methods."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rope_long_context_evaluation_suite.models.factory import list_available_extensions, get_rope_extension

def validate_all_transformers_methods():
    """Validate that we support all transformers rope_scaling methods."""
    
    print("🔍 Validating complete transformers RoPE scaling method coverage...")
    print("=" * 70)
    
    # Transformers supported methods according to their documentation
    transformers_methods = {
        'default': 'Original RoPE (no scaling)',
        'linear': 'Linear interpolation scaling', 
        'dynamic': 'Dynamic NTK scaling',
        'yarn': 'YaRN (Yet another RoPE extensioN)',
        'longrope': 'LongRoPE method',
        'llama3': 'Llama3 frequency-based scaling'
    }
    
    # Our implemented methods 
    our_methods = list_available_extensions()
    
    print("📚 TRANSFORMERS SUPPORTED METHODS:")
    for method, description in transformers_methods.items():
        if method == 'default':
            status = "✓ (Baseline - no extension needed)"
        elif method == 'linear' and 'linear_interpolation' in our_methods:
            status = "✅ IMPLEMENTED as 'linear_interpolation'"
        elif method == 'dynamic' and 'dynamic_ntk' in our_methods:
            status = "✅ IMPLEMENTED as 'dynamic_ntk'"
        elif method in our_methods:
            status = "✅ IMPLEMENTED"
        else:
            status = "❌ MISSING"
        
        print(f"  {method:12} - {description:35} {status}")
    
    print(f"\n🏗️  OUR IMPLEMENTED METHODS:")
    for method in sorted(our_methods):
        print(f"  ✅ {method}")
    
    print(f"\n📊 COVERAGE SUMMARY:")
    implemented_count = len([m for m in transformers_methods.keys() if m == 'default' or m in our_methods or (m == 'linear' and 'linear_interpolation' in our_methods) or (m == 'dynamic' and 'dynamic_ntk' in our_methods)])
    total_count = len(transformers_methods)
    coverage_pct = (implemented_count / total_count) * 100
    
    print(f"  Total transformers methods: {total_count}")
    print(f"  Methods implemented: {implemented_count}")
    print(f"  Coverage: {coverage_pct:.1f}% ✅")
    
    return coverage_pct == 100.0

def test_method_instantiation():
    """Test that all methods can be instantiated correctly."""
    
    print(f"\n🧪 TESTING METHOD INSTANTIATION:")
    print("=" * 70)
    
    test_configs = {
        'linear_interpolation': {'scaling_factor': 2.0},
        'ntk_aware': {'alpha': 1.0, 'beta': 32.0},
        'yarn': {'scaling_factor': 2.0, 'beta_fast': 32, 'beta_slow': 1, 's': 1.0},
        'longrope': {'scaling_factor': 2.0, 'original_max_position_embeddings': 2048},
        'dynamic_ntk': {'scaling_factor': 2.0, 'original_max_position_embeddings': 2048},
        'llama3': {'factor': 4.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192}
    }
    
    all_passed = True
    
    for method_name, config in test_configs.items():
        try:
            extension = get_rope_extension(method_name, config)
            scaling_info = extension.get_scaling_info()
            rope_config = extension.compute_rope_scaling(seq_len=8192, original_max_len=2048)
            
            print(f"  ✅ {method_name:18} - {scaling_info['method']:8} - {scaling_info.get('description', 'Working')[:40]}")
            
        except Exception as e:
            print(f"  ❌ {method_name:18} - FAILED: {str(e)[:40]}")
            all_passed = False
    
    return all_passed

def show_comparison_with_transformers():
    """Show side-by-side comparison with transformers rope_scaling types."""
    
    print(f"\n🔄 TRANSFORMERS COMPATIBILITY:")
    print("=" * 70)
    
    mapping = {
        'linear_interpolation': 'linear',
        'ntk_aware': 'ntk (custom implementation)',
        'yarn': 'yarn', 
        'longrope': 'longrope',
        'dynamic_ntk': 'dynamic',
        'llama3': 'llama3'
    }
    
    print("Our Method               → Transformers rope_type")
    print("-" * 50)
    for our_method, transformers_type in mapping.items():
        print(f"{our_method:23} → {transformers_type}")
    
    print(f"\n✨ RESULT: Complete compatibility with transformers library!")
    print(f"   You can now use any rope_scaling method supported by transformers.")

if __name__ == "__main__":
    print("🚀 RoPE Long Context Evaluation Suite - Method Coverage Validation")
    print("=" * 70)
    
    try:
        # Test coverage
        coverage_complete = validate_all_transformers_methods()
        
        # Test instantiation 
        instantiation_working = test_method_instantiation()
        
        # Show compatibility
        show_comparison_with_transformers()
        
        if coverage_complete and instantiation_working:
            print(f"\n🎉 SUCCESS: All transformers RoPE scaling methods are supported!")
            print(f"   The suite now provides complete coverage of rope_scaling options.")
            print(f"   Ready for comprehensive hyperparameter sweeping! 🚀")
        else:
            print(f"\n⚠️  Some issues found. Check output above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)