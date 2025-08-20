#!/usr/bin/env python3
"""Test script to validate the run_comprehensive.py validation fix."""

import sys
import os
sys.path.append('src')

import yaml
from pathlib import Path

# Import the validation function from run_comprehensive.py
sys.path.insert(0, str(Path(__file__).parent))
from run_comprehensive import validate_config


def test_validation_with_fixed_config():
    """Test validation with the fixed comprehensive config."""
    print("🧪 Testing run_comprehensive.py validation with fixed config...")
    
    try:
        # Load the fixed config
        with open("comprehensive_config_fixed.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded: {config['model']['name']}")
        
        # Test validation
        validate_config(config)
        print("✅ Validation passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_with_missing_fields():
    """Test validation with missing required fields."""
    print("\n🧪 Testing validation with missing fields...")
    
    try:
        # Create config with missing field
        incomplete_config = {
            "model": {"type": "hf_hub", "path": "test"},
            "rope_extension": {"method": "yarn"},
            # Missing: benchmarks, evaluation, data
        }
        
        validate_config(incomplete_config)
        print("❌ Validation should have failed but didn't!")
        return False
        
    except ValueError as e:
        if "Missing required config field" in str(e):
            print(f"✅ Validation correctly failed: {e}")
            return True
        else:
            print(f"❌ Wrong validation error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Run validation tests."""
    print("🧩 Testing run_comprehensive.py Validation Fix\n")
    
    tests = [
        ("Fixed Config Validation", test_validation_with_fixed_config),
        ("Missing Fields Detection", test_validation_with_missing_fields)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Testing: {test_name}")
        print(f"{'='*50}")
        
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("\nYou can now run:")
        print("python run_comprehensive.py --config comprehensive_config_fixed.yaml")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())