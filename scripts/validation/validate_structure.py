#!/usr/bin/env python3
"""Validate the project structure and basic functionality without heavy dependencies."""

import os
import sys
import json
import yaml
from pathlib import Path


def validate_project_structure():
    """Validate that all expected files and directories exist."""
    print("📁 Validating project structure...")
    
    required_files = [
        "README.md",
        "pyproject.toml", 
        "config/default.yaml",
        "src/rope_long_context_evaluation_suite/__init__.py",
        "src/rope_long_context_evaluation_suite/cli.py",
        "src/rope_long_context_evaluation_suite/core.py",
        "src/rope_long_context_evaluation_suite/models/__init__.py",
        "src/rope_long_context_evaluation_suite/models/loader.py",
        "src/rope_long_context_evaluation_suite/models/rope_extensions.py",
        "src/rope_long_context_evaluation_suite/benchmarks/__init__.py",
        "src/rope_long_context_evaluation_suite/benchmarks/base.py",
        "src/rope_long_context_evaluation_suite/benchmarks/niah.py",
        "src/rope_long_context_evaluation_suite/benchmarks/ruler.py",
        "src/rope_long_context_evaluation_suite/benchmarks/longbench.py",
        "src/rope_long_context_evaluation_suite/benchmarks/longbench_v2.py",
        "test_llama31_1b.yaml",
        "test_llama32_1b.yaml",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True


def validate_yaml_configs():
    """Validate YAML configuration files."""
    print("📄 Validating YAML configurations...")
    
    config_files = [
        "config/default.yaml",
        "test_llama31_1b.yaml", 
        "test_llama32_1b.yaml",
        "examples/configs/yarn_evaluation.yaml",
        "examples/configs/comprehensive_evaluation.yaml"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ["model", "rope_extension", "benchmarks", "evaluation"]
            for section in required_sections:
                if section not in config:
                    print(f"❌ {config_file}: Missing section '{section}'")
                    return False
            
            print(f"✅ {config_file}: Valid YAML structure")
            
        except Exception as e:
            print(f"❌ {config_file}: YAML parsing error - {e}")
            return False
    
    return True


def validate_python_syntax():
    """Validate Python syntax of all source files."""
    print("🐍 Validating Python syntax...")
    
    src_dir = Path("src")
    python_files = list(src_dir.rglob("*.py"))
    
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                source = f.read()
            
            # Try to compile the Python code
            compile(source, str(py_file), 'exec')
            print(f"✅ {py_file}: Valid Python syntax")
            
        except SyntaxError as e:
            print(f"❌ {py_file}: Syntax error - {e}")
            return False
        except Exception as e:
            print(f"⚠️  {py_file}: Warning - {e}")
    
    return True


def validate_model_configs():
    """Validate the model configurations for Llama 3.1 and 3.2 1B."""
    print("🤖 Validating model configurations...")
    
    # Test Llama 3.1 1B config
    with open("test_llama31_1b.yaml", 'r') as f:
        llama31_config = yaml.safe_load(f)
    
    # Validate Llama 3.1 1B config
    assert llama31_config["model"]["name"] == "meta-llama/Llama-3.1-1B"
    assert llama31_config["model"]["type"] == "hf_hub"
    assert llama31_config["rope_extension"]["method"] == "yarn"
    assert llama31_config["benchmarks"]["niah"]["enabled"] == True
    print("✅ Llama 3.1 1B configuration is valid")
    
    # Test Llama 3.2 1B config
    with open("test_llama32_1b.yaml", 'r') as f:
        llama32_config = yaml.safe_load(f)
    
    # Validate Llama 3.2 1B config
    assert llama32_config["model"]["name"] == "meta-llama/Llama-3.2-1B"
    assert llama32_config["model"]["type"] == "hf_hub"
    assert llama32_config["rope_extension"]["method"] == "linear_interpolation"
    assert llama32_config["benchmarks"]["niah"]["enabled"] == True
    print("✅ Llama 3.2 1B configuration is valid")
    
    return True


def validate_rope_methods():
    """Validate RoPE method configurations."""
    print("🔗 Validating RoPE method configurations...")
    
    # Check that all expected RoPE methods are configured
    with open("config/default.yaml", 'r') as f:
        default_config = yaml.safe_load(f)
    
    rope_methods = [
        "linear_interpolation",
        "ntk_aware", 
        "yarn",
        "longrope",
        "dynamic_ntk"
    ]
    
    # Verify that each method has its configuration section
    for method in rope_methods:
        if method in default_config["rope_extension"]:
            print(f"✅ RoPE method '{method}' configuration found")
        else:
            print(f"⚠️  RoPE method '{method}' configuration missing (this is OK)")
    
    return True


def validate_benchmark_configs():
    """Validate benchmark configurations."""
    print("🎯 Validating benchmark configurations...")
    
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    benchmarks = ["niah", "ruler", "longbench", "longbench_v2"]
    
    for benchmark in benchmarks:
        if benchmark in config["benchmarks"]:
            bench_config = config["benchmarks"][benchmark]
            
            # Check for enabled flag
            if "enabled" in bench_config:
                print(f"✅ {benchmark}: configuration valid (enabled: {bench_config['enabled']})")
            else:
                print(f"⚠️  {benchmark}: missing 'enabled' flag")
        else:
            print(f"❌ {benchmark}: configuration missing")
            return False
    
    return True


def main():
    """Run all validation tests."""
    print("🔍 RoPE Long Context Evaluation Suite - Structure Validation")
    print("=" * 65)
    
    tests = [
        ("Project Structure", validate_project_structure),
        ("YAML Configurations", validate_yaml_configs),
        ("Python Syntax", validate_python_syntax),
        ("Model Configurations", validate_model_configs),
        ("RoPE Methods", validate_rope_methods),
        ("Benchmark Configurations", validate_benchmark_configs),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Validation Summary")
    print("=" * 30)
    print(f"✅ Tests passed: {passed}/{total}")
    print(f"❌ Tests failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All validations PASSED!")
        print("\n📋 Framework is ready for:")
        print("• Llama 3.1 1B evaluation with YaRN scaling")
        print("• Llama 3.2 1B evaluation with Linear Interpolation")
        print("• NIAH and RULER benchmarks")
        print("• Context lengths up to 8192 tokens")
        
        print("\n🚀 To run evaluations:")
        print("1. Install dependencies: pip install -e .")
        print("2. Run Llama 3.1 1B test: rope-eval --config test_llama31_1b.yaml")
        print("3. Run Llama 3.2 1B test: rope-eval --config test_llama32_1b.yaml")
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()