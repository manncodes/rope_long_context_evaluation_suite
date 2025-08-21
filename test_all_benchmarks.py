#!/usr/bin/env python3
"""
Comprehensive test runner for all benchmarks.
Tests NIAH, RULER, and LongBench separately.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_test_script(script_name):
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,  # Show output in real time
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        
        success = result.returncode == 0
        print(f"\n{'✅' if success else '❌'} {script_name}: {'PASSED' if success else 'FAILED'}")
        return success
        
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {script_name}: TIMEOUT")
        return False
    except Exception as e:
        print(f"\n❌ {script_name}: ERROR - {e}")
        return False

def main():
    """Main test runner."""
    print("🧪 Comprehensive Benchmark Test Suite")
    print("=" * 60)
    print("Testing all official benchmark implementations:")
    print("  • NIAH (Needle in a Haystack)")
    print("  • RULER")
    print("  • LongBench")
    print("=" * 60)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Test scripts to run
    test_scripts = [
        "test_niah.py",
        "test_ruler.py", 
        "test_longbench.py"
    ]
    
    # Run all tests
    results = {}
    start_time = time.time()
    
    for script in test_scripts:
        if not os.path.exists(script):
            print(f"❌ Test script not found: {script}")
            results[script] = False
            continue
            
        results[script] = run_test_script(script)
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("🎉 TEST SUITE SUMMARY")
    print('='*60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for script, success in results.items():
        benchmark_name = script.replace('test_', '').replace('.py', '').upper()
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {benchmark_name:<12}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Duration: {duration:.1f} seconds")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("\n📝 Next Steps:")
        print("  1. Run individual benchmarks:")
        print("     python run_comprehensive_sweep.py --config test_configs/niah_test.yaml --max-runs 1")
        print("     python run_comprehensive_sweep.py --config test_configs/ruler_test.yaml --max-runs 1")
        print("     python run_comprehensive_sweep.py --config test_configs/longbench_test.yaml --max-runs 1")
        print("\n  2. Run comprehensive sweep:")
        print("     python run_comprehensive_sweep.py --config sweep_configs/quick_test_sweep.yaml --max-runs 3")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED!")
        print("\nCheck the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)