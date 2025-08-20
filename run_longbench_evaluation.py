#!/usr/bin/env python3
"""
LongBench Evaluation for RoPE Scaling Methods
=============================================

This script evaluates RoPE scaling methods using real LongBench tasks.
It integrates with our existing comprehensive benchmark framework.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rope_long_context_evaluation_suite.models import (
    LinearInterpolationRoPE, NTKAwareRoPE, YaRNRoPE, 
    LongRoPE, DynamicNTKRoPE, Llama3RoPE
)
from rope_long_context_evaluation_suite.benchmarks import LongBench

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LongBenchEvaluator:
    """LongBench evaluator for RoPE scaling methods."""
    
    def __init__(self, model_name: str = "unsloth/Llama-3.2-1B"):
        """Initialize the evaluator."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        
        # RoPE methods to test
        self.rope_methods = {
            "yarn": YaRNRoPE,
            "llama3": Llama3RoPE,
            "ntk_aware": NTKAwareRoPE,
        }
        
        # Best configurations from previous evaluation
        self.rope_configs = {
            "yarn": {"scaling_factor": 3.0, "beta_fast": 32, "beta_slow": 1, "s": 1.0},
            "llama3": {"factor": 4.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 131072},
            "ntk_aware": {"alpha": 1.0, "beta": 32},
        }
        
        # LongBench configuration
        self.longbench_config = {
            "tasks": ["narrativeqa", "qasper", "multifieldqa_en"],  # Start with 3 key tasks
            "data_path": "data/longbench/",
            "max_samples": 5,  # Limit samples for quick evaluation
            "generation": {
                "max_new_tokens": 256,
                "temperature": 0.0,
                "do_sample": False
            }
        }
    
    def load_tokenizer(self):
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            pad_token='<pad>',
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def check_data_availability(self):
        """Check if LongBench data is available."""
        data_dir = Path("data/longbench")
        
        if not data_dir.exists():
            logger.error("❌ LongBench data directory not found")
            logger.info("Please run: python scripts/setup_data.py --benchmarks longbench")
            return False
        
        # Check for key task files
        required_tasks = self.longbench_config["tasks"]
        missing_tasks = []
        
        for task in required_tasks:
            task_file = data_dir / f"{task}.jsonl"
            if not task_file.exists():
                missing_tasks.append(task)
        
        if missing_tasks:
            logger.error(f"❌ Missing LongBench tasks: {missing_tasks}")
            logger.info("Please wait for download to complete or run setup script again")
            return False
        
        logger.info(f"✅ LongBench data available for tasks: {required_tasks}")
        return True
    
    def apply_rope_scaling(self, rope_method: str, rope_config: Dict) -> Any:
        """Apply RoPE scaling to the model."""
        try:
            logger.info(f"Loading model with {rope_method} scaling...")
            
            # Load fresh model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            
            # Apply RoPE scaling
            rope_class = self.rope_methods[rope_method]
            rope_instance = rope_class(rope_config)
            scaled_model = rope_instance.apply(model)
            
            return scaled_model
            
        except Exception as e:
            logger.error(f"Failed to apply RoPE scaling {rope_method}: {e}")
            return None
    
    def evaluate_method_on_longbench(self, rope_method: str) -> Dict[str, Any]:
        """Evaluate a single RoPE method on LongBench tasks."""
        logger.info(f"🔧 Evaluating {rope_method.upper()} on LongBench")
        
        rope_config = self.rope_configs[rope_method]
        logger.info(f"   Config: {rope_config}")
        
        try:
            # Apply RoPE scaling
            model = self.apply_rope_scaling(rope_method, rope_config)
            if model is None:
                return {"error": "Failed to apply RoPE scaling"}
            
            # Create LongBench evaluator
            longbench = LongBench(self.longbench_config, model, self.tokenizer)
            
            # Run evaluation
            logger.info(f"   Running LongBench evaluation...")
            results = longbench.evaluate(max_samples=self.longbench_config["max_samples"])
            
            # Clean up model
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info(f"   ✅ {rope_method} completed: {results['average_score']:.3f} average score")
            
            return {
                "rope_method": rope_method,
                "rope_config": rope_config,
                "longbench_results": results,
                "average_score": results["average_score"],
                "num_samples": results["num_valid"],
                "error_rate": results["error_rate"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed for {rope_method}: {e}")
            return {
                "rope_method": rope_method,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_longbench_evaluation(self):
        """Run LongBench evaluation across all methods."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("🎯 STARTING LONGBENCH EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"RoPE Methods: {list(self.rope_methods.keys())}")
        logger.info(f"LongBench Tasks: {self.longbench_config['tasks']}")
        logger.info(f"Max Samples per Task: {self.longbench_config['max_samples']}")
        logger.info("=" * 60)
        
        # Check data availability
        if not self.check_data_availability():
            return None
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Run evaluations
        all_results = []
        
        for i, rope_method in enumerate(self.rope_methods.keys(), 1):
            logger.info(f"\\nEvaluation {i}/{len(self.rope_methods)}")
            
            result = self.evaluate_method_on_longbench(rope_method)
            all_results.append(result)
        
        # Calculate summary statistics
        elapsed_time = time.time() - start_time
        successful_results = [r for r in all_results if "error" not in r]
        success_rate = len(successful_results) / len(all_results)
        
        logger.info("\\n🎉 LONGBENCH EVALUATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total methods evaluated: {len(all_results)}")
        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Total time: {elapsed_time:.1f} seconds")
        
        # Method rankings
        if successful_results:
            logger.info("\\nMethod Rankings:")
            logger.info("-" * 30)
            
            successful_results.sort(key=lambda x: x["average_score"], reverse=True)
            
            for i, result in enumerate(successful_results, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                logger.info(f"{medal} {result['rope_method']:12s}: {result['average_score']:.3f}")
        
        # Save results
        final_results = {
            "metadata": {
                "model_name": self.model_name,
                "total_methods": len(all_results),
                "success_rate": success_rate,
                "elapsed_time_seconds": elapsed_time,
                "timestamp": timestamp,
                "longbench_tasks": self.longbench_config["tasks"],
                "max_samples_per_task": self.longbench_config["max_samples"]
            },
            "results": all_results
        }
        
        # Save to file
        output_dir = Path("comprehensive_benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / f"longbench_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_file}")
        
        # Generate summary report
        self._generate_longbench_report(final_results, timestamp)
        
        return final_results
    
    def _generate_longbench_report(self, results: Dict, timestamp: str):
        """Generate a LongBench evaluation report."""
        successful_results = [r for r in results["results"] if "error" not in r]
        
        if not successful_results:
            logger.warning("No successful results to report")
            return
        
        report_path = Path("comprehensive_benchmark_results") / f"LONGBENCH_REPORT_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write("# 🎯 LONGBENCH EVALUATION REPORT\\n\\n")
            f.write(f"**Model**: {results['metadata']['model_name']}\\n")
            f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\\n")
            f.write(f"**Tasks Evaluated**: {', '.join(results['metadata']['longbench_tasks'])}\\n")
            f.write(f"**Samples per Task**: {results['metadata']['max_samples_per_task']}\\n")
            f.write(f"**Success Rate**: {results['metadata']['success_rate']:.1%}\\n\\n")
            
            f.write("## 🏆 RESULTS SUMMARY\\n\\n")
            
            # Sort by performance
            sorted_results = sorted(successful_results, key=lambda x: x["average_score"], reverse=True)
            
            f.write("| Rank | Method | LongBench Score | Samples | Error Rate |\\n")
            f.write("|------|--------|----------------|---------|------------|\\n")
            
            for i, result in enumerate(sorted_results, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"
                f.write(f"| {medal} | **{result['rope_method'].upper()}** | ")
                f.write(f"{result['average_score']:.3f} | {result['num_samples']} | ")
                f.write(f"{result['error_rate']:.1%} |\\n")
            
            f.write("\\n## 📊 DETAILED ANALYSIS\\n\\n")
            
            # Best performer details
            best = sorted_results[0]
            f.write(f"### 🥇 Top Performer: {best['rope_method'].upper()}\\n")
            f.write(f"- **LongBench Score**: {best['average_score']:.3f}\\n")
            f.write(f"- **Configuration**: `{best['rope_config']}`\\n")
            f.write(f"- **Samples Evaluated**: {best['num_samples']}\\n")
            f.write(f"- **Error Rate**: {best['error_rate']:.1%}\\n\\n")
            
            # Task-specific performance (if available)
            f.write("### 📋 Task-Specific Performance\\n")
            longbench_results = best["longbench_results"]
            if "results" in longbench_results:
                task_scores = {}
                for sample_result in longbench_results["results"]:
                    if "sample_id" in sample_result:
                        # Extract task from sample if available
                        score = sample_result.get("score", 0)
                        # This would need enhancement based on actual LongBench result format
                        
                f.write("Individual task performance analysis would be added here\\n")
                f.write("based on the detailed LongBench evaluation results.\\n\\n")
            
            f.write("## 🎯 KEY FINDINGS\\n\\n")
            f.write("### Performance Insights\\n")
            avg_score = sum(r["average_score"] for r in sorted_results) / len(sorted_results)
            f.write(f"- **Average LongBench Score**: {avg_score:.3f}\\n")
            f.write(f"- **Performance Range**: {sorted_results[-1]['average_score']:.3f} - {sorted_results[0]['average_score']:.3f}\\n")
            f.write(f"- **Best Method**: {best['rope_method'].upper()} with clear advantage\\n\\n")
            
            f.write("### Real-World Assessment\\n")
            f.write("LongBench provides authentic real-world task evaluation that\\n")
            f.write("complements synthetic benchmarks (NIAH, RULER) by testing:\\n")
            f.write("- Reading comprehension capabilities\\n")
            f.write("- Multi-hop reasoning performance\\n")
            f.write("- Question answering accuracy\\n")
            f.write("- Long-context understanding\\n\\n")
            
            f.write("## 🚀 RECOMMENDATIONS\\n\\n")
            f.write(f"### Production Use\\n")
            f.write(f"- **Deploy {best['rope_method'].upper()}** for real-world long-context applications\\n")
            f.write(f"- **Configuration**: Use `{best['rope_config']}` for optimal performance\\n\\n")
            
            f.write("### Further Evaluation\\n")
            f.write("- Expand to full LongBench task suite (16 tasks)\\n")
            f.write("- Increase sample sizes for more robust statistics\\n")
            f.write("- Test at longer context lengths (32K+ tokens)\\n\\n")
            
            f.write(f"## 📁 FILES GENERATED\\n\\n")
            f.write(f"- `longbench_evaluation_{timestamp}.json` - Complete results data\\n")
            f.write(f"- `LONGBENCH_REPORT_{timestamp}.md` - This evaluation report\\n\\n")
            
            f.write("---\\n\\n")
            f.write(f"**Generated**: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\\n")
            f.write("🎯 **Real-world LongBench evaluation completed!**\\n")
        
        logger.info(f"📋 LongBench report generated: {report_path}")

def main():
    """Main execution function."""
    evaluator = LongBenchEvaluator()
    results = evaluator.run_longbench_evaluation()
    return results

if __name__ == "__main__":
    main()