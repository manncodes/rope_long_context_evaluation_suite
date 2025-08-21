"""Core evaluation framework for RoPE long context evaluation."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from .benchmarks import NIAHBenchmark, RULERBenchmark, LongBench, LongBenchV2
from .models import ModelLoader, get_rope_extension
from .utils import Config, save_results, setup_logging

logger = logging.getLogger(__name__)


class RoPEEvaluator:
    """Main evaluator for RoPE long context methods."""
    
    def __init__(self, config: DictConfig):
        """Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_loader = ModelLoader(config)
        self.model = None
        self.tokenizer = None
        self.results = {}
        
        # Create output directory
        self.output_dir = Path(config.data.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized RoPE Evaluator")
    
    def load_model(self):
        """Load and prepare the model with RoPE extensions."""
        logger.info("Loading model and tokenizer...")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer()
        
        # Apply RoPE extension if specified
        if self.model is not None:
            rope_config = self.config.rope_extension
            if rope_config.method != "none":
                logger.info(f"Applying RoPE extension: {rope_config.method}")
                
                rope_extension = get_rope_extension(
                    rope_config.method,
                    rope_config[rope_config.method]
                )
                self.model = rope_extension.apply(self.model)
                
                # Store extension info
                self.rope_info = rope_extension.get_scaling_info()
                logger.info(f"RoPE extension applied: {self.rope_info}")
            else:
                self.rope_info = {"method": "none"}
        else:
            self.rope_info = {"method": "api_model"}
        
        logger.info("Model loading completed")
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on all enabled benchmarks.
        
        Returns:
            Dictionary containing all evaluation results
        """
        if self.model is None and self.tokenizer is None:
            self.load_model()
        
        start_time = time.time()
        
        # Initialize results structure
        self.results = {
            "config": self._get_config_summary(),
            "model_info": self.model_loader.get_model_info(),
            "rope_info": self.rope_info,
            "benchmarks": {},
            "summary": {},
        }
        
        # Run benchmarks
        benchmarks_config = self.config.benchmarks
        
        if benchmarks_config.niah.enabled:
            self._run_niah_benchmark()
        
        if benchmarks_config.ruler.enabled:
            self._run_ruler_benchmark()
        
        if benchmarks_config.longbench.enabled:
            self._run_longbench_benchmark()
        
        if benchmarks_config.longbench_v2.enabled:
            self._run_longbench_v2_benchmark()
        
        # Compute summary statistics
        self._compute_summary()
        
        # Save results
        end_time = time.time()
        self.results["evaluation_time"] = end_time - start_time
        
        self._save_results()
        
        return self.results
    
    def _run_niah_benchmark(self):
        """Run NIAH benchmark evaluation."""
        logger.info("Running NIAH benchmark...")
        
        # Create config with generation settings
        niah_config = dict(self.config.benchmarks.niah)
        niah_config['generation'] = self.config.evaluation.generation
        
        benchmark = NIAHBenchmark(
            niah_config,
            self.model,
            self.tokenizer
        )
        
        max_samples = self.config.benchmarks.niah.get("max_samples")
        results = benchmark.evaluate(max_samples)
        
        self.results["benchmarks"]["niah"] = results
        logger.info(f"NIAH evaluation completed. Average score: {results['average_score']:.3f}")
    
    def _run_ruler_benchmark(self):
        """Run RULER benchmark evaluation."""
        logger.info("Running RULER benchmark...")
        
        try:
            # Create config with generation settings
            ruler_config = dict(self.config.benchmarks.ruler)
            ruler_config['generation'] = self.config.evaluation.generation
            
            benchmark = RULERBenchmark(
                ruler_config,
                self.model,
                self.tokenizer
            )
            
            max_samples = self.config.benchmarks.ruler.get("max_samples")
            results = benchmark.evaluate(max_samples)
            
            self.results["benchmarks"]["ruler"] = results
            logger.info(f"RULER evaluation completed. Average score: {results['average_score']:.3f}")
            
        except Exception as e:
            logger.error(f"RULER benchmark failed: {e}")
            self.results["benchmarks"]["ruler"] = {"error": str(e)}
    
    def _run_longbench_benchmark(self):
        """Run LongBench benchmark evaluation."""
        logger.info("Running LongBench benchmark...")
        
        try:
            # Create config with generation settings
            longbench_config = dict(self.config.benchmarks.longbench)
            longbench_config['generation'] = self.config.evaluation.generation
            
            benchmark = LongBench(
                longbench_config,
                self.model,
                self.tokenizer
            )
            
            max_samples = self.config.benchmarks.longbench.get("max_samples")
            results = benchmark.evaluate(max_samples)
            
            self.results["benchmarks"]["longbench"] = results
            logger.info(f"LongBench evaluation completed. Average score: {results['average_score']:.3f}")
            
        except Exception as e:
            logger.error(f"LongBench benchmark failed: {e}")
            self.results["benchmarks"]["longbench"] = {"error": str(e)}
    
    def _run_longbench_v2_benchmark(self):
        """Run LongBench-V2 benchmark evaluation."""
        logger.info("Running LongBench-V2 benchmark...")
        
        try:
            # Create config with generation settings
            longbench_v2_config = dict(self.config.benchmarks.longbench_v2)
            longbench_v2_config['generation'] = self.config.evaluation.generation
            
            benchmark = LongBenchV2(
                longbench_v2_config,
                self.model,
                self.tokenizer
            )
            
            max_samples = self.config.benchmarks.longbench_v2.get("max_samples")
            results = benchmark.evaluate(max_samples)
            
            self.results["benchmarks"]["longbench_v2"] = results
            logger.info(f"LongBench-V2 evaluation completed. Average score: {results['average_score']:.3f}")
            
        except Exception as e:
            logger.error(f"LongBench-V2 benchmark failed: {e}")
            self.results["benchmarks"]["longbench_v2"] = {"error": str(e)}
    
    def _compute_summary(self):
        """Compute summary statistics across all benchmarks."""
        benchmark_results = self.results["benchmarks"]
        
        # Filter out failed benchmarks
        valid_benchmarks = {
            name: result for name, result in benchmark_results.items()
            if "error" not in result and "average_score" in result
        }
        
        if not valid_benchmarks:
            logger.warning("No valid benchmark results to summarize")
            self.results["summary"] = {"error": "No valid benchmark results"}
            return
        
        # Compute aggregate statistics
        scores = [result["average_score"] for result in valid_benchmarks.values()]
        
        summary = {
            "num_benchmarks": len(valid_benchmarks),
            "benchmarks_run": list(valid_benchmarks.keys()),
            "overall_average": sum(scores) / len(scores),
            "best_benchmark": max(valid_benchmarks.items(), key=lambda x: x[1]["average_score"]),
            "worst_benchmark": min(valid_benchmarks.items(), key=lambda x: x[1]["average_score"]),
        }
        
        self.results["summary"] = summary
        
        logger.info(f"Summary - Overall average: {summary['overall_average']:.3f}")
        logger.info(f"Best: {summary['best_benchmark'][0]} ({summary['best_benchmark'][1]['average_score']:.3f})")
        logger.info(f"Worst: {summary['worst_benchmark'][0]} ({summary['worst_benchmark'][1]['average_score']:.3f})")
    
    def _save_results(self):
        """Save evaluation results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"results_{timestamp}.json"
        save_results(self.results, results_file)
        
        # Save summary report
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        summary_data = {
            "config": self.results["config"],
            "model_info": self.results["model_info"],
            "rope_info": self.results["rope_info"],
            "summary": self.results["summary"],
            "benchmark_scores": {
                name: result.get("average_score", 0.0)
                for name, result in self.results["benchmarks"].items()
                if "average_score" in result
            }
        }
        save_results(summary_data, summary_file)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration used."""
        return {
            "model": {
                "type": self.config.model.type,
                "name": self.config.model.name,
                "max_length": self.config.model.max_length,
            },
            "rope_extension": {
                "method": self.config.rope_extension.method,
            },
            "benchmarks_enabled": [
                name for name, config in self.config.benchmarks.items()
                if config.get("enabled", False)
            ],
            "seed": self.config.get("seed", None),
        }
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume evaluation from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory or file
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.is_file():
            checkpoint_file = checkpoint_path
        else:
            # Look for latest checkpoint in directory
            checkpoint_files = list(checkpoint_path.glob("checkpoint_*.json"))
            if not checkpoint_files:
                raise ValueError(f"No checkpoint files found in {checkpoint_path}")
            checkpoint_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Restore results
        self.results = checkpoint_data.get("results", {})
        
        # TODO: Implement more sophisticated checkpoint restoration
        # This would include tracking which samples have been evaluated
        # and resuming from the appropriate point
        
        logger.info("Checkpoint loaded successfully")