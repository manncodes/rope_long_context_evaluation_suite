"""Sweep runner implementations for hyperparameter optimization."""

import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import traceback

import torch
from omegaconf import DictConfig, OmegaConf

from ..core import RoPEEvaluator
from ..models import get_rope_extension
from ..metrics import BaseMetric, PerplexityMetric, PasskeyRetrievalMetric, LongPPLMetric
from .config import SweepConfig

logger = logging.getLogger(__name__)


class SweepRunner:
    """Base class for running hyperparameter sweeps."""
    
    def __init__(self, sweep_config: SweepConfig):
        """Initialize sweep runner.
        
        Args:
            sweep_config: Sweep configuration
        """
        self.config = sweep_config
        self.results = []
        self.cache = {}
        
        # Create output directories
        self.output_dir = Path(self.config.output_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache if enabled
        if self.config.use_cache:
            self._load_cache()
            
        # Initialize metrics
        self.metric_instances = self._create_metric_instances()
        
    def _create_metric_instances(self) -> Dict[str, BaseMetric]:
        """Create metric instances based on configuration.
        
        Returns:
            Dictionary of metric name to instance mappings
        """
        metrics = {}
        
        for metric_name in self.config.metrics:
            if metric_name == "perplexity":
                metrics[metric_name] = PerplexityMetric()
            elif metric_name == "passkey_retrieval":
                metrics[metric_name] = PasskeyRetrievalMetric()
            elif metric_name == "longppl":
                metrics[metric_name] = LongPPLMetric()
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        return metrics
    
    def _load_cache(self):
        """Load cached results if available."""
        cache_file = self.cache_dir / "sweep_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached results")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save current cache to disk."""
        if not self.config.use_cache:
            return
            
        cache_file = self.cache_dir / "sweep_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_experiment_hash(self, experiment_config: Dict[str, Any]) -> str:
        """Generate hash for experiment configuration.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Hash string for the configuration
        """
        config_str = json.dumps(experiment_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _run_single_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment.
        
        Args:
            experiment_config: Configuration for the experiment
            
        Returns:
            Experiment results
        """
        exp_hash = self._get_experiment_hash(experiment_config)
        
        # Check cache
        if self.config.use_cache and exp_hash in self.cache:
            logger.info(f"Using cached result for experiment {exp_hash[:8]}")
            return self.cache[exp_hash]
        
        try:
            start_time = time.time()
            
            # Create evaluator configuration
            eval_config = self._create_evaluator_config(experiment_config)
            
            # Initialize evaluator
            evaluator = RoPEEvaluator(eval_config)
            evaluator.load_model()
            
            # Run metrics
            metric_results = {}
            for metric_name, metric_instance in self.metric_instances.items():
                try:
                    if metric_name in ["perplexity", "longppl"]:
                        # For perplexity-based metrics, we need test data
                        test_texts = self._get_test_texts(experiment_config['context_length'])
                        result = metric_instance.compute(
                            evaluator.model,
                            evaluator.tokenizer,
                            test_texts,
                            context_length=experiment_config['context_length']
                        )
                    else:
                        # For synthetic metrics like passkey retrieval
                        result = metric_instance.compute(
                            evaluator.model,
                            evaluator.tokenizer,
                            context_length=experiment_config['context_length']
                        )
                    
                    metric_results[metric_name] = result
                    
                except Exception as e:
                    logger.error(f"Error computing {metric_name}: {e}")
                    metric_results[metric_name] = {"error": str(e)}
            
            # Cleanup
            del evaluator.model
            del evaluator.tokenizer
            torch.cuda.empty_cache()
            
            # Prepare result
            result = {
                'experiment_config': experiment_config,
                'metrics': metric_results,
                'execution_time': time.time() - start_time,
                'status': 'completed'
            }
            
            # Cache result
            if self.config.use_cache:
                self.cache[exp_hash] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error in experiment {exp_hash[:8]}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'experiment_config': experiment_config,
                'metrics': {},
                'execution_time': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    def _create_evaluator_config(self, experiment_config: Dict[str, Any]) -> DictConfig:
        """Create evaluator configuration from experiment config.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Evaluator configuration
        """
        # Base configuration
        config = {
            'model': {
                'type': experiment_config['model_type'],
                'name': experiment_config['model_name'],
                'path': experiment_config.get('model_path'),
                'max_length': experiment_config['context_length'],
                'device_map': 'auto',
                'torch_dtype': 'auto'
            },
            'rope_extension': {
                'method': experiment_config['rope_method'],
                experiment_config['rope_method']: experiment_config['parameters']
            },
            'benchmarks': {
                'niah': {'enabled': False},
                'ruler': {'enabled': False},
                'longbench': {'enabled': False},
                'longbench_v2': {'enabled': False}
            },
            'data': {
                'output_dir': str(self.output_dir / 'temp'),
                'cache_dir': str(self.cache_dir),
            },
            'logging': {
                'level': 'WARNING'
            }
        }
        
        return OmegaConf.create(config)
    
    def _get_test_texts(self, context_length: int) -> List[str]:
        """Get test texts for perplexity evaluation.
        
        Args:
            context_length: Target context length
            
        Returns:
            List of test texts
        """
        # Generate synthetic test text
        # In practice, you'd load from a proper dataset
        base_text = """
        The study of artificial intelligence has progressed rapidly in recent years, 
        with transformer models achieving remarkable performance across various tasks.
        These models, based on the attention mechanism, can process sequences of arbitrary length
        and have revolutionized natural language processing, computer vision, and other domains.
        However, one limitation of transformer models is their computational complexity,
        which scales quadratically with sequence length due to the self-attention mechanism.
        """
        
        # Repeat text to reach desired length
        target_chars = context_length * 4  # Rough estimate: 4 chars per token
        repetitions = target_chars // len(base_text) + 1
        long_text = (base_text * repetitions)[:target_chars]
        
        return [long_text]
    
    def run(self) -> List[Dict[str, Any]]:
        """Run the hyperparameter sweep.
        
        Returns:
            List of experiment results
        """
        experiments = self.config.generate_experiment_configs()
        logger.info(f"Running {len(experiments)} experiments")
        
        results = []
        
        for i, exp_config in enumerate(experiments):
            logger.info(f"Running experiment {i+1}/{len(experiments)}: "
                       f"{exp_config['rope_method']} @ {exp_config['context_length']}")
            
            result = self._run_single_experiment(exp_config)
            results.append(result)
            
            # Early stopping check
            if self._should_early_stop(results):
                logger.info("Early stopping triggered")
                break
            
            # Save intermediate results
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(results)
                self._save_cache()
        
        # Save final results
        self._save_results(results)
        self._save_cache()
        
        self.results = results
        return results
    
    def _should_early_stop(self, results: List[Dict[str, Any]]) -> bool:
        """Check if early stopping criteria are met.
        
        Args:
            results: Current results
            
        Returns:
            True if should stop early
        """
        if not self.config.early_stopping or len(results) < self.config.early_stopping_patience:
            return False
        
        # Check last N results for the stopping metric
        recent_results = results[-self.config.early_stopping_patience:]
        metric_values = []
        
        for result in recent_results:
            if result['status'] == 'completed':
                metric_result = result['metrics'].get(self.config.early_stopping_metric)
                if metric_result and not isinstance(metric_result, dict) or 'error' not in metric_result:
                    # Extract the main metric value
                    if self.config.early_stopping_metric == 'perplexity':
                        value = metric_result.get('perplexity', float('inf'))
                    elif self.config.early_stopping_metric == 'passkey_retrieval':
                        value = 1.0 - metric_result.get('passkey_accuracy', 0.0)  # Convert to loss
                    else:
                        value = metric_result.get(self.config.early_stopping_metric, float('inf'))
                    
                    metric_values.append(value)
        
        # Check if all recent results are above threshold (for loss-like metrics)
        if len(metric_values) >= self.config.early_stopping_patience:
            return all(v > self.config.early_stopping_threshold for v in metric_values)
        
        return False
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to disk.
        
        Args:
            results: Current results
        """
        output_file = self.output_dir / "intermediate_results.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save final results to disk.
        
        Args:
            results: Final results
        """
        # Save detailed results
        output_file = self.output_dir / "sweep_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary = self._create_summary(results)
        summary_file = self.output_dir / "sweep_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary saved to {summary_file}")
    
    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of sweep results.
        
        Args:
            results: All experiment results
            
        Returns:
            Summary dictionary
        """
        completed_results = [r for r in results if r['status'] == 'completed']
        failed_results = [r for r in results if r['status'] == 'failed']
        
        summary = {
            'total_experiments': len(results),
            'completed_experiments': len(completed_results),
            'failed_experiments': len(failed_results),
            'success_rate': len(completed_results) / len(results) if results else 0,
            'total_execution_time': sum(r.get('execution_time', 0) for r in results)
        }
        
        # Best results per metric
        if completed_results:
            for metric_name in self.config.metrics:
                best_result = self._find_best_result(completed_results, metric_name)
                if best_result:
                    summary[f'best_{metric_name}'] = {
                        'experiment_config': best_result['experiment_config'],
                        'metric_value': best_result['metrics'][metric_name]
                    }
        
        return summary
    
    def _find_best_result(self, results: List[Dict[str, Any]], metric_name: str) -> Optional[Dict[str, Any]]:
        """Find the best result for a specific metric.
        
        Args:
            results: List of completed results
            metric_name: Name of the metric to optimize
            
        Returns:
            Best result or None
        """
        valid_results = []
        
        for result in results:
            metric_result = result['metrics'].get(metric_name)
            if metric_result and 'error' not in metric_result:
                # Extract metric value
                if metric_name == 'perplexity':
                    value = metric_result.get('perplexity', float('inf'))
                    # Lower is better for perplexity
                    valid_results.append((value, result))
                elif metric_name == 'passkey_retrieval':
                    value = metric_result.get('passkey_accuracy', 0.0)
                    # Higher is better for accuracy
                    valid_results.append((-value, result))  # Negate for min operation
                elif metric_name == 'longppl':
                    value = metric_result.get('longppl', float('inf'))
                    # Lower is better for LongPPL
                    valid_results.append((value, result))
        
        if valid_results:
            _, best_result = min(valid_results, key=lambda x: x[0])
            return best_result
        
        return None


class ParallelSweepRunner(SweepRunner):
    """Parallel version of sweep runner using multiple processes/threads."""
    
    def __init__(self, sweep_config: SweepConfig):
        """Initialize parallel sweep runner.
        
        Args:
            sweep_config: Sweep configuration
        """
        super().__init__(sweep_config)
        self.num_workers = min(sweep_config.parallel_jobs, 8)  # Limit to avoid resource issues
    
    def run(self) -> List[Dict[str, Any]]:
        """Run the hyperparameter sweep in parallel.
        
        Returns:
            List of experiment results
        """
        experiments = self.config.generate_experiment_configs()
        logger.info(f"Running {len(experiments)} experiments with {self.num_workers} workers")
        
        results = []
        
        # Use ThreadPoolExecutor for I/O bound tasks, ProcessPoolExecutor for CPU bound
        # Since model evaluation is GPU bound, ThreadPoolExecutor is often better
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(self._run_single_experiment, exp): exp 
                for exp in experiments
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_exp)):
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                    results.append(result)
                    
                    exp_config = future_to_exp[future]
                    logger.info(f"Completed experiment {i+1}/{len(experiments)}: "
                               f"{exp_config['rope_method']} @ {exp_config['context_length']}")
                    
                    # Early stopping check
                    if self._should_early_stop(results):
                        logger.info("Early stopping triggered, cancelling remaining experiments")
                        # Cancel remaining futures
                        for remaining_future in future_to_exp:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    
                    # Save intermediate results
                    if (i + 1) % 10 == 0:
                        self._save_intermediate_results(results)
                        self._save_cache()
                        
                except Exception as e:
                    logger.error(f"Error in parallel experiment: {e}")
                    exp_config = future_to_exp[future]
                    results.append({
                        'experiment_config': exp_config,
                        'metrics': {},
                        'execution_time': 0,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        # Save final results
        self._save_results(results)
        self._save_cache()
        
        self.results = results
        return results