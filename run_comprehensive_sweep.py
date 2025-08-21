#!/usr/bin/env python3
"""Comprehensive parameter sweep evaluation script.

This script performs systematic evaluation across multiple dimensions:
- RoPE scaling methods
- Context lengths 
- Benchmarks
- Model configurations

Results are saved with detailed tracking of all parameter combinations.
"""

import sys
import os
import argparse
import logging
import itertools
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rope_long_context_evaluation_suite.core import RoPEEvaluator
from rope_long_context_evaluation_suite.utils import setup_logging
from omegaconf import OmegaConf, DictConfig
import copy

logger = logging.getLogger(__name__)

class ComprehensiveSweepRunner:
    """Runner for comprehensive parameter sweeps."""
    
    def __init__(self, config_path: str, output_dir: str = None):
        """Initialize the sweep runner.
        
        Args:
            config_path: Path to comprehensive config YAML
            output_dir: Override output directory
        """
        self.base_config = OmegaConf.load(config_path)
        self.output_dir = Path(output_dir) if output_dir else Path(self.base_config.data.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sweep-specific output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = self.output_dir / f"comprehensive_sweep_{timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.failed_runs = []
        
    def get_parameter_combinations(self) -> List[Tuple[str, Dict, Dict, List[str]]]:
        """Generate all parameter combinations to evaluate.
        
        Returns:
            List of tuples: (rope_method, rope_config, context_config, enabled_benchmarks)
        """
        combinations = []
        
        # Get RoPE methods from config
        rope_methods = self.base_config.get('rope_methods', [{'name': 'none', 'config': {}}])
        
        # Get context lengths - use multiple sources
        context_sources = []
        
        # From dataset config
        if 'datasets' in self.base_config:
            datasets_config = self.base_config.datasets
            if 'retrieval' in datasets_config and 'context_lengths' in datasets_config.retrieval:
                context_sources.append(datasets_config.retrieval.context_lengths)
            if 'niah' in datasets_config and 'context_lengths' in datasets_config.niah:
                context_sources.append(datasets_config.niah.context_lengths)
            if 'ruler' in datasets_config and 'context_lengths' in datasets_config.ruler:
                context_sources.append(datasets_config.ruler.context_lengths)
        
        # From benchmark config
        if 'benchmarks' in self.base_config:
            benchmarks_config = self.base_config.benchmarks
            if 'niah' in benchmarks_config and 'context_lengths' in benchmarks_config.niah:
                context_sources.append(benchmarks_config.niah.context_lengths)
        
        # Use the longest list of context lengths found
        all_context_lengths = [4000, 8000, 16000, 32000]  # Default
        if context_sources:
            all_context_lengths = max(context_sources, key=len)
        
        # Get enabled benchmarks
        enabled_benchmarks = []
        if 'benchmarks' in self.base_config:
            for bench_name, bench_config in self.base_config.benchmarks.items():
                if bench_config.get('enabled', False):
                    enabled_benchmarks.append(bench_name)
        
        logger.info(f"Found {len(rope_methods)} RoPE methods")
        logger.info(f"Found {len(all_context_lengths)} context lengths: {all_context_lengths}")
        logger.info(f"Found {len(enabled_benchmarks)} enabled benchmarks: {enabled_benchmarks}")
        
        # Generate all combinations
        for rope_method in rope_methods:
            rope_name = rope_method['name']
            rope_config = rope_method.get('config', {})
            
            # For each context length, create a context config
            for context_length in all_context_lengths:
                context_config = {
                    'max_context_length': context_length,
                    'target_context_length': context_length
                }
                
                combinations.append((rope_name, rope_config, context_config, enabled_benchmarks.copy()))
        
        logger.info(f"Generated {len(combinations)} total parameter combinations")
        return combinations
    
    def create_run_config(self, rope_method: str, rope_config: Dict, 
                         context_config: Dict, benchmarks: List[str]) -> DictConfig:
        """Create a config for a specific parameter combination.
        
        Args:
            rope_method: RoPE scaling method name
            rope_config: RoPE method configuration  
            context_config: Context length configuration
            benchmarks: List of enabled benchmarks
            
        Returns:
            Configuration for this specific run
        """
        # Deep copy base config
        run_config = copy.deepcopy(self.base_config)
        
        # Set RoPE configuration
        if 'rope_extension' not in run_config:
            run_config['rope_extension'] = {}
        run_config.rope_extension.method = rope_method
        
        # Add RoPE method-specific config
        if rope_config and rope_method != 'none':
            if rope_method not in run_config.rope_extension:
                run_config.rope_extension[rope_method] = {}
            run_config.rope_extension[rope_method].update(rope_config)
        
        # Set context length configurations
        max_context = context_config['max_context_length']
        
        # Update model max_length
        if 'model' in run_config:
            run_config.model.max_length = max_context
        
        # Update evaluation config
        if 'evaluation' not in run_config:
            run_config['evaluation'] = {}
        run_config.evaluation.max_context_length = max_context
        
        # Update benchmark configs with context lengths
        if 'benchmarks' in run_config:
            # Disable all benchmarks first
            for bench_name in run_config.benchmarks.keys():
                run_config.benchmarks[bench_name].enabled = False
            
            # Enable only specified benchmarks with context length
            for bench_name in benchmarks:
                if bench_name in run_config.benchmarks:
                    run_config.benchmarks[bench_name].enabled = True
                    
                    # Set context length for benchmarks that support it
                    if bench_name == 'niah':
                        run_config.benchmarks[bench_name].context_lengths = [max_context]
                    elif bench_name == 'ruler':
                        run_config.benchmarks[bench_name].max_length = max_context
                    elif bench_name in ['longbench', 'longbench_v2']:
                        # LongBench uses its own dataset lengths, but we can set a max
                        if 'max_length' not in run_config.benchmarks[bench_name]:
                            run_config.benchmarks[bench_name]['max_length'] = max_context
        
        # Set output directory for this run
        run_name = f"{rope_method}_ctx{max_context}_{'_'.join(benchmarks)}"
        run_output_dir = self.sweep_dir / run_name
        run_config.data.output_dir = str(run_output_dir)
        
        return OmegaConf.create(run_config)
    
    def run_single_evaluation(self, rope_method: str, rope_config: Dict, 
                            context_config: Dict, benchmarks: List[str]) -> Dict[str, Any]:
        """Run a single evaluation with specific parameters.
        
        Args:
            rope_method: RoPE scaling method name
            rope_config: RoPE method configuration
            context_config: Context length configuration  
            benchmarks: List of enabled benchmarks
            
        Returns:
            Evaluation results dictionary
        """
        max_context = context_config['max_context_length']
        run_name = f"{rope_method}_ctx{max_context}_{'_'.join(benchmarks)}"
        
        logger.info(f"🚀 Starting evaluation: {run_name}")
        logger.info(f"   RoPE Method: {rope_method}")
        logger.info(f"   Context Length: {max_context}")
        logger.info(f"   Benchmarks: {benchmarks}")
        
        try:
            # Create run-specific config
            run_config = self.create_run_config(rope_method, rope_config, context_config, benchmarks)
            
            # Run evaluation
            evaluator = RoPEEvaluator(run_config)
            results = evaluator.evaluate()
            
            # Add metadata
            results['sweep_metadata'] = {
                'rope_method': rope_method,
                'rope_config': rope_config,
                'context_length': max_context,
                'benchmarks': benchmarks,
                'run_name': run_name,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"✅ Completed evaluation: {run_name}")
            
            # Log summary results
            if 'summary' in results and 'overall_average' in results['summary']:
                logger.info(f"   Overall Average Score: {results['summary']['overall_average']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed evaluation: {run_name} - {e}")
            error_result = {
                'sweep_metadata': {
                    'rope_method': rope_method,
                    'rope_config': rope_config,
                    'context_length': max_context,
                    'benchmarks': benchmarks,
                    'run_name': run_name,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'error': str(e)
                },
                'error': str(e),
                'failed': True
            }
            self.failed_runs.append(error_result)
            return error_result
    
    def run_comprehensive_sweep(self, max_runs: int = None, 
                              filter_rope_methods: List[str] = None,
                              filter_context_lengths: List[int] = None,
                              filter_benchmarks: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive parameter sweep.
        
        Args:
            max_runs: Maximum number of runs (for testing)
            filter_rope_methods: Only run these RoPE methods
            filter_context_lengths: Only run these context lengths
            filter_benchmarks: Only run these benchmarks
            
        Returns:
            Comprehensive results dictionary
        """
        logger.info("🔥 Starting Comprehensive Parameter Sweep")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Get all parameter combinations
        combinations = self.get_parameter_combinations()
        
        # Apply filters
        if filter_rope_methods:
            combinations = [c for c in combinations if c[0] in filter_rope_methods]
            logger.info(f"Filtered to RoPE methods: {filter_rope_methods}")
        
        if filter_context_lengths:
            combinations = [c for c in combinations if c[2]['max_context_length'] in filter_context_lengths]
            logger.info(f"Filtered to context lengths: {filter_context_lengths}")
        
        if filter_benchmarks:
            combinations = [c for c in combinations if any(b in filter_benchmarks for b in c[3])]
            logger.info(f"Filtered to benchmarks: {filter_benchmarks}")
        
        if max_runs:
            combinations = combinations[:max_runs]
            logger.info(f"Limited to {max_runs} runs for testing")
        
        total_runs = len(combinations)
        logger.info(f"Total evaluation runs: {total_runs}")
        logger.info("=" * 80)
        
        # Run all combinations
        for i, (rope_method, rope_config, context_config, benchmarks) in enumerate(combinations, 1):
            logger.info(f"\n📊 Progress: {i}/{total_runs}")
            
            result = self.run_single_evaluation(rope_method, rope_config, context_config, benchmarks)
            self.results.append(result)
            
            # Save intermediate results
            if i % 5 == 0 or i == total_runs:
                self.save_sweep_results()
        
        # Final save
        self.save_sweep_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Create final summary
        sweep_summary = self.create_sweep_summary(total_time)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 COMPREHENSIVE SWEEP COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total Runs: {total_runs}")
        logger.info(f"Successful: {len([r for r in self.results if not r.get('failed', False)])}")
        logger.info(f"Failed: {len(self.failed_runs)}")
        logger.info(f"Total Time: {total_time/3600:.2f} hours")
        logger.info(f"Results saved to: {self.sweep_dir}")
        
        return sweep_summary
    
    def save_sweep_results(self):
        """Save comprehensive sweep results."""
        # Save detailed results
        results_file = self.sweep_dir / "comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save failed runs
        if self.failed_runs:
            failed_file = self.sweep_dir / "failed_runs.json" 
            with open(failed_file, 'w') as f:
                json.dump(self.failed_runs, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.results)} results to {results_file}")
    
    def create_sweep_summary(self, total_time: float) -> Dict[str, Any]:
        """Create comprehensive sweep summary.
        
        Args:
            total_time: Total execution time in seconds
            
        Returns:
            Summary dictionary
        """
        successful_results = [r for r in self.results if not r.get('failed', False)]
        
        summary = {
            'sweep_info': {
                'total_runs': len(self.results),
                'successful_runs': len(successful_results),
                'failed_runs': len(self.failed_runs),
                'total_time_hours': total_time / 3600,
                'sweep_directory': str(self.sweep_dir),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'performance_analysis': self.analyze_performance(successful_results),
            'best_configurations': self.find_best_configurations(successful_results),
            'rope_method_comparison': self.compare_rope_methods(successful_results),
            'context_length_analysis': self.analyze_context_lengths(successful_results)
        }
        
        # Save summary
        summary_file = self.sweep_dir / "sweep_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def analyze_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze overall performance across all runs."""
        if not results:
            return {'error': 'No successful results to analyze'}
        
        all_scores = []
        benchmark_scores = {}
        
        for result in results:
            if 'summary' in result and 'overall_average' in result['summary']:
                all_scores.append(result['summary']['overall_average'])
            
            # Collect benchmark-specific scores
            if 'benchmarks' in result:
                for bench_name, bench_result in result['benchmarks'].items():
                    if 'average_score' in bench_result:
                        if bench_name not in benchmark_scores:
                            benchmark_scores[bench_name] = []
                        benchmark_scores[bench_name].append(bench_result['average_score'])
        
        analysis = {
            'overall_statistics': {
                'mean_score': sum(all_scores) / len(all_scores) if all_scores else 0,
                'max_score': max(all_scores) if all_scores else 0,
                'min_score': min(all_scores) if all_scores else 0,
                'num_evaluations': len(all_scores)
            },
            'benchmark_statistics': {}
        }
        
        for bench_name, scores in benchmark_scores.items():
            if scores:
                analysis['benchmark_statistics'][bench_name] = {
                    'mean_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'num_evaluations': len(scores)
                }
        
        return analysis
    
    def find_best_configurations(self, results: List[Dict], top_k: int = 5) -> Dict[str, Any]:
        """Find the best performing configurations."""
        if not results:
            return {'error': 'No successful results to analyze'}
        
        # Score each result
        scored_results = []
        for result in results:
            if 'summary' in result and 'overall_average' in result['summary']:
                score = result['summary']['overall_average']
                metadata = result.get('sweep_metadata', {})
                scored_results.append({
                    'score': score,
                    'rope_method': metadata.get('rope_method', 'unknown'),
                    'context_length': metadata.get('context_length', 'unknown'),
                    'benchmarks': metadata.get('benchmarks', []),
                    'config': metadata
                })
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'top_configurations': scored_results[:top_k],
            'worst_configurations': scored_results[-top_k:] if len(scored_results) >= top_k else []
        }
    
    def compare_rope_methods(self, results: List[Dict]) -> Dict[str, Any]:
        """Compare performance across different RoPE methods."""
        if not results:
            return {'error': 'No successful results to analyze'}
        
        rope_performance = {}
        
        for result in results:
            if 'summary' in result and 'overall_average' in result['summary']:
                metadata = result.get('sweep_metadata', {})
                rope_method = metadata.get('rope_method', 'unknown')
                score = result['summary']['overall_average']
                
                if rope_method not in rope_performance:
                    rope_performance[rope_method] = []
                rope_performance[rope_method].append(score)
        
        # Compute statistics for each method
        method_stats = {}
        for method, scores in rope_performance.items():
            if scores:
                method_stats[method] = {
                    'mean_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'num_evaluations': len(scores),
                    'scores': scores
                }
        
        # Rank methods by average performance
        ranked_methods = sorted(method_stats.items(), 
                              key=lambda x: x[1]['mean_score'], 
                              reverse=True)
        
        return {
            'method_statistics': method_stats,
            'ranking': [(method, stats['mean_score']) for method, stats in ranked_methods]
        }
    
    def analyze_context_lengths(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance across different context lengths."""
        if not results:
            return {'error': 'No successful results to analyze'}
        
        context_performance = {}
        
        for result in results:
            if 'summary' in result and 'overall_average' in result['summary']:
                metadata = result.get('sweep_metadata', {})
                context_length = metadata.get('context_length', 'unknown')
                score = result['summary']['overall_average']
                
                if context_length not in context_performance:
                    context_performance[context_length] = []
                context_performance[context_length].append(score)
        
        # Compute statistics for each context length
        length_stats = {}
        for length, scores in context_performance.items():
            if scores:
                length_stats[length] = {
                    'mean_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'num_evaluations': len(scores),
                    'scores': scores
                }
        
        # Sort by context length
        sorted_lengths = sorted(length_stats.items(), key=lambda x: x[0])
        
        return {
            'length_statistics': length_stats,
            'performance_by_length': [(length, stats['mean_score']) for length, stats in sorted_lengths]
        }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive parameter sweep evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="comprehensive_config.yaml",
        help="Path to comprehensive configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Maximum number of runs (for testing)"
    )
    parser.add_argument(
        "--rope-methods",
        nargs="+",
        help="Filter to specific RoPE methods"
    )
    parser.add_argument(
        "--context-lengths", 
        nargs="+",
        type=int,
        help="Filter to specific context lengths"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+", 
        help="Filter to specific benchmarks"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    # Create sweep runner
    runner = ComprehensiveSweepRunner(args.config, args.output_dir)
    
    # Run comprehensive sweep
    summary = runner.run_comprehensive_sweep(
        max_runs=args.max_runs,
        filter_rope_methods=args.rope_methods,
        filter_context_lengths=args.context_lengths,
        filter_benchmarks=args.benchmarks
    )
    
    return summary


if __name__ == "__main__":
    main()