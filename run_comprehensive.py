#!/usr/bin/env python3

"""
Comprehensive RoPE Long Context Evaluation Runner

This script runs the complete evaluation framework with all benchmarks:
- Traditional retrieval tasks
- NIAH (Needle In A Haystack)
- RULER synthetic benchmark
- LongBench real-world tasks

Usage:
    python run_comprehensive.py --config comprehensive_config.yaml
    python run_comprehensive.py --config comprehensive_config.yaml --output results/my_eval
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging(config):
    """Setup logging configuration."""
    log_level = getattr(logging, config.get('output', {}).get('log_level', 'INFO'))
    log_file = config.get('output', {}).get('log_file', 'evaluation.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path):
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config):
    """Validate configuration has required fields."""
    required_fields = ['model', 'datasets', 'rope_methods', 'evaluation', 'output']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate model config
    model_config = config['model']
    if 'path' not in model_config:
        raise ValueError("Model path is required")
    
    # Validate dataset paths
    datasets = config['datasets']
    if 'longbench' in datasets and 'path' in datasets['longbench']:
        longbench_path = Path(datasets['longbench']['path'])
        if not longbench_path.exists():
            logging.warning(f"LongBench data path does not exist: {longbench_path}")

def run_traditional_retrieval(config, evaluator, results):
    """Run traditional retrieval evaluation."""
    logger = logging.getLogger(__name__)
    
    if not config['datasets']['retrieval']['enabled']:
        logger.info("Skipping traditional retrieval evaluation (disabled)")
        return
    
    logger.info("Running traditional retrieval evaluation...")
    
    retrieval_config = config['datasets']['retrieval']
    context_lengths = retrieval_config['context_lengths']
    num_samples = retrieval_config['num_samples']
    
    retrieval_results = {}
    
    for method_config in config['rope_methods']:
        method_name = method_config['name']
        logger.info(f"Evaluating {method_name} on retrieval tasks")
        
        method_results = {}
        
        for context_length in context_lengths:
            logger.info(f"  Context length: {context_length}")
            
            try:
                # Apply RoPE scaling
                evaluator.apply_rope_scaling(method_name, method_config.get('config', {}))
                
                # Generate synthetic retrieval tasks
                from rope_long_context_evaluation_suite.benchmarks.retrieval import generate_retrieval_samples
                samples = generate_retrieval_samples(context_length, num_samples)
                
                # Evaluate samples
                scores = []
                for sample in samples[:min(10, len(samples))]:  # Limit for demo
                    score = evaluator.evaluate_sample(sample['input'], sample['target'])
                    scores.append(score)
                
                avg_score = sum(scores) / len(scores) if scores else 0.0
                method_results[f"context_{context_length}"] = {
                    "average_score": avg_score,
                    "num_samples": len(scores),
                    "scores": scores
                }
                
                logger.info(f"    Average score: {avg_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {method_name} at {context_length}: {e}")
                method_results[f"context_{context_length}"] = {"error": str(e)}
        
        retrieval_results[method_name] = method_results
    
    results['traditional_retrieval'] = retrieval_results
    logger.info("Traditional retrieval evaluation completed")

def run_niah_evaluation(config, evaluator, results):
    """Run NIAH (Needle In A Haystack) evaluation."""
    logger = logging.getLogger(__name__)
    
    if not config['datasets']['niah']['enabled']:
        logger.info("Skipping NIAH evaluation (disabled)")
        return
    
    logger.info("Running NIAH evaluation...")
    
    niah_config = config['datasets']['niah']
    context_lengths = niah_config['context_lengths']
    num_samples = niah_config['num_samples']
    variants = niah_config['variants']
    
    niah_results = {}
    
    for method_config in config['rope_methods']:
        method_name = method_config['name']
        logger.info(f"Evaluating {method_name} on NIAH")
        
        method_results = {}
        
        for variant in variants:
            logger.info(f"  Variant: {variant}")
            variant_results = {}
            
            for context_length in context_lengths:
                try:
                    # Apply RoPE scaling
                    evaluator.apply_rope_scaling(method_name, method_config.get('config', {}))
                    
                    # Generate NIAH samples
                    from rope_long_context_evaluation_suite.benchmarks.niah import NIAHBenchmark
                    benchmark = NIAHBenchmark()
                    samples = benchmark.generate_samples(variant, context_length, min(5, num_samples))
                    
                    # Evaluate samples
                    scores = []
                    for sample in samples:
                        score = evaluator.evaluate_sample(sample['input'], sample['target'])
                        scores.append(score)
                    
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    variant_results[f"context_{context_length}"] = {
                        "average_score": avg_score,
                        "num_samples": len(scores)
                    }
                    
                    logger.info(f"    Context {context_length}: {avg_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error in NIAH {variant} at {context_length}: {e}")
                    variant_results[f"context_{context_length}"] = {"error": str(e)}
            
            method_results[variant] = variant_results
        
        niah_results[method_name] = method_results
    
    results['niah'] = niah_results
    logger.info("NIAH evaluation completed")

def run_ruler_evaluation(config, evaluator, results):
    """Run RULER benchmark evaluation."""
    logger = logging.getLogger(__name__)
    
    if not config['datasets']['ruler']['enabled']:
        logger.info("Skipping RULER evaluation (disabled)")
        return
    
    logger.info("Running RULER evaluation...")
    
    ruler_config = config['datasets']['ruler']
    context_lengths = ruler_config['context_lengths']
    num_samples = ruler_config['num_samples']
    tasks = ruler_config['tasks']
    
    ruler_results = {}
    
    for method_config in config['rope_methods']:
        method_name = method_config['name']
        logger.info(f"Evaluating {method_name} on RULER")
        
        method_results = {}
        
        for task in tasks:
            logger.info(f"  Task: {task}")
            task_results = {}
            
            for context_length in context_lengths:
                try:
                    # Apply RoPE scaling
                    evaluator.apply_rope_scaling(method_name, method_config.get('config', {}))
                    
                    # Generate RULER samples
                    from rope_long_context_evaluation_suite.benchmarks.ruler import RULERBenchmark
                    benchmark = RULERBenchmark()
                    samples = benchmark.generate_samples(task, context_length, min(5, num_samples))
                    
                    # Evaluate samples
                    scores = []
                    for sample in samples:
                        score = evaluator.evaluate_sample(sample['input'], sample['target'])
                        scores.append(score)
                    
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    task_results[f"context_{context_length}"] = {
                        "average_score": avg_score,
                        "num_samples": len(scores)
                    }
                    
                    logger.info(f"    Context {context_length}: {avg_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error in RULER {task} at {context_length}: {e}")
                    task_results[f"context_{context_length}"] = {"error": str(e)}
            
            method_results[task] = task_results
        
        ruler_results[method_name] = method_results
    
    results['ruler'] = ruler_results
    logger.info("RULER evaluation completed")

def run_longbench_evaluation(config, evaluator, results):
    """Run LongBench evaluation."""
    logger = logging.getLogger(__name__)
    
    if 'longbench' not in config['datasets']:
        logger.info("Skipping LongBench evaluation (not configured)")
        return
    
    logger.info("Running LongBench evaluation...")
    
    longbench_config = config['datasets']['longbench']
    data_path = Path(longbench_config['path'])
    tasks = longbench_config['tasks']
    
    if not data_path.exists():
        logger.warning(f"LongBench data path not found: {data_path}")
        results['longbench'] = {"error": "Data path not found"}
        return
    
    longbench_results = {}
    
    for method_config in config['rope_methods']:
        method_name = method_config['name']
        logger.info(f"Evaluating {method_name} on LongBench")
        
        method_results = {}
        
        for task in tasks:
            logger.info(f"  Task: {task}")
            
            try:
                # Apply RoPE scaling
                evaluator.apply_rope_scaling(method_name, method_config.get('config', {}))
                
                # Load LongBench task data
                task_file = data_path / f"{task}.jsonl"
                if not task_file.exists():
                    logger.warning(f"Task file not found: {task_file}")
                    continue
                
                # Load samples (limit to first 5 for demo)
                samples = []
                with open(task_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 5:  # Limit for demo
                            break
                        sample = json.loads(line)
                        samples.append(sample)
                
                # Evaluate samples
                scores = []
                for sample in samples:
                    try:
                        input_text = sample.get('input', sample.get('context', ''))
                        target = sample.get('answers', sample.get('answer', ''))
                        if isinstance(target, list) and target:
                            target = target[0]
                        
                        score = evaluator.evaluate_sample(input_text, str(target))
                        scores.append(score)
                    except Exception as e:
                        logger.warning(f"Error evaluating sample in {task}: {e}")
                
                avg_score = sum(scores) / len(scores) if scores else 0.0
                method_results[task] = {
                    "average_score": avg_score,
                    "num_samples": len(scores)
                }
                
                logger.info(f"    Average score: {avg_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error in LongBench {task}: {e}")
                method_results[task] = {"error": str(e)}
        
        longbench_results[method_name] = method_results
    
    results['longbench'] = longbench_results
    logger.info("LongBench evaluation completed")

def save_results(config, results):
    """Save evaluation results."""
    logger = logging.getLogger(__name__)
    
    output_config = config['output']
    base_dir = Path(output_config['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON results
    if 'json' in output_config['formats']:
        json_file = base_dir / f"comprehensive_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'config': config,
                'results': results,
                'timestamp': timestamp,
                'evaluation_info': {
                    'framework': 'RoPE Long Context Evaluation Suite',
                    'version': '1.0.0'
                }
            }, f, indent=2)
        logger.info(f"Results saved to: {json_file}")
    
    # Save CSV summary if requested
    if 'csv' in output_config['formats']:
        import pandas as pd
        
        # Create summary table
        summary_data = []
        for benchmark_name, benchmark_results in results.items():
            if isinstance(benchmark_results, dict):
                for method_name, method_results in benchmark_results.items():
                    if isinstance(method_results, dict):
                        for task_or_length, task_results in method_results.items():
                            if isinstance(task_results, dict) and 'average_score' in task_results:
                                summary_data.append({
                                    'benchmark': benchmark_name,
                                    'method': method_name,
                                    'task_or_length': task_or_length,
                                    'average_score': task_results['average_score'],
                                    'num_samples': task_results.get('num_samples', 0)
                                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = base_dir / f"comprehensive_summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"CSV summary saved to: {csv_file}")

def run_evaluation(config):
    """Run comprehensive evaluation."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting comprehensive RoPE evaluation...")
        
        # Import and initialize evaluator
        from rope_long_context_evaluation_suite.core import RoPEEvaluator
        from omegaconf import OmegaConf
        
        model_config = config['model']
        logger.info(f"Loading model: {model_config['name']} from {model_config['path']}")
        
        # Convert config to OmegaConf format expected by RoPEEvaluator
        omega_config = OmegaConf.create(config)
        evaluator = RoPEEvaluator(omega_config)
        
        # Run the comprehensive evaluation using the evaluator's built-in method
        logger.info("Running evaluation...")
        results = evaluator.evaluate()
        
        # Save results using the evaluator's built-in save mechanism (already called in evaluate())
        logger.info("Evaluation results saved automatically by RoPEEvaluator")
        
        logger.info("Comprehensive evaluation completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*60)
        
        for benchmark_name, benchmark_results in results.items():
            print(f"\n{benchmark_name.upper()}:")
            if isinstance(benchmark_results, dict):
                for method_name, method_results in benchmark_results.items():
                    if isinstance(method_results, dict):
                        # Calculate average across all tasks/lengths
                        scores = []
                        for task_results in method_results.values():
                            if isinstance(task_results, dict) and 'average_score' in task_results:
                                scores.append(task_results['average_score'])
                        
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            print(f"  {method_name}: {avg_score:.3f} (avg across {len(scores)} tasks)")
        
        print(f"\nResults saved to: {config['output']['base_dir']}")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive RoPE evaluation')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--output', '-o', help='Output directory (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.output:
        config['output']['base_dir'] = args.output
    
    if args.verbose:
        config['output']['log_level'] = 'DEBUG'
    
    # Validate configuration
    validate_config(config)
    
    # Setup logging
    setup_logging(config)
    
    # Run evaluation
    results = run_evaluation(config)
    
    print(f"\nEvaluation completed! Check results in: {config['output']['base_dir']}")

if __name__ == "__main__":
    main()