"""
A comprehensive evaluation framework for long context RoPE extension methods with hyperparameter sweeping
"""

__version__ = "0.2.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import RoPEEvaluator
from .utils import Config, setup_logging

# Import sweep functionality
from .sweep import (
    SweepConfig, SweepRunner, ParallelSweepRunner,
    SweepAnalyzer, SweepVisualizer
)

# Import model functionality
from .models import (
    BaseRoPEExtension, LinearInterpolationRoPE, NTKAwareRoPE,
    YaRNRoPE, LongRoPE, DynamicNTKRoPE, ModelLoader, get_rope_extension
)

# Import metrics
from .metrics import (
    BaseMetric, PerplexityMetric, SlidingWindowPerplexity,
    PasskeyRetrievalMetric, MultiNeedleRetrievalMetric, LongPPLMetric
)

# Import benchmarks
from .benchmarks import NIAHBenchmark, RULERBenchmark, LongBenchV2, LongBench

__all__ = [
    "RoPEEvaluator",
    "Config",
    "setup_logging",
    # Sweep functionality
    "SweepConfig",
    "SweepRunner", 
    "ParallelSweepRunner",
    "SweepAnalyzer",
    "SweepVisualizer",
    # Model functionality
    "BaseRoPEExtension",
    "LinearInterpolationRoPE",
    "NTKAwareRoPE", 
    "YaRNRoPE",
    "LongRoPE",
    "DynamicNTKRoPE",
    "ModelLoader",
    "get_rope_extension",
    # Metrics
    "BaseMetric",
    "PerplexityMetric",
    "SlidingWindowPerplexity",
    "PasskeyRetrievalMetric", 
    "MultiNeedleRetrievalMetric",
    "LongPPLMetric",
    # Benchmarks
    "NIAHBenchmark",
    "RULERBenchmark", 
    "LongBenchV2",
    "LongBench",
]