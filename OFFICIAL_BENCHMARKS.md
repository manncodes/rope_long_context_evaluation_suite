# Official Benchmark Implementations

This document describes the integration of official, standardized benchmark implementations in the RoPE Long Context Evaluation Suite.

## Overview

The framework now supports **official implementations** of the major long-context benchmarks, replacing the previous stub implementations with real, standardized evaluation suites used by research community.

## Available Official Benchmarks

### 1. NIAH (Needle-in-a-Haystack) - Official
- **Repository**: [gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- **Author**: Greg Kamradt
- **Used by**: OpenAI, Anthropic, Google (Gemini), and many research papers
- **Implementation**: `NIAHOfficialBenchmark` in `benchmarks/niah_official.py`

**Features**:
- Configurable context lengths and needle positions
- Multi-needle support
- Real Paul Graham essays as background text
- Exact same implementation used in industry evaluations

### 2. RULER - Official
- **Repository**: [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
- **Author**: NVIDIA Research
- **Paper**: "RULER: What's the Real Context Size of Your Long-Context Language Models?"
- **Implementation**: `RULEROfficialBenchmark` in `benchmarks/ruler_official.py`

**Features**:
- 13 tasks across 4 categories: retrieval, multi-hop, aggregation, QA
- Configurable sequence lengths up to 128K+ tokens
- Synthetic data generation with controlled complexity
- Comprehensive evaluation beyond simple retrieval

### 3. LongBench - Official
- **Repository**: [THUDM/LongBench](https://github.com/THUDM/LongBench)
- **Author**: THUDM (Tsinghua University)
- **Paper**: "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
- **Implementation**: `LongBenchOfficialBenchmark` in `benchmarks/longbench_official.py`

**Features**:
- Both LongBench v1 and v2 support
- Real-world multitask evaluation
- 12+ tasks across multiple domains
- Direct HuggingFace datasets integration

## Setup and Installation

### 1. Clone Official Repositories

The setup script automatically clones the official repositories:

```bash
# Repositories are cloned to third_party/
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git third_party/LLMTest_NeedleInAHaystack
git clone https://github.com/NVIDIA/RULER.git third_party/RULER  
git clone https://github.com/THUDM/LongBench.git third_party/LongBench
```

### 2. Install Dependencies

```bash
pip install -r requirements_benchmarks.txt
```

Key dependencies:
- `datasets` - For LongBench HuggingFace integration
- `anthropic`, `openai`, `cohere` - For NIAH API model support
- Standard ML libraries (torch, transformers, etc.)

## Configuration

### Using Official Benchmarks

The framework automatically tries official implementations first, with fallback to legacy implementations:

```yaml
# comprehensive_config.yaml
benchmarks:
  niah:
    enabled: true
    context_lengths: [4000, 8000, 16000, 32000]
    depth_percents: [10, 50, 90]
    num_tests: 10
    
  ruler:
    enabled: true
    categories: ["niah_single_1", "vt", "cwe", "qa_1"]
    max_length: 32000
    num_samples: 100
    
  longbench:
    enabled: true
    version: "v1"  # or "v2" for LongBench v2
    tasks: ["narrativeqa", "qasper", "multifieldqa_en"]
    max_samples: 100
```

### NIAH Configuration Options

```yaml
niah:
  context_lengths: [1000, 4000, 8000, 16000, 32000]
  depth_percents: [0, 10, 25, 50, 75, 90, 100]
  num_tests: 10
  save_contexts: false
  save_results: false
```

### RULER Configuration Options

```yaml
ruler:
  categories:
    - "niah_single_1"    # Single needle, noise background
    - "niah_single_2"    # Single needle, essay background
    - "niah_multikey_1"  # Multi-key needle task
    - "vt"               # Variable tracking
    - "cwe"              # Common words extraction
    - "qa_1"             # SQuAD QA
    - "qa_2"             # HotpotQA
  max_length: 32000
  num_samples: 100
```

### LongBench Configuration Options

```yaml
longbench:
  version: "v1"  # "v1" or "v2"
  tasks:         # v1 tasks
    - "narrativeqa"
    - "qasper" 
    - "multifieldqa_en"
    - "hotpotqa"
    - "2wikimqa"
  max_samples: 100

# For LongBench v2
longbench_v2:
  enabled: true
  version: "v2"
  max_samples: 100
```

## Usage Examples

### Basic Evaluation

```python
from rope_long_context_evaluation_suite import RoPEEvaluator
from rope_long_context_evaluation_suite.utils import load_config

# Load configuration
config = load_config("comprehensive_config.yaml")

# Initialize evaluator
evaluator = RoPEEvaluator(config)

# Run evaluation (will use official implementations automatically)
results = evaluator.evaluate()
```

### Accessing Official Benchmarks Directly

```python
from rope_long_context_evaluation_suite.benchmarks import (
    NIAHOfficialBenchmark,
    RULEROfficialBenchmark, 
    LongBenchOfficialBenchmark
)

# Initialize specific benchmark
niah = NIAHOfficialBenchmark(config, model, tokenizer)
results = niah.evaluate()
```

## Benchmark Information

### NIAH Results Format

```python
{
    "sample_id": "niah_0",
    "context_length": 16000,
    "depth_percent": 50,
    "score": 1.0,
    "prediction": "The best thing to do in San Francisco...",
    "target": "needle_content",
    "metadata": {
        "test_number": 1,
        "evaluation_method": "official_niah"
    }
}
```

### RULER Results Format

```python
{
    "sample_id": "ruler_niah_single_1_0", 
    "category": "niah_single_1",
    "context_length": 32000,
    "score": 0.85,
    "prediction": "generated_answer",
    "target": "expected_answer",
    "metadata": {
        "task_description": "Single needle in a haystack with noise background",
        "evaluation_method": "official_ruler"
    }
}
```

### LongBench Results Format

```python
{
    "sample_id": "longbench_narrativeqa_0",
    "task": "narrativeqa", 
    "context_length": 15000,
    "score": 0.75,
    "prediction": "generated_answer",
    "target": "reference_answer", 
    "metadata": {
        "version": "v1",
        "evaluation_method": "official_longbench"
    }
}
```

## Fallback Behavior

The framework implements intelligent fallback:

1. **Try Official Implementation**: Attempts to use official benchmark
2. **Fallback to Legacy**: If official fails, uses stub implementation
3. **Error Logging**: Clear logging of which implementation is used

```
INFO - Using official NIAH implementation
INFO - Using official RULER implementation  
WARN - Failed to load official LongBench, using legacy: datasets not available
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `third_party/` repositories are cloned
2. **Missing Dependencies**: Install `requirements_benchmarks.txt`
3. **Dataset Access**: LongBench requires internet for HuggingFace datasets
4. **Memory Issues**: Official benchmarks may use more memory than stubs

### Debugging

Enable debug logging to see detailed benchmark selection:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Benchmark Selection

Force specific implementation:

```python
# Force official implementation
from rope_long_context_evaluation_suite.benchmarks import NIAHOfficialBenchmark
benchmark = NIAHOfficialBenchmark(config, model, tokenizer)

# Force legacy implementation  
from rope_long_context_evaluation_suite.benchmarks import NIAHBenchmark
benchmark = NIAHBenchmark(config, model, tokenizer)
```

## Performance Notes

- **Official benchmarks** are more computationally intensive than stubs
- **NIAH** includes full context generation and needle insertion
- **RULER** generates synthetic data on-the-fly
- **LongBench** downloads datasets from HuggingFace
- Consider **smaller sample sizes** for initial testing

## Contributing

To add new official benchmark integrations:

1. Clone the official repository to `third_party/`
2. Create wrapper class in `benchmarks/[name]_official.py`
3. Update `benchmarks/__init__.py` with imports
4. Add fallback logic in `core.py`
5. Update documentation and configuration examples