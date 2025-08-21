# Installation Guide

## Quick Start

### Basic Installation (Local Models Only)
```bash
pip install -e .
```

### With Official Benchmarks (Recommended)
```bash
pip install -e .[benchmarks]
```

### With API Providers (OpenAI, Anthropic, etc.)
```bash
pip install -e .[api-providers]
```

### Full Installation
```bash
pip install -e .[benchmarks,api-providers]
```

## Optional Dependencies

- **benchmarks**: Official benchmark implementations (langchain, vllm, etc.)
- **api-providers**: API model support (openai, anthropic, cohere, google-generativeai)
- **evaluation**: Additional evaluation metrics (rouge, bert-score, etc.)
- **dev**: Development tools (pytest, black, mypy, etc.)

## Official Benchmarks

The suite now uses only **official, standardized benchmark implementations**:

- **NIAH**: Greg Kamradt's LLMTest_NeedleInAHaystack (used by OpenAI, Anthropic)
- **RULER**: NVIDIA's official implementation
- **LongBench**: THUDM's official HuggingFace datasets integration

No more placeholder or fake implementations!