# TinyLlama 1.1B Comprehensive RoPE Evaluation Report

**Generated**: 2025-08-20 15:41:36

## Executive Summary

- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Total Experiments**: 364
- **Successful**: 364 (100.0%)
- **Failed**: 0 (0.0%)

## Method Performance Summary

| Method | Experiments | Best PPL | Avg PPL | Worst PPL |
|--------|-------------|----------|---------|----------|
| yarn | 100 | 21.882 | 38.074 | 61.308 |
| llama3 | 100 | 22.653 | 37.368 | 59.707 |
| longrope | 20 | 23.370 | 38.332 | 60.080 |
| ntk_aware | 100 | 23.616 | 38.269 | 60.818 |
| dynamic_ntk | 20 | 24.070 | 38.724 | 60.780 |
| linear_interpolation | 24 | 25.863 | 40.133 | 62.721 |

## Top 10 Best Configurations

### 1. yarn (Perplexity: 21.882)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 3.0, 'beta_fast': 32, 'beta_slow': 1, 's': 2.0}
- **Metrics**:
  - Perplexity: 21.882
  - LongPPL: 16.835
  - Passkey Accuracy: 0.981

### 2. yarn (Perplexity: 22.645)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 4.0, 'beta_fast': 32, 'beta_slow': 1, 's': 0.5}
- **Metrics**:
  - Perplexity: 22.645
  - LongPPL: 22.370
  - Passkey Accuracy: 0.969

### 3. llama3 (Perplexity: 22.653)

- **Context Length**: 2048
- **Configuration**: {'factor': 4.0, 'low_freq_factor': 2.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 2048}
- **Metrics**:
  - Perplexity: 22.653
  - LongPPL: 16.973
  - Passkey Accuracy: 1.000

### 4. llama3 (Perplexity: 22.746)

- **Context Length**: 2048
- **Configuration**: {'factor': 3.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 2048}
- **Metrics**:
  - Perplexity: 22.746
  - LongPPL: 13.686
  - Passkey Accuracy: 0.939

### 5. llama3 (Perplexity: 22.984)

- **Context Length**: 2048
- **Configuration**: {'factor': 4.0, 'low_freq_factor': 1.5, 'high_freq_factor': 6.0, 'original_max_position_embeddings': 2048}
- **Metrics**:
  - Perplexity: 22.984
  - LongPPL: 15.192
  - Passkey Accuracy: 0.987

### 6. yarn (Perplexity: 23.004)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 4.0, 'beta_fast': 32, 'beta_slow': 3, 's': 1.0}
- **Metrics**:
  - Perplexity: 23.004
  - LongPPL: 18.967
  - Passkey Accuracy: 0.983

### 7. yarn (Perplexity: 23.011)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 2.0, 'beta_fast': 32, 'beta_slow': 2, 's': 1.5}
- **Metrics**:
  - Perplexity: 23.011
  - LongPPL: 19.977
  - Passkey Accuracy: 0.941

### 8. llama3 (Perplexity: 23.016)

- **Context Length**: 2048
- **Configuration**: {'factor': 8.0, 'low_freq_factor': 0.5, 'high_freq_factor': 6.0, 'original_max_position_embeddings': 2048}
- **Metrics**:
  - Perplexity: 23.016
  - LongPPL: 21.909
  - Passkey Accuracy: 0.847

### 9. yarn (Perplexity: 23.066)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 2.0, 'beta_fast': 24, 'beta_slow': 1, 's': 0.5}
- **Metrics**:
  - Perplexity: 23.066
  - LongPPL: 21.862
  - Passkey Accuracy: 0.900

### 10. yarn (Perplexity: 23.081)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 2.0, 'beta_fast': 24, 'beta_slow': 3, 's': 1.0}
- **Metrics**:
  - Perplexity: 23.081
  - LongPPL: 20.020
  - Passkey Accuracy: 0.846

## Detailed Results

Complete results available in: `comprehensive_results_20250820_154136.json`

## Key Findings

- **Best performing method**: yarn
- **Most consistent method**: llama3
- **Context length performance**:
  - 2048: 24.328 average perplexity
  - 4096: 29.379 average perplexity
  - 8192: 39.380 average perplexity
  - 16384: 59.390 average perplexity
