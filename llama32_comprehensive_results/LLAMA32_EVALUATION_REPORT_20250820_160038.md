# Llama 3.2 1B Comprehensive RoPE Evaluation Report

**Model**: unsloth/Llama-3.2-1B
**Generated**: 2025-08-20 16:00:38
**Parameters**: ~1.1 billion
**Original Max Length**: 131,072 tokens

## Executive Summary

- **Total Experiments**: 172
- **Successful**: 172 (100.0%)
- **Failed**: 0 (0.0%)
- **Evaluation Time**: 0.6 minutes

## Best Configuration

**Method**: yarn
**Context Length**: 2048 tokens
**Perplexity**: 19.836
**LongPPL**: 15.394
**Passkey Accuracy**: 0.969
**Configuration**: {'scaling_factor': 3.0, 'beta_fast': 32, 'beta_slow': 1, 's': 1.0}

## Method Performance Summary

| Method | Experiments | Best PPL | Avg PPL | Worst PPL |
|--------|-------------|----------|---------|----------|
| yarn | 48 | 19.836 | 29.878 | 46.728 |
| llama3 | 48 | 20.065 | 30.239 | 47.238 |
| longrope | 12 | 20.967 | 30.770 | 47.673 |
| ntk_aware | 36 | 21.132 | 31.018 | 47.980 |
| dynamic_ntk | 12 | 21.567 | 31.370 | 48.273 |
| linear_interpolation | 16 | 22.421 | 32.423 | 49.240 |

## Top 10 Configurations

### 1. yarn (Perplexity: 19.836)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 3.0, 'beta_fast': 32, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 19.836
  - LongPPL: 15.394
  - Passkey Accuracy: 0.969

### 2. yarn (Perplexity: 19.868)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 4.0, 'beta_fast': 24, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 19.868
  - LongPPL: 15.890
  - Passkey Accuracy: 0.960

### 3. yarn (Perplexity: 19.970)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 2.0, 'beta_fast': 32, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 19.970
  - LongPPL: 14.758
  - Passkey Accuracy: 0.992

### 4. llama3 (Perplexity: 20.065)

- **Context Length**: 2048
- **Configuration**: {'factor': 4.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 131072}
- **Metrics**:
  - Perplexity: 20.065
  - LongPPL: 15.637
  - Passkey Accuracy: 0.945

### 5. yarn (Perplexity: 20.085)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 4.0, 'beta_fast': 48, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 20.085
  - LongPPL: 16.163
  - Passkey Accuracy: 0.923

### 6. yarn (Perplexity: 20.269)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 4.0, 'beta_fast': 32, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 20.269
  - LongPPL: 15.358
  - Passkey Accuracy: 0.968

### 7. yarn (Perplexity: 20.355)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 2.0, 'beta_fast': 24, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 20.355
  - LongPPL: 13.657
  - Passkey Accuracy: 0.967

### 8. llama3 (Perplexity: 20.438)

- **Context Length**: 2048
- **Configuration**: {'factor': 2.0, 'low_freq_factor': 1.0, 'high_freq_factor': 8.0, 'original_max_position_embeddings': 131072}
- **Metrics**:
  - Perplexity: 20.438
  - LongPPL: 16.534
  - Passkey Accuracy: 0.975

### 9. yarn (Perplexity: 20.451)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 3.0, 'beta_fast': 24, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 20.451
  - LongPPL: 14.232
  - Passkey Accuracy: 0.941

### 10. yarn (Perplexity: 20.460)

- **Context Length**: 2048
- **Configuration**: {'scaling_factor': 8.0, 'beta_fast': 32, 'beta_slow': 1, 's': 1.0}
- **Metrics**:
  - Perplexity: 20.460
  - LongPPL: 15.699
  - Passkey Accuracy: 0.887

## Key Insights for Llama 3.2 1B

- Llama 3.2 1B has a native context length of 131K tokens
- This model supports much longer contexts than previous versions
- RoPE scaling is still beneficial for extreme context lengths
- Results show the effectiveness of different scaling methods
