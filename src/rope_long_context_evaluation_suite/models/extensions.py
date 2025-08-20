"""RoPE extension implementations."""

import math
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .base import BaseRoPEExtension


class LinearInterpolationRoPE(BaseRoPEExtension):
    """Linear interpolation RoPE scaling method."""
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply linear interpolation scaling to the model."""
        scaling_factor = self.config.get("scaling_factor", 2.0)
        
        # Apply linear interpolation scaling
        for name, module in model.named_modules():
            if hasattr(module, 'rotary_emb'):
                self._apply_linear_scaling(module.rotary_emb, scaling_factor)
        
        self.scaling_info = {"scaling_factor": scaling_factor, "method": "linear"}
        return model
    
    def _apply_linear_scaling(self, rotary_emb, scaling_factor: float):
        """Apply linear scaling to rotary embeddings."""
        if hasattr(rotary_emb, 'inv_freq'):
            # Scale the inverse frequencies
            rotary_emb.inv_freq = rotary_emb.inv_freq / scaling_factor
        
        # Update cached length if present
        if hasattr(rotary_emb, 'max_seq_len_cached'):
            rotary_emb.max_seq_len_cached = int(rotary_emb.max_seq_len_cached * scaling_factor)


class NTKAwareRoPE(BaseRoPEExtension):
    """NTK-aware RoPE scaling method."""
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply NTK-aware scaling to the model."""
        scaling_factor = self.config.get("scaling_factor", 2.0)
        alpha = self.config.get("alpha", 8.0)
        
        for name, module in model.named_modules():
            if hasattr(module, 'rotary_emb'):
                self._apply_ntk_scaling(module.rotary_emb, scaling_factor, alpha)
        
        self.scaling_info = {
            "scaling_factor": scaling_factor, 
            "alpha": alpha,
            "method": "ntk_aware"
        }
        return model
    
    def _apply_ntk_scaling(self, rotary_emb, scaling_factor: float, alpha: float):
        """Apply NTK-aware scaling."""
        if hasattr(rotary_emb, 'inv_freq'):
            dim = len(rotary_emb.inv_freq) * 2  # Dimension of embeddings
            base = getattr(rotary_emb, 'base', 10000.0)
            
            # NTK-aware scaling
            new_base = base * (alpha * scaling_factor)**(dim / (dim - 2))
            rotary_emb.inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


class YaRNRoPE(BaseRoPEExtension):
    """YaRN (Yet another RoPE extensioN) scaling method."""
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply YaRN scaling to the model."""
        scaling_factor = self.config.get("scaling_factor", 2.0)
        original_max_position_embeddings = self.config.get("original_max_position_embeddings", 2048)
        attention_factor = self.config.get("attention_factor", 0.1)
        beta_fast = self.config.get("beta_fast", 32)
        beta_slow = self.config.get("beta_slow", 1)
        
        for name, module in model.named_modules():
            if hasattr(module, 'rotary_emb'):
                self._apply_yarn_scaling(
                    module.rotary_emb, 
                    scaling_factor, 
                    original_max_position_embeddings,
                    attention_factor,
                    beta_fast,
                    beta_slow
                )
        
        self.scaling_info = {
            "scaling_factor": scaling_factor,
            "attention_factor": attention_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "method": "yarn"
        }
        return model
    
    def _apply_yarn_scaling(self, rotary_emb, scaling_factor: float, 
                           original_max_pos: int, attention_factor: float,
                           beta_fast: int, beta_slow: int):
        """Apply YaRN scaling."""
        if hasattr(rotary_emb, 'inv_freq'):
            dim = len(rotary_emb.inv_freq) * 2
            
            # YaRN interpolation
            low_freq_wavelen = original_max_pos / beta_fast
            high_freq_wavelen = original_max_pos / beta_slow
            
            freqs = 1.0 / rotary_emb.inv_freq
            wavelens = 2 * math.pi / freqs
            
            # Apply YaRN scaling based on frequency
            new_freqs = torch.where(
                wavelens < high_freq_wavelen,
                freqs,  # High frequency: no scaling
                torch.where(
                    wavelens > low_freq_wavelen,
                    freqs / scaling_factor,  # Low frequency: linear scaling
                    # Medium frequency: smooth interpolation
                    freqs * (1 - attention_factor) / scaling_factor + freqs * attention_factor
                )
            )
            
            rotary_emb.inv_freq = 1.0 / new_freqs


class LongRoPE(BaseRoPEExtension):
    """LongRoPE scaling method with evolutionary search."""
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply LongRoPE scaling to the model."""
        scaling_factor = self.config.get("scaling_factor", 2.0)
        original_max_position_embeddings = self.config.get("original_max_position_embeddings", 2048)
        short_factor = self.config.get("short_factor", [1.0, 1.0])
        long_factor = self.config.get("long_factor", [2.0, 2.0])
        
        for name, module in model.named_modules():
            if hasattr(module, 'rotary_emb'):
                self._apply_longrope_scaling(
                    module.rotary_emb,
                    scaling_factor,
                    original_max_position_embeddings,
                    short_factor,
                    long_factor
                )
        
        self.scaling_info = {
            "scaling_factor": scaling_factor,
            "short_factor": short_factor,
            "long_factor": long_factor,
            "method": "longrope"
        }
        return model
    
    def _apply_longrope_scaling(self, rotary_emb, scaling_factor: float,
                               original_max_pos: int, short_factor: list, long_factor: list):
        """Apply LongRoPE scaling with different factors for different dimensions."""
        if hasattr(rotary_emb, 'inv_freq'):
            dim = len(rotary_emb.inv_freq) * 2
            
            # Apply different scaling factors to different dimensions
            # This is a simplified implementation - real LongRoPE uses evolutionary search
            mid_dim = dim // 2
            
            freqs = 1.0 / rotary_emb.inv_freq
            new_freqs = torch.zeros_like(freqs)
            
            # Apply short factor to first half
            new_freqs[:mid_dim//2] = freqs[:mid_dim//2] / short_factor[0]
            # Apply long factor to second half  
            new_freqs[mid_dim//2:] = freqs[mid_dim//2:] / long_factor[0]
            
            rotary_emb.inv_freq = 1.0 / new_freqs


class DynamicNTKRoPE(BaseRoPEExtension):
    """Dynamic NTK RoPE scaling method."""
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply dynamic NTK scaling to the model."""
        scaling_factor = self.config.get("scaling_factor", 2.0)
        original_max_position_embeddings = self.config.get("original_max_position_embeddings", 2048)
        
        for name, module in model.named_modules():
            if hasattr(module, 'rotary_emb'):
                self._apply_dynamic_ntk_scaling(
                    module.rotary_emb,
                    scaling_factor,
                    original_max_position_embeddings
                )
        
        self.scaling_info = {
            "scaling_factor": scaling_factor,
            "original_max_position_embeddings": original_max_position_embeddings,
            "method": "dynamic_ntk"
        }
        return model
    
    def _apply_dynamic_ntk_scaling(self, rotary_emb, scaling_factor: float, original_max_pos: int):
        """Apply dynamic NTK scaling that adapts based on sequence length."""
        if hasattr(rotary_emb, 'inv_freq'):
            dim = len(rotary_emb.inv_freq) * 2
            base = getattr(rotary_emb, 'base', 10000.0)
            
            # Dynamic scaling based on current sequence length vs original
            alpha = scaling_factor ** (dim / (dim - 2))
            new_base = base * alpha
            
            rotary_emb.inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


class Llama3RoPE(BaseRoPEExtension):
    """Llama 3 official RoPE scaling method."""
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply Llama 3 RoPE scaling to the model."""
        scaling_factor = self.config.get("scaling_factor", 8.0)
        low_freq_factor = self.config.get("low_freq_factor", 1.0)
        high_freq_factor = self.config.get("high_freq_factor", 4.0)
        original_max_position_embeddings = self.config.get("original_max_position_embeddings", 8192)
        
        for name, module in model.named_modules():
            if hasattr(module, 'rotary_emb'):
                self._apply_llama3_scaling(
                    module.rotary_emb,
                    scaling_factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings
                )
        
        self.scaling_info = {
            "scaling_factor": scaling_factor,
            "low_freq_factor": low_freq_factor,
            "high_freq_factor": high_freq_factor,
            "method": "llama3"
        }
        return model
    
    def _apply_llama3_scaling(self, rotary_emb, scaling_factor: float,
                             low_freq_factor: float, high_freq_factor: float,
                             original_max_pos: int):
        """Apply Llama 3 frequency-dependent scaling."""
        if hasattr(rotary_emb, 'inv_freq'):
            # Llama 3 uses frequency-dependent scaling
            freqs = 1.0 / rotary_emb.inv_freq
            
            # Define frequency thresholds
            low_freq_threshold = original_max_pos / (2 * math.pi)
            high_freq_threshold = original_max_pos / (8 * math.pi)
            
            # Apply scaling based on frequency
            new_freqs = torch.where(
                freqs < high_freq_threshold,
                freqs / high_freq_factor,  # High frequency scaling
                torch.where(
                    freqs > low_freq_threshold,
                    freqs / low_freq_factor,   # Low frequency scaling
                    freqs / scaling_factor     # Medium frequency scaling
                )
            )
            
            rotary_emb.inv_freq = 1.0 / new_freqs