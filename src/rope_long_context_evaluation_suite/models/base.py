"""Base classes for RoPE extensions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from transformers import PreTrainedModel


class BaseRoPEExtension(ABC):
    """Base class for all RoPE extension methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RoPE extension with configuration.
        
        Args:
            config: Configuration dictionary for the specific RoPE method
        """
        self.config = config
        self.scaling_info = {}
    
    @abstractmethod
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply the RoPE extension to a model.
        
        Args:
            model: The model to apply RoPE extension to
            
        Returns:
            Model with RoPE extension applied
        """
        pass
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """Get information about the scaling applied.
        
        Returns:
            Dictionary containing scaling information
        """
        return {
            "method": self.__class__.__name__,
            "config": self.config,
            **self.scaling_info
        }
    
    def _modify_rope_embeddings(
        self, 
        model: PreTrainedModel, 
        scaling_factor: float,
        original_max_length: Optional[int] = None
    ):
        """Helper method to modify RoPE embeddings in the model.
        
        Args:
            model: The model to modify
            scaling_factor: Factor to scale position embeddings
            original_max_length: Original maximum sequence length
        """
        # Store scaling info
        self.scaling_info.update({
            "scaling_factor": scaling_factor,
            "original_max_length": original_max_length
        })
        
        # Apply to all layers with rotary embeddings
        for name, module in model.named_modules():
            if hasattr(module, 'rotary_emb'):
                self._apply_to_rotary_emb(module.rotary_emb, scaling_factor, original_max_length)
    
    def _apply_to_rotary_emb(self, rotary_emb, scaling_factor: float, original_max_length: Optional[int]):
        """Apply scaling to a specific rotary embedding module.
        
        Args:
            rotary_emb: The rotary embedding module
            scaling_factor: Scaling factor to apply
            original_max_length: Original maximum sequence length
        """
        # This is a generic implementation - subclasses should override for specific behavior
        if hasattr(rotary_emb, 'max_seq_len_cached'):
            rotary_emb.max_seq_len_cached = int(rotary_emb.max_seq_len_cached * scaling_factor)
        
        if hasattr(rotary_emb, 'scaling_factor'):
            rotary_emb.scaling_factor = scaling_factor