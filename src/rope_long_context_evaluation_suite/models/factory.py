"""Factory for creating RoPE extensions."""

from typing import Any, Dict
from .base import BaseRoPEExtension
from .extensions import (
    LinearInterpolationRoPE,
    NTKAwareRoPE,
    YaRNRoPE, 
    LongRoPE,
    DynamicNTKRoPE,
    Llama3RoPE
)


# Registry of available RoPE extension methods
ROPE_EXTENSIONS = {
    "linear": LinearInterpolationRoPE,
    "linear_interpolation": LinearInterpolationRoPE,
    "ntk_aware": NTKAwareRoPE,
    "ntk": NTKAwareRoPE,
    "yarn": YaRNRoPE,
    "longrope": LongRoPE,
    "dynamic_ntk": DynamicNTKRoPE,
    "llama3": Llama3RoPE,
    "none": None,
}


def get_rope_extension(method: str, config: Dict[str, Any]) -> BaseRoPEExtension:
    """Factory function to create RoPE extension instances.
    
    Args:
        method: Name of the RoPE extension method
        config: Configuration dictionary for the method
        
    Returns:
        Instance of the specified RoPE extension
        
    Raises:
        ValueError: If method is not recognized
    """
    if method.lower() == "none":
        # Return a no-op extension for the "none" method
        return NoOpRoPEExtension(config)
    
    extension_class = ROPE_EXTENSIONS.get(method.lower())
    
    if extension_class is None:
        available_methods = [k for k in ROPE_EXTENSIONS.keys() if k != "none"]
        raise ValueError(
            f"Unknown RoPE extension method: {method}. "
            f"Available methods: {available_methods}"
        )
    
    return extension_class(config)


class NoOpRoPEExtension(BaseRoPEExtension):
    """No-operation RoPE extension that doesn't modify the model."""
    
    def apply(self, model):
        """Return the model unchanged."""
        self.scaling_info = {"method": "none", "scaling_factor": 1.0}
        return model


def list_available_methods():
    """List all available RoPE extension methods.
    
    Returns:
        List of available method names
    """
    return list(ROPE_EXTENSIONS.keys())