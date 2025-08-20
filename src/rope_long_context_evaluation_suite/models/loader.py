"""Model loading utilities."""

import logging
import torch
from typing import Any, Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and setup of language models."""
    
    def __init__(self, config: DictConfig):
        """Initialize model loader with configuration.
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.model_config = config.model
        
    def load_model_and_tokenizer(self) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """Load model and tokenizer based on configuration.
        
        Returns:
            Tuple of (model, tokenizer) or (None, None) for API models
        """
        model_type = self.model_config.get("type", "hf_hub")
        
        if model_type == "api":
            # For API models, we don't need to load actual model/tokenizer
            logger.info(f"Using API model: {self.model_config.name}")
            return None, None
        elif model_type == "hf_hub":
            return self._load_huggingface_model()
        elif model_type == "local":
            return self._load_local_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_huggingface_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from HuggingFace Hub."""
        model_name = self.model_config.get("name") or self.model_config.get("path")
        
        if not model_name:
            raise ValueError("Model name or path must be specified for HuggingFace models")
        
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": self._get_torch_dtype(),
            "device_map": self.model_config.get("device_map", "auto"),
            "trust_remote_code": self.model_config.get("trust_remote_code", False),
            "attn_implementation": self.model_config.get("attn_implementation", "eager"),
        }
        
        # Add max memory if specified
        max_memory_gb = self.model_config.get("max_memory_gb")
        if max_memory_gb:
            max_memory = {0: f"{max_memory_gb}GB"}
            model_kwargs["max_memory"] = max_memory
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            logger.info(f"Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Load tokenizer
        tokenizer_path = self.model_config.get("tokenizer_path", model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.model_config.get("trust_remote_code", False)
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info(f"Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise
        
        return model, tokenizer
    
    def _load_local_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from local path."""
        model_path = self.model_config.get("path")
        
        if not model_path:
            raise ValueError("Model path must be specified for local models")
        
        logger.info(f"Loading local model from: {model_path}")
        
        # Use similar logic as HuggingFace loading but from local path
        model_kwargs = {
            "torch_dtype": self._get_torch_dtype(),
            "device_map": self.model_config.get("device_map", "auto"),
            "trust_remote_code": self.model_config.get("trust_remote_code", False),
            "attn_implementation": self.model_config.get("attn_implementation", "eager"),
        }
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Local model loaded successfully")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load local model from {model_path}: {e}")
            raise
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get the appropriate torch dtype from config."""
        dtype_str = self.model_config.get("torch_dtype", "float16")
        
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto"
        }
        
        if dtype_str == "auto":
            return "auto"
        
        return dtype_mapping.get(dtype_str, torch.float16)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_config.get("name", "unknown"),
            "type": self.model_config.get("type", "unknown"),
            "path": self.model_config.get("path"),
            "max_length": self.model_config.get("max_length"),
            "torch_dtype": self.model_config.get("torch_dtype"),
            "device": self.model_config.get("device", "auto"),
        }