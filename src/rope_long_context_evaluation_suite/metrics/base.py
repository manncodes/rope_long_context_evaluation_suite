"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the metric.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results = []
        
    @abstractmethod
    def compute(self, 
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                text: Union[str, List[str]],
                **kwargs) -> Dict[str, Any]:
        """Compute the metric for given text.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            text: Input text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing metric results
        """
        pass
    
    @abstractmethod
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple metric results.
        
        Args:
            results: List of individual metric results
            
        Returns:
            Aggregated metric results
        """
        pass
    
    def reset(self):
        """Reset the metric state."""
        self.results = []
    
    def update(self, result: Dict[str, Any]):
        """Update metric with a new result.
        
        Args:
            result: New metric result
        """
        self.results.append(result)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all stored results.
        
        Returns:
            List of all results
        """
        return self.results
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        """Get aggregated results.
        
        Returns:
            Aggregated metric results
        """
        if not self.results:
            return {}
        return self.aggregate(self.results)
    
    @staticmethod
    def prepare_inputs(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       text: str,
                       max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model evaluation.
        
        Args:
            model: The model to prepare inputs for
            tokenizer: The tokenizer to use
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of model inputs
        """
        # Tokenize text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs