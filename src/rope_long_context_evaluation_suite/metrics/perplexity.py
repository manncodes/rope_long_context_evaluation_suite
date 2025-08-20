"""Perplexity metrics for evaluating language models."""

import math
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseMetric


class PerplexityMetric(BaseMetric):
    """Standard perplexity metric for language model evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize perplexity metric.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - batch_size: Batch size for processing (default: 1)
                - max_length: Maximum sequence length (default: model's max)
        """
        super().__init__(config)
        self.batch_size = self.config.get('batch_size', 1)
        self.max_length = self.config.get('max_length', None)
        
    def compute(self,
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                text: Union[str, List[str]],
                **kwargs) -> Dict[str, Any]:
        """Compute perplexity for given text.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            text: Input text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with perplexity and related metrics
        """
        if isinstance(text, str):
            text = [text]
            
        total_loss = 0.0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for txt in text:
                # Prepare inputs
                inputs = self.prepare_inputs(model, tokenizer, txt, self.max_length)
                input_ids = inputs['input_ids']
                
                # Compute loss
                outputs = model(**inputs, labels=input_ids)
                loss = outputs.loss
                
                # Accumulate loss and token count
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        result = {
            'perplexity': perplexity,
            'loss': avg_loss,
            'total_tokens': total_tokens,
            'num_sequences': len(text)
        }
        
        self.update(result)
        return result
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate perplexity results.
        
        Args:
            results: List of individual perplexity results
            
        Returns:
            Aggregated perplexity metrics
        """
        if not results:
            return {}
            
        total_loss = sum(r['loss'] * r['total_tokens'] for r in results)
        total_tokens = sum(r['total_tokens'] for r in results)
        total_sequences = sum(r['num_sequences'] for r in results)
        
        avg_loss = total_loss / total_tokens
        avg_perplexity = math.exp(avg_loss)
        
        # Calculate per-sequence statistics
        perplexities = [r['perplexity'] for r in results]
        
        return {
            'mean_perplexity': avg_perplexity,
            'weighted_perplexity': avg_perplexity,
            'std_perplexity': torch.tensor(perplexities).std().item(),
            'min_perplexity': min(perplexities),
            'max_perplexity': max(perplexities),
            'total_tokens': total_tokens,
            'total_sequences': total_sequences
        }


class SlidingWindowPerplexity(BaseMetric):
    """Sliding window perplexity for long sequences."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sliding window perplexity.
        
        Args:
            config: Configuration dictionary with parameters:
                - window_size: Size of sliding window (default: 256)
                - stride: Stride for sliding window (default: window_size)
                - overlap: Overlap between windows (default: 0)
        """
        super().__init__(config)
        self.window_size = self.config.get('window_size', 256)
        self.stride = self.config.get('stride', self.window_size)
        self.overlap = self.config.get('overlap', 0)
        
    def compute(self,
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                text: Union[str, List[str]],
                **kwargs) -> Dict[str, Any]:
        """Compute sliding window perplexity.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            text: Input text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with sliding window perplexity metrics
        """
        if isinstance(text, str):
            text = [text]
            
        all_window_losses = []
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for txt in text:
                # Tokenize full text
                full_inputs = tokenizer(txt, return_tensors="pt", padding=False)
                input_ids = full_inputs['input_ids'][0]  # Remove batch dimension
                
                if len(input_ids) <= self.window_size:
                    # Short text, compute regular perplexity
                    inputs = {k: v.to(next(model.parameters()).device) 
                             for k, v in full_inputs.items()}
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    all_window_losses.append(outputs.loss.item())
                    total_tokens += len(input_ids)
                else:
                    # Long text, use sliding window
                    window_losses = []
                    
                    for i in range(0, len(input_ids) - self.window_size + 1, 
                                  self.stride - self.overlap):
                        # Extract window
                        window_ids = input_ids[i:i + self.window_size]
                        
                        # Prepare inputs
                        window_inputs = {
                            'input_ids': window_ids.unsqueeze(0).to(next(model.parameters()).device)
                        }
                        
                        # Compute loss for this window
                        outputs = model(**window_inputs, labels=window_inputs['input_ids'])
                        window_losses.append(outputs.loss.item())
                        total_tokens += len(window_ids)
                    
                    all_window_losses.extend(window_losses)
        
        # Calculate perplexity from window losses
        if all_window_losses:
            avg_loss = sum(all_window_losses) / len(all_window_losses)
            perplexity = math.exp(avg_loss)
        else:
            avg_loss = float('inf')
            perplexity = float('inf')
        
        result = {
            'sliding_window_perplexity': perplexity,
            'sliding_window_loss': avg_loss,
            'num_windows': len(all_window_losses),
            'window_size': self.window_size,
            'total_tokens': total_tokens,
            'window_losses': all_window_losses
        }
        
        self.update(result)
        return result
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate sliding window perplexity results.
        
        Args:
            results: List of individual sliding window results
            
        Returns:
            Aggregated sliding window perplexity metrics
        """
        if not results:
            return {}
            
        all_losses = []
        for r in results:
            all_losses.extend(r['window_losses'])
        
        total_windows = sum(r['num_windows'] for r in results)
        total_tokens = sum(r['total_tokens'] for r in results)
        
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            avg_perplexity = math.exp(avg_loss)
            std_loss = torch.tensor(all_losses).std().item()
        else:
            avg_loss = float('inf')
            avg_perplexity = float('inf')
            std_loss = 0.0
        
        return {
            'mean_sliding_window_perplexity': avg_perplexity,
            'mean_sliding_window_loss': avg_loss,
            'std_sliding_window_loss': std_loss,
            'total_windows': total_windows,
            'total_tokens': total_tokens,
            'window_size': self.window_size
        }