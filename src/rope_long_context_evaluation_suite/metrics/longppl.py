"""LongPPL metric implementation for long context evaluation."""

import math
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseMetric


class LongPPLMetric(BaseMetric):
    """LongPPL (Long-context Perplexity) metric implementation.
    
    LongPPL focuses on key tokens through a long-short context contrastive method,
    providing better correlation with long-context abilities than standard perplexity.
    
    Paper: "What is Wrong with Perplexity for Long-context Language Modeling?"
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LongPPL metric.
        
        Args:
            config: Configuration dictionary with parameters:
                - short_context_window: Window size for short context (default: 1024)
                - long_context_window: Window size for long context (default: 4096)
                - key_token_threshold: Threshold for identifying key tokens (default: 0.1)
                - discriminator_model: Model for computing LongPPL (default: None, uses eval model)
        """
        super().__init__(config)
        self.short_context_window = self.config.get('short_context_window', 1024)
        self.long_context_window = self.config.get('long_context_window', 4096)
        self.key_token_threshold = self.config.get('key_token_threshold', 0.1)
        self.discriminator_model_name = self.config.get('discriminator_model', None)
        self.discriminator_model = None
        
    def _load_discriminator_model(self) -> PreTrainedModel:
        """Load discriminator model for computing LongPPL.
        
        Returns:
            Discriminator model instance
        """
        if self.discriminator_model is None and self.discriminator_model_name:
            from transformers import AutoModelForCausalLM
            self.discriminator_model = AutoModelForCausalLM.from_pretrained(
                self.discriminator_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        return self.discriminator_model
    
    def _identify_key_tokens(self, 
                           model: PreTrainedModel,
                           tokenizer: PreTrainedTokenizer,
                           text: str) -> List[int]:
        """Identify key tokens using long-short context contrastive method.
        
        Args:
            model: Model to use for identifying key tokens
            tokenizer: Tokenizer for the model
            text: Input text to analyze
            
        Returns:
            List of key token indices
        """
        # Use discriminator model if available, otherwise use the evaluation model
        discriminator = self._load_discriminator_model() or model
        
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs['input_ids'][0]  # Remove batch dimension
        
        if len(input_ids) <= self.short_context_window:
            # Text is too short for contrastive analysis
            return list(range(len(input_ids)))
        
        device = next(discriminator.parameters()).device
        input_ids = input_ids.to(device)
        
        key_token_indices = []
        
        discriminator.eval()
        with torch.no_grad():
            for i in range(self.short_context_window, len(input_ids)):
                # Short context: only recent tokens
                short_start = max(0, i - self.short_context_window)
                short_context = input_ids[short_start:i+1].unsqueeze(0)
                
                # Long context: extended context
                long_start = max(0, i - self.long_context_window)
                long_context = input_ids[long_start:i+1].unsqueeze(0)
                
                # Compute probabilities for target token
                target_token_id = input_ids[i]
                
                # Short context probability
                short_outputs = discriminator(short_context)
                short_logits = short_outputs.logits[0, -1]  # Last position
                short_prob = F.softmax(short_logits, dim=-1)[target_token_id]
                
                # Long context probability
                long_outputs = discriminator(long_context)
                long_logits = long_outputs.logits[0, -1]  # Last position
                long_prob = F.softmax(long_logits, dim=-1)[target_token_id]
                
                # Compute probability difference
                prob_diff = long_prob - short_prob
                
                # Token is "key" if long context significantly helps
                if prob_diff.item() > self.key_token_threshold:
                    key_token_indices.append(i)
        
        return key_token_indices
    
    def _compute_longppl_loss(self,
                             model: PreTrainedModel,
                             tokenizer: PreTrainedTokenizer,
                             text: str,
                             key_token_indices: List[int]) -> float:
        """Compute LongPPL loss focusing on key tokens.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            text: Input text
            key_token_indices: Indices of key tokens
            
        Returns:
            LongPPL loss value
        """
        if not key_token_indices:
            return float('inf')
        
        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs['input_ids']
        
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        model.eval()
        with torch.no_grad():
            # Compute loss for the entire sequence
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits
            
            # Extract losses for key tokens only
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute loss per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Select losses for key tokens (adjust indices for shifting)
            key_losses = []
            for idx in key_token_indices:
                if 0 <= idx - 1 < len(token_losses):  # -1 due to shifting
                    key_losses.append(token_losses[idx - 1].item())
            
            if key_losses:
                longppl_loss = sum(key_losses) / len(key_losses)
            else:
                longppl_loss = float('inf')
        
        return longppl_loss
    
    def compute(self,
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                text: Union[str, List[str]],
                **kwargs) -> Dict[str, Any]:
        """Compute LongPPL for given text.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            text: Input text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with LongPPL metrics
        """
        if isinstance(text, str):
            text = [text]
        
        all_longppl_losses = []
        all_key_token_ratios = []
        total_key_tokens = 0
        total_tokens = 0
        
        for txt in text:
            # Identify key tokens
            key_token_indices = self._identify_key_tokens(model, tokenizer, txt)
            
            # Compute LongPPL loss
            longppl_loss = self._compute_longppl_loss(model, tokenizer, txt, key_token_indices)
            
            if not math.isinf(longppl_loss):
                all_longppl_losses.append(longppl_loss)
            
            # Calculate key token ratio
            total_text_tokens = len(tokenizer(txt)['input_ids'])
            key_token_ratio = len(key_token_indices) / total_text_tokens if total_text_tokens > 0 else 0
            all_key_token_ratios.append(key_token_ratio)
            
            total_key_tokens += len(key_token_indices)
            total_tokens += total_text_tokens
        
        # Aggregate results
        if all_longppl_losses:
            avg_longppl_loss = sum(all_longppl_losses) / len(all_longppl_losses)
            longppl_perplexity = math.exp(avg_longppl_loss)
        else:
            avg_longppl_loss = float('inf')
            longppl_perplexity = float('inf')
        
        avg_key_token_ratio = sum(all_key_token_ratios) / len(all_key_token_ratios) if all_key_token_ratios else 0
        
        result = {
            'longppl': longppl_perplexity,
            'longppl_loss': avg_longppl_loss,
            'key_token_ratio': avg_key_token_ratio,
            'total_key_tokens': total_key_tokens,
            'total_tokens': total_tokens,
            'num_sequences': len(text),
            'valid_sequences': len(all_longppl_losses)
        }
        
        self.update(result)
        return result
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate LongPPL results.
        
        Args:
            results: List of individual LongPPL results
            
        Returns:
            Aggregated LongPPL metrics
        """
        if not results:
            return {}
        
        # Weight by valid sequences
        weighted_loss = 0
        total_weight = 0
        
        for r in results:
            if r['valid_sequences'] > 0 and not math.isinf(r['longppl_loss']):
                weighted_loss += r['longppl_loss'] * r['valid_sequences']
                total_weight += r['valid_sequences']
        
        if total_weight > 0:
            avg_longppl_loss = weighted_loss / total_weight
            avg_longppl = math.exp(avg_longppl_loss)
        else:
            avg_longppl_loss = float('inf')
            avg_longppl = float('inf')
        
        # Aggregate other metrics
        total_key_tokens = sum(r['total_key_tokens'] for r in results)
        total_tokens = sum(r['total_tokens'] for r in results)
        total_sequences = sum(r['num_sequences'] for r in results)
        total_valid = sum(r['valid_sequences'] for r in results)
        
        avg_key_token_ratio = total_key_tokens / total_tokens if total_tokens > 0 else 0
        
        # Individual LongPPL values for statistics
        individual_longppls = [r['longppl'] for r in results if not math.isinf(r['longppl'])]
        
        return {
            'mean_longppl': avg_longppl,
            'mean_longppl_loss': avg_longppl_loss,
            'mean_key_token_ratio': avg_key_token_ratio,
            'total_key_tokens': total_key_tokens,
            'total_tokens': total_tokens,
            'total_sequences': total_sequences,
            'valid_sequences': total_valid,
            'std_longppl': torch.tensor(individual_longppls).std().item() if individual_longppls else 0,
            'min_longppl': min(individual_longppls) if individual_longppls else float('inf'),
            'max_longppl': max(individual_longppls) if individual_longppls else float('inf')
        }