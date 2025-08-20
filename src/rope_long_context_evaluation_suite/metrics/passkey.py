"""Passkey retrieval metrics for long context evaluation."""

import random
import re
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseMetric


class PasskeyRetrievalMetric(BaseMetric):
    """Passkey retrieval metric for evaluating long context understanding."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize passkey retrieval metric.
        
        Args:
            config: Configuration dictionary with parameters:
                - num_samples: Number of samples to generate (default: 10)
                - min_context_length: Minimum context length (default: 1000)
                - max_context_length: Maximum context length (default: 32000)
                - passkey_length: Length of passkey (default: 5)
                - depth_percent: Where to place passkey (0-1, default: random)
                - noise_level: How much noise text to add (default: "medium")
        """
        super().__init__(config)
        self.num_samples = self.config.get('num_samples', 10)
        self.min_context_length = self.config.get('min_context_length', 1000)
        self.max_context_length = self.config.get('max_context_length', 32000)
        self.passkey_length = self.config.get('passkey_length', 5)
        self.depth_percent = self.config.get('depth_percent', None)  # None means random
        self.noise_level = self.config.get('noise_level', "medium")
        
        # Load or generate noise text
        self.noise_texts = self._generate_noise_texts()
        
    def _generate_noise_texts(self) -> List[str]:
        """Generate noise texts for creating haystack.
        
        Returns:
            List of noise text strings
        """
        # Sample noise texts (in practice, you'd load from a corpus)
        noise_samples = [
            "The quick brown fox jumps over the lazy dog. This is a common pangram used in typing practice.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore.",
            "In the beginning was the Word, and the Word was with God, and the Word was God.",
            "To be or not to be, that is the question: Whether 'tis nobler in the mind to suffer.",
            "It was the best of times, it was the worst of times, it was the age of wisdom.",
            "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
            "Happy families are all alike; every unhappy family is unhappy in its own way.",
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms.",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            "Space: the final frontier. These are the voyages of the starship Enterprise."
        ]
        
        # Expand the noise texts
        expanded_noise = []
        for text in noise_samples:
            # Create variations
            expanded_noise.extend([
                text,
                text.replace(".", "!"),
                text.replace("the", "a"),
                f"Furthermore, {text}",
                f"Additionally, {text}",
                f"Moreover, {text}"
            ])
        
        return expanded_noise
    
    def _generate_passkey(self) -> str:
        """Generate a random passkey.
        
        Returns:
            Random passkey string
        """
        digits = ''.join([str(random.randint(0, 9)) for _ in range(self.passkey_length)])
        return digits
    
    def _create_haystack_with_passkey(self, context_length: int, passkey: str, 
                                     depth_percent: float) -> str:
        """Create haystack text with embedded passkey.
        
        Args:
            context_length: Target context length in characters
            passkey: The passkey to embed
            depth_percent: Position to embed passkey (0.0 = start, 1.0 = end)
            
        Returns:
            Haystack text with embedded passkey
        """
        # Create the passkey statement
        passkey_statement = f"The pass key is {passkey}. Remember it."
        
        # Calculate position for passkey
        passkey_position = int(depth_percent * context_length)
        
        # Generate enough noise text
        noise_text = ""
        while len(noise_text) < context_length:
            noise_text += random.choice(self.noise_texts) + " "
        
        # Insert passkey at specified position
        before_passkey = noise_text[:passkey_position]
        after_passkey = noise_text[passkey_position + len(passkey_statement):]
        
        # Combine
        full_text = before_passkey + passkey_statement + after_passkey
        
        # Truncate to desired length
        return full_text[:context_length]
    
    def _create_retrieval_prompt(self, haystack: str) -> str:
        """Create prompt for passkey retrieval.
        
        Args:
            haystack: The haystack text containing the passkey
            
        Returns:
            Full prompt for retrieval task
        """
        prompt = f"""Here is some text:

{haystack}

What is the pass key mentioned in the text above? Please provide only the numeric pass key."""
        
        return prompt
    
    def compute(self,
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                text: Union[str, List[str]] = None,
                **kwargs) -> Dict[str, Any]:
        """Compute passkey retrieval accuracy.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            text: Not used for passkey retrieval (generated internally)
            **kwargs: Additional arguments including context_length
            
        Returns:
            Dictionary with passkey retrieval metrics
        """
        context_length = kwargs.get('context_length', self.max_context_length)
        
        correct_retrievals = 0
        total_samples = 0
        sample_results = []
        
        model.eval()
        with torch.no_grad():
            for i in range(self.num_samples):
                # Generate passkey
                passkey = self._generate_passkey()
                
                # Determine depth
                if self.depth_percent is not None:
                    depth = self.depth_percent
                else:
                    depth = random.uniform(0.1, 0.9)  # Avoid very beginning/end
                
                # Create haystack
                haystack = self._create_haystack_with_passkey(
                    context_length, passkey, depth
                )
                
                # Create retrieval prompt
                prompt = self._create_retrieval_prompt(haystack)
                
                # Tokenize
                inputs = self.prepare_inputs(model, tokenizer, prompt)
                
                # Generate response
                generation_config = {
                    'max_new_tokens': 20,
                    'do_sample': False,
                    'temperature': 0.0,
                    'pad_token_id': tokenizer.eos_token_id
                }
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Decode response (only the generated part)
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Check if passkey is retrieved correctly
                is_correct = self._check_passkey_retrieval(response, passkey)
                
                if is_correct:
                    correct_retrievals += 1
                total_samples += 1
                
                sample_results.append({
                    'passkey': passkey,
                    'depth_percent': depth,
                    'response': response.strip(),
                    'is_correct': is_correct,
                    'context_length': len(haystack)
                })
        
        # Calculate accuracy
        accuracy = correct_retrievals / total_samples if total_samples > 0 else 0.0
        
        result = {
            'passkey_accuracy': accuracy,
            'correct_retrievals': correct_retrievals,
            'total_samples': total_samples,
            'target_context_length': context_length,
            'sample_results': sample_results
        }
        
        self.update(result)
        return result
    
    def _check_passkey_retrieval(self, response: str, expected_passkey: str) -> bool:
        """Check if the response contains the correct passkey.
        
        Args:
            response: Model's response
            expected_passkey: The correct passkey
            
        Returns:
            True if passkey is correctly retrieved
        """
        # Clean the response
        response_clean = response.strip().lower()
        
        # Look for the exact passkey
        if expected_passkey in response_clean:
            return True
        
        # Look for numeric patterns
        numbers = re.findall(r'\d+', response_clean)
        for number in numbers:
            if number == expected_passkey:
                return True
        
        return False
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate passkey retrieval results.
        
        Args:
            results: List of individual passkey results
            
        Returns:
            Aggregated passkey retrieval metrics
        """
        if not results:
            return {}
        
        total_correct = sum(r['correct_retrievals'] for r in results)
        total_samples = sum(r['total_samples'] for r in results)
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Analyze by depth if available
        all_samples = []
        for r in results:
            all_samples.extend(r['sample_results'])
        
        # Group by depth ranges
        depth_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        depth_accuracies = {}
        
        for start, end in depth_ranges:
            range_samples = [s for s in all_samples 
                           if start <= s['depth_percent'] < end]
            if range_samples:
                range_accuracy = sum(s['is_correct'] for s in range_samples) / len(range_samples)
                depth_accuracies[f'depth_{start:.1f}-{end:.1f}'] = range_accuracy
        
        return {
            'overall_passkey_accuracy': overall_accuracy,
            'total_correct_retrievals': total_correct,
            'total_samples': total_samples,
            'depth_accuracies': depth_accuracies,
            'individual_accuracies': [r['passkey_accuracy'] for r in results]
        }


class MultiNeedleRetrievalMetric(PasskeyRetrievalMetric):
    """Multi-needle retrieval metric for more challenging evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-needle retrieval metric.
        
        Args:
            config: Configuration dictionary with additional parameters:
                - num_needles: Number of needles to embed (default: 3)
        """
        super().__init__(config)
        self.num_needles = self.config.get('num_needles', 3)
    
    def _create_haystack_with_multiple_passkeys(self, context_length: int, 
                                              passkeys: List[str]) -> str:
        """Create haystack with multiple embedded passkeys.
        
        Args:
            context_length: Target context length
            passkeys: List of passkeys to embed
            
        Returns:
            Haystack text with multiple passkeys
        """
        # Generate noise text
        noise_text = ""
        while len(noise_text) < context_length:
            noise_text += random.choice(self.noise_texts) + " "
        
        # Insert passkeys at different positions
        positions = [i / (len(passkeys) + 1) for i in range(1, len(passkeys) + 1)]
        
        text_parts = []
        last_pos = 0
        
        for i, (passkey, pos) in enumerate(zip(passkeys, positions)):
            char_pos = int(pos * len(noise_text))
            
            # Add text before passkey
            text_parts.append(noise_text[last_pos:char_pos])
            
            # Add passkey statement
            passkey_statement = f"The pass key {i+1} is {passkey}. Remember it."
            text_parts.append(passkey_statement)
            
            last_pos = char_pos
        
        # Add remaining text
        text_parts.append(noise_text[last_pos:])
        
        full_text = "".join(text_parts)
        return full_text[:context_length]
    
    def compute(self,
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                text: Union[str, List[str]] = None,
                **kwargs) -> Dict[str, Any]:
        """Compute multi-needle retrieval accuracy.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            text: Not used (generated internally)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with multi-needle retrieval metrics
        """
        context_length = kwargs.get('context_length', self.max_context_length)
        
        correct_retrievals = 0
        total_needles = 0
        sample_results = []
        
        model.eval()
        with torch.no_grad():
            for i in range(self.num_samples):
                # Generate multiple passkeys
                passkeys = [self._generate_passkey() for _ in range(self.num_needles)]
                
                # Create haystack
                haystack = self._create_haystack_with_multiple_passkeys(
                    context_length, passkeys
                )
                
                # Test retrieval for each passkey
                sample_correct = 0
                needle_results = []
                
                for j, passkey in enumerate(passkeys):
                    prompt = f"""Here is some text:

{haystack}

What is pass key {j+1} mentioned in the text above? Please provide only the numeric pass key."""
                    
                    # Generate response
                    inputs = self.prepare_inputs(model, tokenizer, prompt)
                    
                    generation_config = {
                        'max_new_tokens': 20,
                        'do_sample': False,
                        'temperature': 0.0,
                        'pad_token_id': tokenizer.eos_token_id
                    }
                    
                    outputs = model.generate(**inputs, **generation_config)
                    
                    input_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    is_correct = self._check_passkey_retrieval(response, passkey)
                    
                    if is_correct:
                        correct_retrievals += 1
                        sample_correct += 1
                    total_needles += 1
                    
                    needle_results.append({
                        'needle_id': j + 1,
                        'passkey': passkey,
                        'response': response.strip(),
                        'is_correct': is_correct
                    })
                
                sample_results.append({
                    'sample_id': i,
                    'total_needles': len(passkeys),
                    'correct_needles': sample_correct,
                    'sample_accuracy': sample_correct / len(passkeys),
                    'needle_results': needle_results,
                    'context_length': len(haystack)
                })
        
        # Calculate overall accuracy
        accuracy = correct_retrievals / total_needles if total_needles > 0 else 0.0
        
        result = {
            'multi_needle_accuracy': accuracy,
            'total_correct_needles': correct_retrievals,
            'total_needles': total_needles,
            'num_needles_per_sample': self.num_needles,
            'target_context_length': context_length,
            'sample_results': sample_results
        }
        
        self.update(result)
        return result