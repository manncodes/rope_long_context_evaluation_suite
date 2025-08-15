"""Base benchmark class for long context evaluation."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, config: Dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Initialize the benchmark.
        
        Args:
            config: Benchmark configuration
            model: The model to evaluate
            tokenizer: The tokenizer to use
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.name = self.__class__.__name__
        
        # Setup generation parameters
        self.generation_config = config.get("generation", {})
        
    @abstractmethod
    def load_data(self) -> List[Dict]:
        """Load benchmark data.
        
        Returns:
            List of data samples
        """
        pass
    
    @abstractmethod
    def prepare_input(self, sample: Dict) -> str:
        """Prepare input text for a sample.
        
        Args:
            sample: Data sample
            
        Returns:
            Formatted input text
        """
        pass
    
    @abstractmethod
    def extract_answer(self, response: str, sample: Dict) -> str:
        """Extract answer from model response.
        
        Args:
            response: Raw model response
            sample: Original data sample
            
        Returns:
            Extracted answer
        """
        pass
    
    @abstractmethod
    def compute_score(self, prediction: str, ground_truth: str, sample: Dict) -> float:
        """Compute score for a single prediction.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            sample: Original data sample
            
        Returns:
            Score (typically between 0 and 1)
        """
        pass
    
    def generate_response(self, input_text: str) -> str:
        """Generate response from model.
        
        Args:
            input_text: Input text to generate from
            
        Returns:
            Generated response
        """
        if self.model is None:
            raise ValueError("Model is None - this benchmark requires a local model")
        
        # Encode and ensure proper device placement
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            padding=False
        )
        
        # Move ALL inputs to the same device as the model
        if self.model is not None:
            # Get device from model parameters
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        if inputs['input_ids'].shape[1] > self.model.config.max_position_embeddings:
            logger.warning(f"Input length {inputs['input_ids'].shape[1]} exceeds model's max position embeddings "
                          f"{self.model.config.max_position_embeddings}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # Unpack the dict with input_ids and attention_mask
                max_new_tokens=self.generation_config.get("max_new_tokens", 256),
                temperature=self.generation_config.get("temperature", 0.0),
                do_sample=self.generation_config.get("do_sample", False),
                top_p=self.generation_config.get("top_p", 1.0),
                top_k=self.generation_config.get("top_k", 50),
                repetition_penalty=self.generation_config.get("repetition_penalty", 1.0),
                length_penalty=self.generation_config.get("length_penalty", 1.0),
                num_beams=self.generation_config.get("num_beams", 1),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def evaluate_sample(self, sample: Dict) -> Dict[str, Any]:
        """Evaluate a single sample.
        
        Args:
            sample: Data sample to evaluate
            
        Returns:
            Evaluation results
        """
        try:
            # Prepare input
            input_text = self.prepare_input(sample)
            
            # Generate response
            response = self.generate_response(input_text)
            
            # Extract answer
            prediction = self.extract_answer(response, sample)
            
            # Compute score
            ground_truth = sample.get("answer", sample.get("target", ""))
            score = self.compute_score(prediction, ground_truth, sample)
            
            return {
                "sample_id": sample.get("id", "unknown"),
                "input_length": len(self.tokenizer.encode(input_text)),
                "prediction": prediction,
                "ground_truth": ground_truth,
                "score": score,
                "raw_response": response,
            }
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample.get('id', 'unknown')}: {e}")
            return {
                "sample_id": sample.get("id", "unknown"),
                "error": str(e),
                "score": 0.0,
            }
    
    def evaluate(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate the benchmark.
        
        Args:
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation of {self.name}")
        
        # Load data
        data = self.load_data()
        
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
            logger.info(f"Limiting evaluation to {len(data)} samples")
        
        # Evaluate samples
        results = []
        for i, sample in enumerate(data):
            logger.info(f"Evaluating sample {i+1}/{len(data)}")
            result = self.evaluate_sample(sample)
            results.append(result)
        
        # Compute aggregate metrics
        valid_results = [r for r in results if "error" not in r]
        scores = [r["score"] for r in valid_results]
        
        if not scores:
            logger.error("No valid results obtained")
            return {
                "benchmark": self.name,
                "num_samples": len(data),
                "num_valid": 0,
                "error_rate": 1.0,
                "average_score": 0.0,
                "results": results,
            }
        
        aggregate_results = {
            "benchmark": self.name,
            "num_samples": len(data),
            "num_valid": len(valid_results),
            "error_rate": (len(data) - len(valid_results)) / len(data),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "results": results,
        }
        
        logger.info(f"Completed evaluation of {self.name}. Average score: {aggregate_results['average_score']:.3f}")
        
        return aggregate_results
    
    def get_info(self) -> Dict[str, Any]:
        """Get benchmark information.
        
        Returns:
            Dictionary containing benchmark information
        """
        return {
            "name": self.name,
            "config": self.config,
            "generation_config": self.generation_config,
        }