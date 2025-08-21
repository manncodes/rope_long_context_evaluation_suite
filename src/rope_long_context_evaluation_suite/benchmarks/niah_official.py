"""Official NIAH benchmark implementation using Greg Kamradt's LLMTest_NeedleInAHaystack."""

import json
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional

# Add third_party to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "third_party" / "LLMTest_NeedleInAHaystack"))

try:
    from needlehaystack.llm_needle_haystack_tester import LLMNeedleHaystackTester
    from needlehaystack.providers.model import ModelProvider
    from needlehaystack.evaluators.evaluator import Evaluator
    NIAH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import NIAH official implementation: {e}")
    LLMNeedleHaystackTester = None
    ModelProvider = object  # Fallback base class
    Evaluator = object  # Fallback base class
    NIAH_AVAILABLE = False

from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class CustomModelProvider(ModelProvider if NIAH_AVAILABLE else object):
    """Custom model provider that wraps our model and tokenizer."""
    
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.model_name = getattr(model.config, 'name_or_path', 'unknown')
        
    def generate_text(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using our model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.get("max_new_tokens", max_tokens),
                    temperature=self.generation_config.get("temperature", 0.0),
                    do_sample=self.generation_config.get("do_sample", False),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def encode_text_to_tokens(self, text: str) -> list:
        """Encode text to tokens."""
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list, context_length: int = None) -> str:
        """Decode tokens to text."""
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    async def evaluate_model(self, prompt: str) -> str:
        """Evaluate model with async interface."""
        return self.generate_text(prompt)
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str:
        """Generate prompt for NIAH test."""
        return f"Context: {context}\n\nQuestion: {retrieval_question}\n\nAnswer:"


class CustomEvaluator(Evaluator if NIAH_AVAILABLE else object):
    """Custom evaluator for NIAH results with regex extraction and LLM judge options."""
    
    CRITERIA = {"accuracy": "Score 1 if the needle is extracted correctly from response, 0 otherwise"}
    
    def __init__(self, needle: str = None, evaluation_method: str = "regex"):
        self.needle = needle or "The secret key is XGXGWKEU42ZKDP and should be remembered for later use."
        self.evaluation_method = evaluation_method  # "regex", "exact", "llm_judge"
        
        # Extract the actual secret key from the needle sentence
        self.secret_key = self._extract_key_from_needle(self.needle)
    
    def _extract_key_from_needle(self, needle: str) -> str:
        """Extract the secret key from the needle sentence."""
        import re
        
        # Pattern to extract key from sentences like "The secret key is XGXGWKEU42ZKDP and..."
        patterns = [
            r'(?:secret key|key|password|code)\s+is\s+([A-Z0-9]+)',
            r'(?:secret key|key|password|code):\s*([A-Z0-9]+)',
            r'remember\s+([A-Z0-9]{8,})',
            r'([A-Z0-9]{8,})',  # Fallback: any long alphanumeric string
        ]
        
        for pattern in patterns:
            match = re.search(pattern, needle, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern matches, return the original needle
        return needle
    
    def evaluate_response(self, response: str) -> int:
        """Evaluate if the needle is correctly extracted from the response."""
        try:
            if self.evaluation_method == "regex":
                return self._evaluate_with_regex(response)
            elif self.evaluation_method == "exact":
                return self._evaluate_exact_match(response)
            elif self.evaluation_method == "llm_judge":
                return self._evaluate_with_llm_judge(response)
            else:
                return self._evaluate_with_regex(response)  # Default
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return 0
    
    def _evaluate_with_regex(self, response: str) -> int:
        """Extract secret key using regex patterns and match with expected key."""
        import re
        
        # Patterns for extracting secret keys from responses
        patterns = [
            # Direct key extraction
            r'\b' + re.escape(self.secret_key) + r'\b',
            # After "secret key is" or similar  
            r'(?:secret key|key|password|code)\s+is\s+([A-Z0-9]+)',
            r'(?:secret key|key|password|code):\s*([A-Z0-9]+)',
            # After "answer is" or similar
            r'(?:answer\s+is|answer:|is)\s*([A-Z0-9]+)',
            # Standalone alphanumeric keys
            r'\b([A-Z0-9]{8,})\b',
            # After question marks
            r'\?\s*([A-Z0-9]+)',
        ]
        
        # First try exact secret key match
        key_pattern = r'\b' + re.escape(self.secret_key) + r'\b'
        if re.search(key_pattern, response, re.IGNORECASE):
            logger.debug(f"NIAH: Found exact secret key '{self.secret_key}' in response")
            return 1
        
        # Then try extracting potential keys
        for pattern in patterns[1:]:  # Skip the exact match pattern we already tried
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                
                # Clean the match
                cleaned_match = str(match).strip().upper()
                key_clean = str(self.secret_key).strip().upper()
                
                if cleaned_match == key_clean:
                    logger.debug(f"NIAH: Extracted '{match}' matches secret key '{self.secret_key}'")
                    return 1
        
        logger.debug(f"NIAH: No valid secret key found. Expected: '{self.secret_key}', Response: '{response[:100]}...'")
        return 0
    
    def _evaluate_exact_match(self, response: str) -> int:
        """Simple exact substring matching (fallback method)."""
        key_lower = self.secret_key.lower()
        response_lower = response.lower()
        return 1 if key_lower in response_lower else 0
    
    def _evaluate_with_llm_judge(self, response: str) -> int:
        """Use LLM as judge to evaluate if response contains the needle."""
        try:
            judge_prompt = f"""Does the text contain the secret key {self.secret_key}?

Text: {response}

Answer: Yes or No?"""

            # Try to use the model that's already loaded for this evaluation
            if hasattr(self, '_judge_model') and self._judge_model:
                # Use the loaded model for evaluation
                import torch
                inputs = self._judge_tokenizer(judge_prompt, return_tensors="pt", max_length=512, truncation=True)
                
                if hasattr(self._judge_model, 'device'):
                    device = next(self._judge_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._judge_model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self._judge_tokenizer.pad_token_id,
                        eos_token_id=self._judge_tokenizer.eos_token_id,
                    )
                
                # Decode only new tokens
                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[0][input_length:]
                judge_response = self._judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                # Parse the judge response
                logger.info(f"NIAH LLM Judge Debug:")
                logger.info(f"  Secret Key: '{self.secret_key}'")
                logger.info(f"  Model Response: '{response[:50]}...'" if len(response) > 50 else f"  Model Response: '{response}'")
                logger.info(f"  Judge Response: '{judge_response}'")
                
                judge_lower = judge_response.lower().strip()
                if (judge_response.startswith('1') or 
                    'yes' in judge_lower or 
                    'correct' in judge_lower or
                    'true' in judge_lower):
                    logger.info(f"  Decision: Found needle - Score: 1")
                    return 1
                else:
                    logger.info(f"  Decision: Needle not found - Score: 0")
                    return 0
            else:
                # Fallback to regex if no judge model available
                logger.warning("No LLM judge model available, falling back to regex evaluation")
                return self._evaluate_with_regex(response)
                
        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {e}")
            logger.warning("LLM judge failed, falling back to regex evaluation")
            return self._evaluate_with_regex(response)


class NIAHOfficialBenchmark(BaseBenchmark):
    """Official NIAH benchmark using Greg Kamradt's implementation."""
    
    def __init__(self, config: Dict, model, tokenizer):
        """Initialize official NIAH benchmark."""
        super().__init__(config, model, tokenizer)
        
        if not NIAH_AVAILABLE:
            raise ImportError(
                "NIAH official implementation not available. "
                "Install dependencies with: pip install -e .[benchmarks] "
                "or run: ./setup_benchmarks.sh"
            )
        
        # Configuration
        self.context_lengths = config.get("context_lengths", [2048, 4096])
        self.document_depth_percents = config.get("depth_percents", [10, 50, 90])
        self.num_tests = config.get("num_tests", 3)
        
        # NIAH specific configuration
        self.needle = config.get("needle", "The secret key is XGXGWKEU42ZKDP and should be remembered for later use.")
        self.retrieval_question = config.get("retrieval_question", "What is the secret key mentioned in the document?")
        self.evaluation_method = config.get("evaluation_method", "regex")
        
        # Setup custom providers
        self.model_provider = CustomModelProvider(model, tokenizer, self.generation_config)
        self.evaluator = CustomEvaluator(self.needle, self.evaluation_method)
        
        # If using LLM judge, provide access to the model
        if self.evaluation_method == "llm_judge":
            self.evaluator._judge_model = model
            self.evaluator._judge_tokenizer = tokenizer
        
        # Initialize NIAH tester
        self.niah_tester = LLMNeedleHaystackTester(
            model_to_test=self.model_provider,
            evaluator=self.evaluator,
            needle=self.needle,
            retrieval_question=self.retrieval_question,
            context_lengths=self.context_lengths,
            document_depth_percents=self.document_depth_percents,
            save_results=False,  # We'll handle results ourselves
            save_contexts=False,
            print_ongoing_status=True,
            num_concurrent_requests=1
        )
        
    def load_data(self) -> List[Dict]:
        """Generate NIAH test configurations."""
        data = []
        test_id = 0
        
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                for i in range(self.num_tests):
                    data.append({
                        "id": f"niah_{test_id}",
                        "context_length": context_length,
                        "depth_percent": depth_percent,
                        "test_number": i + 1,
                        "input": f"NIAH test with {context_length} tokens at {depth_percent}% depth",
                        "expected_output": "needle_content"  # Will be set during testing
                    })
                    test_id += 1
        
        logger.info(f"Generated {len(data)} NIAH test configurations")
        return data
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single NIAH sample using official implementation."""
        try:
            import asyncio
            
            # Configure the tester for this specific test  
            context_length = sample["context_length"]
            depth_percent = sample["depth_percent"]
            
            # Create a new event loop for this evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the official NIAH evaluation
                result = loop.run_until_complete(
                    self.niah_tester.evaluate_and_log(context_length, depth_percent)
                )
                
                # The official implementation doesn't return results directly
                # We need to check if the evaluation was successful by running a test
                context = loop.run_until_complete(
                    self.niah_tester.generate_context(context_length, depth_percent)
                )
                
                prompt = self.model_provider.generate_prompt(context, self.retrieval_question)
                
                # Debug: Show what prompt is actually being used
                logger.info(f"NIAH Prompt Debug:")
                logger.info(f"  Context length: {len(context)} chars")
                logger.info(f"  Needle: '{self.needle}'")
                logger.info(f"  Question: '{self.retrieval_question}'")
                logger.info(f"  Prompt preview: '{prompt[-200:]}...' (last 200 chars)")
                
                response = loop.run_until_complete(self.model_provider.evaluate_model(prompt))
                
                # Use the official evaluator to score the response
                score = self.evaluator.evaluate_response(response)
                
                return {
                    "sample_id": sample["id"],
                    "context_length": context_length,
                    "depth_percent": depth_percent,
                    "score": score,
                    "prediction": response.strip(),
                    "target": self.needle,
                    "metadata": {
                        "test_number": sample["test_number"],
                        "evaluation_method": "official_niah",
                        "needle": self.needle,
                        "retrieval_question": self.retrieval_question
                    }
                }
                
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Error evaluating NIAH sample {sample['id']}: {e}")
            return {
                "sample_id": sample["id"],
                "context_length": sample.get("context_length", 0),
                "depth_percent": sample.get("depth_percent", 0),
                "score": 0.0,
                "prediction": "",
                "target": self.needle,
                "error": str(e),
                "metadata": {
                    "test_number": sample.get("test_number", 0),
                    "evaluation_method": "official_niah_failed"
                }
            }
    
    def prepare_input(self, sample: Dict) -> str:
        """Prepare input for NIAH test (handled by official implementation)."""
        return sample.get("input", "")
    
    def extract_answer(self, response: str, sample: Dict) -> str:
        """Extract answer from response (handled by official implementation)."""
        return response.strip()
    
    def compute_score(self, prediction: str, ground_truth: str, sample: Dict) -> float:
        """Compute score (handled by official implementation)."""
        # Simple exact match for now - official implementation does more sophisticated scoring
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
    
    def get_benchmark_info(self) -> Dict:
        """Get information about the NIAH benchmark."""
        return {
            "name": "NIAH (Official)",
            "description": "Official Needle-in-a-Haystack benchmark by Greg Kamradt",
            "context_lengths": self.context_lengths,
            "depth_percents": self.document_depth_percents,
            "num_tests": self.num_tests,
            "repository": "https://github.com/gkamradt/LLMTest_NeedleInAHaystack",
            "version": "official"
        }