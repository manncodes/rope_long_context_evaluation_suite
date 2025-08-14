"""Needle in a Haystack (NIAH) benchmark implementation."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class NIAHBenchmark(BaseBenchmark):
    """Needle in a Haystack benchmark for evaluating retrieval from long contexts."""
    
    DEFAULT_NEEDLES = [
        "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
        "The secret to happiness is finding joy in the small moments of everyday life.",
        "To solve complex problems, break them down into smaller, manageable pieces.",
        "The most efficient way to learn a new skill is through deliberate practice and repetition.",
        "Innovation happens at the intersection of different fields and perspectives.",
    ]
    
    DEFAULT_HAYSTACK_TEXT = """
    The history of artificial intelligence began in antiquity with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The study of mechanical or "formal" reasoning began with philosophers and mathematicians in antiquity. The study of mathematical logic led directly to Alan Turing's theory of computation, which suggested that a machine, by shuffling symbols as simple as "0" and "1", could simulate any conceivable act of mathematical deduction.
    
    In the 1950s, a generation of scientists, mathematicians, and philosophers had the concept of artificial intelligence (or AI) intellectually formulated. One such person was Alan Turing, a young British polymath who explored the mathematical possibility of artificial intelligence. Turing suggested that humans use available information as well as reason in order to solve problems and make decisions, so why can't machines do the same thing? This was the logical framework of his 1950 paper, Computing Machinery and Intelligence, in which he discussed how to build intelligent machines and how to test their intelligence.
    
    The field of artificial intelligence research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.
    
    Eventually, it became obvious that commercial developers and researchers had been overly optimistic. Getting a computer to play checkers was relatively easy, but getting one that could understand and respond appropriately to human language proved much more difficult. By 1974, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter."
    """
    
    def __init__(self, config: Dict, model, tokenizer):
        """Initialize NIAH benchmark."""
        super().__init__(config, model, tokenizer)
        self.variants = config.get("variants", ["standard"])
        self.context_lengths = config.get("context_lengths", [4000, 8000, 16000, 32000])
        self.num_needles = config.get("num_needles", 1)
        self.depth_intervals = config.get("depth_percent_intervals", 
                                        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # Load or use default needles and haystack
        self._load_needles_and_haystack()
    
    def _load_needles_and_haystack(self):
        """Load needle and haystack texts from files or use defaults."""
        needle_file = self.config.get("needle_file")
        haystack_file = self.config.get("haystack_file")
        
        # Load needles
        if needle_file and Path(needle_file).exists():
            with open(needle_file, 'r', encoding='utf-8') as f:
                self.needles = [line.strip() for line in f if line.strip()]
        else:
            self.needles = self.DEFAULT_NEEDLES
            if needle_file:
                logger.warning(f"Needle file {needle_file} not found, using default needles")
        
        # Load haystack
        if haystack_file and Path(haystack_file).exists():
            with open(haystack_file, 'r', encoding='utf-8') as f:
                self.haystack_base = f.read()
        else:
            self.haystack_base = self.DEFAULT_HAYSTACK_TEXT
            if haystack_file:
                logger.warning(f"Haystack file {haystack_file} not found, using default haystack")
        
        logger.info(f"Loaded {len(self.needles)} needles")
    
    def load_data(self) -> List[Dict]:
        """Generate NIAH evaluation samples."""
        data = []
        sample_id = 0
        
        for variant in self.variants:
            for context_length in self.context_lengths:
                for depth in self.depth_intervals:
                    if variant == "standard":
                        samples = self._generate_standard_samples(sample_id, context_length, depth)
                    elif variant == "multi_needle":
                        samples = self._generate_multi_needle_samples(sample_id, context_length, depth)
                    elif variant == "nolib":
                        samples = self._generate_nolib_samples(sample_id, context_length, depth)
                    else:
                        logger.warning(f"Unknown variant: {variant}")
                        continue
                    
                    data.extend(samples)
                    sample_id += len(samples)
        
        logger.info(f"Generated {len(data)} NIAH samples")
        return data
    
    def _generate_standard_samples(self, start_id: int, context_length: int, depth: float, num_samples: int = 1) -> List[Dict]:
        """Generate standard NIAH samples with single needle."""
        samples = []
        
        for i in range(num_samples):
            needle = random.choice(self.needles)
            context = self._create_context_with_needle(needle, context_length, depth)
            
            sample = {
                "id": start_id + i,
                "variant": "standard",
                "context_length": context_length,
                "depth_percent": depth,
                "needle": needle,
                "context": context,
                "question": f"What is the best thing to do in San Francisco?" if "San Francisco" in needle 
                           else f"What is mentioned about {needle.split()[0].lower()}?",
                "answer": needle,
            }
            samples.append(sample)
        
        return samples
    
    def _generate_multi_needle_samples(self, start_id: int, context_length: int, depth: float, num_samples: int = 1) -> List[Dict]:
        """Generate multi-needle NIAH samples."""
        samples = []
        
        for i in range(num_samples):
            # Select multiple needles
            needles = random.sample(self.needles, min(self.num_needles, len(self.needles)))
            
            # Create context with multiple needles at different positions
            context = self._create_context_with_multiple_needles(needles, context_length, depth)
            
            # Ask about all needles
            question = f"What are the {len(needles)} important facts mentioned in the text?"
            answer = " ".join(needles)
            
            sample = {
                "id": start_id + i,
                "variant": "multi_needle",
                "context_length": context_length,
                "depth_percent": depth,
                "needles": needles,
                "context": context,
                "question": question,
                "answer": answer,
            }
            samples.append(sample)
        
        return samples
    
    def _generate_nolib_samples(self, start_id: int, context_length: int, depth: float, num_samples: int = 1) -> List[Dict]:
        """Generate NoLiMa (Non-Lexical Matching) NIAH samples."""
        samples = []
        
        nolib_pairs = [
            {
                "needle": "The capital of France is Paris.",
                "question": "What is the largest city in the country known for the Eiffel Tower?",
                "answer": "Paris"
            },
            {
                "needle": "Water boils at 100 degrees Celsius at sea level.",
                "question": "At what temperature does H2O transition to vapor under standard atmospheric pressure?",
                "answer": "100 degrees Celsius"
            },
            {
                "needle": "The fastest land animal is the cheetah.",
                "question": "Which terrestrial creature holds the speed record?",
                "answer": "cheetah"
            }
        ]
        
        for i in range(num_samples):
            pair = random.choice(nolib_pairs)
            context = self._create_context_with_needle(pair["needle"], context_length, depth)
            
            sample = {
                "id": start_id + i,
                "variant": "nolib",
                "context_length": context_length,
                "depth_percent": depth,
                "needle": pair["needle"],
                "context": context,
                "question": pair["question"],
                "answer": pair["answer"],
            }
            samples.append(sample)
        
        return samples
    
    def _create_context_with_needle(self, needle: str, target_length: int, depth_percent: float) -> str:
        """Create context with needle inserted at specified depth."""
        # Repeat haystack text to reach target length
        haystack = self._expand_haystack_to_length(target_length - len(needle))
        
        # Calculate insertion position
        insertion_pos = int(len(haystack) * depth_percent)
        
        # Insert needle
        context = haystack[:insertion_pos] + "\\n" + needle + "\\n" + haystack[insertion_pos:]
        
        return context
    
    def _create_context_with_multiple_needles(self, needles: List[str], target_length: int, base_depth: float) -> str:
        """Create context with multiple needles at different positions."""
        total_needle_length = sum(len(needle) for needle in needles)
        haystack = self._expand_haystack_to_length(target_length - total_needle_length)
        
        # Insert needles at different depths around the base depth
        context = haystack
        offset = 0
        
        for i, needle in enumerate(needles):
            # Vary depth slightly for each needle
            depth = base_depth + (i - len(needles)//2) * 0.1
            depth = max(0.0, min(1.0, depth))  # Clamp to [0, 1]
            
            insertion_pos = int(len(context) * depth) + offset
            context = context[:insertion_pos] + "\\n" + needle + "\\n" + context[insertion_pos:]
            offset += len(needle) + 2  # Account for newlines
        
        return context
    
    def _expand_haystack_to_length(self, target_length: int) -> str:
        """Expand haystack text to reach target length."""
        if target_length <= len(self.haystack_base):
            return self.haystack_base[:target_length]
        
        # Repeat haystack until we reach target length
        repetitions = (target_length // len(self.haystack_base)) + 1
        expanded = (self.haystack_base + "\\n\\n") * repetitions
        
        return expanded[:target_length]
    
    def prepare_input(self, sample: Dict) -> str:
        """Prepare input text for NIAH evaluation."""
        context = sample["context"]
        question = sample["question"]
        
        prompt = f"""Read the following text carefully and answer the question based on the information provided.

Text:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def extract_answer(self, response: str, sample: Dict) -> str:
        """Extract answer from model response."""
        # Simple extraction - take first line/sentence of response
        lines = response.strip().split('\\n')
        if lines:
            return lines[0].strip()
        return response.strip()
    
    def compute_score(self, prediction: str, ground_truth: str, sample: Dict) -> float:
        """Compute score for NIAH prediction."""
        prediction = prediction.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # For multi-needle samples, check if all needles are mentioned
        if sample["variant"] == "multi_needle":
            needles = [needle.lower().strip() for needle in sample["needles"]]
            found_needles = sum(1 for needle in needles if needle in prediction)
            return found_needles / len(needles)
        
        # For standard and nolib samples, check for exact or partial match
        if ground_truth in prediction:
            return 1.0
        
        # Check for partial matches (for longer ground truths)
        ground_truth_words = set(ground_truth.split())
        prediction_words = set(prediction.split())
        
        if ground_truth_words and prediction_words:
            overlap = len(ground_truth_words.intersection(prediction_words))
            return overlap / len(ground_truth_words)
        
        return 0.0