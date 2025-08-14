"""RULER benchmark implementation."""

import logging
import random
from typing import Dict, List

from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class RULERBenchmark(BaseBenchmark):
    """RULER benchmark for comprehensive long context evaluation."""
    
    def __init__(self, config: Dict, model, tokenizer):
        """Initialize RULER benchmark."""
        super().__init__(config, model, tokenizer)
        self.categories = config.get("categories", ["retrieval", "multi_hop", "aggregation", "qa"])
        self.max_length = config.get("max_length", 128000)
        self.num_samples = config.get("num_samples", 500)
        
    def load_data(self) -> List[Dict]:
        """Generate RULER evaluation samples."""
        data = []
        sample_id = 0
        
        samples_per_category = self.num_samples // len(self.categories)
        
        for category in self.categories:
            if category == "retrieval":
                samples = self._generate_retrieval_samples(sample_id, samples_per_category)
            elif category == "multi_hop":
                samples = self._generate_multi_hop_samples(sample_id, samples_per_category)
            elif category == "aggregation":
                samples = self._generate_aggregation_samples(sample_id, samples_per_category)
            elif category == "qa":
                samples = self._generate_qa_samples(sample_id, samples_per_category)
            else:
                logger.warning(f"Unknown RULER category: {category}")
                continue
            
            data.extend(samples)
            sample_id += len(samples)
        
        logger.info(f"Generated {len(data)} RULER samples across {len(self.categories)} categories")
        return data
    
    def _generate_retrieval_samples(self, start_id: int, num_samples: int) -> List[Dict]:
        """Generate retrieval task samples."""
        samples = []
        
        for i in range(num_samples):
            # Create a synthetic retrieval task
            context_length = random.choice([4000, 8000, 16000, 32000, 64000])
            target_info = f"Target information #{random.randint(1000, 9999)}"
            
            context = self._create_synthetic_context(context_length, target_info)
            question = f"What is the target information mentioned in the text?"
            
            sample = {
                "id": start_id + i,
                "category": "retrieval",
                "context_length": context_length,
                "context": context,
                "question": question,
                "answer": target_info,
            }
            samples.append(sample)
        
        return samples
    
    def _generate_multi_hop_samples(self, start_id: int, num_samples: int) -> List[Dict]:
        """Generate multi-hop reasoning samples."""
        samples = []
        
        for i in range(num_samples):
            context_length = random.choice([8000, 16000, 32000, 64000])
            
            # Create chain of entities
            entities = [f"Entity_{j}" for j in range(3)]
            relations = [f"relates to", "connects with", "leads to"]
            
            context = self._create_multi_hop_context(context_length, entities, relations)
            question = f"What does {entities[0]} ultimately connect to through the chain of relationships?"
            answer = entities[-1]
            
            sample = {
                "id": start_id + i,
                "category": "multi_hop",
                "context_length": context_length,
                "context": context,
                "question": question,
                "answer": answer,
            }
            samples.append(sample)
        
        return samples
    
    def _generate_aggregation_samples(self, start_id: int, num_samples: int) -> List[Dict]:
        """Generate aggregation task samples."""
        samples = []
        
        for i in range(num_samples):
            context_length = random.choice([8000, 16000, 32000, 64000])
            
            # Create list of numbers to aggregate
            numbers = [random.randint(1, 100) for _ in range(10)]
            context = self._create_aggregation_context(context_length, numbers)
            question = "What is the sum of all the special numbers mentioned in the text?"
            answer = str(sum(numbers))
            
            sample = {
                "id": start_id + i,
                "category": "aggregation",
                "context_length": context_length,
                "context": context,
                "question": question,
                "answer": answer,
            }
            samples.append(sample)
        
        return samples
    
    def _generate_qa_samples(self, start_id: int, num_samples: int) -> List[Dict]:
        """Generate question answering samples."""
        samples = []
        
        qa_pairs = [
            ("What is the capital of artificial intelligence research?", "Silicon Valley"),
            ("Who is considered the father of computer science?", "Alan Turing"),
            ("What year was the Dartmouth Conference held?", "1956"),
        ]
        
        for i in range(num_samples):
            context_length = random.choice([4000, 8000, 16000, 32000])
            qa_pair = random.choice(qa_pairs)
            
            context = self._create_qa_context(context_length, qa_pair)
            question = qa_pair[0]
            answer = qa_pair[1]
            
            sample = {
                "id": start_id + i,
                "category": "qa",
                "context_length": context_length,
                "context": context,
                "question": question,
                "answer": answer,
            }
            samples.append(sample)
        
        return samples
    
    def _create_synthetic_context(self, target_length: int, target_info: str) -> str:
        """Create synthetic context with target information."""
        base_text = """
        In the field of artificial intelligence, researchers have made significant progress in developing
        systems that can process and understand large amounts of information. These systems rely on
        advanced algorithms and neural networks to perform complex tasks.
        """
        
        # Insert target info at random position
        expanded_text = self._expand_text_to_length(base_text, target_length - len(target_info))
        insertion_pos = random.randint(0, len(expanded_text))
        
        context = expanded_text[:insertion_pos] + f"\\n{target_info}\\n" + expanded_text[insertion_pos:]
        return context
    
    def _create_multi_hop_context(self, target_length: int, entities: List[str], relations: List[str]) -> str:
        """Create context with multi-hop entity relationships."""
        base_text = "This document contains information about various entities and their relationships."
        
        # Create relationship chain
        relationship_text = ""
        for i in range(len(entities) - 1):
            relationship_text += f"\\n{entities[i]} {relations[i]} {entities[i+1]}.\\n"
        
        expanded_text = self._expand_text_to_length(base_text, target_length - len(relationship_text))
        
        # Insert relationships at different positions
        parts = [expanded_text[i::3] for i in range(3)]
        context = parts[0] + relationship_text[:len(relationship_text)//2] + parts[1] + relationship_text[len(relationship_text)//2:] + parts[2]
        
        return context
    
    def _create_aggregation_context(self, target_length: int, numbers: List[int]) -> str:
        """Create context with numbers to aggregate."""
        base_text = "This document contains various numerical data points that need to be processed."
        
        # Scatter numbers throughout text
        number_text = ""
        for num in numbers:
            number_text += f"\\nSpecial number: {num}\\n"
        
        expanded_text = self._expand_text_to_length(base_text, target_length - len(number_text))
        
        # Randomly insert numbers
        context = expanded_text
        for num_text in number_text.split('\\n'):
            if num_text.strip():
                pos = random.randint(0, len(context))
                context = context[:pos] + num_text + '\\n' + context[pos:]
        
        return context
    
    def _create_qa_context(self, target_length: int, qa_pair: tuple) -> str:
        """Create context containing answer to question."""
        base_text = """
        The history of computing and artificial intelligence spans many decades and involves
        numerous important figures and events that have shaped the field.
        """
        
        answer_text = f"\\nImportant fact: {qa_pair[1]} is the answer to {qa_pair[0]}\\n"
        expanded_text = self._expand_text_to_length(base_text, target_length - len(answer_text))
        
        insertion_pos = random.randint(0, len(expanded_text))
        context = expanded_text[:insertion_pos] + answer_text + expanded_text[insertion_pos:]
        
        return context
    
    def _expand_text_to_length(self, base_text: str, target_length: int) -> str:
        """Expand text to reach target length."""
        if target_length <= len(base_text):
            return base_text[:target_length]
        
        repetitions = (target_length // len(base_text)) + 1
        expanded = (base_text + "\\n\\n") * repetitions
        
        return expanded[:target_length]
    
    def prepare_input(self, sample: Dict) -> str:
        """Prepare input text for RULER evaluation."""
        context = sample["context"]
        question = sample["question"]
        
        prompt = f"""Please read the following text carefully and answer the question based on the information provided.

Text:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def extract_answer(self, response: str, sample: Dict) -> str:
        """Extract answer from model response."""
        lines = response.strip().split('\\n')
        if lines:
            return lines[0].strip()
        return response.strip()
    
    def compute_score(self, prediction: str, ground_truth: str, sample: Dict) -> float:
        """Compute score for RULER prediction."""
        prediction = prediction.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        category = sample["category"]
        
        if category == "aggregation":
            # For aggregation, check exact numeric match
            try:
                pred_num = float(prediction)
                gt_num = float(ground_truth)
                return 1.0 if abs(pred_num - gt_num) < 1e-6 else 0.0
            except ValueError:
                return 0.0
        
        # For other categories, check substring match
        if ground_truth in prediction:
            return 1.0
        
        # Check for partial word overlap
        pred_words = set(prediction.split())
        gt_words = set(ground_truth.split())
        
        if gt_words and pred_words:
            overlap = len(gt_words.intersection(pred_words))
            return overlap / len(gt_words)
        
        return 0.0