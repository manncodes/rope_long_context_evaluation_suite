"""LongBench benchmark implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, List

from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class LongBench(BaseBenchmark):
    """LongBench benchmark for real-world long context evaluation."""
    
    def __init__(self, config: Dict, model, tokenizer):
        """Initialize LongBench benchmark."""
        super().__init__(config, model, tokenizer)
        self.tasks = config.get("tasks", ["narrativeqa", "qasper", "multifieldqa_en"])
        self.data_path = Path(config.get("data_path", "data/longbench/"))
        self.max_samples = config.get("max_samples", 100)
        
    def load_data(self) -> List[Dict]:
        """Load LongBench evaluation data."""
        data = []
        
        for task in self.tasks:
            task_data = self._load_task_data(task)
            data.extend(task_data)
        
        logger.info(f"Loaded {len(data)} samples from {len(self.tasks)} LongBench tasks")
        return data
    
    def _load_task_data(self, task: str) -> List[Dict]:
        """Load data for a specific LongBench task."""
        task_file = self.data_path / f"{task}.jsonl"
        
        if not task_file.exists():
            logger.warning(f"LongBench task file not found: {task_file}")
            logger.info("Run 'python scripts/setup_data.py --benchmarks longbench' to download the data")
            return self._generate_placeholder_data(task)
        
        samples = []
        with open(task_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    sample["id"] = f"{task}_{i}"
                    sample["task"] = task
                    samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {i+1} in {task_file}")
        
        logger.info(f"Loaded {len(samples)} samples for task {task}")
        return samples
    
    def _generate_placeholder_data(self, task: str) -> List[Dict]:
        """Generate placeholder data when actual data is not available."""
        logger.info(f"Generating placeholder data for {task}")
        
        placeholder_samples = []
        
        if task == "narrativeqa":
            sample = {
                "id": f"{task}_placeholder",
                "task": task,
                "input": "This is a placeholder narrative text. " * 100,
                "context": "Extended narrative context. " * 200,
                "question": "What is the main theme of the story?",
                "answer": "The main theme is placeholder content.",
            }
        elif task == "qasper":
            sample = {
                "id": f"{task}_placeholder", 
                "task": task,
                "input": "This is a placeholder scientific paper abstract. " * 150,
                "context": "Extended scientific paper content. " * 300,
                "question": "What is the main contribution of this paper?",
                "answer": "The main contribution is a placeholder method.",
            }
        else:
            sample = {
                "id": f"{task}_placeholder",
                "task": task,
                "input": f"Placeholder content for {task}. " * 100,
                "context": f"Extended placeholder context for {task}. " * 200,
                "question": f"What is the key information in this {task} example?",
                "answer": "Key information is placeholder content.",
            }
        
        placeholder_samples.append(sample)
        return placeholder_samples
    
    def prepare_input(self, sample: Dict) -> str:
        """Prepare input text for LongBench evaluation."""
        task = sample["task"]
        
        if task == "narrativeqa":
            return self._prepare_narrativeqa_input(sample)
        elif task == "qasper":
            return self._prepare_qasper_input(sample)
        elif task == "multifieldqa_en":
            return self._prepare_multifieldqa_input(sample)
        elif task in ["hotpotqa", "2wikimqa", "musique"]:
            return self._prepare_multihop_qa_input(sample)
        elif task in ["gov_report", "qmsum", "multi_news", "vcsum"]:
            return self._prepare_summarization_input(sample)
        elif task in ["trec", "triviaqa"]:
            return self._prepare_single_qa_input(sample)
        elif task == "samsum":
            return self._prepare_dialogue_input(sample)
        elif task in ["passage_count", "passage_retrieval_en"]:
            return self._prepare_synthetic_input(sample)
        elif task == "lcc":
            return self._prepare_code_input(sample)
        elif task == "repobench-p":
            return self._prepare_repo_input(sample)
        else:
            return self._prepare_generic_input(sample)
    
    def _prepare_narrativeqa_input(self, sample: Dict) -> str:
        """Prepare input for NarrativeQA task."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "")
        
        prompt = f"""Read the following story and answer the question.

Story:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _prepare_qasper_input(self, sample: Dict) -> str:
        """Prepare input for QASPER task."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "")
        
        prompt = f"""Read the following research paper and answer the question based on the paper's content.

Paper:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _prepare_multifieldqa_input(self, sample: Dict) -> str:
        """Prepare input for MultiFieldQA task."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "")
        
        prompt = f"""Read the following document and answer the question.

Document:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _prepare_multihop_qa_input(self, sample: Dict) -> str:
        """Prepare input for multi-hop QA tasks."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "")
        
        prompt = f"""Read the following passages and answer the question that requires reasoning across multiple pieces of information.

Passages:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _prepare_summarization_input(self, sample: Dict) -> str:
        """Prepare input for summarization tasks."""
        context = sample.get("context", sample.get("input", ""))
        
        prompt = f"""Please provide a concise summary of the following text.

Text:
{context}

Summary:"""
        return prompt
    
    def _prepare_single_qa_input(self, sample: Dict) -> str:
        """Prepare input for single-turn QA tasks."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "")
        
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _prepare_dialogue_input(self, sample: Dict) -> str:
        """Prepare input for dialogue summarization."""
        context = sample.get("context", sample.get("input", ""))
        
        prompt = f"""Summarize the following conversation.

Conversation:
{context}

Summary:"""
        return prompt
    
    def _prepare_synthetic_input(self, sample: Dict) -> str:
        """Prepare input for synthetic tasks."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "")
        
        prompt = f"""Complete the following task based on the given text.

Text:
{context}

Task: {question}

Answer:"""
        return prompt
    
    def _prepare_code_input(self, sample: Dict) -> str:
        """Prepare input for code completion."""
        context = sample.get("context", sample.get("input", ""))
        
        prompt = f"""Complete the following code:

{context}

Completion:"""
        return prompt
    
    def _prepare_repo_input(self, sample: Dict) -> str:
        """Prepare input for repository understanding."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "")
        
        prompt = f"""Based on the following code repository, answer the question.

Repository:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _prepare_generic_input(self, sample: Dict) -> str:
        """Generic input preparation."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("question", "Please provide relevant information from the text.")
        
        prompt = f"""Based on the following text, {question.lower()}

Text:
{context}

Response:"""
        return prompt
    
    def extract_answer(self, response: str, sample: Dict) -> str:
        """Extract answer from model response."""
        lines = response.strip().split('\\n')
        if lines:
            return lines[0].strip()
        return response.strip()
    
    def compute_score(self, prediction: str, ground_truth: str, sample: Dict) -> float:
        """Compute score for LongBench prediction."""
        task = sample["task"]
        
        prediction = prediction.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        if task in ["gov_report", "qmsum", "multi_news", "vcsum", "samsum"]:
            # For summarization tasks, use ROUGE-like scoring
            return self._compute_rouge_score(prediction, ground_truth)
        elif task in ["passage_count"]:
            # For counting tasks, check exact match
            try:
                pred_num = int(prediction.split()[0])
                gt_num = int(ground_truth.split()[0])
                return 1.0 if pred_num == gt_num else 0.0
            except (ValueError, IndexError):
                return 0.0
        else:
            # For QA tasks, check substring match and word overlap
            if ground_truth in prediction:
                return 1.0
            
            # Word-level F1 score
            pred_words = set(prediction.split())
            gt_words = set(ground_truth.split())
            
            if not gt_words:
                return 0.0
            
            if not pred_words:
                return 0.0
            
            overlap = len(pred_words.intersection(gt_words))
            precision = overlap / len(pred_words)
            recall = overlap / len(gt_words)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * precision * recall / (precision + recall)
            return f1
    
    def _compute_rouge_score(self, prediction: str, ground_truth: str) -> float:
        """Compute a simple ROUGE-like score."""
        pred_words = prediction.split()
        gt_words = ground_truth.split()
        
        if not gt_words:
            return 0.0
        
        # Compute ROUGE-1 (unigram overlap)
        pred_unigrams = set(pred_words)
        gt_unigrams = set(gt_words)
        
        overlap = len(pred_unigrams.intersection(gt_unigrams))
        recall = overlap / len(gt_unigrams) if gt_unigrams else 0.0
        precision = overlap / len(pred_unigrams) if pred_unigrams else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1