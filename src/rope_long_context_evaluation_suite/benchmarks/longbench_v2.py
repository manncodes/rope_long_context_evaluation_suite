"""LongBench-V2 benchmark implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, List

from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class LongBenchV2(BaseBenchmark):
    """LongBench-V2 benchmark for challenging long context evaluation."""
    
    def __init__(self, config: Dict, model, tokenizer):
        """Initialize LongBench-V2 benchmark."""
        super().__init__(config, model, tokenizer)
        self.tasks = config.get("tasks", ["single_doc_qa", "multi_doc_qa", "long_icl"])
        self.data_path = Path(config.get("data_path", "data/longbench_v2/"))
        self.max_samples = config.get("max_samples", 100)
        
    def load_data(self) -> List[Dict]:
        """Load LongBench-V2 evaluation data."""
        data = []
        
        for task in self.tasks:
            task_data = self._load_task_data(task)
            data.extend(task_data)
        
        logger.info(f"Loaded {len(data)} samples from {len(self.tasks)} LongBench-V2 tasks")
        return data
    
    def _load_task_data(self, task: str) -> List[Dict]:
        """Load data for a specific LongBench-V2 task."""
        task_file = self.data_path / f"{task}.jsonl"
        
        if not task_file.exists():
            logger.warning(f"LongBench-V2 task file not found: {task_file}")
            logger.info("LongBench-V2 data may not be publicly available yet. Using placeholder data.")
            return self._generate_placeholder_data(task)
        
        samples = []
        with open(task_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    sample["id"] = f"{task}_v2_{i}"
                    sample["task"] = task
                    samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {i+1} in {task_file}")
        
        logger.info(f"Loaded {len(samples)} samples for LongBench-V2 task {task}")
        return samples
    
    def _generate_placeholder_data(self, task: str) -> List[Dict]:
        """Generate placeholder data for LongBench-V2 tasks."""
        logger.info(f"Generating challenging placeholder data for {task}")
        
        placeholder_samples = []
        
        if task == "single_doc_qa":
            sample = {
                "id": f"{task}_v2_placeholder",
                "task": task,
                "context": self._generate_long_document(50000),  # ~50k tokens
                "question": "Based on the comprehensive analysis in sections 15-23, what are the three main methodological limitations identified in the comparative studies, and how do they relate to the theoretical framework presented in the introduction?",
                "answer": "The three main methodological limitations are sample size constraints, measurement validity issues, and confounding variable control problems, which directly challenge the theoretical framework's core assumptions about causal relationships.",
                "options": [
                    "Sample size, validity, confounding issues challenging theoretical assumptions",
                    "Data collection, analysis methods, and interpretation problems",
                    "Experimental design, statistical power, and generalizability concerns",
                    "Measurement errors, bias, and reliability issues affecting conclusions"
                ]
            }
        elif task == "multi_doc_qa":
            sample = {
                "id": f"{task}_v2_placeholder", 
                "task": task,
                "contexts": [
                    self._generate_research_paper_section(20000),  # Document 1
                    self._generate_research_paper_section(25000),  # Document 2
                    self._generate_research_paper_section(30000),  # Document 3
                ],
                "question": "Synthesizing evidence from all three papers, what convergent themes emerge regarding the effectiveness of the proposed methodologies, and what contradictory findings require further investigation?",
                "answer": "Convergent themes include improved performance metrics and scalability benefits, while contradictory findings center on computational efficiency claims and generalizability across different domains.",
                "options": [
                    "Performance and scalability convergence, with efficiency and generalizability contradictions",
                    "Methodology validation and implementation ease, with cost and complexity disputes",
                    "Accuracy improvements and user satisfaction, with training time and resource conflicts",
                    "Technical feasibility and practical utility, with maintenance and adoption disagreements"
                ]
            }
        elif task == "long_icl":
            sample = {
                "id": f"{task}_v2_placeholder",
                "task": task,
                "context": self._generate_icl_context(40000),  # Long in-context learning examples
                "question": "Following the pattern established in the examples above, solve this complex multi-step problem:",
                "answer": "Following the established pattern of systematic decomposition and iterative refinement, the solution involves three phases with validation checkpoints.",
                "problem": "Given the constraints and objectives shown in examples 47-52, optimize the resource allocation for the new scenario with updated parameters."
            }
        elif task == "dialogue":
            sample = {
                "id": f"{task}_v2_placeholder",
                "task": task,
                "context": self._generate_long_dialogue(35000),
                "question": "Analyze the conversation patterns and identify the key decision points that led to the final resolution, including the underlying motivations of each participant.",
                "answer": "The key decision points occurred at minutes 15, 32, and 48, where participants shifted from positional bargaining to interest-based negotiation, ultimately resolving through mutual value creation.",
                "options": [
                    "Decision points at 15, 32, 48 minutes with shift to interest-based negotiation",
                    "Critical moments at 8, 25, 41 minutes focused on compromise solutions", 
                    "Key transitions at 12, 28, 35 minutes emphasizing collaborative problem-solving",
                    "Pivotal exchanges at 20, 38, 52 minutes highlighting competitive dynamics"
                ]
            }
        elif task == "code_repo":
            sample = {
                "id": f"{task}_v2_placeholder",
                "task": task,
                "context": self._generate_code_repository(60000),
                "question": "Analyze the architectural dependencies and identify potential scalability bottlenecks in the data processing pipeline, considering the interaction between modules A, C, and F.",
                "answer": "The main scalability bottleneck is in the synchronous communication pattern between modules A and C, which creates a blocking dependency that prevents parallel processing in module F.",
                "options": [
                    "Synchronous A-C communication blocks parallel processing in F",
                    "Memory allocation patterns in module C limit overall throughput",
                    "Database connection pooling in module A restricts concurrent operations",
                    "Thread management issues in module F reduce processing efficiency"
                ]
            }
        elif task == "structured_data":
            sample = {
                "id": f"{task}_v2_placeholder",
                "task": task,
                "context": self._generate_structured_data(45000),
                "question": "Based on the longitudinal analysis of the dataset, identify the three most significant trend reversals and explain their correlation with the external factors mentioned in the metadata.",
                "answer": "The three significant trend reversals occur in Q2 2019, Q4 2020, and Q3 2022, correlating with policy changes, market disruptions, and technological adoption patterns respectively.",
                "options": [
                    "Q2 2019, Q4 2020, Q3 2022 tied to policy, market, and technology factors",
                    "Q1 2018, Q3 2021, Q1 2023 linked to economic, social, and regulatory changes",
                    "Q4 2019, Q2 2021, Q4 2022 connected to industry, environmental, and competitive shifts",
                    "Q3 2018, Q1 2020, Q2 2023 associated with demographic, cultural, and innovation trends"
                ]
            }
        else:
            sample = {
                "id": f"{task}_v2_placeholder",
                "task": task,
                "context": f"This is an extremely challenging long context task for {task}. " * 2000,
                "question": f"Provide a comprehensive analysis of the complex patterns and relationships present in this {task} scenario.",
                "answer": "The comprehensive analysis reveals multiple interconnected patterns with significant implications for understanding the underlying mechanisms.",
            }
        
        placeholder_samples.append(sample)
        return placeholder_samples
    
    def _generate_long_document(self, target_length: int) -> str:
        """Generate a long, coherent document for testing."""
        sections = [
            "# Introduction\\n\\nThis document presents a comprehensive analysis of modern computational approaches to complex problem-solving methodologies.",
            "# Literature Review\\n\\nExtensive review of existing research reveals several methodological approaches and their comparative effectiveness.",
            "# Methodology\\n\\nOur approach combines multiple established techniques with novel innovations in data processing and analysis.",
            "# Results\\n\\nExtensive experimental validation demonstrates significant improvements across multiple evaluation metrics.",
            "# Discussion\\n\\nThe implications of these findings extend beyond the immediate domain to broader applications in related fields.",
            "# Conclusions\\n\\nThis work establishes a new foundation for future research and practical applications in computational methods."
        ]
        
        # Expand each section to reach target length
        content = ""
        for section in sections:
            expanded_section = section
            while len(expanded_section) < target_length // len(sections):
                expanded_section += "\\n\\nAdditional detailed analysis and comprehensive examination of the underlying principles and methodologies reveals complex interactions and emergent properties that require careful consideration and systematic investigation."
            content += expanded_section + "\\n\\n"
        
        return content[:target_length]
    
    def _generate_research_paper_section(self, target_length: int) -> str:
        """Generate a research paper section."""
        base_content = """
        Abstract: This research investigates novel approaches to computational efficiency in large-scale systems.
        
        1. Introduction
        The field of computational systems has evolved significantly in recent years, with increasing demands for 
        scalability, efficiency, and reliability. This paper presents a comprehensive analysis of emerging 
        methodologies and their practical applications.
        
        2. Related Work
        Previous research in this domain has established several foundational principles and methodological 
        approaches that inform current best practices and guide future research directions.
        
        3. Methodology
        Our experimental design incorporates multiple evaluation criteria and comparative analysis frameworks
        to ensure robust validation of the proposed approaches and methodologies.
        """
        
        # Expand to target length
        repetitions = (target_length // len(base_content)) + 1
        expanded = base_content * repetitions
        return expanded[:target_length]
    
    def _generate_icl_context(self, target_length: int) -> str:
        """Generate in-context learning examples."""
        examples = []
        example_template = """
        Example {i}:
        Problem: Given constraints X={x} and objectives Y={y}, find optimal solution Z.
        Solution: Through systematic analysis of the constraint space and objective function optimization, 
        the optimal solution Z={z} is achieved by decomposing the problem into subcomponents and applying 
        iterative refinement techniques.
        
        Explanation: The solution process involves three key phases: initial decomposition, constraint 
        satisfaction, and objective optimization, each requiring careful analysis of the problem structure.
        """
        
        for i in range(50):
            example = example_template.format(i=i+1, x=i*2, y=i*3, z=i*5)
            examples.append(example)
        
        content = "\\n".join(examples)
        
        # Expand to target length if needed
        while len(content) < target_length:
            content += "\\n" + content[:target_length - len(content)]
        
        return content[:target_length]
    
    def _generate_long_dialogue(self, target_length: int) -> str:
        """Generate a long dialogue for analysis."""
        speakers = ["Alice", "Bob", "Carol", "David"]
        topics = ["project planning", "resource allocation", "timeline management", "quality assurance"]
        
        dialogue = "Meeting Transcript - Strategic Planning Session\\n\\n"
        
        for i in range(100):
            speaker = speakers[i % len(speakers)]
            topic = topics[i % len(topics)]
            
            dialogue += f"[{i+1:02d}:00] {speaker}: I think we need to address the {topic} concerns that were raised earlier. The implications for our overall strategy are significant and require careful consideration of multiple factors and stakeholder perspectives.\\n\\n"
        
        # Expand to reach target length
        while len(dialogue) < target_length:
            dialogue += dialogue[:target_length - len(dialogue)]
        
        return dialogue[:target_length]
    
    def _generate_code_repository(self, target_length: int) -> str:
        """Generate code repository content."""
        modules = ["module_a.py", "module_b.py", "module_c.py", "module_d.py", "module_e.py", "module_f.py"]
        
        repo_content = "# Repository Structure and Code Analysis\\n\\n"
        
        for module in modules:
            repo_content += f"## {module}\\n\\n"
            repo_content += f"""
```python
# {module} - Core functionality implementation
import asyncio
import concurrent.futures
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    batch_size: int = 1000
    max_workers: int = 10
    timeout: int = 300
    retry_attempts: int = 3

class {module.replace('.py', '').title().replace('_', '')}Processor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)
    
    async def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Complex processing logic with error handling and retry mechanisms
        results = []
        for batch in self._create_batches(data):
            try:
                batch_result = await self._process_batch(batch)
                results.extend(batch_result)
            except Exception as e:
                # Error handling and retry logic
                await self._handle_processing_error(e, batch)
        return results
    
    def _create_batches(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        # Batch creation logic for efficient processing
        batches = []
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            batches.append(batch)
        return batches
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Actual batch processing implementation
        processed_items = []
        for item in batch:
            processed_item = await self._process_single_item(item)
            processed_items.append(processed_item)
        return processed_items
```
\\n\\n"""
        
        # Expand to target length
        while len(repo_content) < target_length:
            repo_content += repo_content[:min(len(repo_content), target_length - len(repo_content))]
        
        return repo_content[:target_length]
    
    def _generate_structured_data(self, target_length: int) -> str:
        """Generate structured data content."""
        data_content = """
# Longitudinal Dataset Analysis
# Period: 2018-2023
# Records: 500,000+ observations
# Variables: 200+ features across multiple domains

## Dataset Metadata
- Source: Multi-institutional collaboration
- Collection Method: Automated sensors and manual surveys
- Quality Assurance: Triple validation process
- Missing Data: <2% across all variables

## Temporal Trends Analysis
### Q1 2018 - Q4 2018
- Baseline establishment period
- Initial measurement protocols implemented
- Data collection infrastructure deployed

### Q1 2019 - Q4 2019  
- Significant trend reversal observed in Q2
- Correlation with Policy Change #147 implementation
- 15% increase in primary metrics

### Q1 2020 - Q4 2020
- COVID-19 impact period
- Major trend reversal in Q4
- Market disruption effects documented

### Q1 2021 - Q4 2021
- Recovery and adaptation phase
- New baseline establishment
- Technology adoption acceleration

### Q1 2022 - Q4 2022
- Stabilization period
- Trend reversal in Q3 due to technological shifts
- Long-term pattern emergence

### Q1 2023 - Q4 2023
- Current observation period
- Predictive modeling validation
- Future trend projection

## External Factors Correlation Matrix
- Policy Changes: 0.73 correlation with primary metrics
- Market Conditions: 0.65 correlation with volatility measures
- Technology Adoption: 0.82 correlation with efficiency indicators
- Environmental Factors: 0.45 correlation with operational metrics
- Regulatory Changes: 0.58 correlation with compliance measures
"""
        
        # Expand to target length with additional detailed statistics
        while len(data_content) < target_length:
            additional_content = "\\n\\n## Additional Statistical Analysis\\nDetailed breakdown of correlations, regression analyses, and predictive modeling results reveal complex interdependencies and emergent patterns across multiple temporal scales and dimensional perspectives."
            data_content += additional_content
        
        return data_content[:target_length]
    
    def prepare_input(self, sample: Dict) -> str:
        """Prepare input text for LongBench-V2 evaluation."""
        task = sample["task"]
        
        if task == "single_doc_qa":
            context = sample["context"]
            question = sample["question"]
            options = sample.get("options", [])
            
            if options:
                option_text = "\\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                prompt = f"""Read the following document carefully and answer the multiple-choice question.

Document:
{context}

Question: {question}

Options:
{option_text}

Answer:"""
            else:
                prompt = f"""Read the following document carefully and answer the question.

Document:
{context}

Question: {question}

Answer:"""
        
        elif task == "multi_doc_qa":
            contexts = sample["contexts"]
            question = sample["question"] 
            options = sample.get("options", [])
            
            doc_text = ""
            for i, context in enumerate(contexts):
                doc_text += f"\\n\\n## Document {i+1}:\\n{context}"
            
            if options:
                option_text = "\\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                prompt = f"""Read the following documents carefully and answer the question by synthesizing information across all documents.

{doc_text}

Question: {question}

Options:
{option_text}

Answer:"""
            else:
                prompt = f"""Read the following documents carefully and answer the question by synthesizing information across all documents.

{doc_text}

Question: {question}

Answer:"""
        
        elif task == "long_icl":
            context = sample["context"]
            question = sample["question"]
            problem = sample.get("problem", "")
            
            prompt = f"""{context}

{question}

{problem}

Following the established pattern, provide your solution:"""
        
        else:
            # Generic preparation for other tasks
            context = sample.get("context", "")
            question = sample.get("question", "")
            
            prompt = f"""Based on the following information, answer the question.

Information:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def extract_answer(self, response: str, sample: Dict) -> str:
        """Extract answer from model response."""
        # For multiple choice, try to extract the letter
        if "options" in sample:
            for line in response.strip().split('\\n'):
                line = line.strip()
                if line and line[0] in 'ABCD':
                    return line
        
        # Otherwise return first meaningful line
        lines = response.strip().split('\\n')
        for line in lines:
            if line.strip():
                return line.strip()
        
        return response.strip()
    
    def compute_score(self, prediction: str, ground_truth: str, sample: Dict) -> float:
        """Compute score for LongBench-V2 prediction."""
        prediction = prediction.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # For multiple choice questions
        if "options" in sample:
            # Extract letter from prediction if present
            pred_letter = None
            if prediction and prediction[0] in 'abcd':
                pred_letter = prediction[0]
            
            # Extract letter from ground truth
            gt_letter = None  
            if ground_truth and ground_truth[0] in 'abcd':
                gt_letter = ground_truth[0]
            
            if pred_letter and gt_letter:
                return 1.0 if pred_letter == gt_letter else 0.0
        
        # For open-ended questions, use sophisticated matching
        if ground_truth in prediction:
            return 1.0
        
        # Word-level F1 with higher threshold for LongBench-V2
        pred_words = set(prediction.split())
        gt_words = set(ground_truth.split())
        
        if not gt_words or not pred_words:
            return 0.0
        
        overlap = len(pred_words.intersection(gt_words))
        precision = overlap / len(pred_words)
        recall = overlap / len(gt_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        # Higher threshold for LongBench-V2 due to increased difficulty
        return f1 if f1 > 0.3 else 0.0