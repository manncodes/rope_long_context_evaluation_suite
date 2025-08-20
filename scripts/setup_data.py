#!/usr/bin/env python3
"""Setup script to download and prepare benchmark datasets."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

import requests
from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def setup_longbench_data(data_dir: Path) -> None:
    """Download and setup LongBench dataset."""
    logger.info("Setting up LongBench dataset...")
    
    longbench_dir = data_dir / "longbench"
    longbench_dir.mkdir(parents=True, exist_ok=True)
    
    # Key LongBench tasks to download
    key_tasks = [
        'narrativeqa',        # Reading comprehension
        'qasper',            # Scientific QA  
        'multifieldqa_en',   # Multi-domain QA
        'hotpotqa',          # Multi-hop reasoning
        '2wikimqa',          # Multi-hop QA
        'qmsum',             # Meeting summarization
        'trec',              # Question classification
        'triviaqa',          # Knowledge QA
        'samsum',            # Dialogue summary
        'passage_retrieval_en',  # Information retrieval
        'passage_count',     # Counting task
        'lcc',               # Code completion
    ]
    
    logger.info(f"Downloading {len(key_tasks)} LongBench tasks...")
    
    success_count = 0
    for task in key_tasks:
        try:
            logger.info(f"  Downloading {task}...")
            dataset = load_dataset("THUDM/LongBench", task, cache_dir=str(longbench_dir), trust_remote_code=True)
            
            # Save to JSONL format for easier access
            task_file = longbench_dir / f"{task}.jsonl"
            
            # Convert dataset to JSONL
            with open(task_file, 'w', encoding='utf-8') as f:
                for split in dataset:
                    for example in dataset[split]:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            logger.info(f"  ✅ {task} saved to {task_file}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"  ❌ Failed to download {task}: {e}")
    
    logger.info(f"LongBench setup completed: {success_count}/{len(key_tasks)} tasks downloaded to {longbench_dir}")


def setup_longbench_v2_data(data_dir: Path) -> None:
    """Download and setup LongBench-V2 dataset."""
    logger.info("Setting up LongBench-V2 dataset...")
    
    longbench_v2_dir = data_dir / "longbench_v2"
    longbench_v2_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for LongBench-V2 (replace with actual URLs when available)
    urls = [
        # "https://github.com/THUDM/LongBench/releases/download/v2.0/longbench_v2.jsonl",
    ]
    
    if not urls:
        logger.warning("LongBench-V2 URLs not yet available. Please check the official repository.")
        return
    
    for url in urls:
        filename = Path(url).name
        output_path = longbench_v2_dir / filename
        
        if not output_path.exists():
            try:
                download_file(url, output_path)
                logger.info(f"Downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")


def setup_ruler_data(data_dir: Path) -> None:
    """Download and setup RULER benchmark data."""
    logger.info("Setting up RULER dataset...")
    
    ruler_dir = data_dir / "ruler"
    ruler_dir.mkdir(parents=True, exist_ok=True)
    
    # RULER is typically generated synthetically
    # Create placeholder files
    placeholder_file = ruler_dir / "README.md"
    with open(placeholder_file, 'w') as f:
        f.write("""# RULER Dataset

The RULER benchmark generates synthetic data on-demand.
No pre-downloaded files are required.

The benchmark will create the following synthetic tasks:
- Needle-in-a-haystack variants
- Variable tracking
- Common words extraction  
- Frequent words extraction
- Question answering

For more information, see the original paper:
RULER: What's the Real Context Size of Your Long-Context Language Models?
""")
    
    logger.info(f"RULER setup completed at {ruler_dir}")


def setup_niah_data(data_dir: Path) -> None:
    """Setup NIAH (Needle in a Haystack) data."""
    logger.info("Setting up NIAH data...")
    
    niah_dir = data_dir / "niah"
    niah_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default needles file
    needles_file = niah_dir / "needles.txt"
    default_needles = [
        "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
        "The secret to happiness is finding joy in the small moments of everyday life.",
        "To solve complex problems, break them down into smaller, manageable pieces.",
        "The most efficient way to learn a new skill is through deliberate practice and repetition.",
        "Innovation happens at the intersection of different fields and perspectives.",
        "The key to successful teamwork is clear communication and mutual respect.",
        "Creativity flourishes when we step outside our comfort zones and embrace uncertainty.",
        "The most valuable skill in the digital age is the ability to learn and adapt quickly.",
        "True leadership is about empowering others to achieve their full potential.",
        "The foundation of any strong relationship is trust, honesty, and empathy.",
    ]
    
    with open(needles_file, 'w') as f:
        for needle in default_needles:
            f.write(needle + '\\n')
    
    # Create default haystack file
    haystack_file = niah_dir / "haystack.txt"
    default_haystack = """
The history of artificial intelligence began in antiquity with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The study of mechanical or "formal" reasoning began with philosophers and mathematicians in antiquity. The study of mathematical logic led directly to Alan Turing's theory of computation, which suggested that a machine, by shuffling symbols as simple as "0" and "1", could simulate any conceivable act of mathematical deduction.

In the 1950s, a generation of scientists, mathematicians, and philosophers had the concept of artificial intelligence (or AI) intellectually formulated. One such person was Alan Turing, a young British polymath who explored the mathematical possibility of artificial intelligence. Turing suggested that humans use available information as well as reason in order to solve problems and make decisions, so why can't machines do the same thing? This was the logical framework of his 1950 paper, Computing Machinery and Intelligence, in which he discussed how to build intelligent machines and how to test their intelligence.

The field of artificial intelligence research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

Eventually, it became obvious that commercial developers and researchers had been overly optimistic. Getting a computer to play checkers was relatively easy, but getting one that could understand and respond appropriately to human language proved much more difficult. By 1974, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter."

The development of personal computers in the 1980s and the subsequent rise of the Internet in the 1990s created new opportunities for AI research and development. Machine learning algorithms became more sophisticated, and researchers began to explore neural networks and other approaches to artificial intelligence.

The 21st century has seen unprecedented advances in artificial intelligence, driven by improvements in computing power, the availability of large datasets, and breakthroughs in deep learning. Today, AI systems can perform tasks that were once thought to be the exclusive domain of human intelligence, from recognizing images and understanding natural language to playing complex games and driving cars.
"""
    
    with open(haystack_file, 'w') as f:
        f.write(default_haystack.strip())
    
    logger.info(f"NIAH data setup completed at {niah_dir}")


def create_directory_structure(data_dir: Path) -> None:
    """Create the basic directory structure."""
    subdirs = [
        "niah",
        "ruler", 
        "longbench",
        "longbench_v2",
    ]
    
    for subdir in subdirs:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files
        with open(data_dir / subdir / ".gitkeep", 'w') as f:
            f.write("")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup benchmark datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store datasets (default: data)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        choices=["niah", "ruler", "longbench", "longbench_v2", "all"],
        default=["all"],
        help="Which benchmarks to setup (default: all)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Setting up data in {data_dir}")
    
    # Create directory structure
    create_directory_structure(data_dir)
    
    # Setup benchmarks
    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["niah", "ruler", "longbench", "longbench_v2"]
    
    if "niah" in benchmarks:
        setup_niah_data(data_dir)
    
    if "ruler" in benchmarks:
        setup_ruler_data(data_dir)
    
    if "longbench" in benchmarks:
        setup_longbench_data(data_dir)
    
    if "longbench_v2" in benchmarks:
        setup_longbench_v2_data(data_dir)
    
    logger.info("Data setup completed!")
    logger.info(f"Dataset files are available in: {data_dir}")
    logger.info("You can now run evaluations using the configured benchmarks.")


if __name__ == "__main__":
    main()