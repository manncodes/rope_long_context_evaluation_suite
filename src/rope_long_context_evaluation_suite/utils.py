"""Utility functions for the RoPE evaluation suite."""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration object
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)


def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any numpy types to Python types for JSON serialization
    results = convert_numpy_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(results_path: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Results dictionary
    """
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices.
    
    Returns:
        Device information dictionary
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "device_count": 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        device_info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info["devices"].append({
                "id": i,
                "name": device_props.name,
                "memory_total": device_props.total_memory,
                "memory_reserved": torch.cuda.memory_reserved(i),
                "memory_allocated": torch.cuda.memory_allocated(i),
            })
    
    return device_info


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_time(seconds: float) -> str:
    """Format time in human readable format.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using given tokenizer.
    
    Args:
        text: Text to count tokens for
        tokenizer: Tokenizer to use
        
    Returns:
        Number of tokens
    """
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback to rough word-based estimation
        return len(text.split()) * 1.3


class Config:
    """Configuration wrapper class."""
    
    def __init__(self, config: Union[DictConfig, Dict, str, Path]):
        """Initialize configuration.
        
        Args:
            config: Configuration object, dict, or path to config file
        """
        if isinstance(config, (str, Path)):
            self._config = load_config(config)
        elif isinstance(config, dict):
            self._config = OmegaConf.create(config)
        else:
            self._config = config
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration attribute."""
        return getattr(self._config, name)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration item."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration item."""
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return OmegaConf.select(self._config, key, default=default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        save_config(self._config, output_path)