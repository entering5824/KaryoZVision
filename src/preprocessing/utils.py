"""
Utility functions for I/O, grid mapping, and logging.
Shared utilities across all parts of the pipeline.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import cv2
import numpy as np


def get_image_files(input_path: str) -> List[str]:
    """
    Get list of image files from input path (file or directory).
    
    Args:
        input_path: Path to image file or directory
    
    Returns:
        List of image file paths
    """
    path = Path(input_path)
    
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(path.glob(f'*{ext}'))
        return sorted([str(f) for f in image_files])
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save dictionary to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(input_path: str) -> Dict[str, Any]:
    """Load JSON file to dictionary."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_output_dirs(base_dir: Path, chromosome_numbers: List[str]) -> None:
    """
    Create output directories for chromosomes (1-22, X, Y).
    
    Args:
        base_dir: Base output directory
        chromosome_numbers: List of chromosome identifiers (e.g., ['1', '2', ..., '22', 'X', 'Y'])
    """
    for chrom in chromosome_numbers:
        (base_dir / chrom).mkdir(parents=True, exist_ok=True)


def get_chromosome_numbers() -> List[str]:
    """Get list of chromosome numbers: 1-22, X, Y."""
    return [str(i) for i in range(1, 23)] + ['X', 'Y']

