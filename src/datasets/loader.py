"""
Data Loading Module for Chromosome Classification

Loads labeled and unlabeled chromosome images from directory structure.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os

from .. import config


def load_labeled_data(data_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load labeled chromosome images from directory structure.
    
    Expected structure:
        data_dir/
            1/          # Chromosome 1
                img1.png
                img2.png
                ...
            2/          # Chromosome 2
                ...
            22/         # Chromosome 22
            X/          # Chromosome X
            Y/          # Chromosome Y
    
    Args:
        data_dir: Path to labeled data directory (default: config.LABELED_DATA_DIR)
    
    Returns:
        Tuple of (images, labels) where:
            - images: List of grayscale images as numpy arrays
            - labels: List of integer labels (0-22)
    """
    if data_dir is None:
        data_dir = config.LABELED_DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    images = []
    labels = []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP'}
    
    # Load images from each class directory
    for class_name in config.CLASS_NAMES:
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Get label index
        label_idx = config.CLASS_TO_IDX[class_name]
        
        # Load all images in this directory
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(f'*{ext}')))
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"Loading {len(image_files)} images from class {class_name} (label {label_idx})...")
        
        for img_path in sorted(image_files):
            # Read image as grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            images.append(img)
            labels.append(label_idx)
    
    print(f"\nLoaded {len(images)} labeled images from {len(set(labels))} classes")
    
    if len(images) == 0:
        raise ValueError(f"No images loaded from {data_dir}")
    
    return images, labels


def load_unlabeled_data(data_dir: Optional[str] = None) -> List[np.ndarray]:
    """
    Load unlabeled chromosome images from directory.
    
    Args:
        data_dir: Path to unlabeled data directory (default: config.UNLABELED_DATA_DIR)
    
    Returns:
        List of grayscale images as numpy arrays
    """
    if data_dir is None:
        data_dir = config.UNLABELED_DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"Warning: Unlabeled data directory does not exist: {data_dir}")
        return []
    
    images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP'}
    
    # Load all images from directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_dir.glob(f'*{ext}')))
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {data_dir}")
        return []
    
    print(f"Loading {len(image_files)} unlabeled images...")
    
    for img_path in sorted(image_files):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        images.append(img)
    
    print(f"Loaded {len(images)} unlabeled images")
    
    return images

