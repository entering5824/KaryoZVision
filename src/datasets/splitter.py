"""
Data Splitting Module for Semi-Supervised Chromosome Classification

Splits labeled data into train/val/test sets with stratification.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def split_data(
    images: List[np.ndarray],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[
    Tuple[List[np.ndarray], List[int]],  # train
    Tuple[List[np.ndarray], List[int]],  # val
    Tuple[List[np.ndarray], List[int]]   # test
]:
    """
    Split labeled data into train, validation, and test sets.
    
    For small datasets, automatically handles stratification constraints.
    
    Args:
        images: List of images
        labels: List of labels
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data) where each is (images, labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    labels_array = np.array(labels)
    unique_classes = np.unique(labels_array)
    n_classes = len(unique_classes)
    n_samples = len(images)
    
    # Check if stratified splitting is feasible
    # Stratified split requires at least 2 samples per class (1 in each split)
    # and for 3-way split, we need at least 3 samples per class ideally
    min_samples_per_class = np.min([np.sum(labels_array == cls) for cls in unique_classes])
    test_size = int(n_samples * test_ratio)
    
    # Determine if we can use stratified splitting
    use_stratify = True
    if test_size < n_classes:
        print(f"Warning: Test set size ({test_size}) < number of classes ({n_classes})")
        print(f"  Using non-stratified splitting for small dataset")
        use_stratify = False
    elif min_samples_per_class < 3:
        print(f"Warning: Some classes have < 3 samples (min: {min_samples_per_class})")
        print(f"  Using non-stratified splitting to avoid stratification constraints")
        use_stratify = False
    
    # First split: separate test set
    if use_stratify:
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_ratio,
            random_state=random_state,
            stratify=labels
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_ratio,
            random_state=random_state,
            stratify=None
        )
    
    # Second split: separate train and validation from remaining data
    # Adjust val_ratio relative to the remaining data
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    
    if use_stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=None
        )
    
    print(f"\nData splitting:")
    print(f"  Train: {len(X_train)} images ({len(X_train)/len(images)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} images ({len(X_val)/len(images)*100:.1f}%)")
    print(f"  Test: {len(X_test)} images ({len(X_test)/len(images)*100:.1f}%)")
    
    # Show class distribution
    train_dist = Counter(y_train)
    val_dist = Counter(y_val)
    test_dist = Counter(y_test)
    print(f"\nClass distribution:")
    print(f"  Train: {len(train_dist)} classes represented")
    print(f"  Validation: {len(val_dist)} classes represented")
    print(f"  Test: {len(test_dist)} classes represented")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_unlabeled_data(
    train_images: List[np.ndarray],
    train_labels: List[int],
    unlabeled_ratio: float = 0.35,
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Create unlabeled dataset (D_U) by removing labels from a subset of training data.
    
    This simulates the semi-supervised learning setting where we have unlabeled data.
    The original labels are kept separately for evaluation purposes only.
    
    IMPORTANT: This is a simulation approach. In a real scenario, D_U would come
    from external unlabeled chromosome images. We use this approach to evaluate
    the effectiveness of semi-supervised learning while preserving a held-out test set.
    
    Args:
        train_images: Training images
        train_labels: Training labels
        unlabeled_ratio: Proportion of training data to convert to unlabeled (default: 0.35)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (remaining_train_images, remaining_train_labels, unlabeled_images, true_unlabeled_labels)
        where true_unlabeled_labels are kept for evaluation but NOT used in training
    """
    # Check if stratified splitting is feasible
    labels_array = np.array(train_labels)
    unique_classes = np.unique(labels_array)
    n_classes = len(unique_classes)
    n_samples = len(train_images)
    unlabeled_size = int(n_samples * unlabeled_ratio)
    
    use_stratify = True
    if unlabeled_size < n_classes or (n_samples - unlabeled_size) < n_classes:
        print(f"Warning: Cannot use stratified splitting for unlabeled data creation")
        print(f"  Using non-stratified splitting")
        use_stratify = False
    
    # Split training data: keep some labeled, convert some to unlabeled
    if use_stratify:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
            train_images, train_labels,
            test_size=unlabeled_ratio,
            random_state=random_state,
            stratify=train_labels
        )
    else:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
            train_images, train_labels,
            test_size=unlabeled_ratio,
            random_state=random_state,
            stratify=None
        )
    
    print(f"\nUnlabeled data creation:")
    print(f"  Remaining labeled (D_L_train): {len(X_labeled)} images")
    print(f"  Unlabeled (D_U): {len(X_unlabeled)} images ({unlabeled_ratio*100:.1f}% of original train)")
    print(f"  Note: D_U > D_L_train to demonstrate semi-supervised benefit")
    
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled_true

