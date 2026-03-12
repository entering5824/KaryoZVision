"""
Evaluation Script for Chromosome Classification

Evaluates trained models on test set and generates visualizations.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.loader import load_labeled_data
from src.datasets.splitter import split_data
from src.features.pca import FeatureExtractor
from src.models.mlp import ChromosomeMLP
from src import config
from src.evaluation.metrics import calculate_metrics, compare_models, print_comparison_table
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_per_class_performance
)
from src.utils.model_utils import get_device


def main():
    """Main evaluation pipeline."""
    
    print("=" * 80)
    print("CHROMOSOME CLASSIFICATION - EVALUATION")
    print("=" * 80)
    
    device = get_device()
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU (GPU not available or USE_GPU=False)")
    
    # Load data
    print("\nLoading data...")
    images, labels = load_labeled_data(config.DATA_DIR)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        images, labels, config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO, config.RANDOM_STATE
    )
    
    class_names = [str(i) for i in range(1, 23)] + ['X', 'Y']
    num_classes = len(class_names)
    
    # Load feature extractor
    print("\nLoading feature extractor...")
    extractor = FeatureExtractor()
    extractor.load(config.PCA_MODEL_PATH)
    
    # Extract test features
    print("Extracting test features...")
    features_test = extractor.get_combined_features(
        X_test,
        fit_pca=False,
        fit_scaler=False,
        extended=config.USE_EXTENDED_FEATURES,
        include_texture=config.INCLUDE_TEXTURE_FEATURES,
        include_histogram=config.INCLUDE_HISTOGRAM_FEATURES
    )
    
    # Load models
    print("\nLoading models...")
    input_dim = features_test.shape[1]
    
    # Supervised model
    supervised_checkpoint = torch.load(config.SUPERVISED_MODEL_PATH, map_location=device)
    supervised_model = ChromosomeMLP(input_dim, num_classes, config.HIDDEN_DIMS).to(device)
    supervised_model.load_state_dict(supervised_checkpoint['model_state_dict'])
    
    # Semi-supervised model
    semi_checkpoint = torch.load(config.SEMI_SUPERVISED_MODEL_PATH, map_location=device)
    semi_model = ChromosomeMLP(input_dim, num_classes, config.HIDDEN_DIMS).to(device)
    semi_model.load_state_dict(semi_checkpoint['model_state_dict'])
    
    # Evaluate supervised model
    print("\n" + "=" * 80)
    print("EVALUATING SUPERVISED MODEL")
    print("=" * 80)
    supervised_results = calculate_metrics(
        supervised_model,
        features_test,
        np.array(y_test),
        class_names,
        device=device
    )
    
    print(f"\nSupervised Model Results:")
    print(f"  Accuracy: {supervised_results['accuracy']*100:.2f}%")
    print(f"  Precision: {supervised_results['precision']*100:.2f}%")
    print(f"  Recall: {supervised_results['recall']*100:.2f}%")
    print(f"  F1-Score (Weighted): {supervised_results['f1_score']*100:.2f}%")
    print(f"  F1-Score (Macro): {supervised_results.get('f1_macro', 0.0)*100:.2f}%")
    
    # Evaluate semi-supervised model
    print("\n" + "=" * 80)
    print("EVALUATING SEMI-SUPERVISED MODEL")
    print("=" * 80)
    semi_results = calculate_metrics(
        semi_model,
        features_test,
        np.array(y_test),
        class_names,
        device=device
    )
    
    print(f"\nSemi-Supervised Model Results:")
    print(f"  Accuracy: {semi_results['accuracy']*100:.2f}%")
    print(f"  Precision: {semi_results['precision']*100:.2f}%")
    print(f"  Recall: {semi_results['recall']*100:.2f}%")
    print(f"  F1-Score (Weighted): {semi_results['f1_score']*100:.2f}%")
    print(f"  F1-Score (Macro): {semi_results.get('f1_macro', 0.0)*100:.2f}%")
    
    # Compare models
    comparison = compare_models(
        supervised_results,
        semi_results,
        save_path=config.RESULTS_JSON_PATH
    )
    print_comparison_table(comparison)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrices
    cm_supervised = np.array(supervised_results['confusion_matrix'])
    plot_confusion_matrix(
        cm_supervised,
        class_names,
        save_path=config.CONFUSION_MATRIX_PATH.replace('.png', '_supervised.png')
    )
    
    cm_semi = np.array(semi_results['confusion_matrix'])
    plot_confusion_matrix(
        cm_semi,
        class_names,
        save_path=config.CONFUSION_MATRIX_PATH.replace('.png', '_semi_supervised.png')
    )
    
    # Per-class performance
    plot_per_class_performance(
        supervised_results,
        class_names,
        save_path=config.PER_CLASS_PERFORMANCE_PATH.replace('.png', '_supervised.png')
    )
    
    plot_per_class_performance(
        semi_results,
        class_names,
        save_path=config.PER_CLASS_PERFORMANCE_PATH.replace('.png', '_semi_supervised.png')
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()

