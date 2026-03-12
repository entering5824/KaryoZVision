"""
Training Script for Semi-Supervised Chromosome Classification

Main script to train supervised and semi-supervised models.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.loader import load_labeled_data, load_unlabeled_data
from src.datasets.splitter import split_data, create_unlabeled_data
from src.features.pca import FeatureExtractor
from src.training.supervised import train_supervised
from src.training.semi_supervised import self_training_loop
from src import config
from src.utils.model_utils import save_model
from src.evaluation.visualization import plot_training_curves, plot_pca_variance


def main():
    """Main training pipeline."""
    
    print("=" * 80)
    print("SEMI-SUPERVISED CHROMOSOME CLASSIFICATION - TRAINING")
    print("=" * 80)
    
    # Set device
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # STEP 1: Data Preparation
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80)
    
    # Load labeled data (D_L) từ các folder 1-22, X, Y trong data/
    images, labels = load_labeled_data(config.LABELED_DATA_DIR)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        images, labels, config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO, config.RANDOM_STATE
    )
    
    # Load unlabeled data (D_U)
    # Priority: 1) Load from unlabeled folder if exists, 2) Create from train split
    X_unlabeled = load_unlabeled_data(config.UNLABELED_DATA_DIR)
    
    if len(X_unlabeled) == 0:
        # If no unlabeled folder exists, create unlabeled data from training split
        print("\nNo unlabeled data folder found. Creating unlabeled data from training split...")
        X_labeled, y_labeled, X_unlabeled, y_unlabeled_true = create_unlabeled_data(
            X_train, y_train, config.UNLABELED_RATIO, config.RANDOM_STATE
        )
    else:
        # Use all training data as labeled, unlabeled from folder
        print(f"\nLoaded {len(X_unlabeled)} unlabeled images from {config.UNLABELED_DATA_DIR}")
        X_labeled = X_train
        y_labeled = y_train
        y_unlabeled_true = None  # True labels unknown for external unlabeled data
    
    class_names = [str(i) for i in range(1, 23)] + ['X', 'Y']
    num_classes = len(class_names)
    
    print(f"\nDataset Summary:")
    print(f"  D_L_train: {len(X_labeled)} images")
    print(f"  D_L_val: {len(X_val)} images")
    print(f"  D_L_test: {len(X_test)} images")
    print(f"  D_U: {len(X_unlabeled)} images")
    
    # STEP 2: Feature Extraction
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE EXTRACTION")
    print("=" * 80)
    
    extractor = FeatureExtractor(pca_variance_threshold=config.PCA_VARIANCE_THRESHOLD)
    
    # Extract features from training set (FIT PCA and scaler)
    print("\nExtracting features from training set...")
    features_train = extractor.get_combined_features(
        X_labeled,
        fit_pca=True,
        fit_scaler=True,
        extended=config.USE_EXTENDED_FEATURES,
        include_texture=config.INCLUDE_TEXTURE_FEATURES,
        include_histogram=config.INCLUDE_HISTOGRAM_FEATURES
    )
    
    # Extract features from validation set (TRANSFORM only)
    features_val = extractor.get_combined_features(
        X_val,
        fit_pca=False,
        fit_scaler=False,
        extended=config.USE_EXTENDED_FEATURES,
        include_texture=config.INCLUDE_TEXTURE_FEATURES,
        include_histogram=config.INCLUDE_HISTOGRAM_FEATURES
    )
    
    # Extract features from unlabeled set
    features_unlabeled = extractor.get_combined_features(
        X_unlabeled,
        fit_pca=False,
        fit_scaler=False,
        extended=config.USE_EXTENDED_FEATURES,
        include_texture=config.INCLUDE_TEXTURE_FEATURES,
        include_histogram=config.INCLUDE_HISTOGRAM_FEATURES
    )
    
    print(f"\nFeature dimensions:")
    print(f"  Training: {features_train.shape}")
    print(f"  Validation: {features_val.shape}")
    print(f"  Unlabeled: {features_unlabeled.shape}")
    
    # Save PCA model
    Path(config.PCA_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    extractor.save(config.PCA_MODEL_PATH)
    
    # Plot PCA variance
    if extractor.pca is not None:
        plot_pca_variance(
            extractor.pca.explained_variance_ratio_,
            save_path=config.PCA_VARIANCE_PATH
        )
    
    # STEP 3: Supervised Baseline Training
    print("\n" + "=" * 80)
    print("STEP 3: SUPERVISED BASELINE TRAINING")
    print("=" * 80)
    
    # Supervised initialization: 50 epochs as recommended in the plan
    # This provides a good baseline before semi-supervised learning
    supervised_model, supervised_history = train_supervised(
        features_train, np.array(y_labeled),
        features_val, np.array(y_val),
        num_classes=num_classes,
        hidden_dims=config.HIDDEN_DIMS,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        num_epochs=50,  # Initial supervised training (50 epochs as per plan)
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        weight_decay=1e-4,
        use_lr_scheduler=True,
        device=device,
        verbose=config.VERBOSE
    )
    
    # Save supervised model
    save_model(
        supervised_model,
        config.SUPERVISED_MODEL_PATH,
        metadata={'input_dim': features_train.shape[1], 'num_classes': num_classes}
    )
    
    # STEP 4: Semi-Supervised Self-Training
    print("\n" + "=" * 80)
    print("STEP 4: SEMI-SUPERVISED SELF-TRAINING")
    print("=" * 80)
    
    final_model, semi_supervised_history = self_training_loop(
        features_train, np.array(y_labeled),
        features_val, np.array(y_val),
        features_unlabeled,
        num_classes=num_classes,
        hidden_dims=config.HIDDEN_DIMS,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        epochs_per_iteration=config.EPOCHS_PER_ITERATION,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        max_iterations=config.MAX_ITERATIONS,
        confidence_thresholds=config.CONFIDENCE_THRESHOLDS,
        top_k_per_class=config.TOP_K_PER_CLASS,
        min_total_epochs=config.MIN_TOTAL_EPOCHS,
        device=device,
        verbose=config.VERBOSE
    )
    
    # Save final model
    save_model(
        final_model,
        config.SEMI_SUPERVISED_MODEL_PATH,
        metadata={'input_dim': features_train.shape[1], 'num_classes': num_classes}
    )
    
    # Plot training curves
    plot_training_curves(
        supervised_history,
        semi_supervised_history,
        save_path=config.TRAINING_CURVES_PATH
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Models saved to: {config.MODELS_DIR}/")
    print(f"Results saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()

