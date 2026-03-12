"""
Inference Script for Chromosome Classification

Predicts chromosome classes for a single image or a directory of unlabeled images.
"""

import argparse
import os
import sys
import torch
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.classifier import ChromosomeClassifier
from src import config


def run_single_image(classifier: ChromosomeClassifier, image_path: str) -> None:
    """Run inference on a single image and print result in demo-friendly format."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)
    pred_class, confidence = classifier.predict_single(img)
    print(f"Chromosome {pred_class} (confidence {confidence:.2f})")


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Predict chromosome class for image(s).",
        epilog="Examples:\n  python scripts/infer.py --image sample.png\n  python scripts/infer.py --dir data/unlabeled",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single chromosome image (demo: prints 'Chromosome N (confidence 0.xx)')",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/unlabeled",
        help="Directory of images to predict (default: data/unlabeled). Ignored if --image is set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path for batch predictions (default: results/predictions.csv). Only used with --dir.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CHROMOSOME CLASSIFICATION - INFERENCE")
    print("=" * 80)

    print("\nLoading trained model and feature extractor...")
    checkpoint = torch.load(config.SEMI_SUPERVISED_MODEL_PATH, map_location="cpu")
    if "metadata" not in checkpoint or "input_dim" not in checkpoint["metadata"]:
        print("Error: Model checkpoint missing metadata.input_dim.", file=sys.stderr)
        sys.exit(1)
    input_dim = checkpoint["metadata"]["input_dim"]

    classifier = ChromosomeClassifier(
        model_path=config.SEMI_SUPERVISED_MODEL_PATH,
        pca_model_path=config.PCA_MODEL_PATH,
        num_classes=config.NUM_CLASSES,
        hidden_dims=config.HIDDEN_DIMS,
        input_dim=input_dim,
    )

    if args.image is not None:
        run_single_image(classifier, args.image)
        return

    out_path = args.output or os.path.join(config.RESULTS_DIR, "predictions.csv")
    print(f"\nPredicting on images in: {args.dir}")
    results_df = classifier.predict_batch_from_directory(
        image_dir=args.dir,
        output_path=out_path,
    )
    if len(results_df) > 0:
        print("\nPredictions (first 10):")
        print(results_df.head(10))
        print(f"\nTotal predictions: {len(results_df)}")
        print("\nClass distribution:")
        print(results_df["predicted_class"].value_counts())
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)
    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()

